"""
Eval judge: labels (query, candidate) pairs with an independent model.

These labels are the ground truth used to score P@5/NDCG for BOTH
pipelines (see eval.py / run_eval.py). They must never be read by
pipelines.py or used to drive the Pipeline 2 filter — that's what
llm_judge.py's system labels (system_labels.json) are for. Keeping the
judge that grades the filter separate from the judge that built it is
what makes Pipeline 2's P@5 gain a real result instead of a tautology.

The eval judge is a DIFFERENT model from the system judge (Sonnet drives
the filter; this judge is Haiku), so cross-model agreement / Cohen's kappa
measure how much two independent judges concur. The model name is baked into
the cache key, so swapping the eval model (e.g. back to GPT) never collides
with existing labels — set JUDGE_PROVIDER=openai + OPENAI_JUDGE_MODEL to do so.
"""

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
LABELS_PATH = RESULTS_DIR / "eval_labels.json"

# The eval judge provider/model. Default: Claude Haiku via the Anthropic key.
# Set JUDGE_PROVIDER=openai (+ OPENAI_JUDGE_MODEL, OPENAI_API_KEY) to use GPT.
PROVIDER = os.environ.get("JUDGE_PROVIDER", "anthropic")
if PROVIDER == "openai":
    MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
else:
    MODEL = os.environ.get("ANTHROPIC_JUDGE_MODEL", "claude-haiku-4-5-20251001")
PROMPT_VERSION = "v1"

# Parallel API calls. Labeling is I/O-bound (network latency dominates), so a
# thread pool cuts wall-clock time roughly by MAX_WORKERS with no CPU cost.
# Keep this at or below the provider tier's requests-per-minute headroom — on a
# low tier a high worker count just produces a 429 storm.
MAX_WORKERS = int(os.environ.get("JUDGE_MAX_WORKERS", "8"))

# Per-request retry budget for transient rate limits. We handle backoff ourselves
# and disable the client's blind auto-retries, so a burst of 429s waits and
# recovers instead of silently burning a daily request budget.
MAX_RETRIES = int(os.environ.get("JUDGE_MAX_RETRIES", "6"))

JUDGE_PROMPT = """
You are evaluating search relevance for a streaming platform.
The platform has a large catalog including many niche and long-tail titles.

Query: "{query}"
Title: "{title}"
Release Year: {release_year}
Genres: {genres}
Plot: "{plot}"

Is this title relevant to the search query?

Relevant: The title meaningfully satisfies the search intent.
A user who typed this query would be satisfied finding this title.

Irrelevant: The title has no meaningful connection to this query.
A user who typed this query would be confused or disappointed
finding this title in top results.

Return only valid JSON with no other text:
{{"label": "Relevant" | "Irrelevant", "reasoning": "one sentence"}}
"""


def cache_key(query, title):
    return f"{MODEL}|||{PROMPT_VERSION}|||{query}|||{title}"


def _make_client():
    if PROVIDER == "openai":
        from openai import OpenAI
        # max_retries=0: we own retry/backoff below.
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0)


def _rate_limit_errors():
    if PROVIDER == "openai":
        from openai import RateLimitError
        return (RateLimitError,)
    import anthropic
    return (anthropic.RateLimitError,)


def _raw_completion(client, prompt):
    if PROVIDER == "openai":
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    response = client.messages.create(
        model=MODEL,
        max_tokens=128,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _call_judge(client, prompt):
    # Retry transient per-minute rate limits with exponential backoff + jitter.
    # A daily-cap 429 (requests-per-day) won't clear within the run, so we stop
    # retrying it and let the caller skip the pair (uncached, retried next run).
    rate_errs = _rate_limit_errors()
    for attempt in range(MAX_RETRIES):
        try:
            raw = _raw_completion(client, prompt)
            break
        except rate_errs as e:
            if "per day" in str(e).lower() or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _load_cache():
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return {}


def _prompt_for(query, candidate):
    genres_str = ", ".join(candidate["genres"]) if candidate["genres"] else "Unknown"
    plot = candidate["overview"][:300]
    return JUDGE_PROMPT.format(
        query=query,
        title=candidate["title"],
        release_year=candidate.get("release_year", "Unknown"),
        genres=genres_str,
        plot=plot,
    )


def label_pairs(query_candidate_pairs):
    """
    query_candidate_pairs: list of (query, query_type, candidate_dict)
    Returns: dict keyed by cache_key with label info

    API errors are NOT cached — a failed call is skipped and retried on the
    next run, rather than being recorded as a fabricated "Irrelevant" label.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    cache = _load_cache()

    loaded = sum(
        1 for (q, _, c) in query_candidate_pairs
        if cache_key(q, c["title"]) in cache
    )
    print(f"Running eval judge ({PROVIDER}:{MODEL})...      loaded {loaded} labels from cache")

    client = _make_client()

    # De-duplicate the work: many (query, title) pairs repeat across the pair list.
    todo = {}
    for query, _, candidate in query_candidate_pairs:
        key = cache_key(query, candidate["title"])
        if key not in cache and key not in todo:
            todo[key] = (query, candidate)

    if not todo:
        return cache

    lock = threading.Lock()
    counters = {"new": 0, "errors": 0}

    def worker(item):
        key, (query, candidate) = item
        prompt = _prompt_for(query, candidate)
        try:
            r = _call_judge(client, prompt)
        except Exception as e:
            with lock:
                counters["errors"] += 1
            print(f"  WARNING: judge failed for '{candidate['title']}' / '{query}': {e} (will retry next run)")
            return
        with lock:
            cache[key] = {
                "label": r.get("label", "Irrelevant"),
                "reasoning": r.get("reasoning", ""),
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
            }
            counters["new"] += 1
            # Save periodically to preserve progress without thrashing disk.
            if counters["new"] % 20 == 0:
                LABELS_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(worker, todo.items()))

    # Final flush.
    LABELS_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    if counters["new"]:
        print(f"                          {counters['new']} new labels generated via API")
    if counters["errors"]:
        print(f"                          {counters['errors']} calls failed and were left uncached")

    return cache


def get_labels_for_query(query, candidates, cache):
    """Return a dict of title -> label for these candidates. Used by eval.py only."""
    return {
        c["title"]: cache.get(cache_key(query, c["title"]), {}).get("label", "Irrelevant")
        for c in candidates
    }
