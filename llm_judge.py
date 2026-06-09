import json
import os
from pathlib import Path

from dotenv import load_dotenv
import anthropic

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
LABELS_PATH = RESULTS_DIR / "llm_labels.json"

JUDGE_PROMPT = """
You are evaluating search relevance for a streaming platform.
The platform has a large catalog including many niche and long-tail titles.

Query: "{query}"
Title: "{title}"
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

JUDGE_PROMPT_V2 = """
You are a search quality evaluator for a video streaming service.

Search query entered by user: "{query}"
Title shown in search results: "{title}"
Genres: {genres}
Description: "{plot}"

Should this title appear in search results for this query?

Yes (Relevant): title matches what the user is looking for
No (Irrelevant): title does not match what the user is looking for

Return only JSON:
{{"label": "Relevant" | "Irrelevant", "reasoning": "one sentence"}}
"""


def _cache_key(query, title):
    return f"{query}|||{title}"


def _call_judge(client, prompt):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def label_pairs(query_candidate_pairs):
    """
    query_candidate_pairs: list of (query, query_type, candidate_dict)
    Returns: dict keyed by cache_key with label info
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load existing cache
    if LABELS_PATH.exists():
        cache = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    else:
        cache = {}

    loaded = sum(
        1 for (q, _, c) in query_candidate_pairs
        if _cache_key(q, c["title"]) in cache
    )
    print(f"Running LLM judge...      loaded {loaded} labels from cache")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    new_calls = 0

    for query, _, candidate in query_candidate_pairs:
        key = _cache_key(query, candidate["title"])
        if key in cache:
            continue

        genres_str = ", ".join(candidate["genres"]) if candidate["genres"] else "Unknown"
        plot = candidate["overview"][:300]

        p1 = JUDGE_PROMPT.format(
            query=query, title=candidate["title"],
            genres=genres_str, plot=plot
        )
        p2 = JUDGE_PROMPT_V2.format(
            query=query, title=candidate["title"],
            genres=genres_str, plot=plot
        )

        try:
            r1 = _call_judge(client, p1)
            r2 = _call_judge(client, p2)
        except Exception as e:
            print(f"  WARNING: judge failed for '{candidate['title']}' / '{query}': {e}")
            cache[key] = {
                "label_v1": "Irrelevant",
                "label_v2": "Irrelevant",
                "reasoning_v1": "API error",
                "consistent": True,
            }
            continue

        cache[key] = {
            "label_v1": r1.get("label", "Irrelevant"),
            "label_v2": r2.get("label", "Irrelevant"),
            "reasoning_v1": r1.get("reasoning", ""),
            "consistent": r1.get("label") == r2.get("label"),
        }
        new_calls += 1

        # Save after each call to preserve progress
        LABELS_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    if new_calls:
        print(f"                          {new_calls} new labels generated via API")

    return cache


def get_labels_for_query(query, candidates, cache):
    """Return a dict of title -> label_v1 for these candidates."""
    return {
        c["title"]: cache.get(_cache_key(query, c["title"]), {}).get("label_v1", "Irrelevant")
        for c in candidates
    }


def get_consistency_for_query(query, candidates, cache):
    """Return fraction of consistent labels for these candidates."""
    entries = [
        cache[_cache_key(query, c["title"])]
        for c in candidates
        if _cache_key(query, c["title"]) in cache
    ]
    if not entries:
        return 1.0
    return sum(1 for e in entries if e.get("consistent", True)) / len(entries)


def label_on_demand(query, candidates):
    """
    Label all candidates for a single query in real time.
    Called from explore.py for unlabeled queries.
    Saves to cache after every API call.
    Returns updated cache.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    if LABELS_PATH.exists():
        cache = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    else:
        cache = {}

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    to_label = [c for c in candidates if _cache_key(query, c["title"]) not in cache]

    if not to_label:
        print("  All candidates already labeled.")
        return cache

    print(f"  Labeling {len(to_label)} candidates with Claude...")
    for i, candidate in enumerate(to_label, 1):
        key = _cache_key(query, candidate["title"])
        genres_str = ", ".join(candidate["genres"]) if candidate["genres"] else "Unknown"
        plot = candidate["overview"][:300]

        print(f"  {i}/{len(to_label)}: {candidate['title'][:40]}", end="", flush=True)

        p1 = JUDGE_PROMPT.format(
            query=query, title=candidate["title"],
            genres=genres_str, plot=plot
        )
        p2 = JUDGE_PROMPT_V2.format(
            query=query, title=candidate["title"],
            genres=genres_str, plot=plot
        )

        try:
            r1 = _call_judge(client, p1)
            r2 = _call_judge(client, p2)
            label = r1.get("label", "Irrelevant")
            consistent = r1.get("label") == r2.get("label")
            print(f" → {label} {'✓' if consistent else '~'}")
            cache[key] = {
                "label_v1": label,
                "label_v2": r2.get("label", "Irrelevant"),
                "reasoning_v1": r1.get("reasoning", ""),
                "consistent": consistent,
            }
        except Exception as e:
            print(f" → ERROR: {e}")
            cache[key] = {
                "label_v1": "Irrelevant",
                "label_v2": "Irrelevant",
                "reasoning_v1": "API error",
                "consistent": True,
            }

        LABELS_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return cache
