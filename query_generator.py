import json
import random
import os
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
import anthropic

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
QUERIES_PATH = RESULTS_DIR / "queries.json"
MAX_COMPOUND = 8


def _genre_counts(catalog):
    counts = defaultdict(int)
    for item in catalog:
        for g in item["genres"]:
            counts[g] += 1
    return counts


def _genre_queries(catalog):
    counts = _genre_counts(catalog)
    return [
        {"query": genre.lower(), "type": "genre"}
        for genre, count in counts.items()
        if count >= 10
    ]


def _compound_queries(catalog):
    counts = _genre_counts(catalog)
    eligible = [g for g, c in counts.items() if c >= 10]

    # Build co-occurrence counts
    co = defaultdict(int)
    for item in catalog:
        item_genres = [g for g in item["genres"] if g in set(eligible)]
        for i in range(len(item_genres)):
            for j in range(i + 1, len(item_genres)):
                pair = tuple(sorted([item_genres[i], item_genres[j]]))
                co[pair] += 1

    results = []
    seen_pairs = set()
    for (g1, g2), count in sorted(co.items(), key=lambda x: -x[1]):
        if count >= 5 and len(results) < MAX_COMPOUND:
            pair_key = tuple(sorted([g1.lower(), g2.lower()]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                results.append({"query": f"{g1.lower()} {g2.lower()}", "type": "compound"})
    return results


def _decade_queries(catalog):
    decade_counts = defaultdict(int)
    for item in catalog:
        year = item.get("release_year", 0)
        if year >= 1900:
            decade = (year // 10) * 10
            decade_counts[decade] += 1

    queries = []
    for decade, count in sorted(decade_counts.items()):
        if count >= 20:
            label = f"{str(decade)[2:]}s movies"  # e.g. 1980 → "80s movies"
            queries.append({"query": label, "type": "decade"})
    return queries


def _thematic_mood_longtail_queries(catalog):
    random.seed(42)
    sample = random.sample(catalog, min(60, len(catalog)))
    plots = [item["overview"] for item in sample if item["overview"]][:60]

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sample_plots = "\n\n".join(f"- {p}" for p in plots)

    prompt = f"""Given these movie plot summaries from a streaming platform catalog,
generate search queries that real users would type.

Generate exactly:
- 6 thematic queries (2-4 words, about themes or subject matter)
  Examples: "based on true story", "revenge thriller", "heist gone wrong", "mind bending"

- 4 mood-based queries (how a user describes what they feel like watching)
  Examples: "feel good movies", "dark psychological", "movies to cry to", "light hearted"

- 4 long-tail subgenre queries (niche styles a casual user might search)
  Examples: "spaghetti western", "japanese action", "slow burn mystery", "70s blaxploitation"

All queries must be derivable from the actual plots and genres in this catalog.
Do not invent queries for content that clearly isn't represented.

Plot summaries:
{sample_plots}

Return only valid JSON, no other text:
{{"thematic": ["...", "...", "...", "...", "...", "..."],
  "mood": ["...", "...", "...", "..."],
  "longtail": ["...", "...", "...", "..."]}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())

    thematic = [{"query": t.lower(), "type": "thematic"} for t in data.get("thematic", [])[:6]]
    mood = [{"query": t.lower(), "type": "mood"} for t in data.get("mood", [])[:4]]
    longtail = [{"query": t.lower(), "type": "longtail"} for t in data.get("longtail", [])[:4]]
    return thematic, mood, longtail


def generate_queries(catalog):
    genre_qs = _genre_queries(catalog)
    compound_qs = _compound_queries(catalog)
    decade_qs = _decade_queries(catalog)
    thematic_qs, mood_qs, longtail_qs = _thematic_mood_longtail_queries(catalog)

    all_queries = genre_qs + compound_qs + decade_qs + thematic_qs + mood_qs + longtail_qs

    RESULTS_DIR.mkdir(exist_ok=True)
    QUERIES_PATH.write_text(json.dumps(all_queries, indent=2), encoding="utf-8")

    print(f"Generating queries...     {len(all_queries)} queries generated")
    print(
        f"                          {len(genre_qs)} genre | "
        f"{len(compound_qs)} compound | "
        f"{len(decade_qs)} decade | "
        f"{len(thematic_qs)} thematic | "
        f"{len(mood_qs)} mood | "
        f"{len(longtail_qs)} longtail"
    )
    return all_queries


def load_or_generate_queries(catalog, force_regenerate=False):
    if QUERIES_PATH.exists() and not force_regenerate:
        queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
        counts = {}
        for q in queries:
            counts[q["type"]] = counts.get(q["type"], 0) + 1
        print(f"Generating queries...     {len(queries)} queries loaded from cache")
        print(
            f"                          {counts.get('genre', 0)} genre | "
            f"{counts.get('compound', 0)} compound | "
            f"{counts.get('decade', 0)} decade | "
            f"{counts.get('thematic', 0)} thematic | "
            f"{counts.get('mood', 0)} mood | "
            f"{counts.get('longtail', 0)} longtail"
        )
        return queries
    return generate_queries(catalog)
