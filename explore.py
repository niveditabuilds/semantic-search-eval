"""
Interactive search relevance explorer.
Type any query — see Pipeline1 vs Pipeline2 side by side.
Labeled queries show ✓/✗ and P@5. Unlabeled queries offer live Claude labeling.

Usage:
    python3 explore.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set.")
    sys.exit(1)

RESULTS_DIR = Path(__file__).parent / "results"
LABELS_PATH = RESULTS_DIR / "llm_labels.json"
QUERIES_PATH = RESULTS_DIR / "queries.json"

W = 26  # column width for side-by-side display


def _load_cache():
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return {}


def _cache_key(query, title):
    return f"{query}|||{title}"


def _get_labels(query, candidates, cache):
    return {
        c["title"]: cache.get(_cache_key(query, c["title"]), {}).get("label_v1")
        for c in candidates
    }


def _fmt(title, label):
    if label == "Relevant":
        mark = "✓"
    elif label == "Irrelevant":
        mark = "✗"
    else:
        mark = "?"
    truncated = title[:W - 3] + ".." if len(title) > W - 1 else title
    return f"{truncated} {mark}"


def _consistency(query, candidates, cache):
    entries = [
        cache[_cache_key(query, c["title"])]
        for c in candidates
        if _cache_key(query, c["title"]) in cache
    ]
    if not entries:
        return None
    return round(sum(1 for e in entries if e.get("consistent", True)) / len(entries) * 100)


def _detect_query_type(query, known_queries):
    for q in known_queries:
        if q["query"] == query.lower():
            return q["type"]
    return "unknown"


def _display_results(query, qtype, p3_out, p4_out, labels, cache, candidates):
    labeled_count = sum(1 for c in candidates if _cache_key(query, c["title"]) in cache)
    consistency = _consistency(query, candidates, cache)

    print()
    print("═" * 60)
    print(f'  Query: "{query}"  [{qtype}]')
    if labeled_count > 0:
        cons_str = f"  |  Consistency: {consistency}%" if consistency is not None else ""
        print(f"  Labels: {labeled_count} cached{cons_str}")
    else:
        print("  Labels: none cached")
    print("═" * 60)
    print()

    p3r = p3_out["results"][:5]
    p4r = p4_out["results"][:5]
    impact = p4_out["filter_impact"]

    p3_header = "PIPELINE 1  (Retrieval → Rerank)"
    p4_header = f"PIPELINE 2  (+ LLM Filter)"
    if impact:
        p4_header += f"  removed {impact['removed']}/{impact['total']}"

    print(f"  {p3_header:<{W+8}} {p4_header}")
    print(f"  {'─' * (W + 6)} {'─' * (W + 6)}")

    for i in range(5):
        if i < len(p3r):
            lbl3 = labels.get(p3r[i]["title"])
            pt = _fmt(p3r[i]["title"], lbl3)
        else:
            pt = ""
        if i < len(p4r):
            lbl4 = labels.get(p4r[i]["title"])
            qt = _fmt(p4r[i]["title"], lbl4)
        else:
            qt = ""
        print(f"  {i+1}. {pt:<{W+4}} {i+1}. {qt:<{W+4}}")

    if labeled_count > 0:
        from eval import precision_at_5, ndcg_at_5
        p3_p5 = precision_at_5(p3r, labels)
        p4_p5 = precision_at_5(p4r, labels)
        p3_ndcg = ndcg_at_5(p3r, labels)
        p4_ndcg = ndcg_at_5(p4r, labels)
        delta = round(p4_p5 - p3_p5, 2)
        sign = "+" if delta >= 0 else ""

        print()
        print(f"  P@5:   Pipeline1 {p3_p5:.2f}   Pipeline2 {p4_p5:.2f}   delta {sign}{delta:.2f}")
        print(f"  NDCG:  Pipeline1 {p3_ndcg:.2f}   Pipeline2 {p4_ndcg:.2f}")

    print()
    print("─" * 60)


def main():
    print()
    print("=" * 60)
    print("  TUBI SEARCH RELEVANCE EXPLORER")
    print("  Pipeline1 vs Pipeline2 — side by side")
    print("=" * 60)
    print()
    print("  Loading models and index...")

    try:
        from catalog import CATALOG
    except ImportError:
        print("ERROR: catalog.py not found. Run: python3 data/fetch_catalog.py")
        sys.exit(1)

    from rankers import precompute_embeddings, build_bm25_index
    from pipelines import Pipeline1, Pipeline2

    embeddings = precompute_embeddings(CATALOG)
    build_bm25_index(CATALOG)

    cache = _load_cache()
    known_queries = []
    if QUERIES_PATH.exists():
        known_queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    print(f"\n  {len(cache)} labels cached  |  {len(known_queries)} queries in query set")
    print()
    print("  Commands:")
    print("    <query>   search and compare pipelines")
    print("    list      show all labeled queries by type")
    print("    quit      exit")
    print()

    while True:
        try:
            raw = input("Enter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue

        if raw.lower() == "quit":
            print("Bye.")
            break

        if raw.lower() == "list":
            if not known_queries:
                print("  No queries cached yet.")
                continue
            by_type = defaultdict(list)
            for q in known_queries:
                by_type[q["type"]].append(q["query"])
            for qtype, qs in sorted(by_type.items()):
                print(f"\n  [{qtype.upper()}]")
                for q in qs:
                    labeled = sum(
                        1 for title_key in cache
                        if title_key.startswith(f"{q}|||")
                    )
                    print(f"    \"{q}\"  ({labeled} labeled)")
            print()
            continue

        query = raw.lower()
        qtype = _detect_query_type(query, known_queries)

        # Get candidates
        from pipelines import _retrieve_union
        candidates = _retrieve_union(query, CATALOG, embeddings, top_n=20)

        # Check label coverage
        cache = _load_cache()
        labels = _get_labels(query, candidates, cache)
        labeled_count = sum(1 for v in labels.values() if v is not None)

        # Run pipelines
        p3_out = Pipeline1.run(query, candidates)
        p4_out = Pipeline2.run(query, candidates, labels)

        _display_results(query, qtype, p3_out, p4_out, labels, cache, candidates)

        if labeled_count == 0:
            try:
                ans = input("  No labels cached. Label this query with Claude now? (y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue

            if ans == "y":
                from llm_judge import label_on_demand
                cache = label_on_demand(query, candidates)
                labels = _get_labels(query, candidates, cache)

                # Re-run Pipeline2 with fresh labels
                p4_out = Pipeline2.run(query, candidates, labels)
                print()
                print("  Results with labels:")
                _display_results(query, qtype, p3_out, p4_out, labels, cache, candidates)


if __name__ == "__main__":
    main()
