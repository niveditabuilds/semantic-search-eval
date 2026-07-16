"""
Interactive search relevance explorer.
Type any query — see Pipeline1 vs Pipeline2 side by side.

Two independent judges, matching the non-circular eval:
  - system judge (Sonnet, system_labels.json) drives the Pipeline 2 filter
  - eval judge  (Haiku/GPT, eval_labels.json)  scores the ✓/✗ marks and P@5

The filter never grades its own output — the ✓/✗ and P@5 you see come from a
different model than the one that decided what to filter out.

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

import llm_judge
import eval_judge

RESULTS_DIR = Path(__file__).parent / "results"
SYSTEM_LABELS_PATH = llm_judge.LABELS_PATH
EVAL_LABELS_PATH = eval_judge.LABELS_PATH
QUERIES_PATH = RESULTS_DIR / "queries.json"

W = 26  # column width for side-by-side display


def _load(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _get_labels(query, candidates, cache, module):
    """Labels from one judge's cache. module is llm_judge or eval_judge."""
    return {
        c["title"]: cache.get(module.cache_key(query, c["title"]), {}).get("label")
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


def _detect_query_type(query, known_queries):
    for q in known_queries:
        if q["query"] == query.lower():
            return q["type"]
    return "unknown"


def _display_results(query, qtype, p3_out, p4_out, eval_labels, sys_cache, eval_cache, candidates):
    sys_count = sum(1 for c in candidates if llm_judge.cache_key(query, c["title"]) in sys_cache)
    eval_count = sum(1 for c in candidates if eval_judge.cache_key(query, c["title"]) in eval_cache)

    print()
    print("═" * 60)
    print(f'  Query: "{query}"  [{qtype}]')
    if sys_count or eval_count:
        print(f"  Filter labels (system judge): {sys_count}   "
              f"Scoring labels (eval judge): {eval_count}")
        print(f"  ✓/✗ and P@5 below are scored by the independent eval judge")
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
            lbl3 = eval_labels.get(p3r[i]["title"])
            pt = _fmt(p3r[i]["title"], lbl3)
        else:
            pt = ""
        if i < len(p4r):
            lbl4 = eval_labels.get(p4r[i]["title"])
            qt = _fmt(p4r[i]["title"], lbl4)
        else:
            qt = ""
        print(f"  {i+1}. {pt:<{W+4}} {i+1}. {qt:<{W+4}}")

    if eval_count > 0:
        from eval import precision_at_5, ndcg_at_5
        p3_p5 = precision_at_5(p3r, eval_labels)
        p4_p5 = precision_at_5(p4r, eval_labels)
        p3_ndcg = ndcg_at_5(p3r, eval_labels)
        p4_ndcg = ndcg_at_5(p4r, eval_labels)
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

    sys_cache = _load(SYSTEM_LABELS_PATH)
    eval_cache = _load(EVAL_LABELS_PATH)
    known_queries = []
    if QUERIES_PATH.exists():
        known_queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    print(f"\n  {len(sys_cache)} filter labels | {len(eval_cache)} scoring labels"
          f"  |  {len(known_queries)} queries in query set")
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
                        1 for title_key in eval_cache
                        if f"|||{q}|||" in title_key
                    )
                    print(f"    \"{q}\"  ({labeled} scored)")
            print()
            continue

        query = raw.lower()
        qtype = _detect_query_type(query, known_queries)

        # Get candidates
        from pipelines import _retrieve_union
        candidates = _retrieve_union(query, CATALOG, embeddings, top_n=20)

        # Two independent label sets: system judge drives the filter, eval judge scores.
        sys_cache = _load(SYSTEM_LABELS_PATH)
        eval_cache = _load(EVAL_LABELS_PATH)
        sys_labels = _get_labels(query, candidates, sys_cache, llm_judge)
        eval_labels = _get_labels(query, candidates, eval_cache, eval_judge)
        eval_count = sum(1 for v in eval_labels.values() if v is not None)

        # Run pipelines — the filter reads system labels only.
        p3_out = Pipeline1.run(query, candidates)
        p4_out = Pipeline2.run(query, candidates, sys_labels)

        _display_results(query, qtype, p3_out, p4_out, eval_labels, sys_cache, eval_cache, candidates)

        if eval_count == 0:
            try:
                ans = input("  No scoring labels cached. Label this query with both judges now? (y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue

            if ans == "y":
                # Label with both judges so the filter and the scoring both have data.
                pairs = [(query, qtype, c) for c in candidates]
                sys_cache = llm_judge.label_pairs(pairs)
                eval_cache = eval_judge.label_pairs(pairs)
                sys_labels = _get_labels(query, candidates, sys_cache, llm_judge)
                eval_labels = _get_labels(query, candidates, eval_cache, eval_judge)

                # Re-run Pipeline2 with fresh filter labels
                p4_out = Pipeline2.run(query, candidates, sys_labels)
                print()
                print("  Results with labels:")
                _display_results(query, qtype, p3_out, p4_out, eval_labels, sys_cache, eval_cache, candidates)


if __name__ == "__main__":
    main()
