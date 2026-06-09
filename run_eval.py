import json
import sys
import os
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
    sys.exit(1)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 56)
    print("SEMANTIC SEARCH RELEVANCE EVAL V2")
    print("Catalog: TMDb (1000 titles)")
    print("Comparing: Two-Stage Pipeline vs Two-Stage + LLM Filter")
    print("LLM-as-Judge filtering vs cross-encoder reranking")
    print("=" * 56)
    print()

    # --- Catalog ---
    print("Loading catalog...        ", end="", flush=True)
    try:
        from catalog import CATALOG
    except ImportError:
        print("\nERROR: catalog.py not found. Run: python3 data/fetch_catalog.py")
        sys.exit(1)
    print(f"{len(CATALOG)} titles loaded")

    # --- Queries ---
    from query_generator import load_or_generate_queries
    queries = load_or_generate_queries(CATALOG)

    # --- Rankers and pipelines (models load here) ---
    from rankers import precompute_embeddings, build_bm25_index
    from pipelines import Pipeline3, Pipeline4

    embeddings = precompute_embeddings(CATALOG)
    build_bm25_index(CATALOG)

    # --- Candidate collection (retrieval only — no cross-encoder here) ---
    print("Collecting candidates...  ", end="", flush=True)
    from pipelines import _retrieve_union
    judge_pairs = []
    query_candidates = {}

    for q in queries:
        candidates = _retrieve_union(q["query"], CATALOG, embeddings, top_n=20)
        query_candidates[q["query"]] = candidates
        for c in candidates:
            judge_pairs.append((q["query"], q["type"], c))

    print(f"{len(judge_pairs)} (query, title) pairs to label")

    # --- LLM Judge ---
    from llm_judge import label_pairs, get_labels_for_query, get_consistency_for_query
    cache = label_pairs(judge_pairs)

    # --- Eval loop ---
    print()
    print("=" * 56)
    print("QUERY RESULTS")
    print("=" * 56)
    print()

    all_results = []

    for q in queries:
        query = q["query"]
        qtype = q["type"]
        candidates = query_candidates[query]
        labels = get_labels_for_query(query, candidates, cache)
        consistency = get_consistency_for_query(query, candidates, cache)

        from eval import score_ranker, label_distribution, filter_impact

        p3_out = Pipeline3.run(query, candidates)
        p4_out = Pipeline4.run(query, candidates, labels)

        p3_scores = score_ranker(p3_out["results"], labels)
        p4_scores = score_ranker(p4_out["results"], labels)
        dist = label_distribution(candidates, labels)
        impact = p4_out["filter_impact"]

        delta_p5 = round(p4_scores["p5"] - p3_scores["p5"], 3)
        delta_ndcg = round(p4_scores["ndcg5"] - p3_scores["ndcg5"], 3)

        result = {
            "query": query,
            "type": qtype,
            "llm_consistency": round(consistency, 3),
            "label_distribution": dist,
            "pipeline3": p3_scores,
            "pipeline4": p4_scores,
            "filter_impact": impact,
            "delta_p5": delta_p5,
            "top5": {
                "pipeline3": [c["title"] for c in p3_out["results"][:5]],
                "pipeline4": [c["title"] for c in p4_out["results"][:5]],
            }
        }
        all_results.append(result)

        # --- Print query block ---
        print(f'[{qtype.upper()}] "{query}"')
        print(f"  Label distribution: {dist['relevant']} Relevant / {dist['irrelevant']} Irrelevant  ({dist['total']} candidates)")
        print(f"  LLM consistency:    {round(consistency * 100)}%")
        if impact:
            print(f"  Filter removed:     {impact['removed']}/{impact['total']} candidates ({round(impact['pct_removed']*100)}%)")
        print()
        print(f"                    P@5    NDCG@5")
        print(f"  {'Pipeline3':<16}  {p3_scores['p5']:.2f}   {p3_scores['ndcg5']:.2f}")
        print(f"  {'Pipeline4':<16}  {p4_scores['p5']:.2f}   {p4_scores['ndcg5']:.2f}")
        sign = "+" if delta_p5 >= 0 else ""
        print(f"  {'─' * 29}")
        print(f"  Filter delta      {sign}{delta_p5:.2f}   {'+' if delta_ndcg >= 0 else ''}{delta_ndcg:.2f}")

        _w = 24
        def _fmt(title, lbl):
            mark = "✓" if lbl == "Relevant" else "✗"
            t = title[:_w - 3] + ".." if len(title) > _w - 1 else title
            return f"{t} {mark}"

        p3r = p3_out["results"][:5]
        p4r = p4_out["results"][:5]
        print()
        print(f"  {'PIPELINE 3':<{_w+4}} {'PIPELINE 4 (+ LLM Filter)':<{_w+4}}")
        for i in range(5):
            pt = _fmt(p3r[i]["title"], labels.get(p3r[i]["title"], "Irrelevant")) if i < len(p3r) else ""
            qt = _fmt(p4r[i]["title"], labels.get(p4r[i]["title"], "Irrelevant")) if i < len(p4r) else ""
            print(f"  {i+1}. {pt:<{_w+2}} {i+1}. {qt:<{_w+2}}")

        print()
        print("─" * 56)
        print()

    # --- Aggregate by query type ---
    print("=" * 56)
    print("AGGREGATE BY QUERY TYPE")
    print("=" * 56)
    print()

    type_groups = defaultdict(list)
    for r in all_results:
        type_groups[r["type"]].append(r)

    def avg(rows, key, subkey):
        vals = [r[key][subkey] for r in rows if r[key]]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def avg_filter_pct(rows):
        vals = [r["filter_impact"]["pct_removed"] for r in rows if r["filter_impact"]]
        return round(sum(vals) / len(vals) * 100) if vals else 0

    header = f"{'Type':<12} {'Queries':>7}  {'P3 P@5':>7}  {'P4 P@5':>7}  {'Filter%':>7}  {'Delta':>7}"
    print(header)
    print("─" * 58)

    overall_rows = []
    for qtype in ["genre", "compound", "decade", "thematic", "mood", "longtail"]:
        rows = type_groups.get(qtype, [])
        if not rows:
            continue
        p3 = avg(rows, "pipeline3", "p5")
        p4 = avg(rows, "pipeline4", "p5")
        flt = avg_filter_pct(rows)
        delta = round(p4 - p3, 3)
        sign = "+" if delta >= 0 else ""
        print(f"{qtype:<12} {len(rows):>7}  {p3:>7.2f}  {p4:>7.2f}  {flt:>6}%  {sign}{delta:>6.2f}")
        overall_rows.extend(rows)

    print("─" * 58)
    p3_all = avg(overall_rows, "pipeline3", "p5")
    p4_all = avg(overall_rows, "pipeline4", "p5")
    flt_all = avg_filter_pct(overall_rows)
    delta_all = round(p4_all - p3_all, 3)
    sign_all = "+" if delta_all >= 0 else ""
    print(f"{'OVERALL':<12} {len(overall_rows):>7}  {p3_all:>7.2f}  {p4_all:>7.2f}  {flt_all:>6}%  {sign_all}{delta_all:>6.2f}")
    print()

    avg_consistency = round(
        sum(r["llm_consistency"] for r in all_results) / len(all_results) * 100
    ) if all_results else 0
    print(f"Avg LLM label consistency: {avg_consistency}%")
    print()

    # --- Key Finding (dynamic) ---
    print("=" * 56)
    print("KEY FINDING")
    print("=" * 56)
    print()

    # Query type with most filter benefit
    type_deltas = {
        qtype: round(avg(rows, "pipeline4", "p5") - avg(rows, "pipeline3", "p5"), 3)
        for qtype, rows in type_groups.items() if rows
    }
    best_type = max(type_deltas, key=lambda t: type_deltas[t])
    worst_type = min(type_deltas, key=lambda t: type_deltas[t])
    genre_delta = type_deltas.get("genre", 0)

    print(f"LLM filter adds +{type_deltas[best_type]:.2f} P@5 on [{best_type}] queries")
    sign_g = "+" if genre_delta >= 0 else ""
    print(f"but only {sign_g}{genre_delta:.2f} on [genre] queries.")
    print()
    print("Recommendation: deploy filter selectively on non-genre queries.")
    print("For genre queries, the retrieval layer already surfaces relevant")
    print("content — the filter adds latency without meaningful lift.")
    print()

    # Query with biggest gain
    best_query = max(all_results, key=lambda r: r["delta_p5"])
    print(f"Biggest single-query gain: \"{best_query['query']}\" [{best_query['type']}]")
    print(f"  Pipeline3: {best_query['pipeline3']['p5']:.2f} P@5 → Pipeline4: {best_query['pipeline4']['p5']:.2f} P@5")
    if best_query["filter_impact"]:
        print(f"  Filter removed {best_query['filter_impact']['removed']} of {best_query['filter_impact']['total']} candidates")
    print()
    print("=" * 56)

    # --- Save report ---
    report = {
        "per_query": all_results,
        "aggregate": {
            qtype: {
                "count": len(rows),
                "pipeline3_p5": avg(rows, "pipeline3", "p5"),
                "pipeline4_p5": avg(rows, "pipeline4", "p5"),
                "avg_filter_pct": avg_filter_pct(rows),
                "delta_p5": round(avg(rows, "pipeline4", "p5") - avg(rows, "pipeline3", "p5"), 3),
            }
            for qtype, rows in type_groups.items()
        },
        "overall": {
            "count": len(overall_rows),
            "pipeline3_p5": p3_all,
            "pipeline4_p5": p4_all,
            "avg_filter_pct": flt_all,
            "delta_p5": delta_all,
            "avg_llm_consistency_pct": avg_consistency,
        }
    }
    report_path = RESULTS_DIR / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
