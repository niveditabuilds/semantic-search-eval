"""
Build docs/data.json for the live demo site from the current label caches
and eval_report.json. Run after run_eval.py to refresh the interactive demo:

    python3 build_site_data.py

The site (docs/index.html) is a static, self-contained page that reads this file.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from catalog import CATALOG
from rankers import precompute_embeddings, build_bm25_index
from pipelines import Pipeline1, Pipeline2, _retrieve_union
import llm_judge
import eval_judge
from eval import precision_at_5, ndcg_at_5

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
TYPE_ORDER = ["genre", "compound", "decade", "thematic", "mood", "longtail"]


def main():
    queries = json.loads((RESULTS / "queries.json").read_text())
    report = json.loads((RESULTS / "eval_report.json").read_text())

    emb = precompute_embeddings(CATALOG)
    build_bm25_index(CATALOG)
    sys_cache = llm_judge._load_cache()
    ev_cache = eval_judge._load_cache()

    def pack(rows, ev_labels):
        return [{"title": r["title"], "year": r.get("release_year"),
                 "label": ev_labels.get(r["title"], "Irrelevant")} for r in rows]

    out_queries = []
    for q in queries:
        query, qtype = q["query"], q["type"]
        cands = _retrieve_union(query, CATALOG, emb, top_n=20)
        sys_labels = llm_judge.get_labels_for_query(query, cands, sys_cache)
        ev_labels = eval_judge.get_labels_for_query(query, cands, ev_cache)
        p1 = Pipeline1.run(query, cands)["results"][:5]
        p2o = Pipeline2.run(query, cands, sys_labels)
        p2 = p2o["results"][:5]
        imp = p2o["filter_impact"]
        agr = next((r["cross_model_agreement"] for r in report["per_query"] if r["query"] == query), None)
        out_queries.append({
            "query": query, "type": qtype,
            "filter_removed": imp["removed"], "filter_total": imp["total"],
            "filter_pct": round(imp["pct_removed"] * 100),
            "p1_p5": precision_at_5(p1, ev_labels), "p2_p5": precision_at_5(p2, ev_labels),
            "p1_ndcg": ndcg_at_5(p1, ev_labels), "p2_ndcg": ndcg_at_5(p2, ev_labels),
            "delta": round(precision_at_5(p2, ev_labels) - precision_at_5(p1, ev_labels), 2),
            "agreement": round(agr * 100) if agr is not None else None,
            "p1": pack(p1, ev_labels), "p2": pack(p2, ev_labels),
        })

    out = {
        "overall": report["overall"],
        "aggregate": {t: report["aggregate"][t] for t in TYPE_ORDER if t in report["aggregate"]},
        "type_order": TYPE_ORDER,
        "queries": out_queries,
    }
    dest = ROOT / "docs" / "data.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"Wrote {dest}  ({len(out_queries)} queries)")


if __name__ == "__main__":
    main()
