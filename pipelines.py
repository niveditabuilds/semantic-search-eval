"""
Two production pipeline architectures compared in this eval.

Pipeline3: Retrieval → Cross-encoder Rerank
  Standard two-stage pipeline. BM25 + semantic retrieval feeds a cross-encoder.

Pipeline4: Retrieval → LLM Filter → Cross-encoder Rerank
  The Etsy insight: suppress irrelevant candidates before reranking.
  Filter is conservative — only removes candidates with a cached Irrelevant label.
  Unlabeled candidates pass through.
"""

from rankers import bm25_retrieve, semantic_rank, cross_encoder_rerank
from eval import filter_impact


def _retrieve_union(query, catalog, embeddings, top_n=20):
    bm25_results = bm25_retrieve(query, catalog, top_n=top_n)
    semantic_results = semantic_rank(query, catalog, embeddings=embeddings)[:top_n]
    seen_ids = set()
    union = []
    for c in bm25_results + semantic_results:
        if c["id"] not in seen_ids:
            seen_ids.add(c["id"])
            union.append(c)
    return union


class Pipeline3:
    name = "Two-Stage (Retrieval → Rerank)"
    short = "Pipeline3"

    @staticmethod
    def run(query, candidates):
        """candidates: precomputed union from _retrieve_union"""
        reranked = cross_encoder_rerank(query, candidates)
        return {
            "results": reranked,
            "candidates": candidates,
            "filter_impact": None,
        }


class Pipeline4:
    name = "Two-Stage + LLM Filter (Retrieval → Filter → Rerank)"
    short = "Pipeline4"

    @staticmethod
    def run(query, candidates, labels):
        """candidates: precomputed union from _retrieve_union"""
        # Conservative filter: only remove confirmed Irrelevant candidates
        filtered = [
            c for c in candidates
            if labels.get(c["title"], "keep") != "Irrelevant"
        ]

        impact = filter_impact(candidates, filtered)

        # If filter removed everything (edge case), fall back to full set
        if not filtered:
            filtered = candidates

        reranked = cross_encoder_rerank(query, filtered)
        return {
            "results": reranked,
            "candidates": candidates,
            "filtered_candidates": filtered,
            "filter_impact": impact,
        }
