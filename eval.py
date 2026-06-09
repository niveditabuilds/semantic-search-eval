import math

LABEL_SCORE = {"Relevant": 1, "Irrelevant": 0}


def precision_at_5(ranked_titles, labels):
    top5 = ranked_titles[:5]
    relevant = sum(
        1 for t in top5
        if labels.get(t["title"], "Irrelevant") == "Relevant"
    )
    return round(relevant / 5, 3)


def ndcg_at_5(ranked_titles, labels):
    def dcg(titles):
        score = 0.0
        for i, t in enumerate(titles[:5]):
            gain = LABEL_SCORE.get(labels.get(t["title"], "Irrelevant"), 0)
            score += gain / math.log2(i + 2)
        return score

    actual_dcg = dcg(ranked_titles)
    ideal_order = sorted(
        ranked_titles,
        key=lambda t: LABEL_SCORE.get(labels.get(t["title"], "Irrelevant"), 0),
        reverse=True,
    )
    ideal_dcg = dcg(ideal_order)
    if ideal_dcg == 0:
        return 0.0
    return round(actual_dcg / ideal_dcg, 3)


def label_distribution(candidates, labels):
    relevant = sum(
        1 for c in candidates
        if labels.get(c["title"], "Irrelevant") == "Relevant"
    )
    irrelevant = len(candidates) - relevant
    return {"relevant": relevant, "irrelevant": irrelevant, "total": len(candidates)}


def score_ranker(ranked, labels):
    return {
        "p5": precision_at_5(ranked, labels),
        "ndcg5": ndcg_at_5(ranked, labels),
    }


def filter_impact(candidates_before, candidates_after):
    removed = len(candidates_before) - len(candidates_after)
    pct = round(removed / len(candidates_before), 3) if candidates_before else 0.0
    return {"removed": removed, "total": len(candidates_before), "pct_removed": pct}
