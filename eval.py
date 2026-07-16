import math

from sklearn.metrics import cohen_kappa_score

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


def label_pairs_for_query(candidates, system_labels, eval_labels):
    """Zip each candidate's system (Claude) and eval (GPT) label together."""
    return [
        (system_labels.get(c["title"], "Irrelevant"), eval_labels.get(c["title"], "Irrelevant"))
        for c in candidates
    ]


def cross_model_agreement(candidates, system_labels, eval_labels):
    """Fraction of candidates where the system judge and eval judge agree."""
    pairs = label_pairs_for_query(candidates, system_labels, eval_labels)
    if not pairs:
        return 1.0
    agree = sum(1 for s, e in pairs if s == e)
    return round(agree / len(pairs), 3)


def cohens_kappa(label_pairs):
    """
    Cohen's kappa over a list of (system_label, eval_label) tuples.
    Returns None when kappa is undefined (fewer than 2 pairs, or only one
    label value appears across both raters — chance agreement is then 1.0).
    """
    if len(label_pairs) < 2:
        return None
    system_vals = [s for s, _ in label_pairs]
    eval_vals = [e for _, e in label_pairs]
    if len(set(system_vals) | set(eval_vals)) < 2:
        return None
    return round(float(cohen_kappa_score(system_vals, eval_vals)), 3)
