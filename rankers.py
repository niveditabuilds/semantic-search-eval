import random
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Load once at module level
print("Loading sentence model... ", end="", flush=True)
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("all-MiniLM-L6-v2 ready")

print("Loading cross-encoder...  ", end="", flush=True)
CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("ms-marco-MiniLM-L-6-v2 ready")

# BM25 index — built once over full catalog
_BM25_INDEX = None
_BM25_CATALOG = None


def build_bm25_index(catalog):
    global _BM25_INDEX, _BM25_CATALOG
    print("Building BM25 index...    ", end="", flush=True)
    corpus = []
    for item in catalog:
        text = f"{item['title']} {' '.join(item['genres'])} {item['overview']}"
        corpus.append(text.lower().split())
    _BM25_INDEX = BM25Okapi(corpus)
    _BM25_CATALOG = catalog
    print(f"{len(catalog)} documents indexed")
    return _BM25_INDEX


def bm25_retrieve(query, catalog, top_n=20):
    global _BM25_INDEX, _BM25_CATALOG
    if _BM25_INDEX is None or _BM25_CATALOG is not catalog:
        build_bm25_index(catalog)
    tokens = query.lower().split()
    scores = _BM25_INDEX.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [catalog[i] for i in top_indices]


def cross_encoder_rerank(query, candidates):
    if not candidates:
        return candidates
    pairs = [
        (query, f"{c['title']}. {' '.join(c['genres'])}. {c['overview'][:200]}")
        for c in candidates
    ]
    scores = CROSS_ENCODER.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked]

ALPHA = {
    "genre":    0.40,
    "compound": 0.70,
    "decade":   0.35,
    "thematic": 0.75,
    "default":  0.55,
}


def precompute_embeddings(catalog):
    """Encode all catalog items once. Returns dict of id -> embedding."""
    print("Precomputing embeddings... ", end="", flush=True)
    texts = [
        f"{c['title']}. {' '.join(c['genres'])}. {c['overview'][:150]}"
        for c in catalog
    ]
    embs = MODEL.encode(texts, batch_size=64, show_progress_bar=False)
    result = {c["id"]: embs[i] for i, c in enumerate(catalog)}
    print(f"{len(result)} embeddings ready")
    return result


def _get_candidate_embs(candidates, embeddings):
    return np.array([embeddings[c["id"]] for c in candidates])


def random_rank(query, candidates, seed=42):
    shuffled = candidates.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    return shuffled


def popularity_rank(query, candidates):
    return sorted(candidates, key=lambda x: x["mock_popularity"], reverse=True)


def semantic_rank(query, candidates, embeddings=None):
    if not candidates:
        return candidates
    query_emb = MODEL.encode([query])
    if embeddings is not None:
        candidate_embs = _get_candidate_embs(candidates, embeddings)
    else:
        texts = [f"{c['title']}. {' '.join(c['genres'])}. {c['overview'][:150]}" for c in candidates]
        candidate_embs = MODEL.encode(texts)
    scores = cosine_similarity(query_emb, candidate_embs)[0]
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked]


def hybrid_rank(query, candidates, query_type, embeddings=None):
    if not candidates:
        return candidates
    alpha = ALPHA.get(query_type, ALPHA["default"])

    query_emb = MODEL.encode([query])
    if embeddings is not None:
        candidate_embs = _get_candidate_embs(candidates, embeddings)
    else:
        texts = [f"{c['title']}. {' '.join(c['genres'])}. {c['overview'][:150]}" for c in candidates]
        candidate_embs = MODEL.encode(texts)
    semantic_scores = cosine_similarity(query_emb, candidate_embs)[0]

    pop_scores = np.array([c["mock_popularity"] for c in candidates], dtype=float)
    if pop_scores.max() > pop_scores.min():
        pop_scores = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min())

    final_scores = alpha * semantic_scores + (1 - alpha) * pop_scores
    ranked = sorted(zip(candidates, final_scores), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked]
