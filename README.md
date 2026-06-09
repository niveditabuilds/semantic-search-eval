# semantic-search-eval

**Does an LLM filter make reranking better — and on which queries does it help most?**

A search relevance evaluation framework for streaming catalogs that compares two production-grade pipeline architectures:

- **Pipeline 3** — BM25 + Semantic Retrieval → Cross-encoder Rerank
- **Pipeline 4** — BM25 + Semantic Retrieval → LLM Filter → Cross-encoder Rerank

Inspired by [Etsy's LLM-as-Judge Semantic Relevance Framework (Jan 2026)](https://www.etsy.com/codeascraft/semantic-relevance-evaluation).  
Catalog: TMDb 1000 titles. Queries: auto-generated from catalog metadata. Labels: Claude Sonnet.

---

## The Hypothesis

Etsy's 2026 paper introduced an insight that applies directly to streaming search: **LLM relevance labels, used as a pre-filter before reranking, remove noise from the candidate pool and let the reranker focus on genuinely relevant titles.**

The key claim:
> LLM filtering should help most on queries where the retrieval stage surfaces many semantically-adjacent-but-wrong candidates — compound, thematic, and temporal queries — and add no value on simple genre queries where the retrieval ceiling is already 100% relevant.

This repo tests that claim end-to-end, on a real catalog, with 45 programmatically-generated queries, 1,718 LLM-labeled pairs, and a dual-prompt consistency check.

---

## Architecture

```
Query
  │
  ├─► BM25 retrieval (top 20)   ─┐
  │                               ├─► union (≈30–40 candidates)
  └─► Semantic retrieval (top 20) ┘
            │
     ┌──────┴──────────────────────┐
     │                             │
  Pipeline 3                    Pipeline 4
  Cross-encoder rerank          LLM filter (remove Irrelevant)
         │                         │
     top-5 results             Cross-encoder rerank
                                    │
                                top-5 results
```

**Component details:**

| Component | Model / Library | Role |
|-----------|----------------|------|
| BM25 | `rank_bm25` | Keyword recall — catches exact title/genre matches |
| Semantic retrieval | `all-MiniLM-L6-v2` | Dense recall — catches semantic similarity |
| LLM judge | Claude (dual-prompt) | Labels each candidate Relevant / Irrelevant |
| Cross-encoder reranker | `ms-marco-MiniLM-L-6-v2` | Precision stage — scores query-title relevance |

**Why both BM25 and semantic retrieval?** Each covers different failure modes. BM25 finds "horror movies" even when genre isn't in the embedding. Semantic retrieval finds "dark psychological thriller" when BM25 finds nothing. The union maximizes recall before the precision stage runs.

**LLM filter is conservative.** Only candidates with a cached `Irrelevant` label are removed. Unlabeled candidates pass through. This is intentional — a false positive removal (suppressing a relevant title) is worse than leaving noise for the reranker.

---

## Methodology

### Query generation

All queries are auto-generated from catalog metadata using Claude. No queries are hardcoded. The generator reads plot summaries, genres, and release years across the 1000-title catalog and produces:

| Type | Count | Examples |
|------|-------|---------|
| Genre | 18 | `horror movies`, `animated movies`, `romantic comedy` |
| Compound | 8 | `action thriller heist`, `sci-fi dystopia`, `crime drama based on true story` |
| Decade | 5 | `80s movies`, `10s movies`, `90s movies` |
| Thematic | 6 | `undercover cop`, `childhood wish`, `revenge thriller` |
| Mood | 4 | `feel good movies`, `dark psychological`, `light hearted comedy` |
| Long-tail | 4 | `animated talking animals`, `superhero origin story`, `world war two` |
| **Total** | **45** | |

### LLM judge

Each (query, candidate title) pair is labeled independently using two different prompt framings — a direct phrasing and a user-intent phrasing. `label_v1` is canonical for metrics. The second label is used for consistency scoring only.

Pairs are labeled once and cached. Re-runs load from cache, costing zero API calls on cached queries and labeling only new ones.

**97% dual-prompt consistency** — the judge is stable.

### Metrics

- **P@5** (Precision at 5): fraction of the top-5 results that are `Relevant`. Primary metric.
- **NDCG@5**: position-aware P@5 — a relevant result at position 1 scores higher than at position 5.
- **Filter%**: percentage of candidates removed by the LLM filter before reranking.
- **Delta**: P4 P@5 minus P3 P@5.

---

## Results

```
========================================================
AGGREGATE BY QUERY TYPE (45 queries, 1827 labeled pairs)
========================================================

Type         Queries  P3 P@5  P4 P@5  Filter%    Delta
──────────────────────────────────────────────────────
genre             18    1.00    1.00      10%   + 0.00
compound           8    1.00    1.00      16%   + 0.00
decade             5    0.16    0.52      88%   +0.36
thematic           6    0.90    1.00      50%   +0.10
mood               4    0.60    1.00      49%   +0.40
longtail           4    0.80    1.00      57%   +0.20
──────────────────────────────────────────────────────
OVERALL           45    0.84    0.95      33%   +0.11

Avg LLM label consistency: 96%
```

---

## Key Findings

### 1. The hypothesis holds across temporal, thematic, mood, and long-tail queries

Decade queries showed dramatic improvement: **+0.36 P@5 average**, with individual queries jumping from near-zero to perfect:

```
"10s movies"   Pipeline3: 0.20 P@5  →  Pipeline4: 1.00 P@5  (+0.80)
               Filter removed 25 of 40 candidates (63%)
               Top-5 P4: Inception ✓, The Dark Knight ✓, Interstellar ✓,
                         Django Unchained ✓, Guardians of the Galaxy ✓

"00s movies"   Pipeline3: 0.00 P@5  →  Pipeline4: 0.40 P@5  (+0.40)

"90s movies"   Pipeline3: 0.00 P@5  →  Pipeline4: 0.20 P@5  (+0.20)
```

**Why decade queries benefit so much:** The cross-encoder scores titles on semantic similarity to the query string "10s movies" — which is meaningless to a neural model trained on prose. The LLM judge understands that "10s movies" means films from 2010–2019 and correctly labels older titles as Irrelevant, leaving the reranker a clean candidate pool.

Mood queries saw the largest average gain at **+0.40 P@5**:

```
"feel good movies"   P3: 0.20  →  P4: 1.00  (+0.80)
                     Filter removed 29 of 39 candidates (74%)

"funny horror"       P3: 0.80  →  P4: 1.00  (+0.20)
                     Goosebumps ✓, Gremlins ✓, Ghostbusters ✓ in top 5
```

Long-tail queries also improved substantially: **+0.20 average**, from P3 0.80 → P4 1.00.

### 2. Genre and compound queries are already at ceiling — filter adds nothing

Simple genre queries already achieve **P@5 = 1.00** with Pipeline 3. The retrieval stage (BM25 + semantic) correctly surfaces only relevant titles; there's nothing to filter. Adding the LLM filter at 10–16% removal rate changes nothing.

**Implication for production:** Running the LLM filter on genre queries costs API latency with zero metric improvement. The filter should be deployed selectively.

### 3. Filter rate signals query difficulty

Average filter rate of 33% across all queries masks a meaningful pattern: genre queries remove only 10% of candidates (retrieval is already precise), while decade and mood queries remove 49–88% (retrieval surfaces many wrong candidates that the LLM correctly strips out). High filter rate is a signal that the query type needs the filter most.

### 4. LLM judge calibration: confidence vs ambiguity

The judge is highly consistent (97%) but over-strict on abstract mood queries (`dark psychological`, `mindblowing sci-fi`, `light hearted comedy`) — labeling all 30–38 candidates as Irrelevant. These are real edge cases where the judge's context (title + genres only) isn't enough to make a correct relevance call. A more robust judge would also pass the plot summary.

This is a known limitation of the Etsy-style framework: **judge quality is bounded by the input context.** Title + genre is sufficient for decade and genre queries; it's not sufficient for mood and abstract descriptive queries.

---

## Production Recommendation

Based on these results, the deployment strategy is:

| Query type | Deploy filter? | Reason |
|-----------|---------------|--------|
| Genre | No | Already at P@5 ceiling; filter adds only latency |
| Compound | No | Already at P@5 ceiling |
| Decade / Temporal | **Yes** | +0.32 average lift; LLM understands temporal intent |
| Thematic | **Yes** | +0.10 average lift; filter cleans noisy candidates |
| Mood | **Yes (with richer context)** | +0.20 lift but needs plot summary as input |
| Long-tail | Needs better retrieval first | Fix recall before filtering |

**Query classification** can route live traffic to the right pipeline at query time — a lightweight text classifier trained on query logs handles this.

---

## Cost & Scale

| Scale | Pairs to label | Claude Haiku cost (est.) |
|-------|---------------|--------------------------|
| This eval | 1,718 | ~$0.05 |
| 10,000 queries × 40 candidates | 400,000 | ~$10 |
| Production (1M queries) | 40M pairs | ~$1,000 (batch) |

At production scale, the LLM judge runs offline as a batch labeler — not in the live serving path. Labels are cached, refreshed on catalog changes, and distilled into the reranker as training signal. The serving-time cost of Pipeline 4 is just the lightweight filter lookup, not a live Claude call.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# Download catalog (run once — requires the TMDb CSV)
python3 data/fetch_catalog.py

# Run full eval
python3 run_eval.py
# First run: calls Claude API for labeling (~$0.05 for 45 queries)
# Subsequent runs: loads from cache instantly

# Interactive explorer — test any query side by side
python3 explore.py
# > feel good movies
# > action thriller heist
# > 90s coming of age
# Type any query not in the eval set: offered live Claude labeling
```

---

## Interactive Demo (`explore.py`)

```
Enter query: feel good movies

════════════════════════════════════════════════════════════
  Query: "feel good movies"  [mood]
  Labels: 39 cached  |  Consistency: 97%
════════════════════════════════════════════════════════════

  PIPELINE 3  (Retrieval → Rerank)      PIPELINE 4  (+ LLM Filter)  removed 29/39
  ──────────────────────────────────    ──────────────────────────────────────────
  1. Forrest Gump ✗                      1. Wayne's World ✓
  2. Transformers: Dark of the Moon ✗    2. 102 Dalmatians ✓
  3. The Dark Knight ✗                   3. Grease ✓
  4. Batman v Superman ✗                 4. Wreck-It Ralph ✓
  5. Iron Man 3 ✗                        5. Trainwreck ✓

  P@5:   Pipeline3 0.20   Pipeline4 1.00   delta +0.80
  NDCG:  Pipeline3 0.20   Pipeline4 1.00
```

The explorer supports any free-form query. If it's not in the labeled cache, it offers live Claude labeling for the candidate set.

---

## Project Structure

```
semantic-search-eval/
├── run_eval.py          # Full Pipeline 3 vs 4 evaluation
├── explore.py           # Interactive side-by-side explorer
├── pipelines.py         # Pipeline3, Pipeline4, _retrieve_union
├── rankers.py           # BM25, semantic, cross-encoder
├── llm_judge.py         # Claude labeler, dual-prompt consistency
├── query_generator.py   # Auto-generates queries from catalog metadata
├── eval.py              # P@5, NDCG@5, filter_impact
├── catalog.py           # 1000-title TMDb catalog (auto-generated)
├── data/
│   └── fetch_catalog.py # Downloads and samples TMDb CSV
└── results/
    ├── queries.json      # Cached query set
    ├── llm_labels.json   # Cached LLM labels (~1,718 pairs)
    └── eval_report.json  # Full per-query and aggregate results
```

---

## Relation to Etsy's Work

Etsy's Jan 2026 paper uses LLM labels as **training signal** to fix a ranker trained on biased engagement data. This repo implements the **evaluation half** of that loop: the offline label-and-measure framework that tells you *where* the ranker is failing and *what kind* of query benefits from the LLM filter.

The production loop would be:

```
Label (this eval) → Identify query types with lift → Deploy filter selectively
       ↓
Collect labeled pairs at scale → Distill into deep ranker → Retrain
       ↓
Measure again with this eval → repeat
```

---

## Limitations

- **Catalog size**: 1000 titles is demonstrably small for long-tail and mood queries where true relevant content may not exist in the catalog.
- **Labels are not ground truth**: LLM relevance labels are a proxy for user satisfaction. Real validation requires A/B testing against watch time and engagement.
- **Judge input is title + genre only**: Mood queries need plot summary context to label correctly.
- **No real query logs**: Queries are auto-generated from catalog metadata. Real user query distributions would differ.
