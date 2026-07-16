# semantic-search-eval

**Does an LLM filter make reranking better — and on which queries does it help most?**

A search relevance evaluation framework for streaming catalogs that compares two production-grade pipeline architectures head-to-head:

- **Pipeline 1** — BM25 + Semantic Retrieval → Cross-encoder Rerank
- **Pipeline 2** — BM25 + Semantic Retrieval → LLM Filter → Cross-encoder Rerank

Pipeline 1 is the standard two-stage retrieval-rerank setup. Pipeline 2 adds one step: an LLM relevance filter that removes irrelevant candidates before the reranker runs. The question this eval answers is whether that extra step is worth it — and on which query types.

**▶ [Live interactive demo](https://niveditabuilds.github.io/semantic-search-eval/)** — explore all 45 queries with Pipeline 1 vs Pipeline 2 side by side, scored by the independent judge.

Catalog: TMDb 1000 titles. Queries: auto-generated from catalog metadata. Labels: two independent judges — Claude Sonnet drives the filter, a separate judge (Claude Haiku by default, GPT optional) scores the result.

---

## The Hypothesis

**LLM relevance labels, used as a pre-filter before reranking, remove noise from the candidate pool and let the reranker focus on genuinely relevant titles.**

The key claim:
> LLM filtering should help most on queries where the retrieval stage surfaces many semantically-adjacent-but-wrong candidates — compound, thematic, and temporal queries — and add no value on simple genre queries where the retrieval ceiling is already 100% relevant.

This repo tests that claim end-to-end, on a real catalog, with 45 programmatically-generated queries and two independent LLM judges: a **system judge** (Claude Sonnet) whose labels drive the Pipeline 2 filter, and a separate **eval judge** (a different model — Claude Haiku by default, or GPT) whose labels are the *only* labels used to score P@5/NDCG for both pipelines. Cross-model agreement and Cohen's kappa (by query type) report how much the two judges agree — without ever letting the filter's own judge grade its output.

> **Why two judges?** Using a single judge for both jobs would be circular — a filter that deletes everything its judge calls Irrelevant, then gets graded by that same judge, is structurally guaranteed to look good. Splitting the labels into two independent stores (a different model for scoring, which never sees the filter's decisions) is what makes Pipeline 2's measured lift a real result instead of a tautology.

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
  Pipeline 1                    Pipeline 2
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
| System judge | Claude Sonnet (`llm_judge.py`) | Labels each candidate Relevant / Irrelevant — drives the Pipeline 2 filter only |
| Eval judge | Independent model (`eval_judge.py`) — Claude Haiku default, GPT optional | Independently labels each candidate — used only to score P@5/NDCG for both pipelines |
| Cross-encoder reranker | `ms-marco-MiniLM-L-6-v2` | Precision stage — scores query-title relevance |

**Why both BM25 and semantic retrieval?** Each covers different failure modes. BM25 finds "horror movies" even when genre isn't in the embedding. Semantic retrieval finds "dark psychological thriller" when BM25 finds nothing. The union maximizes recall before the precision stage runs.

**LLM filter is conservative.** Only candidates with a cached `Irrelevant` label are removed. Unlabeled candidates pass through. This is intentional — a false positive removal (suppressing a relevant title) is worse than leaving noise for the reranker.

**Why pre-filter at all — why not let the reranker handle it?** Cross-encoders are strong at ordering but they don't discard — they score everything relative to everything else in the candidate pool. A score of 0.7 means something different in a clean pool of 10 candidates vs a noisy pool of 35. When retrieval surfaces many wrong candidates, the reranker's score distribution gets pulled by the noise and genuinely relevant titles can get crowded out. The filter clears the noise first so the reranker operates on a signal-rich pool. The "feel good movies" result demonstrates this directly: the cross-encoder kept promoting titles with "good" in the name not because it was broken, but because retrieval handed it a pool where those were the closest matches.

**This approach vs retraining the ranker.** Pre-filtering is a fast, zero-retraining improvement — you can ship it as soon as you have a label cache. The longer-term play, as shown in the production loop below, is distilling these labels into the ranker as training signal so the ranker itself learns to handle noisy queries. Pre-filtering and ranker distillation are complementary: filter buys you the improvement today, distillation makes the filter unnecessary over time.

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

### LLM judges

Each (query, candidate title) pair is labeled by two independent judges, each writing to its own cache:

- **System judge** (`llm_judge.py`, Claude Sonnet) → `results/system_labels.json`. These labels drive the Pipeline 2 filter — `pipelines.py` reads only this file.
- **Eval judge** (`eval_judge.py`, a *different* model) → `results/eval_labels.json`. These labels are the sole ground truth for P@5/NDCG — `eval.py`'s scoring functions read only this file. Defaults to Claude Haiku; set `JUDGE_PROVIDER=openai` (with `OPENAI_JUDGE_MODEL` + `OPENAI_API_KEY`) to run it as GPT for a fully cross-provider check.

Separating the two prevents the eval from being circular: if the same labels both filtered candidates out and graded the result, Pipeline 2 could never lose — it would just be scored against its own decisions. Each judge writes to its own store, keyed by model name + prompt version, so the two never collide and swapping the eval model never contaminates the filter's labels.

The judge prompt includes the title's **release year** (so decade queries like "80s movies" don't rely on the model recalling release dates from memory), runs at **`temperature=0`**, and caches under a key that includes the **model name and prompt version**, so a prompt or model change invalidates stale entries automatically instead of silently reusing them.

Both judges are labeled once and cached. Re-runs load from cache, costing zero API calls on cached queries and labeling only new ones. A failed API call is **never cached as a fabricated `Irrelevant` label** — it's skipped and retried on the next run, so a transient outage or exhausted quota can't quietly poison the label set.

**Cross-model agreement and Cohen's kappa** (by query type) are the judge-quality signal — see Results below. Agreement between two independent models is a real inter-annotator signal: it measures whether the labels are *correct*, not merely that one model is internally consistent.

> **Note on the results below:** these were produced with the default eval judge (Claude Haiku) — cross-*model* but same-provider as the Sonnet filter. That still breaks the circularity (different model, separate store, scoring never sees the filter's labels). For a fully cross-*provider* run, set `JUDGE_PROVIDER=openai` and re-run; labels are keyed by model, so the GPT run coexists with the Haiku one rather than overwriting it.

### Metrics

- **P@5** (Precision at 5): fraction of the top-5 results that are `Relevant`. Primary metric.
- **NDCG@5**: position-aware P@5 — a relevant result at position 1 scores higher than at position 5.
- **Filter%**: percentage of candidates removed by the LLM filter before reranking.
- **Delta**: Pipeline 2 P@5 minus Pipeline 1 P@5.

---

## Results

Scored against the **independent eval judge** (1,602 (query, title) pairs, each labeled by both judges).

**Column guide:**
- **P1 P@5** — Pipeline 1 (Retrieval → Rerank, no filter): fraction of top-5 results that are relevant
- **P2 P@5** — Pipeline 2 (Retrieval → LLM Filter → Rerank): same metric after the LLM filter runs
- **Filter%** — percentage of candidates Pipeline 2 removed before reranking
- **Delta** — P2 minus P1: how much the LLM filter improved precision

```
========================================================
AGGREGATE BY QUERY TYPE (45 queries, 1602 labeled pairs)
========================================================

Type         Queries  P1 P@5  P2 P@5  Filter%    Delta
──────────────────────────────────────────────────────
genre             18    1.00    1.00      14%   + 0.00
compound           8    1.00    1.00      18%   + 0.00
decade             5    0.24    0.48      93%   +0.24
thematic           6    0.90    0.93      61%   +0.03
mood               4    0.65    1.00      60%   +0.35
longtail           4    0.70    0.95      59%   +0.25
──────────────────────────────────────────────────────
OVERALL           45    0.84    0.93      38%   +0.09
```

**Judge-quality check** (system judge vs independent eval judge):

```
Avg cross-model agreement:  92%
Cohen's kappa (overall):    0.820   (substantial agreement)

Cohen's kappa by query type
──────────────────────────────
compound   0.845   (n=287)
longtail   0.822   (n=139)
genre      0.767   (n=612)
mood       0.710   (n=143)
thematic   0.665   (n=221)
decade     0.504   (n=200)   ← weakest agreement
```

Full per-query detail is written to `results/eval_report.json`.

---

## Key Findings

### 1. The filter adds nothing on 26 of 45 queries — and that's the honest result

Every **genre** (18) and **compound** (8) query is already at **P@5 = 1.00 with retrieval alone**, and the filter moves them **+0.00** — even when it removes a lot (documentary −62%, western −36%, music −33%). Retrieval is already at ceiling; there's nothing left to fix.

This is the cleanest evidence the eval is honest. A ceiling *cannot* be inflated by self-grading, so genre/compound reading exactly +0.00 under an independent judge is what a trustworthy eval should show — the filter isn't credited for work retrieval already did.

### 2. Where the filter earns its keep: mood, long-tail, decade

| Type | Delta | Flagship query |
|------|------:|----------------|
| mood | **+0.35** | `feel good movies` 0.20 → **1.00** (filter cut 27 of 39) |
| longtail | **+0.25** | `superhero origin story` 0.60 → 1.00 |
| decade | **+0.24** | `00s movies` 0.40 → 1.00 (biggest single-query gain) |

These are queries where retrieval surfaces *semantically-adjacent-but-wrong* candidates — a title containing "good", or a popular film from the wrong era — and the filter strips them before the reranker locks them into the top-5. `feel good movies` is the canonical case: Pipeline 1 matched on the literal word "good" (*A Good Day to Die Hard*, *No Good Deed*, *The Good German*); the filter removed them and Pipeline 2 reached 1.00.

### 3. Decade is a filter *success* and a recall *failure* — and they're separable

Decade queries have the **highest filter rates in the eval (85–98%)** but the **lowest P@5 (0.48 avg)**. The split within the category is the story:

```
"00s movies"   P1 0.40 → P2 1.00  (+0.60)   filter cut 85%   — enough right-era films exist
"90s movies"   P1 0.00 → P2 0.40  (+0.40)   filter cut 95%
"80s movies"   P1 0.20 → P2 0.40  (+0.20)   filter cut 95%
"70s movies"   P1 0.20 → P2 0.20  (+0.00)   filter cut 98%   — nothing left to promote
"10s movies"   P1 0.40 → P2 0.40  (+0.00)   filter cut 92%   — judges disagree (see #4)
```

When the catalog holds only a handful of era-correct films, the filter can strip *all* the noise and there still aren't five right answers to fill the top-5. **The filter can't add what retrieval never retrieved.** High filter rate signals query *difficulty*, not filter *benefit* — the two are decoupled.

### 4. Cross-model agreement is where the honest caveat lives

Overall the two independent judges agree **92%** of the time (**kappa 0.820, substantial**), so the eval labels aren't arbitrary. But agreement is uneven, and the weak spot is pointed:

- **decade has the lowest kappa (0.504)**, and `10s movies` alone drops to **0.50 agreement** — the two models genuinely can't agree on what counts as a 2010s film. This is the exact category where the filter does the *most* work (92–98% removal). **The filter's heaviest lifting rests on its judges' shakiest agreement** — a flag that only a two-model agreement check can surface, since a single model agreeing with itself says nothing about whether the label is right.
- Abstract queries also dip: `mindblowing sci fi` 0.74, `childhood wish` 0.74, `music` 0.77 — subjective intent is harder for two models to converge on.

Because the eval judge never sees the filter's decisions, it sometimes calls a *filter-kept* title Irrelevant — which is why Pipeline 2 lands below a perfect 1.00 on decade (0.48), thematic (0.93), and long-tail (0.95) rather than looking artificially flawless.

---

## Production Recommendation

**Ship the filter** — labels are computed offline and cached, so at serving time it's a dictionary lookup, not a live API call. There's no latency cost to leaving it on for every query, including genre queries where it's inert.

Where the filter actually moves the metric:

| Query type | P@5 lift | Inter-judge kappa | What the filter does |
|-----------|---------:|------------------:|----------------------|
| Genre | +0.00 | 0.767 | No effect — retrieval already at ceiling |
| Compound | +0.00 | 0.845 | No effect — retrieval already at ceiling |
| Mood | +0.35 | 0.710 | Largest gain — removes literal keyword matches (`good` → `feel good`) |
| Long-tail | +0.25 | 0.822 | Cleans niche candidates that lack the subgenre signal |
| Decade | +0.24 | **0.504** | Strips wrong-era titles — but see caveat |
| Thematic | +0.03 | 0.665 | Marginal — retrieval was already strong |

**Two caveats the eval surfaces:**

1. **Decade is the least defensible win.** It has the biggest filter deltas *and* the lowest inter-judge agreement (kappa 0.504). Before leaning on it in production, validate temporal relevance against real release-date metadata rather than model judgment.
2. **Decade needs a recall fix, not more filtering.** The filter already removes 85–98% of decade candidates and P@5 is still only 0.48 — the bottleneck is that retrieval surfaces too few era-correct films. Better temporal retrieval / metadata filtering will move decade P@5 far more than the LLM filter can.

Deploy it — just with eyes open about where its decisions are solid (mood, long-tail) versus shaky (decade).

---

## Cost & Scale

Every pair is labeled twice — once by the system judge (Sonnet) and once by the independent eval judge (Haiku or GPT) — so total labeling cost is roughly 2× a single-judge setup.

| Scale | Pairs to label | System judge (est.) | Eval judge (est.) |
|-------|---------------|--------------------|------------------|
| This eval | 1,602 | ~$0.05 | ~$0.02 |
| 10,000 queries × 40 candidates | 400,000 | ~$10 | ~$5 |
| Production (1M queries) | 40M pairs | ~$1,000 (batch) | ~$500 (batch) |

At production scale, only the **system judge** needs to run continuously — it's the one that drives the live filter. The **eval judge** only runs over a representative sample to keep measuring whether the filter still helps; it's an offline measurement judge, not a serving-path dependency. Labels are cached, refreshed on catalog changes, and distilled into the reranker as training signal. The serving-time cost of Pipeline 2 is just the filter lookup, not a live API call.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key. The default eval judge (Claude Haiku) uses the same Anthropic key
# as the Sonnet system judge — one key runs the whole eval.
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=your_key_here
#
# Optional — for a fully cross-provider eval judge (GPT instead of Haiku):
#   export JUDGE_PROVIDER=openai
#   Edit .env: OPENAI_API_KEY=your_key_here

# Download catalog (run once — requires the TMDb CSV)
python3 data/fetch_catalog.py

# Run full eval
python3 run_eval.py
# First run: labels each pair with both judges (~$0.07 for 45 queries)
# Subsequent runs: loads from cache instantly; only new/failed pairs are re-labeled

# Interactive explorer — test any query side by side
python3 explore.py
# > feel good movies
# > action thriller heist
# > 90s coming of age
# Type any query not in the eval set: offered live Claude labeling
```

---

## Interactive Demo (`explore.py`)

Titles marked ✓/✗ below reflect the **independent eval-judge** labels — the same labels used to score P@5.

**Example 1 — Mood query: "feel good movies"** (filter removed 27/39)

```
  PIPELINE 1  (Retrieval → Rerank)       PIPELINE 2  (+ LLM Filter)
  ───────────────────────────────────    ───────────────────────────────
  1. A Good Day to Die Hard        ✗     1. 102 Dalmatians              ✓
  2. Midnight in the Garden of G.. ✗     2. Pitch Perfect 2             ✓
  3. No Good Deed                  ✗     3. Trainwreck                  ✓
  4. The Good German               ✗     4. The Good Dinosaur           ✓
  5. Good Boy!                     ✓     5. The Sweetest Thing          ✓

  P@5:   Pipeline 1: 0.20   Pipeline 2: 1.00   delta +0.80
  NDCG:  Pipeline 1: 0.39   Pipeline 2: 1.00
```

Pipeline 1 matched on the literal word "good" — four of its top five are action/drama films with "good" in the title. The filter removed them, and Pipeline 2's reranker operated on a clean pool of genuinely feel-good titles.

---

**Example 2 — Decade query: "00s movies"** (filter removed 34/40)

```
  PIPELINE 1  (Retrieval → Rerank)       PIPELINE 2  (+ LLM Filter)
  ───────────────────────────────────    ───────────────────────────────
  1. Dr. No                        ✗     1. Straightheads               ✓
  2. The Expendables               ✓     2. Pirates of the Caribbean..  ✓
  3. Ghostbusters                  ✗     3. The Dark Knight             ✓
  4. Batman v Superman: Dawn of .. ✗     4. Inception                   ✓
  5. Straightheads                 ✓     5. Funny Games                 ✓

  P@5:   Pipeline 1: 0.40   Pipeline 2: 1.00   delta +0.60
  NDCG:  Pipeline 1: 0.62   Pipeline 2: 1.00
```

Pipeline 1 surfaced popular films from the wrong era — *Dr. No* (1962), *Ghostbusters* (1984). The filter used the release year in the judge prompt to strip everything outside 2000–2009, and Pipeline 2's top five are all genuine 2000s films. (This is the eval's biggest single-query gain — but note decade queries also have the lowest cross-model agreement; see Key Finding #4.)

---

**Example 3 — Long-tail query: "superhero origin story"** (filter removed 26/34)

```
  PIPELINE 1  (Retrieval → Rerank)       PIPELINE 2  (+ LLM Filter)
  ───────────────────────────────────    ───────────────────────────────
  1. Deadpool                      ✓     1. Deadpool                    ✓
  2. Iron Man 2                    ✗     2. Megamind                    ✓
  3. Megamind                      ✓     3. Superman                    ✓
  4. Superman                      ✓     4. Captain America: The First. ✓
  5. Batman v Superman: Dawn of .. ✗     5. The Incredible Hulk         ✓

  P@5:   Pipeline 1: 0.60   Pipeline 2: 1.00   delta +0.40
  NDCG:  Pipeline 1: 0.91   Pipeline 2: 1.00
```

Pipeline 1 mixed in superhero *sequels* (*Iron Man 2*, *Batman v Superman*) that aren't origin stories. The filter removed them, leaving only true origin films.

The explorer supports any free-form query. If it's not in the labeled cache, it offers live labeling for the candidate set.

---

## Project Structure

```
semantic-search-eval/
├── run_eval.py          # Full Pipeline 1 vs 2 evaluation
├── explore.py           # Interactive side-by-side explorer
├── pipelines.py         # Pipeline1, Pipeline2, _retrieve_union
├── rankers.py           # BM25, semantic, cross-encoder
├── llm_judge.py         # System judge (Claude Sonnet) — labels feed the Pipeline 2 filter only
├── eval_judge.py        # Eval judge (Haiku/GPT) — labels feed P@5/NDCG scoring only
├── query_generator.py   # Auto-generates queries from catalog metadata
├── eval.py              # P@5, NDCG@5, filter_impact, cross-model agreement, Cohen's kappa
├── catalog.py           # 1000-title TMDb catalog (auto-generated)
├── data/
│   └── fetch_catalog.py # Downloads and samples TMDb CSV
└── results/
    ├── queries.json       # Cached query set
    ├── system_labels.json # Cached Sonnet labels — read only by pipelines.py  (gitignored, regenerated)
    ├── eval_labels.json   # Cached eval-judge labels — read only by eval.py   (gitignored, regenerated)
    └── eval_report.json   # Full per-query and aggregate results
```

---

## The Production Loop

This repo implements the **evaluation half** of a larger system: the offline label-and-measure framework that tells you *where* the ranker is failing and *what kind* of query benefits from the LLM filter.

```
Label (this eval) → Identify query types with lift → Deploy filter selectively
       ↓
Collect labeled pairs at scale → Distill into deep ranker → Retrain
       ↓
Measure again with this eval → repeat
```

---

## Limitations

- **Catalog size**: 1000 titles is small — recall is limited for niche and temporal queries. This is the binding constraint on decade queries (see Key Finding #3), where the filter can't promote era-correct films that retrieval never surfaced.
- **Labels are not ground truth**: LLM relevance labels are a proxy for user satisfaction. Real validation requires A/B testing against watch time and engagement. Cross-model agreement (kappa 0.82) tells you the two judges concur, not that they're *right*.
- **Same-provider eval judge by default**: the default eval judge (Claude Haiku) is a different model but the same provider as the Sonnet filter. This breaks the circularity, but a fully cross-provider check (`JUDGE_PROVIDER=openai`) is a stronger guard against shared model biases — run it when quota allows.
- **Judge input is title, genres, release year, and plot summary**: still bounded — abstract mood/thematic queries remain the hardest to label (their lower kappa reflects this), and no amount of metadata fully resolves subjective intent.
- **No real query logs**: Queries are auto-generated from catalog metadata. Real user query distributions would differ.
