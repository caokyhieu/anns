# Filtered vector search: six papers checked against our measurements

This note reads six recent papers on hybrid (filter + vector) search in full, lists what
each one claims, checks the claims against what we measured in FILTERED_SEARCH.md, and
says what to take into the Lance AV retrieval design. Numbers are the papers' own unless
marked **ours**.

| Paper | Venue | Kind |
| --- | --- | --- |
| Li et al., *Attribute Filtering in ANN Search: An In-depth Experimental Study* | SIGMOD 2026 (arXiv 2508.16263) | Benchmark of 12 methods, 0.1–100% selectivity |
| Amanbayev et al., *Filtered ANN Search in Vector Databases: System Design and Performance Analysis* | arXiv 2602.11443, 2026 | System study of FAISS, Milvus, pgvector |
| Wu & Tang, *PathFinder: Conjunctions and Disjunctions for Filtered ANNS* | arXiv 2511.00995, 2025 (VLDB-format preprint) | Planner over per-attribute graph indexes |
| Li et al., *SIEVE: Filtered Vector Search with Collection of Indexes* | PVLDB 18(11), 2025 | Workload-driven set of sub-indexes |
| Gupta et al., *CAPS: A Practical Partition Index for Filtered Similarity Search* | arXiv 2308.15014, 2023 (preprint) | IVF with per-partition attribute sub-partitions |
| Sehgal & Salihoğlu, *NaviX: Native Vector Index for Graph DBMSs* | PVLDB 18(11), 2025 | Adaptive filtered HNSW search, disk-based |

## 1. Li et al., SIGMOD 2026: experimental study

**Setup.** 10 algorithms (range, label and arbitrary filters) on 4 datasets up to 10M
vectors, all in memory, recall@10 target 0.9. Attributes are synthetic and **uniform**,
or real labels for two datasets. Multi-predicate tests use one label plus one range.

**Claims.**
1. Faiss-HNSW and ACORN cannot reach 0.9 recall at 0.1% selectivity. Faiss-IVFPQ handles
   0.1% on every dataset: graph search is "monotonic" and stops early, while IVF keeps
   scanning the probed partitions.
2. Partitioning makes narrow filters viable (Milvus partitions by one attribute):
   inside a partition, the filter is much less selective.
3. Above 50% selectivity, filter-aware indexes add nothing; plain Faiss-HNSW is best.
4. RNG-style edge pruning breaks down at low selectivity. Keeping plain nearest
   neighbors (KGraph style) does better below about 5%.
5. More bottom-layer entry points (30 or 300 instead of 3) always help; the HNSW
   hierarchy only helps above 50% selectivity.
6. Their guide for arbitrary filters: Faiss-HNSW above 50%, ACORN in between, a scan
   (Milvus-HNSW) below 1%.
7. Open problems they name: arbitrary filters are unsolved; uniform attributes are
   unrealistic and the vector-attribute relationship is unstudied; only one method
   supports updates.

**Against our measurements.**
- Claim 1 matches the shape but not the reason we saw. Our hnswlib run kept recall 1.0 at
  0.1%, but only by touching all 200,000 nodes (193 ms), so at a fixed cost it fails as
  they report. Our Lance IVF run did **not** stay reliable at 0.1% (recall 0.67 with
  `maximum_nprobes`). The difference: they tune nprobe until recall is met, which at
  0.1% amounts to scanning the passing rows. Our planner does the same deliberately and
  reaches 1.0.
- Claims 2 and 6 match our regimes (graph for broad filters, scan for narrow ones) and
  support the per-label tables in FILTERED_SEARCH.md §4.3.
- Claim 4 matches our degree test (below): the pruned HNSW graph with base degree 32
  gives 2-hop search 0.28–0.49 recall at 1%; degree 64 gives 0.55–0.82.
- Open problem 7 is the main caveat for us: with uniform attributes, they never test
  the anti-correlated case, which was our worst case (`tunnel`, query outside). Their
  thresholds are optimistic for AV labels.

## 2. Amanbayev et al., 2026: system study

**Setup.** FAISS, Milvus 2.6 and pgvector 0.8 with HNSW and IVFFlat; MoReVec (IMDb movie
and review text, 768-d), up to 2.6M vectors; **scalar range filters only**, on attributes
they measured to be nearly independent of the embeddings; in memory; k = 1 to 100.

**Claims.**
1. **GLS correlation**: r = σ_local / σ_global, where σ_local is the pass rate among the
   query's unfiltered k nearest neighbors; ρ = (r − 1) / (r + 1) in [−1, 1). Lower ρ means
   lower recall at the same latency. Mean ρ is about 0, so global selectivity is enough
   to choose a plan, and ρ explains per-query variance.
2. IVFFlat beats HNSW below about 5% selectivity; HNSW wins above about 20%. Keep both
   for mixed workloads.
3. Milvus keeps recall high because of three mechanisms, not segmentation: a
   **dual-pool** graph search (separate queues for valid results and for navigation
   nodes, so filtered-out nodes do not use up `ef`); a **brute-force switch when more
   than 93% of rows are filtered out**; and a **safety net** that re-runs exactly when
   the graph search returns fewer than k rows.
4. pgvector's cost model often picks an approximate index scan where a filter-then-exact
   plan would give recall 1.0 at similar latency.
5. Recall falls as k grows unless `ef`/`nprobe` grow with k.
6. A denser HNSW graph (larger M) gives little recall at high build cost.

**Against our measurements.**
- Claim 3's dual-pool rule is what our `traverse` policy and hnswlib's filter do (the
  result queue holds only passing rows). It never lost recall in our test, but its work
  grew as about ef / s, as their "latency floor" remark says.
- Claim 3's 93% switch and safety net fix exactly the two Lance problems we found:
  prefiltered IVF returns k rows with low recall, and nothing checks for that.
- Claim 6 holds for dual-pool search (density hardly matters when every node can be
  walked through) but not for 2-hop search on the passing subgraph, where our degree
  test shows density decides recall at 1%.
- Claim 1: their data has ρ near 0 because they picked independent attributes; they list
  "highly correlated metadata" as outside their validity. AV scenario labels (tunnel,
  night, highway) are likely correlated with the embeddings, positively or negatively per
  query, so per-query ρ matters more for us than for them.

## 3. PathFinder, 2025: planning AND / OR over per-attribute indexes

**Method.** The administrator builds indexes on chosen attributes: a **tree index** for a
numeric attribute (a B+-tree-like range tree with a Vamana graph per node) or a **hash
index** for a categorical attribute (one graph per value), plus a root graph over all
rows. A filter is rewritten in DNF. For each AND clause, the optimizer takes up to two
candidate plans per index (the deepest single graph that covers the clause, and a set of
smaller child graphs) and keeps the plan with the highest **search utility**

  U(G, p) = card(R, p) / (Σ card(g) · |G|^α), α = 0.4,

which rewards graphs dense in passing rows and penalizes searching many graphs. Ranking
needs only precomputed graph sizes, not cardinality estimates. OR clauses are merged:
duplicate graphs are removed and sets of graphs can be replaced by a common ancestor if
that raises utility. **Index borrowing** serves a predicate on an unindexed attribute
through a correlated indexed attribute. Search uses out-of-range (dual-pool) traversal.

**Claims.** Recall close to 1.0 on all workloads (four datasets, 132K–6.9M rows, two or
three attributes per filter). Up to 18.4× QPS at recall 0.9 on conjunctions and 9.8× at
recall 0.95 on mixed AND/OR over the best baseline (a random choice among the same
indexes); optimizer overhead 0.1–5.4% of query time; index borrowing up to 2.44× when
attributes are strongly correlated. Memory is higher than ACORN and Vamana; going from 7
to 2 tree levels cuts it 3.7× and still beats all baselines.

**Limits.** Multi-valued label sets (our `scenario_tags`) are not indexed: "label data
via the root graph" is left for future work. Categorical attributes hold one value per
row. In memory, single node.

**Against our design.** This is the anchor-plus-mask plan for AND/OR I described, with a
cost function: pick the index whose graph is densest in passing rows, apply the rest as a
mask, take the union for OR. Lance gives exact counts from bitmap intersections cheaply,
so we can compute card(R, p) and the per-structure pass rate directly instead of relying
on the ranking trick. The tree index for numeric attributes is what our time partitions
already are.

## 4. SIEVE, PVLDB 2025: a workload-driven collection of sub-indexes

**Method.** Candidates are HNSW sub-indexes over the rows matching filters seen in a
historical workload, arranged in a DAG by logical subsumption (A∨B subsumes A). Cost
model (Table 3 of the paper):
- indexed search on sub-index h for filter f: log(card(h)) · sef · (card(h)/card(f))^cor;
- brute force: γ · card(f);
- index size: M · card(h), with M scaled down as M∞ · log(card(h)) / log(N).

A greedy algorithm adds the sub-index with the best marginal benefit per byte until a
memory budget (3× one HNSW index) is reached. The full-table index is always kept as the
fallback. At query time, SIEVE picks the smallest sub-index that subsumes the filter,
scales `sef` down the same way, and runs brute force instead if the model says it is
cheaper.

**Claims.** Up to 8.06× QPS over ACORN-γ at recall 0.9 (YFCC-10M); memory up to 2.15×
of hnswlib; build time 0.8–9.5% of ACORN-γ's. Fitting on 25% of the workload gives 96% of
the QPS of fitting on all of it. A cold start reaches 97% after three 5K-query slices.
After a complete workload shift, performance drops to 92%, 71% and 49% of a refitted
collection depending on how large the attribute space is; refitting is faster than a
rebuild because the base index stays.

**Limits.** Assumes stable filters, one constant correlation factor (0.5), logical
subsumption only, and one sub-index per query (no union for OR).

**Against our design.** This is the per-label tables of FILTERED_SEARCH.md §4.3, with a
principled way to choose them (benefit per byte under a budget, from the query log) and
the flat-or-index decision of our planner (γ · card(f) against the index cost). Two
differences for us:
- Our sub-tables copy embeddings, which dominate storage at petabyte scale, so the
  budget is in vector bytes, not graph bytes. That favors rare filters even more.
- Labels change every annotation release, so sub-index membership changes; SIEVE's
  incremental refit (keep the base, rebuild only changed sub-indexes) fits the release
  pipeline.

## 5. CAPS, 2023 preprint: attribute sub-partitions inside IVF

**Method.** IVF (balanced k-means, or BLISS learned partitions), then inside each
partition an **Attribute Frequency Tree**: split off the rows with the most frequent
attribute value, repeat on the rest for h levels, and keep a remainder bucket. A query
probes the m nearest partitions, picks the matching sub-partitions in O(1), checks
attributes exactly, and scores only those rows. Supports AND over a variable number of
attributes, rare attributes and inserts.

**Claims.** Better recall-QPS than NHQ, AIRSHIP and Filtered-DiskANN in the high-recall
range; index overhead about 10× smaller than the next smallest (NHQ); beats Amazon's
production system in an auto-complete case study (relative numbers only, 8M × 768-d).

**Limits.** Synthetic, independently drawn attributes (3 attributes); no correlation
tests; single-threaded comparisons. SIEVE later measured CAPS 10.61× slower than SIEVE on
its Paper dataset and notes CAPS handles fewer than 256 attributes and only conjunctive
attribute matches.

**Against our design.** This is the Lance-native option: it keeps IVF and only changes
the row order inside each partition, so a filtered scan reads matching row ranges
instead of masking whole partitions. It extends the partition × label counts of
FILTERED_SEARCH.md §4.2 from "skip empty partitions" to "read only matching rows". It
does not fix partition choice: the m nearest partitions are still chosen without looking
at the filter, so anti-correlated queries still need many probes. Combine it with
probing only partitions whose label count is non-zero.

## 6. NaviX, PVLDB 2025: adaptive filtered search on an unmodified HNSW

**Method.** Evaluate the filter first (a bitmap), keep HNSW unchanged, and choose per
candidate how to expand from its **local selectivity** (share of its neighbors that pass,
a bit lookup per neighbor):
- above 50%: `onehop-s`, expand only passing 1-hop neighbors;
- otherwise `directed`: sort 1-hop neighbors by distance and expand their 2-hop
  neighbors in that order;
- or `blind`: ACORN-style 2-hop in scan order, chosen when the expected passing 2-hop
  count σ · (M + 1) · M is below 3M.

**Claims.** `directed` is best between 50% and 5% (up to 2× over `blind`), `blind` below
5%. Local adaptivity beats global adaptivity by up to 1.7× on correlated workloads (69 ms
to 41 ms at 5%, negatively correlated). Implemented on FAISS, it beats ACORN-10 by 1.65×
and ACORN-1 by 5.45× at about 20% selectivity, and by 5.31× and 29× in the negatively
correlated case. ACORN-1, which drops edge pruning, needed M = 64 to reach the recall
target: pruning matters for edge diversity. Weaviate's latency jumps at its 40% switch
(164 ms against 18 ms for NaviX); Milvus switches to brute force at 5%. NaviX beats
Filtered-DiskANN at 6–70% label selectivity. Up to 15.4M × 1024-d, disk-based through a
buffer manager; recall target 0.95 at k = 100.

**Against our measurements.** Our `acorn` policy is NaviX's `blind` on an unmodified
HNSW graph (uncapped 2-hop). With base degree 32 it failed at 1% (0.29 recall); NaviX
builds base degree 64. Our degree test:

| Base-layer degree | Filter | ef = 200 | ef = 800 | ef = 2000 |
| --- | --- | --- | --- | --- |
| 32 (M = 16) | pedestrian 10% | 0.985 / 972 | 1.00 / 3,106 | 1.00 / 6,176 |
| 32 (M = 16) | cut_in 1% | 0.28 / 262 | 0.49 / 811 | 0.48 / 823 |
| 64 (M = 32) | pedestrian 10% | 0.99 / 1,620 | 1.00 / 4,816 | 1.00 / 8,666 |
| 64 (M = 32) | cut_in 1% | 0.55 / 380 | 0.78 / 1,028 | 0.82 / 1,548 |

(**ours**, `filtered_degree_check.py`: recall@100 / distance computations, 20 queries,
200K rows, 8 passing entry points.) At degree 32, recall stops improving past ef = 800,
which is the passing subgraph running out of connected nodes. Doubling the degree raises
1% recall from about 0.5 to about 0.8. This supports NaviX and ACORN (2-hop needs a dense
enough graph) and Li et al. claim 4, and qualifies Amanbayev claim 6 (density matters for
2-hop, not for dual-pool search).

On pruning, the papers disagree: NaviX finds dropping pruning (ACORN-1) hurts, while Li
et al. find KGraph-style "keep nearest" edges help below 5%. We did not test pruning.

## 7. Where the papers agree, disagree, and what none of them covers

**Agree.**
- No single index or search rule wins across selectivities; all six end up with a plan
  choice (thresholds, a cost model, or per-step adaptivity).
- Below some selectivity, scan the passing rows exactly: Milvus at more than 93%
  filtered out, SIEVE's γ · card(f), Li et al. below 1%, NaviX's baselines.
- Building structures over subsets (partitions, sub-indexes, per-attribute graphs) is
  what makes narrow filters fast: Li et al., PathFinder, SIEVE, CAPS.
- Correlation between filter and query changes recall at a fixed cost: Amanbayev, NaviX,
  SIEVE's correlation term. Most benchmarks still use uniform attributes (Li et al.).

**Disagree.**
- IVF or graph for narrow filters. Li et al. and Amanbayev find IVF more robust; NaviX,
  SIEVE and PathFinder make graphs win with adaptivity or sub-indexes. Both are
  consistent: IVF is robust because with enough probes it turns into a scan of the
  passing rows, and graphs win once the subset has its own structure.
- Edge pruning for filtered graphs (NaviX against Li et al.).

**Not covered by any of them, but central for us.**
- **Scale.** Largest test: 15.4M vectors (NaviX). We target 10^9–10^12 rows on object
  storage. None measures object-store I/O or distributed fan-out.
- **Mutable labels.** SIEVE refits; the others assume fixed attributes. Our labels change
  every annotation release while vectors stay.
- **Multi-valued label sets in AND/OR plans.** PathFinder leaves them out; SIEVE handles
  them only through sub-indexes over observed filters.
- **Large k.** All use k ≤ 100; training-data mining uses k in the thousands. Amanbayev's
  claim 5 says search effort must grow with k.
- **Correlated, real labels.** Only NaviX (Wiki joins) and the GLS paper (as a metric)
  look at correlation.

## 8. What to apply to the Lance AV design

In order of value for effort:

1. **Safety net and exact fallback (Amanbayev / Milvus).** After any approximate plan,
   check that k rows came back and that the plan was allowed for this many passing rows;
   otherwise re-run as filter-then-exact. In Lance: decide with `count_rows(filter)`, and
   never treat a full result from `maximum_nprobes` as proof of recall.
2. **Cost-based flat-or-index choice (SIEVE).** Use SIEVE's two formulas, indexed cost
   against γ · card(f), and calibrate γ and the correlation exponent on our own tables
   with `filtered_search_experiment.py`-style probes, instead of a fixed threshold.
3. **AND/OR planning (PathFinder).** Rewrite to DNF; per AND clause, rank the available
   structures (time partitions, per-label tables, the full table) by pass rate weighted
   by how many structures are searched, U = card(R, p) / (Σ card(g) · |G|^0.4), using
   exact bitmap counts; take the union for OR and merge into a shared parent where that
   scores higher. Use index borrowing for correlated attributes (for example a time-of-day
   partition serving `night`).
4. **Choosing per-label tables (SIEVE).** Pick which label or label-combination tables to
   build by benefit per byte under a storage budget, from the query log, with the full
   table always as fallback. Refit at each annotation release; only tables whose
   membership changed are rebuilt.
5. **CAPS-style row order inside IVF partitions.** When writing the index, order rows in
   each partition by the most frequent labels, and store per-partition label offsets.
   Filtered scans then read only matching row ranges. Add the step CAPS lacks: probe
   only partitions with a non-zero count for the filter.
6. **Filtered graph on the hot tier (NaviX, Li et al.).** Use an unmodified HNSW with base
   degree about 64, per-candidate adaptive expansion from local selectivity, a dual-pool
   result queue, and many entry points drawn from the passing rows.
7. **Measure correlation per label (Amanbayev GLS).** For each label and release, compute
   ρ = (r − 1)/(r + 1), r = pass rate among sampled queries' unfiltered top-k divided by the
   global pass rate. Labels with strongly negative ρ for common queries are the ones every
   index handles badly; give them per-label tables first.
8. **Scale effort with k (Amanbayev).** For mining jobs with k in the thousands, scale
   `ef` / `nprobes` with k, or use the filter-first batch plan (FILTERED_SEARCH.md §4.5).

## Run

```bash
uv run --locked --extra lance --extra compare python filtered_degree_check.py
```

FILTERED_METHODS.md implements these methods (NaviX, ACORN, Milvus's fallbacks, CAPS,
SIEVE, PathFinder, GLS ρ) on our HNSW, fastpipnn, k-NN and IVF indexes and reports
which claims hold on this data.
