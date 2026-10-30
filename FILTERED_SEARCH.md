# Hybrid search: label filters and vector search together

The Lance architecture proposal (observations table, denormalized labels, separate
annotation tables) treats the label filter as one line:
`.where("weather = 'rain' AND has_pedestrian", prefilter=True)`. Once scenarios are
labeled, though, almost every retrieval is **filter ∧ similarity**: "scenes like this
one, *with* `cut_in`, *at night*, *human-verified*". The filter decides recall, latency
and the physical layout as much as the vector index does. This note measures how Lance
behaves under filters and shows what the design has to add.

## 1. Two properties of a filter decide everything

- **Selectivity** s = passing rows / all rows. Coarse filters (release, region, weather)
  have s of 5–50%. Scenario labels on a long-tail dataset have s of 0.01–1%, and
  conjunctions of them go lower still.
- **Correlation with the query.** If rows carrying the label sit in the same embedding
  region as the query (a `tunnel` query and `tunnel` rows), the IVF partitions nearest
  the query already contain the passing rows. If they do not (the query is "a highway
  merge like this one" and the label is `emergency_vehicle`, which appears everywhere),
  the nearest partitions hold few or none of them. Labels that are independent of
  appearance (weather against a trajectory embedding, `annotation_status`, vehicle ID)
  are the common case, not the exception.

## 2. Measured on Lance 14 (`filtered_search_experiment.py`)

200,000 rows, 64-d clustered embeddings (400 clusters), IVF_FLAT with 256 partitions
(no quantization, so every recall loss comes from the filter), base nprobes = 16,
k = 100, 50 queries per row. Filters use BITMAP (weather) and LABEL_LIST (tags) indexes.
Recall is against exact filtered search computed in NumPy. Cells are
**recall / rows returned / median ms**.

| Filter | Passing rows | post | pre_fixed | pre_expand | flat | planner |
| --- | ---: | --- | --- | --- | --- | --- |
| none | 200,000 | – | 1.00 / 100 / 8.9 | 1.00 / 100 / 7.2 | 1.00 / 100 / 24 | – |
| `weather = 'rain'` (20%) | 40,013 | 0.20 / 20 / 7.8 | 0.996 / 100 / 6.6 | 0.996 / 100 / 6.6 | 1.00 / 100 / 35 | 1.00 / 100 / 12 |
| `pedestrian` (10%) | 20,007 | 0.10 / 10 / 8.0 | 0.93 / 100 / 6.0 | 0.93 / 100 / 6.4 | 1.00 / 100 / 53 | 1.00 / 100 / 17 |
| `cut_in` (1%) | 1,969 | 0.01 / 1 / 8.2 | 0.57 / 100 / 5.5 | 0.57 / 100 / 6.0 | 1.00 / 100 / 26 | 1.00 / 100 / 26 |
| `rain AND cut_in` (0.2%) | 399 | 0.00 / 0 / 8.4 | 0.32 / 55 / 5.3 | 0.50 / 100 / 6.6 | 1.00 / 100 / 10 | 1.00 / 100 / 11 |
| `emergency_vehicle` (0.1%) | 193 | 0.00 / 0 / 7.0 | 0.25 / 31 / 4.5 | 0.67 / 100 / 7.7 | 1.00 / 100 / 7.5 | 1.00 / 100 / 8.2 |
| `tunnel` (1%), query inside | 1,989 | 1.00 / 100 / 8.3 | 1.00 / 100 / 4.9 | 1.00 / 100 / 4.9 | 1.00 / 100 / 23 | 1.00 / 100 / 24 |
| `tunnel` (1%), query outside | 1,989 | 0.00 / 0 / 6.6 | 0.17 / 34 / 3.7 | 0.51 / 100 / 5.4 | 1.00 / 100 / 28 | 1.00 / 100 / 27 |

Strategies:
- **post**: ANN first, then filter (`prefilter=False`).
- **pre_fixed**: `prefilter=True`, nprobes fixed at 16.
- **pre_expand**: `prefilter=True`, `minimum_nprobes=16`, `maximum_nprobes=256`.
- **flat**: `prefilter=True, use_index=False`. The scalar index finds the passing rows,
  and those rows are scored exactly.
- **planner**: `count_rows(filter)` (answered from the scalar indexes). At or below
  10,000 passing rows, use flat; otherwise use pre_expand with nprobes = 16 / s.

IVF training is not seeded by the script, so recall moves by about ±0.01 between runs.

## 3. What the numbers say

1. **Postfiltering is unusable for labels.** It returns about s·k rows. For a 1% label
   that is one row out of 100.
2. **`prefilter=True` guarantees the row count, not the recall.** Lance probes the
   nprobes partitions nearest the query and keeps the passing rows in them. With
   `maximum_nprobes` it keeps opening partitions **until it has k passing rows, then
   stops**. Those k rows are the first ones it found, not the nearest ones: 100 rows at
   recall 0.5–0.67 for 0.1–1% filters. A result list that is full and plausible but half
   wrong is worse than a short one, because nothing signals the problem.
3. **Recall falls with selectivity even when the label is uncorrelated.** Probing a
   fixed number of partitions examines a fixed number of rows, and only s of them pass.
   To see the same number of passing candidates, the probe count has to grow as 1/s.
   The planner does that: at 10% it probes 160 of 256 partitions and gets recall 1.0 in
   17 ms.
4. **Correlation dominates.** The same 1% `tunnel` filter gives recall 1.0 when the
   query is near tunnel scenes, and 0.17 (fixed) or 0.51 (expanding) when it is not.
   Unfiltered recall benchmarks hide this completely.
5. **For selective filters, exact search over the passing rows is the right plan.**
   Below a few thousand passing rows, flat is exact and about as fast as the index
   (7.5 ms for 193 rows). It reads only the passing rows, so its cost grows with s·N,
   while the index needs about 1/s probes. The two curves cross. In this setup the
   crossover is between 1% and 10%; it depends on N, the dimension, quantization, and
   the cost of a random read compared with a sequential partition read on your storage.

## 4. What this changes at petabyte scale

At N = 10^9 observations, "selective" filters are still large: s = 0.1% is 10^6 passing
rows, which is 1.5 GB of 768-d fp16 vectors. Flat is fine for an offline mining job
and too slow for an interactive query. Neither "prefilter and hope" nor "always flat"
works. The design needs a query planner and a physical layout chosen for the filters.

### 4.1 A selectivity-aware planner in the query router

The router already exists in the design (§3.3, shard fan-out). Give it a planning step:

```
count = passing rows per shard            # bitmap / label_list cardinalities: cheap
if count == 0:            skip the shard
elif count <= FLAT_ROWS:  exact scan of the passing rows (quantized codes, then refine)
elif label has a materialized sub-index:  search that sub-index unfiltered (§4.3)
else:                     IVF + prefilter, nprobes = base / s (capped), refine_factor > 1
```

- Calibrate `FLAT_ROWS` and the probe scaling per shard size and embedding type, by
  recall probes (§4.6), not by guessing.
- Estimate s from index cardinalities for single labels. For conjunctions, intersect
  the bitmaps; do not multiply marginals, because labels in driving data are strongly
  correlated (`rain` and `wet_road`, `night` and `headlight_glare`).
- Never set nprobes only from latency targets. A prefiltered result with k rows is not
  evidence of good recall (§3, point 2).

### 4.2 Filter-aware physical layout

The cheapest filter is one that removes whole shards, fragments or index segments
before any vector work.
- **Partition by the filters people actually use**, not only by ingestion convenience.
  The release / region / time hierarchy in the design is right for time and geography.
  Measure the query log: if most queries pin `sensor_rig` or `embedding_model_version`,
  those belong in the routing key too.
- **Write in filter order** inside a partition (time, then drive), so zone maps and
  fragment statistics prune on time ranges and drives.
- **Partition-level label statistics.** Keep a small matrix `count[ivf_partition, label]`
  (P × L integers; for P = 10^5 and L = 500 that is 5×10^7 cells). The scanner can then
  rank partitions by centroid distance and skip those with zero passing rows. Only
  partitions that actually contain passing rows count toward nprobes. Lance does not do
  this today; it belongs in the custom batch scanner from RETRIEVAL_DESIGN.md §7.3.
  Refresh the matrix at each annotation release, together with the label summaries.

### 4.3 Materialized sub-indexes for rare, frequently queried labels

For the long-tail scenario labels that mining jobs keep asking for (`cut_in`,
`unprotected_left`, `emergency_vehicle`, `jaywalking`, `construction_zone`), build a small
per-label table, `scenario_<label>`, holding `sample_id`, the embedding columns, and the
few scalar columns used with it. Give it its own index, or no index if it is small
enough for flat.
- A filtered query on a rare label becomes an **unfiltered** search on a dense table:
  full recall, no probe scaling, no anti-correlation problem.
- Cost is proportional to the label's frequency, so all rare labels together are a few
  percent of the main table.
- They are rebuilt in the annotation release pipeline (design §5.2 and §10), in the same
  step that recomputes `has_pedestrian` and similar columns, and are pinned in the
  release manifest. A label correction never touches the vectors, only which sub-table
  a row is in.

This is the IVF counterpart of the label-specific entry points in Filtered-DiskANN and
of per-label partitions in other filtered-ANN systems. It is chosen per label, from the
query log, not for every label.

### 4.4 Labels are hard predicates; captions are scores

"Hybrid search" mixes two different things, and they should not go through one fusion
step:
- **Flags and labels** (`has_pedestrian`, `weather`, `scenario_tags`,
  `annotation_status`) are **constraints**. A scene without the label is a wrong answer,
  whatever its similarity. Run them as filters with the planner above. Do not feed them
  into reciprocal rank fusion as one more ranked list.
- **Free text** (VLM captions, BM25 over the `INVERTED` index) and the embedding are
  **evidence**. Combine those by rank fusion or a reranker, *inside* the set allowed by
  the constraints.

A query then has the form `constraints → (vector ⊕ text) → rerank`. The CARIM finding in
RETRIEVAL_DESIGN.md §2b ("pure similarity misses conjunctive constraints") is the same
point.

### 4.5 Batch mining jobs: filter first

Training-data jobs run many seed queries with large k over the same label constraint.
For them, invert the order:
1. evaluate the constraint once with the scalar indexes, giving the passing set;
2. if it is small enough, score all seed queries against it in one pass (exact, GPU);
3. otherwise, route the queries through the shared IVF centroids and read each needed
   partition once, applying the passing-set mask (the posting-major scanner,
   RETRIEVAL_DESIGN.md §4.3).

Lance 14 accepts a 2-D `q` in `nearest` (one call, many queries, with a `query_index`
column), so step 2 can use Lance directly for moderate sizes.

### 4.6 Measure filtered recall, per filter bucket

Add to the release validation (design §10–11):
- recall@k against exact filtered search, bucketed by **selectivity** (≥10%, 1–10%,
  0.1–1%, <0.1%) and by **correlation** (passing rows among the query's nearest
  partitions, compared with s);
- a fixed probe set of real production filters, re-run at each annotation release,
  because label edits change selectivity and correlation without touching a single
  vector;
- "returned < k" and "plan chosen" counters from the router.

Unfiltered Recall@K (design §11) is necessary but tells you almost nothing about the
queries people actually run.

## 5. Changes to the proposal, in short

| Proposal says | Change |
| --- | --- |
| `.where(..., prefilter=True)` with fixed `nprobes` | Selectivity-aware planner: flat / sub-index / IVF with nprobes ∝ 1/s |
| Recall@K as the quality metric | Filtered recall per selectivity × correlation bucket |
| Partition by release, region, time | Also partition by the most frequent filter keys in the query log |
| Scalar indexes on filter columns | Plus partition × label counts so the scanner skips empty partitions |
| Rare scenarios found by filtering the main table | Materialized per-label sub-tables, rebuilt per annotation release |
| "Hybrid" = vector + metadata | Labels are hard constraints; text and vectors are fused only inside them |
| Batch lookup after the vector search | Batch mining jobs evaluate the constraint first, then score |

## Run

```bash
uv run --locked --extra lance python filtered_search_experiment.py   # writes filtered-search-results.json
```

Options: `--n`, `--dim`, `--clusters`, `--partitions`, `--nprobes`, `--k`, `--queries`,
`--flat-rows`. Synthetic data on one machine: the numbers show the mechanisms (recall
against selectivity and correlation, the shape of the flat/IVF crossover), not
production latency.
