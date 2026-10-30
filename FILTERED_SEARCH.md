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
| One vector index type per column | Add a filtered graph (in-search or ACORN-style) on the hot tier for broad filters (§6) |

## 6. Filtered search inside a graph index (HNSW, DiskANN, ACORN)

Graph indexes offer filtering in three different ways, and they behave very differently:

| Approach | Where | How it filters | What it needs |
| --- | --- | --- | --- |
| **In-search filter** | hnswlib `filter=`, FAISS `IDSelector`, most vector DBs | Walks the normal graph through every node, keeps only passing nodes as results, stops once it holds `ef` passing nodes | Nothing at build time; any predicate |
| **Predicate-agnostic denser graph (ACORN)** | ACORN (SIGMOD 2024); Weaviate implements it | Walks only passing nodes, plus 2-hop expansion through failing neighbors; the graph is built γ× denser so the passing subgraph stays connected down to s ≈ 1/γ | γ× more edges; any predicate |
| **Label-aware construction (Filtered-DiskANN)** | FilteredVamana / StitchedVamana (WWW 2023); UNG (SIGMOD 2024) builds one graph per label set | Edges are kept per label during build, and each label has its own entry point | Labels known at build time; equality on one label (or an OR of labels); graph edits when a label changes |

### Measured (`filtered_graph_experiment.py`)

Same 200,000 rows and filters as §2. hnswlib builds an HNSW graph (M = 16,
ef_construction = 200). `hnswlib` is the library's own filtered search. The other three
are re-implemented on hnswlib's base-layer graph so their work can be counted in
**distance computations**: `traverse` (hnswlib's rule), `strict` (walk passing nodes
only, i.e. the graph is not built for the filter), and `acorn` (strict plus 2-hop
expansion, ACORN-1 on an ordinary HNSW graph, γ = 1). Strict and acorn start from 8
passing nodes, as Filtered-DiskANN's per-label entry points would. Cells are
**recall@100 / distance computations per query** at ef = 200; for hnswlib the second
number is filter-callback calls, followed by median ms. The two right columns are the
work of the plans from §2 (flat = passing rows; IVF planner = rows in 16/s probed
partitions of 256).

| Filter | hnswlib | traverse | strict | acorn (γ = 1) | flat | IVF planner |
| --- | --- | --- | --- | --- | ---: | ---: |
| none | 0.999 / 326 / 0.3 ms | 0.999 / 797 | 0.999 / 800 | 0.999 / 782 | 200,000 | 12,500 |
| `weather = 'rain'` (20%) | 0.999 / 1,483 / 1.4 ms | 0.999 / 3,043 | 0.69 / 464 | 0.999 / 1,418 | 40,013 | 62,500 |
| `pedestrian` (10%) | 0.998 / 3,223 / 2.6 ms | 0.998 / 6,710 | 0.10 / 295 | 0.986 / 997 | 20,007 | 125,000 |
| `cut_in` (1%) | 0.999 / 29,520 / 25 ms | 0.999 / 48,079 | 0.01 / 10 | 0.29 / 272 | 1,969 | 200,000 |
| `rain AND cut_in` (0.2%) | 1.00 / 117,541 / 106 ms | 1.00 / 142,300 | 0.02 / 8 | 0.03 / 12 | 399 | 200,000 |
| `emergency_vehicle` (0.1%) | 1.00 / 199,896 / 193 ms | 1.00 / 199,896 | 0.05 / 8 | 0.05 / 9 | 193 | 200,000 |
| `tunnel` (1%), query inside | 1.00 / 337 / 0.2 ms | 1.00 / 785 | 0.97 / 507 | 1.00 / 513 | 1,989 | 200,000 |
| `tunnel` (1%), query outside | 0.987 / 34,657 / 26 ms | 0.987 / 50,721 | 0.83 / 614 | 0.95 / 635 | 1,989 | 200,000 |

With ef = 800 (in `filtered-graph-results.json`), acorn reaches 0.50 on `cut_in`
and 1.00 on `tunnel` outside, while hnswlib's work on `cut_in` grows to 99,073.

### What it shows

1. **For broad filters (≥ ~10%), a filtered graph is the best plan by far.** hnswlib keeps
   recall ≈ 1.0 with 1.5–3k evaluations, against 40–125k rows for the IVF planner.
   ACORN-style search is cheaper still (1.0k at 10%). This is the range where Lance's
   prefiltered IVF lost recall in §2.
2. **In-search filtering never loses recall, but its work grows as about ef / s.** It
   stops only after collecting `ef` passing nodes, so at 1% it evaluates 30k nodes, and
   when fewer than `ef` rows pass at all (0.1%) it walks **the entire graph**:
   200k evaluations to return 193 rows that a bitmap plus flat scan finds with 193
   distance computations.
3. **Walking only the passing subgraph breaks on uncorrelated labels.** An ordinary HNSW
   graph has no reason to connect rows that share `pedestrian`, so strict search finds
   0.10 recall at 10%. It works when the label is correlated with the embedding
   (`tunnel`: rows that share the label are also neighbors), especially with a passing
   entry point, which is exactly the Filtered-DiskANN observation.
4. **2-hop expansion (ACORN) fixes the 10–20% range on an ordinary graph, not the 1%
   range.** ACORN gets to low selectivity by building the graph γ× denser (γ ≈ 1/s_min),
   which multiplies the graph's memory by γ: for s_min = 1%, roughly 100× the edges of a
   normal graph, or an aggressive compromise between γ and a flat fallback. The ACORN
   paper itself falls back to pre-filtering below 1/γ.

### Can we just use it in the indexing layer?

Yes, as **one of the planner's plans**, not as a replacement for the planner:

| Selectivity of the filter | Best plan |
| --- | --- |
| ≥ ~10% (weather, road type, day/night, verified, region) | Filtered graph search (in-search filter, or ACORN-style) |
| ~1–10% | ACORN with γ sized for this range, a per-label sub-index (§4.3), or IVF with nprobes ∝ 1/s |
| below the flat threshold | Bitmap → exact scan of the passing rows |
| Time and geography ranges | Partition pruning (§4.2); range-filtered graphs exist (SeRF, iRangeGraph) but are not needed when data is written in time order |

Things specific to this workload that limit graph filtering:
- **Labels change; vectors do not.** Every annotation release edits labels. In-search
  filtering and ACORN do not use labels at build time, so a label edit only changes the
  bitmap. Label-aware graphs (Filtered-DiskANN, payload-edge schemes) must re-link
  nodes on every label change, and only support single-label or OR filters, not the
  conjunctions (`rain AND cut_in AND verified`) the queries actually use. For this
  dataset, choose the predicate-agnostic kind.
- **Graphs want RAM or local NVMe.** Each hop is a dependent random read. On object
  storage (where the Lance tables live) that is one round trip per hop. A filtered
  graph belongs on the hot, interactive tier (per-shard graphs held in memory or on
  NVMe, as in Lance Enterprise's cache), not on the cold petabyte table. DISTRIBUTED.md
  covers why a single graph across shards is hard.
- **Batch mining jobs still prefer posting lists.** A graph walk serves one query at a
  time; a training job with thousands of seed queries and k in the thousands is cheaper
  as filter-first plus a posting-major scan (§4.5, RETRIEVAL_DESIGN.md §4.3).
- **On Lance, check what `IVF_HNSW_SQ` / `IVF_HNSW_PQ` do with a prefilter before relying
  on it.** There HNSW lives inside each IVF partition, so the partitions to probe are
  still chosen without looking at the filter (the §2 problem), whatever the HNSW search
  inside them does.

## 7. Literature

FILTERED_SEARCH_LITERATURE.md reads six recent papers (the SIGMOD 2026 experimental
study, the GLS-metric system study, PathFinder, SIEVE, CAPS, NaviX), checks their claims
against the measurements above, and lists what to apply.

## Run

```bash
uv run --locked --extra lance python filtered_search_experiment.py   # writes filtered-search-results.json
uv run --locked --extra lance --extra compare python filtered_graph_experiment.py   # writes filtered-graph-results.json
```

Options: `--n`, `--dim`, `--clusters`, `--partitions`, `--nprobes`, `--k`, `--queries`,
`--flat-rows`. Synthetic data on one machine: the numbers show the mechanisms (recall
against selectivity and correlation, the shape of the flat/IVF crossover), not
production latency.
