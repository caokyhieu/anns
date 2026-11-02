# Sharded index designs with PiPNN: experiment plan and results

DISTRIBUTED.md designed one global PiPNN graph spread over machines. At the target scale
(10^11–10^12 vectors) the index has to be **sharded** instead: every shard is built and
served independently, and a query is answered by a few or all shards. This note lists
the sharded designs we compare, with the LanceDB 10B architecture as the baseline, and
records the measurements.

Code: `native/src/shard_bench.cpp` (all designs, one binary), `native/scripts/run_shards.sh`
(step 1 runs), `native/scripts/summarize_shards.py` (matched-recall tables),
`lance_shard_check.py` (the same baseline on real Lance).

## 1. Designs

### D6 – baseline: LanceDB at 10B (row segments, shared k-means, HNSW router over centroids)

From [How LanceDB accelerates vector search at 10 billion scale](https://www.lancedb.com/blog/how-lancedb-accelerates-vector-search-at-10-billion-scale)
(Yang Cen, April 2026). The table is split by rows into segments; each segment has its own
IVF index built by its own worker. Centroids are trained once (a fixed cost). The query node
routes with an HNSW over the centroids, and every segment scans the same nprobe partitions.

```
 BUILD
   1. train once (fixed)      sample ─► k-means ─► C centroids  (+ RaBitQ rotation)
   2. split rows              rows 0–1B │ 1B–2B │ … │ 9B–10B
   3. one worker / segment    assign own rows to the shared centroids, RaBitQ-encode
                              ── seg 1 IVF ── seg 2 IVF ── … ── seg S IVF
   4. merge / commit          one logical index (fixed)

 QUERY
   query node: HNSW over C centroids ─► nprobe partition IDs, rotate query once
        │ same partition list to EVERY segment
   executor 1 … executor S    scan own rows of those partitions (RaBitQ binary codes)
        └── top-k·refine each ─► rerank with full vectors ─► merge top-k
```

Reported: 10B random 1536-d float32 vectors, 10 segments × 1B on 10 nodes, nprobe 20,
top-100: p50 18 ms, p95 20 ms, p99 21 ms end to end. Index build about 5× faster with 10
workers; encoding scales near-linearly, centroid training and merge stay fixed. No recall is
reported. Lance only builds the centroid HNSW when C·d ≥ 10^6; below that it scans centroids.

### D1 – row shards, one fastpipnn graph per shard, query all shards

```
 vectors ──hash(id)─► shard 1 ─► fastpipnn      (no communication during build)
                      shard 2 ─► fastpipnn
                      shard S ─► fastpipnn
 query ─► coordinator ─► ALL S shards (beam b) ─► top-k each ─► merge
```

Same shape as D6 (rows split, every query hits every shard), but a graph replaces
IVF + RaBitQ inside the shard. HNSW per shard is run as the graph baseline.

### D5 – D6 with PiPNN partitioning instead of k-means

```
 sample ─► PiPNN ball carving (random leaders, one GEMM pass per level) ─► leaves
        ─► leaf means = C centroids ─► router (exact / HNSW / fastpipnn)
 per segment: assign + RaBitQ-encode own rows (same as D6) ; query as D6
```

Isolates the partition step: is carving cheaper than k-means, and are its lists as good?

### D6c – D6 with a fastpipnn graph inside each IVF partition  *(step 2)*
### D6b – partitions placed on shards by locality, query only owning shards  *(step 2)*

```
 centroids clustered into S groups ─► group g = shard g
 query ─► router ─► nprobe partitions ─► only the shards that own them
```

### D3 – PiPNN level-0 leader groups as shards, route to p shards  *(step 2)*

```
 PiPNN partition tree: level 0 leaders ─► leader groups = SHARDS (1–2 copies at the boundary)
                       levels 1..L     ─► normal PiPNN inside each shard
 router = fastpipnn over level-0 leaders; query ─► top-p shards ─► local beam search ─► merge
```

### D4 – D3 plus a few cross-shard bridge edges  *(step 3)*

```
   shard A graph           shard B graph
   ●──●──◉ ═══ bridge ═══ ◉──●──●     ≤ r_x inter-shard edges per boundary vector
 query starts in the top shard, may cross ≤ h bridges (r_x = 0 is D3; no cap is one global graph)
```

### Map of what each comparison isolates

| Comparison | Changes only | Question |
|---|---|---|
| D6 → D5 | k-means → PiPNN carving | cheaper partitioning at equal recall? |
| D6 → D1 | IVF + RaBitQ → fastpipnn graph per shard | which in-shard index is cheaper per unit recall? |
| D6 / D1 → D6b / D3 | all shards → routed shards | from which S does routing pay off? |
| D3 → D4 | adds bridges | recall regained per cross-shard hop |
| router study | HNSW vs fastpipnn over centroids | does PiPNN's build advantage hold for routers? |

## 2. Setup (step 1)

- Data: `native/scripts/synthetic.py --n 1000000 --queries 1000` – 1M × 128-d uint8,
  128-component Gaussian mixture, 1,000 held-out queries, exact top-100. BIGANN could not
  be downloaded from this environment (dl.fbaipublicfiles.com blocked); rerun with
  `run_shards.sh data/bigann10m/...` where it is reachable.
- Machine: 4 cores, 15 GB. Builds use 4 threads per shard; shards are built one after
  another and timed separately, so "wall (S workers)" = fixed cost + slowest shard and
  "total CPU" = fixed + sum of shards. Queries run on one thread; latency = routing +
  slowest shard + merge; CPU/query = routing + all shards + merge.
- D1: fastpipnn degree 64, leaf 1024, fanout (10, 3), local_k 2, α 1.2; HNSW M 32,
  efConstruction 200. D6/D5: k-means 10 iterations on 256 × C sampled rows; carving on the
  same sample with leaf size 1.6 × sample / C; 1-bit RaBitQ codes with a 4-bit query
  (popcount kernel), refine 10 (k · 10 candidates reranked **per segment**, as in Lance).

RESULTS_PLACEHOLDER
