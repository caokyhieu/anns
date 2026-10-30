# Distributed PiPNN for ~1 trillion vectors

Design notes for indexing an autonomous-driving data lake: 2–3 EB of raw video, lidar,
radar and logs, represented by about **10^12 embedding vectors** (1,000 billion). The
vectors and index are about 2 PB. The exabytes stay in object storage, and the index holds pointers
into them. The notes are based on three papers,
read in the requested order, and on a simulated prototype (`dpipnn.py`,
`distributed_experiment.py`, `test_dpipnn.py`) that tests the parts we can test on one machine.

> **Sources.** DistributedANN and SPANN were read in full, from the PDFs Microsoft
> Research hosts. SPIRE is only on arxiv.org, which this environment cannot reach, so
> its details come from the abstract and secondary summaries; those statements are
> marked *(summary)*. RETRIEVAL_DESIGN.md surveys related work and gives the
> design for training-data retrieval.

## 1. What the papers say, and what each means for PiPNN

### DistributedANN: serving one graph across thousands of machines
[Adams et al., ICML 2025 VecDB workshop, arXiv:2509.06046](https://arxiv.org/abs/2509.06046)

- One DiskANN graph over a 50-billion-vector slice of the Bing index (384-d int8),
  spread across more than 1,000 machines. Median latency is 26 ms (p99 35 ms) at over
  100k QPS. It replaced the partitioned production system in Bing, which serves
  "hundreds of billions" of vectors.
- **Layout.** A distributed key-value store holds the graph nodes and is treated as a
  shared disk. Each node stores the full vector, up to R = 72 neighbor IDs (8 bytes
  each), and the **OPQ codes of all its neighbors**. That is about 10× space
  amplification, and it means each hop needs one read. A **head index** is built from
  the top layers of the graph, collected by breadth-first traversal: 2.5B vectors in a
  sharded in-memory index. It supplies entry points.
- **Near-data scoring.** A node-scoring service on every key-value host reads the
  requested nodes, computes exact distances, and scores neighbors from their OPQ codes.
  It returns only scored candidates better than the caller's current threshold, which
  saves about 6× network bandwidth. An orchestration service runs H = 5 rounds of beam
  width 128 and uses hedged requests.
- **Build: stitching, not insertion.** Vectors are partitioned into clusters with SPANN
  closure assignment (boundary vectors go to several partitions). A DiskANN graph is
  built per partition. The global graph is the union of each replicated vector's
  neighbor lists across partitions, and the head comes from the union of the
  partitions' top layers so everything stays reachable. The authors say the stitched
  graph is lower quality than an incrementally built one, but good enough and much
  faster to build.
- **Results against clustered partitioning (40 of 203 partitions searched, ~200M
  vectors each), at equal footprint for 15k QPS:**

  | Metric | DistributedANN | Clustered partitioning |
  | --- | ---: | ---: |
  | Recall@5 | 90.8% | 83.0% |
  | Recall@200 | 71.9% | 67.4% |
  | IOs per query | 640 | 4,800 |
  | SSD | 780 TiB | 270 TiB |
  | RAM | 42 TiB | 18 TiB |
  | Throughput | >100k QPS (SSD-space bound) | ~15k QPS (IO bound) |

  Hosts had 256–768 GiB RAM, 5–10 TiB SSD and 40 Gb/s networking.
- **Lessons the authors report.** The head index becomes the CPU bottleneck. Recall
  degrades gracefully when node reads fail (90.8% → 87.0% at 96% availability).
  Clustered partitions create hot spots. Future work they list: several nearby vectors
  per graph node, HDD tiering for cold data, and GPU for the head.
- **For large k, note recall@200 = 71.9%.** Graph search at this scale is tuned for
  small k. That matters for training-data retrieval (RETRIEVAL_DESIGN.md).

**For PiPNN:** DistributedANN's build is already PiPNN-shaped: closure-replicated
partitions, a local graph per partition, then a merge of each vector's neighbor lists.
PiPNN does the same thing more principled. Its merge is HashPrune plus RobustPrune,
which bounds the degree instead of taking a union and truncating to 72. And its local
step is a GEMM over leaves, not incremental insertion (§3).

### SPIRE: distributed construction and the accuracy cost of partitioning
[Xu et al., arXiv:2512.17264](https://arxiv.org/abs/2512.17264)

- Problem: with a graph sharded across nodes, cross-node traversal dominates. In a
  100M-vector HNSW sharded over 5 nodes, cross-node steps were over 80% of search cost.
- **Partition granularity is a trade-off.** With coarse partitions (low density), each
  query must read more vectors to keep Recall@5. With fine partitions, it takes more
  cross-node hops. SPIRE chooses the granularity by sampling (searching 1–1,000
  vectors per partition).
- **Accuracy-preserving construction:** distributed k-means splits the data across
  nodes, and **vectors near k-means boundaries are replicated** so cross-boundary
  neighbor relations survive. Building is recursive and bottom-up: balanced clustering
  repeats until the top level fits in memory *(summary)*, giving a multi-level index
  with a predictable number of reads per level.
- Results go up to 8B vectors on 46 nodes, with up to 9.64× higher throughput than the
  state of the art. Larger scales (up to 1024B vectors) were simulated, not run.

**For PiPNN:** PiPNN's top-level fanout (each point routed to several leaders) is
already boundary replication. On one machine it costs memory. Across machines it costs
network and storage, multiplied by the replication factor. So the fanout has to be
split into an *inter-shard* part (small, SPANN/SPIRE closure rule) and an
*intra-shard* part (large, cheap). SPIRE's recursion is also what makes the head index
cheap to build and keep in memory (§2).

### SPANN: inverted lists instead of a graph
[Chen et al., NeurIPS 2021, arXiv:2111.08566](https://arxiv.org/abs/2111.08566)

- Memory–disk hybrid. Posting-list centroids are in memory, searched with an SPTAG
  index (trees plus a relative-neighborhood graph). Each centroid is replaced by the
  real vector nearest to it, so routing work also scores real candidates. Posting
  lists are on SSD.
- **Hierarchical balanced clustering (HBC).** It clusters into a small number k of
  balanced clusters recursively until each list is under its size limit. That reduces
  clustering cost from O(N·d·#lists) to O(N·d·k·log_k #lists), and balance keeps
  per-list reads and latency predictable. It clustered 1M points into 160K lists in
  about 50 s on 64 threads, and built the whole index in about 2 minutes. A GPU build
  ran more than 5× faster.
- **About 16% of points as centroids.** Recall stops improving beyond 16%. Posting
  lists are capped at 12 KB (byte vectors) or 48 KB (float).
- **Closure assignment:** add x to list j if Dist(x, c_j) ≤ (1+ε1)·Dist(x, c_1), with
  ε1 = 10 and at most 8 replicas (more than 8 brings no gain). An **RNG rule** then skips
  list j if Dist(c_j, x) > Dist(c_{j-1}, c_j), so the replicas point in different
  directions instead of duplicating nearby lists.
- **Query-aware pruning:** search list j only if Dist(q, c_j) ≤ (1+ε2)·Dist(q, c_1),
  with ε2 = 0.6 for recall@1 and 7.0 for recall@10. Query difficulty varies a lot: on
  SIFT1M, 80% of queries need 6 lists for recall@1, but 99% need 114.
- **Results:** 2× faster than DiskANN to 90% recall at equal memory on SIFT1B, DEEP1B
  and SPACEV1B. 90% recall@1 and recall@10 in about 1 ms with 32 GB of memory.
- **Distributed extension:** balanced partitions with closure assignment, plus query
  pruning to limit how many machines each query reaches. Partitions are made smaller
  than machines and bin-packed onto machines by historical query access, to avoid hot
  spots. Closure adds about 20% more vectors per machine.

**For PiPNN:** PiPNN's leaves *are* balanced, overlapping posting lists, so PiPNN
can produce a SPANN index from its own partition step. The graph-versus-postings choice
then comes down to storage: graphs need many small random reads, postings need a few
large sequential ones (§5).

## 2. Sizing: 10^12 vectors from 2–3 EB of sensor data

The raw sensor data is not indexed. Embeddings are: per frame, per clip or scene, per
object crop, per lidar sweep. Each vector carries an ID that points back to the
raw-data blob, timestamp and vehicle in object storage.

**10^12 is beyond any published single-graph run.** It is 20× DistributedANN's 50B
evaluation (its Bing deployment covers "hundreds of billions") and 125× SPIRE's largest
real run (8B). SPIRE simulated up to 1,024B. Expect to validate recall at 10^10 and
10^11 before trusting extrapolation to 10^12.

Assume 768-d fp16 embeddings (1.5 KB). Replace this with your model's dimension.

| Item | Per vector | Total for 10^12 | Where it lives |
| --- | ---: | ---: | --- |
| Full vectors (768-d fp16) | 1.5 KB | 1.54 PB | NVMe, read for re-ranking (int8 halves it) |
| Adjacency (degree 64 × 6-byte IDs) | 384 B | 384 TB | NVMe, beside the vector |
| PQ codes for traversal (32–64 B) | 32–64 B | 32–64 TB | RAM, spread over the serving nodes |
| Head index (0.1% = 10^9 vectors, PQ 64 B + degree-32 graph) | — | ~230 GB | RAM, one large-memory node, replicated |
| PQ codes duplicated per neighbor (DistributedANN style) | 2–4 KB | 2–4 PB | not affordable (see below) |

What follows:

- **IDs need 40 bits.** 2^40 = 1.1×10^12 leaves no headroom, so use 6-byte IDs, or a
  32-bit shard-local ID plus a shard number.
- **About 1.9 PB of NVMe per copy.** With dense storage nodes (about 120 TB NVMe each)
  that is about 16 nodes per copy, or 35–50 nodes with 2–3 copies. With 30 TB nodes it
  is about 65 per copy. The 32–64 TB of PQ codes must fit in aggregate RAM: about
  1 TB per node at 40–60 nodes, or 256–512 GB per node at 130–200 nodes. RAM may
  decide the node count rather than disk. Use 32-byte codes or keep codes on NVMe
  beside the adjacency if RAM is tight.
- **Do not duplicate codes into graph nodes.** At degree 64 that adds 2–4 PB, more than
  the vectors themselves.
- **The head still fits on one machine if compressed.** A 0.1% sample (10^9 vectors) as
  64-byte codes plus a degree-32 graph is about 230 GB. Replicate it for QPS. With full
  fp16 vectors it would be 1.5 TB, so either compress it or use a two-level head
  (10^8 on top, then 10^9 sharded): SPIRE's recursion, one extra level.
- **Build compute is about 1 day on 100 CPU nodes.** Leaf GEMM work at `fastpipnn`
  defaults (about 30 leaf memberships per point, leaf 1024, d = 768) is 4.7×10^19
  FLOPs, roughly 110 node-days at 5 TFLOP/s or about 70 GPU-hours. Cutting
  intra-shard fanout to 8 memberships with leaf 256 is about 15× less.
- **Build memory is the real constraint.** HashPrune reservoirs (160 slots × about
  8 B) need about 1.3 PB of RAM on top of 1.54 PB of vectors. So the build is sharded
  into thousands of tasks (§3) and run in waves on 100–300 machines. A smaller reservoir
  (64–96 slots) cuts that roughly in half. The prototype uses 64.
- **Build network traffic is a few PB per full build.** With 2× inter-shard replication
  (top-2, the cheapest setting that worked in §6), about 1.5 PB of extra vector copies
  and about 0.5 PB of edge shuffle (about 500 B per vector, measured in §6) move once.
  That is about 15 minutes at an ideal 2.5 TB/s (200 nodes × 100 Gb/s); expect hours in
  practice. So full rebuilds should be rare, and new data goes into segments (§4).

Questions that change the design more than the algorithm choice:

1. **How many embedding spaces?** Camera, lidar, radar and text-aligned (CLIP-style)
   embeddings live in different vector spaces. Each model needs its own index, so 10^12
   is probably several indexes of 10^11. Each of those is close to demonstrated scale.
2. **Filters.** Scenario mining almost always combines similarity with metadata, for
   example "pedestrian crossing, night, rain, city X, sensor rig v3, last 6 months".
   Partition first on coarse, frequently filtered attributes (time window, region,
   sensor generation). That is exact and costs no recall. Apply fine filters during
   traversal or re-ranking.
3. **Near-duplicates.** Consecutive frames at 10–30 Hz produce near-identical vectors.
   Deduplicate or subsample at ingest (for example one vector per clip segment, keeping
   frame-level vectors only for interesting segments). Otherwise duplicates crowd out the
   top-k, and duplicate clusters stress PiPNN's partitioner (its random-split guard
   exists for this case).
4. **Workload.** Engineers mining scenarios issue few, expensive, often batched queries.
   This is not Bing's 100k QPS. Batch queries reward throughput-oriented serving and
   allow higher beams and exact re-ranking. Training-data curation may also need very
   large k (thousands of results), which favors postings or graph search with large beams.
5. **Ingestion rate.** New drives arrive daily. Index them as new time-partitioned
   segments built by the same pipeline, and merge or rebuild segments periodically.

## 3. Distributed PiPNN construction

PiPNN's build is already MapReduce shaped, and the prototype implements exactly this
pipeline (`dpipnn.build`):

```
 sample ─► balanced k-means ─► S shard centroids (broadcast)
 route:    each vector → primary shard (capacity-capped) + closure replicas      [shuffle 1: vectors]
 map:      per shard: PiPNN ball carving (intra-shard fanout) + leaf GEMM k-NN
 combine:  per shard: HashPrune candidates per source → partial reservoirs
 reduce:   by owner of source: merge partial reservoirs ─► RobustPrune           [shuffle 2: edges]
           (candidate vectors not stored on the owner are fetched)               [shuffle 3: vectors]
 write:    node records to the KV/graph store; build head from sample/leaders
```

Why this works for PiPNN in particular:

1. **No incremental insertion.** Vamana/HNSW build a graph by searching the graph
   under construction, which turns into cross-machine random access at scale.
   DistributedANN stitches per-cluster graphs together for this reason. PiPNN has no
   such dependency. Leaf k-NN is independent per leaf, and the global graph appears
   directly from the reduce step.
2. **HashPrune is exactly mergeable.** It keeps the nearest candidate per hash bucket,
   then the `reservoir` nearest buckets, under a total (distance, ID) order. A bucket
   that a partial reservoir drops is beaten by `reservoir` buckets that are at least as
   near in the union, so `hashprune(A ∪ B) = hashprune(hashprune(A) ∪ hashprune(B))`.
   This means each shard can pre-reduce its edges (a combiner) with **no change to the
   final graph**. `test_dpipnn.py` checks both claims: the merge identity on random
   edges with many ties, and a bit-identical graph with and without the combiner.
3. **The shuffle is bounded.** After the combiner, each shard sends at most
   `reservoir` records per source it touched, about 18 bytes each.
4. **The build is deterministic.** Partitioning randomness is seeded per shard, and
   the hash planes come from a shared seed. Reruns and retries of failed tasks produce
   the same graph, which matters for idempotent recovery when tasks fail or are preempted.

Two-level fanout is the key change. On one machine `fastpipnn` routes each point to
10 × 3 leaves, about 30 memberships (`native/results/compare-synthetic1m`: 30.0M
leaf points for 1M vectors). Across machines, every extra membership is a copy of
the full vector over the network. So:

- **inter-shard:** primary shard plus closure replicas within (1+ε) of the nearest
  shard centroid, capped (2–4). This is SPANN's rule at shard level, and SPIRE's
  boundary replication.
- **intra-shard:** PiPNN's usual multi-level fanout, in memory, as large as you like.

Shard granularity follows SPIRE. Size build shards to fit one machine's RAM with
reservoirs (about 3 KB per vector in total, so about 3×10^8 vectors per 1 TB). With
2× replication there are about 2×10^12 memberships, so roughly 7,000–10,000 shards of
2–3×10^8. Process them in waves. Use many more *virtual*
shards than machines, mapped to machines by a consistent map, so you can rebalance or
re-run one shard without touching the rest. Top-level balanced k-means needs about
10^4 centroids. Run it on a sample (10^8 points), and route the full data to the centroids
with a small in-memory index over them (PiPNN or brute force on GPUs). That is one level
of SPIRE recursion.

## 4. Serving

Recommended layout, following DistributedANN:

- **Graph store:** node records keyed by vector ID, placed on the primary shard's
  machine (each record holds the vector or its code, plus adjacency). Primary ownership
  follows the build partition, so most hops stay on one machine. In the experiment
  about half of the final edges cross shards at 8 shards, which is the price of one global graph.
- **Head:** an in-memory tier. A PiPNN graph over a 0.01–0.1% sample (or over the
  partition leaders, which PiPNN already computed), replicated for QPS, since
  DistributedANN reports the head as the bottleneck. At 10^12 a 0.1% head (10^9
  vectors) fits on one large-memory node only as PQ codes (about 230 GB). Otherwise
  use two levels.
- **Codes in RAM:** PQ codes (32–64 TB) are spread over the serving nodes' RAM, so
  neighbor scoring during traversal does not touch disk.
- **Query:** the coordinator runs beam search. Each round expands W nodes (DiskANN
  beam width) and batches the reads by owning shard. Scoring is pushed to the shard,
  which returns scored neighbors. Final top-k is re-ranked with full vectors.
- **Updates:** immutable segments, LSM-style. New data goes into new segments built by
  the same pipeline, deletes are tombstones, and segments merge periodically. The
  in-place repair in `pipnn.py` is fine for small edits inside a shard, but it is not
  how 10^12 vectors evolve. Time-partitioned segments also match how AV data
  arrives and how it is filtered.

## 5. Graph or inverted lists?

| | Global PiPNN graph (DistributedANN-style) | Postings (SPANN/SPIRE-style) |
| --- | --- | --- |
| Reads per query | ~beam random 4 KB reads | a few large sequential reads |
| Storage medium | NVMe (needs IOPS) | HDD or object storage OK (needs bandwidth) |
| Memory | head + PQ codes (~32–64 TB) | centroids (10^10 lists ≈ 15 TB at 100 vectors per list, so the centroid index is itself distributed) |
| Build | distributed PiPNN (§3) | balanced clustering + closure. PiPNN partition gives this |
| Recall vs. cost | fewest reads per unit of recall in our test | more bytes per query, fewer IOs |
| Updates | segment rebuilds | append to postings, rebalance (SPFresh) |

Recommendation for about 10^12 vectors:

1. **Split first by embedding model.** Each model needs its own index anyway, since
   models live in different vector spaces.
2. **Then split by time window, so each graph holds 10^10–10^11 vectors.** That size
   is demonstrated territory. Each segment is one global graph built with distributed
   PiPNN and served DistributedANN-style: head and PQ codes in RAM, adjacency and
   vectors on NVMe. Time segments also match ingestion (new drives become new segments)
   and the most common filter.
   - The cost: a query with no time filter must search every segment, which is
     partition-and-route over P segments. §6 measured that at about 4–5× the reads of
     one global graph at 8 partitions.
   - So pick the segment size from the query log. If most queries filter on time,
     region or sensor generation, segments are nearly free. If most queries scan all
     history, use fewer, larger segments (up to one graph per model). That needs
     validating recall at 10^11–10^12 first.
3. **Budget roughly 2 PB of NVMe per copy, and 32–64 TB of RAM for codes.** That is
   about 50–200 serving nodes depending on density and copies. It is a small fraction
   of what storing the 2–3 EB lake costs.
4. **Use SPANN-style postings where they fit better:** very large k for training-data
   curation, or an archive tier of old segments on HDD or object storage. PiPNN's
   partition step can produce those postings from the same build.

## 6. Prototype results

`distributed_experiment.py` simulates S = 8 shards in one process on a 512-component
Gaussian mixture (100,000 × 32-d, 200 held-out queries, exact top-10). PiPNN
parameters: leaf 256, intra-shard fanout (4, 2), local k 4, reservoir 64, degree 32,
α 1.2. Every configuration starts from the same 1% head sample with 4 entry points.
"Reads" counts expanded nodes (one graph-store read each). "Shards" counts the distinct
shards a query touched. Raw records are in `distributed-results.json`.

**Build (S = 8 except the single-machine row):**

| Index | Replication | Mean degree | Cross-shard edges | Edge shuffle | Replica vectors |
| --- | ---: | ---: | ---: | ---: | ---: |
| single machine | 1.00 | 16.5 | — | — | — |
| no replicas (partitioned) | 1.00 | 15.2 | 0% | 0 | 0 |
| global, top-2 shards | 2.00 | 21.8 | 46% | 49.9 MB | 25.7 MB |
| global, ε = 0.05 | 2.16 | 21.2 | 49% | 57.0 MB | 29.7 MB |
| global, ε = 0.1 | 3.21 | 23.3 | 51% | 112.5 MB | 56.6 MB |
| global, ε = 0.1, no combiner | 3.21 | 23.3 | 51% | 233.0 MB | 56.6 MB |
| global, ε = 0.2 | 3.93 | 24.1 | 48% | 149.8 MB | 75.0 MB |

ε rows cap replication at 4 shards. The combiner halved the edge shuffle and produced a
bit-identical graph (`identical_to_combined: true`). The dataset is 25.6 MB of raw
vectors, so at ε = 0.1 the build moves about 6.6× the data volume over the network.
This network cost is the main build cost to watch at 10^12.

**Search (recall@10, reads per query, distinct shards touched per query):**

| Index | Setting | Recall@10 | Reads | Distance evals | Shards |
| --- | --- | ---: | ---: | ---: | ---: |
| single machine | beam 32 | 0.958 | 33.6 | 391 | 1 |
| single machine | beam 64 | 0.981 | 65.3 | 613 | 1 |
| global ε = 0.05 | beam 32 | 0.972 | 33.4 | 452 | 3.8 |
| global ε = 0.05 | beam 64 | 0.988 | 65.3 | 734 | 4.4 |
| global ε = 0.1 | beam 32 | 0.978 | 33.4 | 464 | 3.7 |
| global ε = 0.1 | beam 64 | 0.993 | 65.1 | 747 | 4.4 |
| global top-2 | beam 64 | 0.984 | 65.1 | 766 | 4.3 |
| global, no replicas | beam 128 | 0.720 | 132.8 | 1081 | 1.8 |
| partition-and-route | 4 probes, beam 16 | 0.929 | 74.9 | 1105 | 4 |
| partition-and-route | 8 probes, beam 16 | 0.971 | 153.0 | 2525 | 8 |
| partition-and-route | 8 probes, beam 32 | 0.988 | 275.8 | 4182 | 8 |
| SPANN-style | 8 probes | 0.968 | 15.4 pages (400 vectors) | 400 | — |
| SPANN-style | 32 probes | 0.993 | 42.0 pages (1098 vectors) | 1098 | — |

What this shows:

1. **The global graph beats partition-and-route by about 4–5× in reads.** At about
   0.97–0.99 recall, partition-and-route needs 153–276 reads against 33–65 for the
   global graph (5.6× more distance evaluations at 0.99). This is the same direction
   as DistributedANN's reported 6×.
2. **Partitioning without boundary replication breaks the graph.** With no replicas,
   shards become disconnected components, and recall stalls at 0.72 even at beam 128.
   Closure replication is required, which matches SPIRE's argument.
3. **The distributed graph matches or beats the single-machine one at equal reads.**
   Replicas add candidates and mean degree goes from 16.5 to 23.3, so this costs about
   20% more distance evaluations. It is not a free lunch. The modest ε = 0.05 or
   top-2 already captures most of the gain at about 2× replication.
4. **A query touches about 4–5 of the 8 shards, and its reads hop between shards
   about 28 times at beam 64.** That is SPIRE's cross-node-hop problem. It argues for
   batching reads per round and pushing scoring to the shard (DistributedANN). It also
   argues for coarser virtual shards on a machine, so consecutive hops land locally.
5. **SPANN-style postings reach 0.99 with about 42 sequential pages per query** but scan
   about 1,100 vectors, compared with 65 random reads for the graph. Postings trade bytes
   for IOs. The right choice depends on the storage medium, as discussed in §5.

## 7. What the prototype does not show

- It does not measure network time, tail latency, stragglers or failures. It counts
  reads, shards and bytes.
- The data is 100k synthetic 32-d vectors. Real AV embeddings have different
  boundary geometry, and closure replication grows quickly with dimension (ε = 0.1
  already gave 3.2× replication here, capped at 4). Tune ε and the cap on real samples.
- The head is a brute-force sample. Real heads are graphs and add their own recall loss.
- SPANN here is simplified: flat k-means, no SPTAG, a relative ε form, and one ε2.
  Treat its row as indicative only.
- Next steps: port `dpipnn.build` stages onto `native/src/fastpipnn.hpp` (leaf phase
  plus partial-reservoir export), run them under Ray or Spark on a real 1–10B sample,
  and measure shuffle volume, build time and recall against single-machine fastpipnn.
