# Distributed PiPNN at exabyte scale

Design notes for running PiPNN over 2–3 EB of vectors. They are based on three papers,
read in the requested order, and on a simulated prototype (`dpipnn.py`,
`distributed_experiment.py`, `test_dpipnn.py`) that tests the parts we can test on one machine.

> **Source caveat.** arxiv.org was not reachable from the environment that wrote
> these notes. Paper details below come from the abstracts and from secondary
> summaries found by web search. Statements marked *(summary)* should be checked
> against the PDFs before any decision depends on them.

## 1. What the papers say, and what each means for PiPNN

### DistributedANN: serving one graph across thousands of machines
[Adams et al., ICML 2025 VecDB workshop, arXiv:2509.06046](https://arxiv.org/abs/2509.06046)

- One 50-billion-vector DiskANN graph spread across more than 1,000 machines. Median
  latency is 26 ms at over 100k QPS. It is deployed in Bing over hundreds of billions of vectors.
- It is built from two well-understood components: a **distributed key-value store
  used as a large shared disk** (it holds the graph nodes) and an **in-memory head
  index** (built from the graph's top layers) that supplies beam-search entry points.
- The authors report it is **about 6× more efficient than partition-and-route** (sending
  each query to a subset of independently indexed partitions).
- *(summary)* Compressed (PQ) neighbor vectors are duplicated into each graph node, and
  node scoring runs on the storage hosts, so a hop returns scored candidates rather
  than raw bytes. The global graph is built by stitching cluster-partitioned graphs.
  The head index is reported as the CPU bottleneck.

**For PiPNN:** this is the serving model to aim for. One global graph, not N
independent shard graphs. The hard part DistributedANN had to work around, building
one graph over data that no single machine holds, is the part PiPNN's design
handles naturally (§3).

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
workable at 10^15 vectors (§2).

### SPANN: inverted lists instead of a graph
[Chen et al., NeurIPS 2021, arXiv:2111.08566](https://arxiv.org/abs/2111.08566)

- Memory–disk hybrid. Posting-list centroids are in memory (searched with an SPTAG
  graph). Posting lists are on disk.
- Build: hierarchical **balanced** clustering keeps posting lengths even. Then a
  **closure assignment** adds a boundary vector to every posting whose centroid is
  within (1+ε1) of its nearest one, capped at 8 replicas. An **RNG rule** skips
  postings whose centroid is close to an already-chosen one.
- Query: probe the nearest K postings, and drop any whose centroid is more than (1+ε2)
  farther than the nearest (query-aware dynamic pruning).
- Reaches 90% recall about 2× faster than DiskANN at equal memory. Recall@1 and
  Recall@10 of 90% in about 1 ms with 32 GB of memory on billion-scale data.

**For PiPNN:** PiPNN's leaves *are* balanced, overlapping posting lists, so PiPNN
can produce a SPANN index from its own partition step. The graph-versus-postings choice
then comes down to storage: graphs need many small random reads, postings need a few
large sequential ones (§5).

## 2. What 2–3 EB means in practice

| Vector format | 2 EB | 3 EB |
| --- | ---: | ---: |
| 768-d fp32 (3 KB) | 6.5×10^14 | 9.8×10^14 |
| 768-d fp16 (1.5 KB) | 1.3×10^15 | 2.0×10^15 |
| 128-d uint8 (128 B) | 1.6×10^16 | 2.3×10^16 |

That is about 20,000× DistributedANN's 50B and about 100,000× SPIRE's largest real run.
No published system has been evaluated at this scale. Consequences for N ≈ 10^15:

- **IDs need 50 bits.** Store a shard-local 32-bit ID plus a shard prefix only on
  cross-shard edges, or use 5–8-byte IDs. Degree 64 × 8 B = 0.5 EB of adjacency alone.
  Degree 32 × 5 B = 0.16 EB.
- **DistributedANN-style PQ duplication does not scale here.** 64 neighbors × 32 B
  codes = 2 KB per node, or 2 EB in total, more than the vectors themselves. Use smaller
  degree and codes (32 × 16 B = 0.5 EB), or store codes once and pay a second lookup.
- **The head index cannot be one machine.** A 0.1% sample with 32-byte codes is
  32 TB, so the head must be a distributed in-memory tier, or a SPIRE-style multi-level
  hierarchy (3–4 levels at this N).
- **Build compute is large but parallel.** Leaf GEMM work is about
  N × (replication) × (leaf size) × d × 2 FLOPs. With `fastpipnn` defaults (about 30
  leaf memberships per point, leaf 1024, d = 768), that is 4.7×10^22 FLOPs, roughly
  110k node-days at 5 TFLOP/s. With replication 8 and leaf 256 it drops to about 7k
  node-days. Leaf work is embarrassingly parallel and fits GPUs, so the FLOPs are not
  the hard part. Moving data across the network is (§3).
- **Hardware:** about 2.1 KB per vector (fp16 vector, adjacency, code), ×2 for
  availability, is about 4 EB. At about 123 TB of NVMe per node, that is roughly 35k
  storage nodes. If the budget only allows HDD or object storage, random 4 KB graph reads
  are not viable, and the design moves toward SPANN/SPIRE (§5).

Two questions to answer before building anything. **Does every query really need all 10^15
vectors?** If data splits by tenant, time, language or modality, route on that
metadata first: it is exact, costs no recall, and divides the problem.
**What recall and latency do you need?** Top-10 out of 10^15 depends heavily on
quantization error, so plan for exact re-ranking.

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
   the same graph, which matters for idempotent recovery on 10^4–10^5 nodes.

Two-level fanout is the key change. On one machine `fastpipnn` routes each point to
10 × 3 leaves, about 30 memberships (`native/results/compare-synthetic1m`: 30.0M
leaf points for 1M vectors). Across machines, every extra membership is a copy of
the full vector over the network. So:

- **inter-shard:** primary shard plus closure replicas within (1+ε) of the nearest
  shard centroid, capped (2–4). This is SPANN's rule at shard level, and SPIRE's
  boundary replication.
- **intra-shard:** PiPNN's usual multi-level fanout, in memory, as large as you like.

Shard granularity follows SPIRE. Use many more *virtual* shards than machines (for
example 10^5–10^6 virtual shards, each 10^9–10^10 vectors), placed onto machines by a
consistent map. The top-level k-means over 10^6 centroids itself needs an ANN index,
which is where recursion (PiPNN over centroids) comes in.

## 4. Serving

Recommended layout, following DistributedANN:

- **Graph store:** node records keyed by vector ID, placed on the primary shard's
  machine (each record holds the vector or its code, plus adjacency). Primary ownership
  follows the build partition, so most hops stay on one machine. In the experiment
  about half of the final edges cross shards at 8 shards, which is the price of one global graph.
- **Head:** an in-memory tier. A PiPNN graph over a 0.01–0.1% sample (or over the
  partition leaders, which PiPNN already computed), replicated for QPS, since
  DistributedANN reports the head as the bottleneck. At 10^15 the head is itself
  sharded, giving a 3-level SPIRE-style hierarchy.
- **Query:** the coordinator runs beam search. Each round expands W nodes (DiskANN
  beam width) and batches the reads by owning shard. Scoring is pushed to the shard,
  which returns scored neighbors. Final top-k is re-ranked with full vectors.
- **Updates:** immutable segments, LSM-style. New data goes into new segments built by
  the same pipeline, deletes are tombstones, and segments merge periodically. The
  in-place repair in `pipnn.py` is fine for small edits inside a shard, but it is not
  how 10^15 vectors evolve.

## 5. Graph or inverted lists?

| | Global PiPNN graph (DistributedANN-style) | Postings (SPANN/SPIRE-style) |
| --- | --- | --- |
| Reads per query | ~beam random 4 KB reads | a few large sequential reads |
| Storage medium | NVMe (needs IOPS) | HDD or object storage OK (needs bandwidth) |
| Memory | head tier only | centroids, which at 10^15 must be hierarchical |
| Build | distributed PiPNN (§3) | balanced clustering + closure. PiPNN partition gives this |
| Recall vs. cost | fewest reads per unit of recall in our test | more bytes per query, fewer IOs |
| Updates | segment rebuilds | append to postings, rebalance (SPFresh) |

Recommendation for 2–3 EB:

1. If queries span everything and NVMe is affordable, use **one global graph built
   with distributed PiPNN, served DistributedANN-style**, with a hierarchical head.
2. If cost forces HDD or object storage, use a **SPIRE-style multi-level index**:
   coarse postings at the bottom on cheap storage, and PiPNN graphs over the centroids at
   upper levels. PiPNN's ball carving produces the balanced, overlapping postings.
3. Most likely you will need both tiers. Hot or recent data goes in the graph on NVMe.
   The cold majority goes in postings on cheap storage. Fan out to both and merge.

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
This network cost is the number to cut at exabyte scale.

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
- The data is 100k synthetic 32-d vectors. Real embeddings at 10^15 have different
  boundary geometry, and closure replication grows quickly with dimension (ε = 0.1
  already gave 3.2× replication here, capped at 4). Tune ε and the cap on real samples.
- The head is a brute-force sample. Real heads are graphs and add their own recall loss.
- SPANN here is simplified: flat k-means, no SPTAG, a relative ε form, and one ε2.
  Treat its row as indicative only.
- Next steps: port `dpipnn.build` stages onto `native/src/fastpipnn.hpp` (leaf phase
  plus partial-reservoir export), run them under Ray or Spark on a real 1–10B sample,
  and measure shuffle volume, build time and recall against single-machine fastpipnn.
