# Mutable PiPNN demo

A small Python/NumPy implementation with build, add, delete, update, graph search,
and recall monitoring. This is an educational prototype, not the authors implementation.

## Run with uv

Requires Python 3.10+ and uv. The checked-in uv.lock pins the dependency resolution.

```bash
uv sync --locked
uv run --locked python -m unittest -v
uv run --locked python experiment.py
```

The experiment writes results.json. To change the workload:

```bash
uv run --locked python experiment.py --n 1000 --dim 32 --queries 100 --batch 50 --seeds 7 19 31 --beam 64 --output larger-results.json
```

## API

```python
import numpy as np
from pipnn import PiPNN

rng = np.random.default_rng(7)
index = PiPNN(leaf_size=64, local_k=12, degree=24).build(
    rng.normal(size=(500, 16))
)
new_id = index.add(rng.normal(size=16))
index.update(new_id, rng.normal(size=16))
neighbors = index.search(rng.normal(size=16), k=10, beam=48)
index.delete(new_id)

queries = rng.normal(size=(40, 16))
report = index.monitor(
    queries, min_recall=0.90, max_mutations=100, rebuild=True
)
index.validate()  # expensive correctness checks; use outside timed workloads
```

Methods return stable integer IDs. Build also accepts explicit IDs. Updates
preserve IDs; automatic IDs are not reused after deletion or rebuild.
Inputs are copied and must be finite vectors of the original dimension.
Unknown IDs raise KeyError; invalid inputs raise ValueError. Do not mutate
the exposed internal dictionaries or vector arrays directly.

## Relation to the paper

Based on [PiPNN, arXiv:2602.21247v3](https://arxiv.org/html/2602.21247v3),
Sections 3 and 4: randomized leader partitioning with multilevel fanout,
leaf-local k-NN with bidirected candidates, residual-hyperplane HashPrune,
and RobustPrune. Dense NumPy matrix products compute leaf distances.
We omit small-leaf merging, packed reservoirs, parallelism, and optimized
partial sorting. The mutation protocol below is an extension, not a
dynamic-index algorithm evaluated in that paper.

## How mutations work

1. **Route:** traverse saved leader coordinates using the same fanout as build.
   Leaders are immutable routing anchors, so deleting a leader ID is safe.
2. **Compute:** insertion computes distances within the routed leaves.
3. **Local neighbors:** update their top-k lists and identify displaced edges.
   Deletion recomputes affected leaf lists to find replacement neighbors.
   Update performs delete then insert with the same ID and new coordinates.
4. **HashPrune:** maintain reference counts for bidirected candidates shared by
   overlapping leaves. Stream additions. If the last reference to a candidate
   disappears, rebuild that source reservoir from its remaining candidates.
5. **Final graph:** prune reservoirs, maintain reverse references to these
   intermediate edges, then prune outgoing plus reciprocal candidates.
   Only affected sources and their reciprocal dependents are repaired.
6. **Monitor:** compare held-out queries against exact neighbors of live vectors.
   A low recall or mutation-count threshold can trigger a full rebuild.
   Monitoring is explicitly called by the caller; there is no background worker.

Final edges are directed: reciprocal edges are proposed, but degree enforcement
can prune one direction. Strict symmetry and connectivity are not guaranteed.
The configured degree bound always applies to outgoing edges.

## Experiment and interpretation

Default: two seeds, 500 vectors in 16 dimensions, 40 held-out queries,
25 operations per stage, and beam width 48. The data are correlated Gaussian
vectors; updates and mixed mutations introduce a coordinate shift.

Each build/add/delete/update/churn stage:
- independently recomputes exact top-10 neighbors on the current live dataset;
- measures recall@10 and graph-search time;
- compares with a fresh build using identical live IDs, vectors, and parameters;
- reconstructs local lists, candidate reference counts, reservoirs, reverse edges,
  and final adjacency lists to check consistency;
- records operation time, fresh-build time, degree, leaf growth, and query work.

Validation, exact ground truth, and fresh-baseline work are excluded from mutation
and query timings. Query distance-evaluation counts cover graph vectors only,
excluding routing-anchor comparisons. Mutation times are batch means, not latency
percentiles. No speedup or large-scale claim follows from this experiment.
The navigation graph intentionally favors directional diversity; its adjacency
lists are not exact top-k lists. Reported recall measures search results.

## Comparing with HNSW

compare_hnsw.py runs this PiPNN and hnswlib (installed by the optional
`compare` extra) on identical vectors, queries, and a pre-generated
build/add/delete/update/churn workload. After each stage it recomputes exact
top-10 neighbors once and sweeps beam width (PiPNN `beam`, HNSW `ef`).

```bash
uv sync --locked --extra compare
uv run --locked --extra compare python compare_hnsw.py --seeds 7 19
```

Defaults: 1,000 vectors, 32 dimensions, 100 queries, 50 operations per stage,
beams 10/16/32/64/128. PiPNN uses degree 24; HNSW uses M=12 (layer-0 degree 24)
and ef_construction=200. Both run single-threaded. HNSW deletion is a tombstone
(`mark_deleted`); HNSW update re-adds the existing label.

Smallest beam reaching mean recall@10 targets over seeds 7 and 19
(raw records in compare-results.json):

| Stage | Algorithm | mutation ms/op | beam@90% | beam@95% | beam@99% |
| --- | --- | ---: | ---: | ---: | ---: |
| build | pipnn | — | 16 | 16 | 32 |
| build | hnsw | — | 16 | 16 | 32 |
| add | pipnn | 147.05 | 16 | 16 | 32 |
| add | hnsw | 0.10 | 16 | 32 | 32 |
| delete | pipnn | 158.24 | 16 | 16 | 32 |
| delete | hnsw | 0.00 | 16 | 32 | 32 |
| update | pipnn | 268.70 | 16 | 16 | 32 |
| update | hnsw | 0.59 | 16 | 32 | 64 |
| churn | pipnn | 156.21 | 16 | 16 | 32 |
| churn | hnsw | 0.20 | 16 | 32 | 32 |

Read recall, not time: this is pure Python against C++ hnswlib, so timings are
three orders of magnitude apart for reasons unrelated to the algorithms.
PiPNN search also starts from one representative per routed leaf, while HNSW
descends from a single entry point, which helps PiPNN at small beams.
For a compiled, static comparison against upstream PiPNN use native/
(`native/scripts/run_benchmark.py`).

## Native from-scratch PiPNN (fastpipnn)

native/src/fastpipnn.hpp is a standalone C++17 PiPNN builder (Eigen + std::thread only),
written against the paper and the authors' reference code but sharing no code with it:
randomized ball carving with fanout 10,3,1 and small-cluster merging, one Gram GEMM per
leaf with bidirected 2-NN candidates, HashPrune reservoirs (12-bit hash of q - p, 160
slots), and RobustPrune to degree 64. The build is deterministic: HashPrune uses a total
(distance, ID) order, so the final graph does not depend on thread interleaving.

Engineering beyond the reference implementation:
- leaf k-NN: branchless AVX-512 masked-blend insertion updates 16 top-k lists per step;
- uint8 data: exact bfloat16 dot-product Gram kernel (`vdpbf16ps`); 0..255 are exact
  in bf16 and fp32 accumulation is exact below 2^24, so graphs match the fp32 path bit for bit;
- HashPrune: unsorted structure-of-arrays reservoirs with SIMD hash lookup, one
  hash compare instruction per offer, and offers grouped per target (one lock per leaf)
  with software prefetch.

```bash
python3 native/scripts/bootstrap.py
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j --target ann_bench fastpipnn_test
native/build/fastpipnn_test
python3 native/scripts/run_benchmark.py --base native/results/synthetic1m/base.u8bin \
  --queries native/results/synthetic1m/query.u8bin --gt native/results/synthetic1m/gt.ibin \
  --threads 4 --query-count 200 --allow-small --output native/results/my-run
```

Recorded run: clustered synthetic uint8 data (native/scripts/synthetic.py), 128
dimensions, 200 queries, 4 threads (Xeon @ 2.1 GHz, AVX-512), same parameters for both
PiPNN builds, single-threaded queries. Both PiPNN graphs are searched from the medoid.
Raw records: native/results/compare-synthetic100k and compare-synthetic1m.

| n | Algorithm | Build s | recall@10 beam 32 / 64 / 128 / 256 |
| ---: | --- | ---: | --- |
| 100k | fastpipnn | 1.67 | 0.957 / 0.995 / 1.000 / 1.000 |
| 100k | upstream PiPNN | 3.69 | 0.948 / 0.990 / 0.995 / 1.000 |
| 100k | hnswlib (M=32, efC=200) | 4.81 | 0.976 / 0.999 / 1.000 / 1.000 |
| 1M | fastpipnn | 26.9 | 0.836 / 0.936 / 0.988 / 1.000 |
| 1M | upstream PiPNN | 44.3 | 0.832 / 0.942 / 0.990 / 1.000 |
| 1M | hnswlib (M=32, efC=200) | 130.6 | 0.791 / 0.923 / 0.980 / 0.993 |

At 1M, fastpipnn builds 1.65x faster than upstream PiPNN and 4.9x faster than hnswlib at
similar recall. Build phases at 1M: partition 4.1 s, leaves 17.9 s, prune 4.9 s. HNSW
uses far less build memory (630 vs 1902 MiB peak; PiPNN reservoirs dominate) and gives
higher QPS at matched recall on the 100k set. Timings are from a shared 4-core cloud VM;
they are one run each, not a controlled benchmark.

Differences from the reference code: small-cluster merging is a single greedy pass; the
duplicate guard splits randomly at depth 8 (reference: 3) or when a child equals its
parent; distances in reservoirs are bfloat16 like the reference, but final pruning
recomputes exact distances. No graph save/load, no MIPS, no int8/float benchmarks yet.

## Distributed PiPNN (~10^12 vectors)

DISTRIBUTED.md covers adapting PiPNN to about 10^12 vectors (embeddings of a 2–3 EB
autonomous-driving data lake), drawing on DistributedANN, SPIRE and SPANN:
scale arithmetic, a MapReduce-shaped distributed build (balanced shards, closure
replicas, per-shard leaf k-NN, exactly mergeable HashPrune combiner, owner-side
RobustPrune), serving, and graph vs. inverted lists. dpipnn.py simulates it in one
process, counting reads, shards touched and shuffle bytes:

```bash
uv run --locked python -m unittest -v test_dpipnn
uv run --locked python distributed_experiment.py   # writes distributed-results.json
```

## Limitations

- Python dictionaries and retained local candidate lists use more memory than
  the static algorithm. Candidate storage scales with leaf memberships and local
  k; only reservoirs and final outgoing degrees have fixed bounds.
- The hierarchy stays fixed during mutations. Leaves can grow or become empty;
  rebuild periodically under drift or growth. Unsplittable duplicate vectors
  and the recursion-depth guard can also produce leaves above leaf_size.
- Deletions rebuild affected leaf distance matrices, so large leaves cost more.
  Insertion re-ranks existing local neighbor lists for clarity.
- Empty leaves receive insertions if their saved routing anchors are selected.
  Search starts from one live representative per routed leaf plus a global
  representative; this differs from the single-entry query in the paper.
- No persistence, concurrent mutation, transactions, GPU kernels, or automatic
  leaf splitting. Floating-point arithmetic can be inaccurate for extreme
  magnitudes or nearly equal distances; use appropriately scaled vectors.
- Recall is empirical and can degrade with distribution shift. A rebuild is
  not guaranteed to meet a requested recall threshold.

## Recorded run

Executed with uv, Python 3.12.3, NumPy 2.5.4. All six unit tests passed.

The default 500-vector workload achieved 100% recall@10 at every stage for both seeds.

A harder run used 1,000 vectors, 32 dimensions, 40 queries, seed 31, and beam 16:

| Stage | Mutable recall@10 | Fresh recall@10 | Operation seconds | Graph distances/query |
| --- | ---: | ---: | ---: | ---: |
| build | 95.75% | 95.75% | 2.280 | 235.1 |
| add | 95.75% | 97.00% | 3.508 | 236.9 |
| delete | 95.75% | 96.25% | 3.802 | 235.1 |
| update | 95.50% | 96.00% | 5.037 | 237.1 |
| churn | 95.50% | 95.00% | 2.721 | 236.6 |

All stage invariants passed. Recall remained above the configured 90% monitoring threshold, so these experiment runs did not trigger a rebuild; unit tests exercise both recall-triggered and periodic rebuilds. Batch mutation costs exceeded fresh-build costs at this scale. The results support prototype correctness and useful recall, not a performance advantage.

Raw records: results.json and results-beam16.json. Timings are machine-specific and are not a controlled performance benchmark.
