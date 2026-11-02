# PiPNN on disk vs HNSW on disk (and LanceDB's HNSW)

Goal: serve a PiPNN graph from SSD with little RAM, and compare it with HNSW served the same
way and with LanceDB's HNSW index. Code: `native/src/disk_index.hpp` (layout + search),
`native/src/disk_bench.cpp` (driver), `native/scripts/run_disk.sh`,
`native/scripts/run_disk_entry.sh`, `lance_hnsw_check.py`. Raw results:
`native/results/disk-synthetic1m/` (`results.jsonl`, `results-entry.jsonl`, `lance-hnsw.jsonl`).

## 1. Design

```
 RAM (small)                                    SSD (large)
 ┌──────────────────────────────┐     ┌──────────────────────────────────────────┐
 │ PQ codes (64 B per vector)   │     │ node file, 4 KB sectors, 10 records each:│
 │ ID map (4 B per vector)      │     │ [vector 128 B | degree | 64 neighbor IDs]│
 │ entry: medoid, + optional    │     │ a record never straddles a sector:       │
 │  hot-node cache (PiPNN) or   │     │ one node = at most one 4 KB read          │
 │  HNSW upper layers (HNSW)    │     └──────────────────────────────────────────┘
 └──────────────────────────────┘
 query: beam search ordered by PQ distance; each round expands W = 4 candidates, reading
        their sectors with O_DIRECT (no page cache). The full vector in each fetched record
        gives the exact distance, so the final top-k is exact without extra reads.
 page expansion (Starling-style): every other node in a freshly read sector is expanded too.
 layouts: none (row order), bfs (BFS from the entry over the graph), carve (PiPNN leaves of 32).
```

Both graphs go through the same file format and search code, so differences are the graphs'.
HNSW is stored as its level-0 graph; its upper layers (3.2% of nodes, 6.7 MB) are kept in RAM
and descended greedily per query to choose the level-0 entry, as an HNSW-on-disk system would.
For equal RAM PiPNN gets a 16,500-node hot cache (BFS around the medoid) of the same size.

fastpipnn also gained `save_graph` / `load_graph`, and the input is memory-mapped.

## 2. Setup

Synthetic 1M × 128 uint8 (128-cluster Gaussian mixture, `native/scripts/synthetic.py`), 1,000
queries, k = 10, exact ground truth. 4 cores, virtual disk (`/dev/vda`), one 4 KB `O_DIRECT`
read ≈ 0.07–0.13 ms. Graphs: fastpipnn degree 64 (mean 35), HNSW M 32 / efC 200 (level-0
degree 64, mean 33). PQ 64 B (2 dimensions per sub-quantizer). Latency is one thread; QPS uses
4 threads sharing the file. The LanceDB numbers go through Python (`ds.to_table`), which adds
a few ms per query.

## 3. Results

### 3.1 Headline: equal RAM (≈ 75 MB), best layout (BFS + page expansion), O_DIRECT

| Index | Build | Recall 0.95 | Recall 0.99 | Recall 0.998 |
|---|---:|---|---|---|
| **PiPNN on disk** (16.5k-node cache) | **35.7 s** | beam 32: 0.941, 35 reads, 3.4 ms | beam 64: 0.989, 63 reads, 5.8 ms | beam 128: 0.9986, 115 reads, 9.1 ms |
| **HNSW on disk** (upper layers in RAM) | 133.8 s | beam 32: 0.947, 34 reads, 3.5 ms | beam 64: 0.991, 62 reads, 5.4 ms | beam 128: 0.9973, 117 reads, 9.1 ms |
| LanceDB IVF_HNSW_SQ, 1 partition, warm | 320 s | ef 64: 0.938, 3.4 ms | ef 128: 0.987, 3.7 ms | ef 256: 0.997, 4.6 ms |
| LanceDB IVF_HNSW_SQ, 64 partitions, warm | 209 s | ef 64, nprobe 4: 0.941, 4.6 ms | ef 128, nprobe 4: 0.990, 4.6 ms | ef 256, nprobe 4: 0.999, 6.0 ms |

RAM: PiPNN/HNSW on disk ≈ 75 MB; the in-RAM graph needs ≈ 384 MB (vectors + adjacency);
LanceDB warm keeps its 430–465 MB index in its cache. Disk: 410 MB node file for both graphs.

### 3.2 LanceDB's HNSW is not a disk-resident graph

| LanceDB IVF_HNSW_SQ, cold (index cache disabled, reopened per query) | Bytes read / query | Latency | Recall |
|---|---:|---:|---:|
| 1 partition (one HNSW over 1M) | 265 MB | 984 ms | 0.946 |
| 64 partitions, nprobe 1 | 6 MB | 25 ms | 0.930 |
| 64 partitions, nprobe 4 | 35 MB | 55 ms | 0.934 |
| 64 partitions, nprobe 16 | 121 MB | 162 ms | 0.934 |
| *PiPNN on disk, beam 64, no cache (68 MB RAM)* | *0.3 MB (75 reads × 4 KB)* | *7.8 ms* | *0.995* |

Lance stores IVF partitions on disk and loads a probed partition's whole HNSW sub-graph into
its index cache on first use. Warm, it is an in-memory HNSW (the whole index in RAM); cold, a
query pays for loading entire partitions. A paged graph reads only the nodes it visits.

### 3.3 What helps on disk (PiPNN, no cache, 68 MB RAM)

| Layout / mode | Reads for recall ≥ 0.99 | Latency |
|---|---:|---:|
| row order, no page expansion | 149 (beam 128, 0.986) / 276 (beam 256) | 10.7 / 19.1 ms |
| BFS order, no page expansion | 129 (beam 128, 0.986) | 11.8 ms |
| **BFS order + page expansion** | **75 (beam 64, 0.9945)** | **7.8 ms** |
| carve order + page expansion | 76 (beam 64, 0.992) | 6.9 ms |
| BFS + page expansion + 10k cache (72 MB) | 65 (beam 64, 0.989) | 5.8 ms |

- Page expansion on a locality-preserving layout halves the reads per recall level. Only
  0.3–0.5% of edges land in the same sector, but sector-mates are BFS siblings or carving
  leaf-mates, i.e. close in space, so expanding them is useful work.
- On row order (random), page expansion costs CPU (10× more nodes expanded) for no gain.
- I/O dominates: the same search with the node file in RAM takes 1.8 ms instead of 7.8 ms.
- PQ size matters on this data: 32 B codes (4 dims per sub-quantizer) cap recall at 0.81–0.88
  even at beam 256; 64 B codes match the in-RAM recall.

### 3.4 Entry point matters for HNSW, not for PiPNN

| HNSW on disk, BFS + page expansion | Recall at beam 16 | Reads for 0.99 |
|---|---:|---:|
| entry = medoid (level 0 only) | 0.54 | 146 (beam 128) |
| entry = descent through upper layers in RAM | 0.82 | 62 (beam 64) |

HNSW's long-range links live in the upper layers; its level 0 alone navigates poorly from a
fixed entry on clustered data. PiPNN's flat graph reaches 0.82 from the medoid with page
expansion and needs no extra structure.

## 4. Conclusions

1. **PiPNN works as a disk index.** With 64 B PQ codes in RAM (≈ 7% of the vector + graph
   size) it keeps its in-RAM recall and reaches 0.99 with about 63–75 4 KB reads per query.
2. **On disk, PiPNN and HNSW are equal on query cost at equal RAM** (within ~5% in reads and
   latency at 0.95–0.998 recall), **and PiPNN builds 3.7× faster** (35.7 s vs 133.8 s here;
   LanceDB's IVF_HNSW_SQ took 209–320 s).
3. **LanceDB's HNSW needs the probed partitions in RAM.** Warm it is fast (3.7–4.6 ms at 0.99,
   including Python overhead) but holds the whole index in memory; cold queries read 6–265 MB
   each. To serve data much larger than RAM with HNSW-quality recall, a paged graph (this
   layout) reads about 0.3 MB per query instead.
4. **Layout + page expansion is the main disk optimisation** (2× fewer reads), and PiPNN's own
   carving gives a layout as good as BFS order.

## 5. Limits and next steps

- Synthetic data, 1M vectors, one machine with a virtual disk; real NVMe latency and queue
  depth differ. W = 4 reads per round are issued one after another (no io_uring), so latency
  is roughly reads × read time; parallel reads per round would cut it.
- LanceDB was measured through Python; a Rust-side benchmark would remove a few ms per query.
- Next: BIGANN / real embeddings at 10M+ where the node file exceeds RAM; io_uring for parallel
  reads; out-of-core PiPNN build (shard → spill edges → merge by source, per DISTRIBUTED.md).
