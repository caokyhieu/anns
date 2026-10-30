# Training-data retrieval over ~10^12 AV embeddings

Design for a distributed vector retrieval system that builds training datasets from an
autonomous-driving data lake (2–3 EB of video, lidar and radar; about 10^12 embedding
vectors). It is optimized for **indexing time** and **throughput** when each query
returns a lot of data. It draws on the three requested papers, related work, and
measurements from this repository's prototype (`dpipnn.py`, `retrieval_experiment.py`).
DISTRIBUTED.md has the detailed paper notes and the distributed PiPNN build.

## 1. The workload decides the design

Training-data retrieval is not web search:

| | Web search (DistributedANN's target) | Training-data retrieval (this system) |
| --- | --- | --- |
| Results per query | 5–200 | 10^3–10^6 (or "everything above a similarity threshold") |
| Queries | 100k QPS, independent | jobs of 10^2–10^5 seed queries (example clips, text prompts), submitted together |
| Latency | ~10 ms | seconds to minutes per job |
| Filters | few | almost always: time, region, weather, sensor rig, labels |
| Output | a ranked page | a dataset manifest (IDs, then blob pointers), deduplicated and balanced |
| Data | updated continuously | new drives daily; appends dominate |

Three consequences follow.
1. **Large k favors posting lists over graphs.** Graph beam search needs a beam of at
   least k, so random reads grow with k. DistributedANN reports recall@200 of only
   71.9% at 50B, against 90.8% at recall@5.
2. **Batching is the throughput lever.** Queries in one job overlap, so each posting
   can be read once and scored against every query that wants it.
3. **Indexing must be streaming and incremental.** Rebuilding a 10^12-vector graph is
   measured in days; appending to postings is not.

## 2. What the literature says

| Work | Key result | Use here |
| --- | --- | --- |
| **DistributedANN** (Adams et al., 2025) | One 50B DiskANN graph on 1,000+ machines. 6× throughput of clustered partitioning. 640 vs 4,800 IOs/query. But 780 TiB SSD (OPQ codes copied into every node, ~10× amplification), head index is the CPU bottleneck, recall@200 = 71.9% | The interactive tier (§5.4). Its build (closure partitions plus stitching) matches our distributed PiPNN build |
| **SPIRE** (Xu et al., 2025) *(summary only)* | Partition granularity trades vector reads against cross-node hops. Recursive balanced clustering until the top fits in memory. Boundary replication. 8B vectors on 46 nodes, up to 9.64× throughput | Multi-level routing. Coarse postings for read-heavy, large-k queries |
| **SPANN** (Chen et al., 2021) | Postings on SSD, centroids in memory. Hierarchical balanced clustering (1M vectors in ~2 min, 5× faster on GPU). Closure assignment (≤8 replicas, RNG rule). Query-aware pruning. 2× faster than DiskANN at 90% recall. Distributed version bin-packs partitions by query history | The bulk tier's index structure and build |
| **SPFresh** (SOSP 2023) | In-place updates to SPANN with LIRE (split, merge, reassign of boundary vectors). Avoids 2–5-day rebuilds at billion scale. 1% DRAM at 1% daily updates | Daily ingestion without rebuilds |
| **SOAR** (Google, NeurIPS 2023) | Spilled (multiple) assignment trained so the second assignment covers the first one's failures | Alternative to closure replication with fewer copies |
| **RaBitQ / Extended RaBitQ** (SIGMOD 2024/2025) | 1–B bits per dimension with error bounds. Fast-scan kernels | Compressed codes in postings: scan more vectors per byte |
| **IVF-RaBitQ on GPU** (PVLDB 2026) | IVF with RaBitQ reaches about 3× CAGRA QPS at ~0.95 recall in GPU batch mode | GPU batch scan of postings |
| **CAGRA / cuVS** | GPU graph build 2–27× faster than HNSW. cuVS IVF builds up to 4.7× faster than Faiss GPU | GPU build of the router graph and clustering |
| **BatANN** (2025) | Passes query state to the server owning the next neighborhood. 1.4–2.1× DistributedANN throughput at 1B on 10 servers | If the interactive graph tier is distributed |
| **Graph partitioning for ANN** (Gottesbüren et al., VLDB 2025) | Neighborhood-preserving shards with learned routers. Up to 1.72× QPS at 90% recall | Shard placement that keeps graph hops local |
| **Pyramid**, **HARMONY** | Partition-and-route frameworks. HARMONY combines dimension and vector partitioning | Baselines. Scatter-gather costs grow with partitions |
| **Filtered-DiskANN**, **ACORN** | Filters inside graph search. Needed degree grows with 1/selectivity | Shows why filters are simpler on postings (scan, then mask) |
| **Automatic data curation** (Vo et al., TMLR 2024) | Hierarchical k-means, then balanced sampling across clusters, matches or beats manual curation for self-supervised pretraining | The same posting hierarchy gives concept-balanced training sets |
| **Driving-RAG**, scenario embeddings | Scenario embeddings plus vector search for AV scenario retrieval | Confirms the use case. The embedding model matters as much as the index |

## 2b. Related work on AV / VLA training data

| Work | What it does | Implication |
| --- | --- | --- |
| **Cosmos-Embed1** (NVIDIA, 2025) | Video-text dual encoder for physical AI. 768-d (448p/336p) or 256-d (224p) clip embeddings for text-to-video and video-to-video search and semantic dedup. Text-to-video R@1 ≈ 30–35 on the OpenDV AV benchmark (NVIDIA-reported) | A ready-made camera-clip embedding. Its dimension sets index size (256-d is 3× smaller) |
| **NeMo Curator video pipeline** (NVIDIA) | Splits video into clips, embeds them with Cosmos-Embed1 to Parquet, and runs semantic dedup as k-means followed by pairwise similarity inside each cluster, on Ray | Dedup is the same clustering as our posting lists. Do it inside each posting during the build |
| **LidarCLIP** (WACV 2024) | Trains a LiDAR encoder into CLIP's space using paired camera and LiDAR frames, so text queries reach point clouds. LiDAR retrieval is on par with image retrieval, with complementary strengths | LiDAR can share the camera/text embedding space, so one text query searches both |
| **BEV-TSR** (AAAI 2025) | Text-to-scene retrieval in BEV space: BEV features plus an LLM text encoder, contrastive and captioning losses. nuScenes-Retrieval benchmark, ~87.7% top-1 text-to-scene | Fused BEV features are a natural scene-level embedding for fused camera, LiDAR and radar |
| **CARIM** (ICCV 2025) | Scene retrieval via VLM/LLM captions, turning text-to-video into text-to-text matching. "Inclusive" matching: return only scenes that satisfy every condition in the query | Pure similarity misses conjunctive constraints. Add caption text and structured attributes as filters, with hybrid text plus vector search |
| **Multimodal scenario similarity search** (2026); **ScenarioFormer**; **Driving-RAG** (2025); LLM-informed trajectory embeddings (2025) | Trajectory embeddings (contrastive transformers, VRAE) work well for motion events (cut-ins, turns). Visual and trajectory embeddings combined beat either alone. Driving-RAG uses HNSW plus typical-scenario sampling | Index a trajectory or scenario embedding next to the visual one, and fuse rankings |
| **Hierarchical segment indexing** and **multi-vector index compression** (2025–2026) | Long videos indexed coarse-to-fine (video, then segment), or as several vectors per item compressed to a budget. Multi-vector video can cost ~10 MB per video uncompressed | Use the segment (a few seconds) as the index unit, with a clip or drive-level summary vector above it |
| **WOD-E2E** (Waymo, 2025) | Mines long-tail end-to-end driving segments with rules plus multimodal LLMs. Mining kept ~0.1% of data, and ~0.03% were final long-tail scenarios | Training retrieval is a funnel: cheap filters and vector recall over everything, then expensive VLM scoring on a small candidate set |
| **Alpamayo-R1** (NVIDIA, 2025) | VLA training clips chosen for explicit driving decisions, with per-scenario balancing for diversity | Balancing and dedup belong in the retrieval output, not afterwards by hand |
| **Mining the Long Tail** (2025) | Uncertainty-weighted curation beats uniform sampling on rare safety-critical events | Store model signals (uncertainty, disagreement) as columns, to filter and rank by |
| **Automatic data curation** (Vo et al., TMLR 2024) | Hierarchical k-means, then balanced sampling across clusters | The posting hierarchy gives concept-balanced sampling for free |
| **Tesla trigger classifiers** (patents, Karpathy) | On-vehicle classifiers on intermediate network outputs decide which data to upload | Filtering at the edge reduces N before indexing |
| **Lance format** (ByteDance Volcano Engine, WeRide case studies) | Columnar lakehouse format for images, point clouds and video blobs plus vector and scalar indexes on object storage. PB-scale AV data lake (vendor-reported). WeRide reports scenario mining going from 1 week to 1 hour | A candidate storage layer for postings, metadata columns and manifests (benchmark before adopting) |
| **CoReVLA**, **OpenLongTail** (2025–2026) | Fill long-tail gaps through simulation takeovers or generative video | Retrieval also finds gaps: when a query returns too little, it becomes a generation or collection request |

**What this changes in the design:**

1. **Index unit and vectors.** Index clip segments of a few seconds, with several
   embeddings per segment, each in its own posting index under a shared segment ID:
   - a video-text embedding (Cosmos-Embed1-style; LiDAR can be trained into the same
     space, LidarCLIP-style);
   - a fused BEV scene embedding;
   - a trajectory or scenario embedding.

   Put drive or scene-level summary vectors above the segments for coarse-to-fine
   search. N in §4 then counts segments × embedding types, after deduplication.
2. **Hybrid queries.** Combine vector similarity (one or more embeddings, fused by
   rank), caption full-text search, and structured filters (time, geography, weather,
   rig, labels, model uncertainty). Use CARIM-style conjunctive matching for "all of
   these conditions" requests. In posting scans these are column masks.
3. **A retrieval funnel.** Recall over 10^12 vectors in postings, then re-rank 10^5–10^7
   candidates with full vectors, then VLM or LLM scoring of the top candidates (as in
   WOD-E2E), then balanced sampling and dedup. Write the result as a manifest.
4. **Dedup and balance during the build.** Run NeMo-Curator-style semantic dedup
   inside each posting. Sample across the hierarchy for balance (Vo et al.,
   Alpamayo-R1).
5. **Storage.** Postings, metadata columns and manifests on a columnar format over
   object storage (Lance or Parquet), with NVMe caching for the hot postings.

## 3. Measurements from the prototype

100,000 vectors × 32-d (512-component Gaussian mixture), 1,000 held-out queries, exact
ground truth. The graph is PiPNN (degree 32). Postings are SPANN-style: 781 lists of
about 128 vectors, closure replication 1.06. Raw records: `retrieval-results.json`
and `distributed-results.json`.

**Build time (one process, Python/NumPy):**

| Index | Build | Recall@10 at beam 64 / 64 probes |
| --- | ---: | ---: |
| PiPNN graph, fanout (4, 2), reservoir 64 | 45.9 s | 0.989 |
| PiPNN graph, fanout (2, 1), reservoir 32 ("fast") | 8.0 s | 0.941 |
| Postings (balanced k-means + closure) | 4.6 s | 1.000 |
| PiPNN router graph over the 781 centroids | 0.13 s | same recall as brute-force routing |

**Recall as k grows (per query):**

| k | Index | Setting | Recall@k | Random reads | Vectors scored |
| ---: | --- | --- | ---: | ---: | ---: |
| 100 | graph | beam 200 | 0.982 | 201 | 1,645 |
| 100 | postings | 32 probes | 0.987 | 32 | 4,621 |
| 1000 | graph | beam 2000 | 0.992 | 2,000 | 19,341 |
| 1000 | graph (fast build) | beam 2000 | 0.918 | 2,001 | 10,499 |
| 1000 | postings | 128 probes | 0.963 | 128 | 18,432 |
| 1000 | postings | 256 probes | 0.995 | 256 | 36,753 |
| 1000 | postings, PiPNN router | 256 probes | 0.995 | 256 | 36,753 |

**Batching (k = 1000, postings read once per batch):**

| Probes (recall@1000) | Batch 1 | Batch 10 | Batch 100 | Batch 1000 |
| --- | ---: | ---: | ---: | ---: |
| 64 (0.875) | 9,259 vectors/query | 5,566 | 1,023 | 106 |
| 128 (0.963) | 18,445 | 7,909 | 1,050 | 106 |

What this shows:
- **Large k:** graph random reads grow linearly with k (about 2k for 0.99 recall).
  Postings reach the same recall with about 8× fewer, sequential reads, at the cost of
  scoring about 2× more vectors. GEMM and fast-scan kernels make that scoring cheap. At
  k = 10^4–10^6, a graph would need 10^4–10^6 dependent random reads per query.
- **Build:** postings build 10× faster than the graph. The "fast" graph build is 5.7×
  faster but loses a lot of recall at large k (0.918 vs 0.992).
- **Routing:** a PiPNN graph over the centroids routes exactly as well as brute force,
  so routing can scale to 10^8 centroids without a full scan.
- **Batching:** at batch 100, reads per query drop 10–18×. At batch 1000 every posting
  is read about once per batch, so the batch costs one full scan (106 vectors per query
  = 1.06 replication × 100k / 1000). That crossover tells the planner when to switch to
  a full scan.
- **Caveats:** these are 32-d synthetic vectors at 100k scale, and k = 1000 is 1% of
  the data, which is much harsher than k = 10^4 out of 10^12. The shapes of the curves
  carry over. The constants must be re-measured on real embeddings at 10^9 scale.

## 4. Architecture

```
                         ┌──────────────── router (RAM, replicated) ────────────────┐
  job: Q seed queries ──►│ L0: 1e5 centroids (brute force, GPU)                       │
  + filters + k / τ      │ L1: 1e8 centroids — PiPNN graph over RaBitQ codes          │
                         └───────────────┬──────────────────────────────────────────┘
                                         │ posting IDs per query
                     batch planner: invert to posting → {queries}; if coverage high → full scan
                                         │
     ┌──────────────── storage nodes (NVMe, postings bin-packed by access) ─────────┐
     │ posting = [IDs | RaBitQ codes | metadata columns (time, geo, rig, labels)]   │
     │ stream each needed posting once → filter mask → GPU/AVX scoring vs. its queries│
     │ → per-query top-k heap or threshold hits                                      │
     └───────────────────────────────┬──────────────────────────────────────────────┘
                                     │ shuffle candidates by query
          merge → exact re-rank of the top candidates (full int8 vectors, object store)
          → dedup / diversify → manifest (Parquet: IDs → blob URIs, timestamps) → training
```

### 4.1 Storage layout at 10^12 (assuming 768-d embeddings)

| Item | Size | Medium |
| --- | ---: | --- |
| Posting entries: 1-bit RaBitQ (96 B) + 6-byte ID + ~10 B metadata, ×1.3 replication | ~150 TB | NVMe |
| Optional 4-bit codes for a second scoring pass | ~0.5 PB | NVMe or SSD tier |
| Full vectors (int8) for exact re-rank of final results | ~0.77 PB | object storage or HDD |
| L1 router: 10^8 centroids (codes + PiPNN graph) | ~30 GB | RAM, per router replica |
| L0 router: 10^5 centroids, fp16 | ~150 MB | RAM or GPU |

Postings hold about 10^4 vectors each (about 1 MB), coarser than SPANN's
latency-oriented 16% centroids, as SPIRE's granularity argument suggests for read-heavy
queries. One posting is one large sequential read, which suits NVMe and object-storage
range reads. This is far smaller than the ~2 PB of the graph-everything design in
DISTRIBUTED.md (§2), which needs no codes duplicated per neighbor and no 1.3 PB of
build-time reservoirs.

### 4.2 Build pipeline (indexing time)

1. **Deduplicate at ingest (the biggest lever).** Consecutive frames at 10–30 Hz are
   near-duplicates. Keep one vector per short clip segment (for example 1–2 s), plus a
   segment → frames mapping. Index frame-level vectors only for flagged segments.
   This plausibly cuts N by 5–20×, and every later cost scales with N.
2. **Train the hierarchy on a sample.** Hierarchical balanced k-means (SPANN's HBC) on
   10^8–10^9 sampled vectors, on GPUs (cuVS k-means): hours, not days.
3. **One streaming assignment pass.** Each vector descends the hierarchy, about 1–5k
   distance evaluations, roughly 2–8×10^18 FLOPs in total, or tens of GPU-hours. It is
   then appended to its posting plus closure replicas (RNG rule, at most 2–4, or SOAR
   spilling). The pass is IO bound: reading 1.5 PB of fp16 embeddings at 500 GB/s
   (100 nodes × 5 GB/s) takes about 50 minutes, so the whole build is a few hours on
   about 100 GPU nodes. For comparison, a PiPNN graph over all 10^12 vectors needs
   4.7×10^19 FLOPs, about 1.3 PB of reservoir RAM and about 2 PB of shuffle
   (DISTRIBUTED.md §2).
4. **Encode** RaBitQ codes during the same pass, and write postings sorted for
   sequential layout.
5. **Build the router graph** over the 10^8 centroids with PiPNN (GEMM-based; on one
   GPU node, or distributed with the pipeline from DISTRIBUTED.md §3).
6. **Ingest daily** with SPFresh-style updates: append to postings, split oversized
   ones, reassign nearby boundary vectors. No global rebuild. Retrain the hierarchy
   only when the embedding model changes, which needs a full re-embed anyway.

### 4.3 Query execution (throughput)

- **Route the whole job at once** through the router (GPU batch).
- **Invert** to posting → queries. Each storage node streams each needed posting once
  and scores it against all its queries. With RaBitQ codes this is a fast-scan or GEMM
  kernel, and filters are a column mask applied during the scan. This avoids the graph
  connectivity problems Filtered-DiskANN and ACORN must work around.
- **Pick between index and full scan** with the measured crossover. A job reads about
  min(Σ per-query reads, all postings). When probes cover most postings, stream
  everything: exact for the codes, and with perfectly sequential IO. Full scan of
  150 TB at 200 GB/s (20 nodes × 10 GB/s) takes about 12–13 minutes for any number of
  queries.
- **Support range queries** ("all clips with similarity ≥ τ"), which suit postings
  naturally and are often what curation wants instead of a fixed k.
- **Re-rank only the top candidates** with full vectors from the object store, then
  dedup and diversify. Balanced sampling across the posting hierarchy (Vo et al.)
  avoids thousands of near-identical clips. Write manifests, not raw bytes. The
  training loader fetches blobs.
- **Rough capacity.** If scanning about c·k vectors gives target recall (c ≈ 20–40 in
  §3; measure on real data), a k = 10^4 query reads about 20–40 postings (about
  20–40 MB). Twenty NVMe nodes at 200 GB/s then serve about 5–10k such queries per
  second (5–10×10^7 results/s), before batch overlap reduces reads further. A graph
  would need at least 2×10^4 dependent random reads for the same query.

### 4.4 Interactive tier (optional)

For engineers browsing scenarios (small k, low latency), keep a PiPNN graph over a
deduplicated or recent subset (10^10–10^11 segment vectors), served DistributedANN- or
BatANN-style (DISTRIBUTED.md §3–4). It uses the same embeddings and IDs. It is a cache
for exploration, not the system of record.

## 5. Where PiPNN fits

- **Router graph over centroids:** a GEMM-based build, fast, and measured here to route
  as well as brute force.
- **Interactive tier graph:** the distributed PiPNN build (closure partitions, mergeable
  HashPrune) for the 10^10–10^11 hot subset.
- **Near-duplicate detection inside postings:** PiPNN's leaf k-NN (one GEMM per leaf)
  finds near-duplicates for dedup and diversity.
- **Not** a graph over all 10^12 vectors: for large-k batch retrieval it costs more to
  build and serves worse.

## 6. Validation plan

1. Embed a 10^9 sample. Measure c (vectors scanned per result at target recall) for
   k = 10^3–10^5, posting sizes 10^3–10^4, replication 1–4, 1-bit vs 4-bit codes.
2. Replay real curation jobs: measure posting overlap within a job to set the batch
   planner's crossover.
3. Measure the dedup ratio on real drive logs, since it sets N.
4. Benchmark the GPU assignment pass and posting writes at 10^10 to calibrate the
   10^12 build estimate.
5. Compare against Milvus or cuVS IVF-RaBitQ at 10^9 as an off-the-shelf baseline
   before building custom components.

## Sources

- DistributedANN: [MSR PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/07/DistributedANN_Adams_VecDBICML.pdf), [arXiv 2509.06046](https://arxiv.org/abs/2509.06046)
- SPIRE: [arXiv 2512.17264](https://arxiv.org/abs/2512.17264) (abstract and summaries only)
- SPANN: [MSR PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/SPANN_finalversion1.pdf), [arXiv 2111.08566](https://arxiv.org/abs/2111.08566)
- SPFresh: [MSR page](https://www.microsoft.com/en-us/research/?p=962919), [arXiv 2410.14452](https://arxiv.org/abs/2410.14452)
- SOAR: [arXiv 2404.00774](https://arxiv.org/abs/2404.00774), [Google Research blog](https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/)
- RaBitQ extended: [arXiv 2405.12497](https://arxiv.org/abs/2405.12497); IVF-RaBitQ GPU: [PVLDB](https://www.vldb.org/pvldb/vol19/p3454-shi.pdf)
- CAGRA: [arXiv 2308.15136](https://arxiv.org/html/2308.15136v2); cuVS: [docs](https://docs.rapids.ai/api/cuvs/stable/indexes/cagra)
- BatANN: [arXiv 2512.09331](https://arxiv.org/abs/2512.09331)
- Graph partitioning for ANN: [PVLDB 18](https://vldb.org/pvldb/volumes/18/paper/Unleashing%20Graph%20Partitioning%20for%20Large-Scale%20Nearest%20Neighbor%20Search)
- Pyramid: [arXiv 1906.10602](https://arxiv.org/abs/1906.10602); HARMONY: [arXiv 2506.14707](https://arxiv.org/abs/2506.14707)
- ACORN and Filtered-DiskANN benchmark: [arXiv 2509.07789](https://arxiv.org/abs/2509.07789)
- Automatic data curation: [arXiv 2405.15613](https://arxiv.org/abs/2405.15613)
- Driving-RAG: [arXiv 2504.04419](https://arxiv.org/abs/2504.04419)
- DSANN (distributed storage ANN): [arXiv 2510.17326](https://arxiv.org/abs/2510.17326)
- Cosmos-Embed1: [Hugging Face](https://huggingface.co/nvidia/Cosmos-Embed1-448p), [TAO docs](https://docs.nvidia.com/tao/tao-toolkit/latest/text/embedding/cosmos_embed1.html)
- NeMo Curator video dedup: [docs](https://docs.nvidia.com/nemo/curator/v25.09/curate-video/process-data/dedup)
- LidarCLIP: [WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Hess_LidarCLIP_or_How_I_Learned_To_Talk_to_Point_Clouds_WACV_2024_paper.pdf)
- BEV-TSR: [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32782), [arXiv 2401.01065](https://arxiv.org/abs/2401.01065)
- CARIM: [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ki_CARIM_Caption-Based_Autonomous_Driving_Scene_Retrieval_via_Inclusive_Text_Matching_ICCV_2025_paper.html)
- Multimodal scenario similarity search: [arXiv 2607.09428](https://arxiv.org/abs/2607.09428); trajectory embedding retrieval: [SCITEPRESS 2025](https://www.scitepress.org/PublishedPapers/2025/132765); KEPT: [arXiv 2509.02966](https://arxiv.org/pdf/2509.02966)
- Hierarchical segment indexing: [arXiv 2510.09553](https://arxiv.org/html/2510.09553v1); multi-vector index compression: [arXiv 2602.21202](https://arxiv.org/abs/2602.21202)
- WOD-E2E: [arXiv 2510.26125](https://arxiv.org/html/2510.26125v1); Alpamayo-R1: [arXiv 2511.00088](https://arxiv.org/pdf/2511.00088); Mining the Long Tail: [arXiv 2508.18397](https://arxiv.org/html/2508.18397v2)
- CoReVLA: [arXiv 2509.15968](https://arxiv.org/abs/2509.15968); OpenLongTail: [arXiv 2607.09655](https://arxiv.org/html/2607.09655); VLA for driving survey: [arXiv 2512.16760](https://arxiv.org/pdf/2512.16760)
- Tesla training-data patent: [EP3850549B1](https://patents.google.com/patent/EP3850549B1)
- Lance AV data lake: [Volcano Engine](https://www.lancedb.com/blog/volcano-engine-autonomous-driving-data-lake-solution), [WeRide](https://www.lancedb.com/blog/werides-data-platform-transformation-how-lancedb-fuels-model-development-velocity)
