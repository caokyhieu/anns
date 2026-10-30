# How a Lance-based pipeline prepares multimodal AV training data

This is how a training-data retrieval pipeline is built on the open Lance format,
the stack ByteDance's Volcano Engine (LAS) and WeRide report using for driving data.
Each stage names the Lance feature it uses, cites Lance's docs or code
(`lance-format/lance`, commit `47540d0`; tested with `pylance` 14.0.2), and is
exercised end to end by `lance_pipeline_demo.py` on synthetic AV-style data.
RETRIEVAL_DESIGN.md §7 covers what has to be added at 10^12 vectors.

```bash
uv sync --locked --extra lance
uv run --locked --extra lance python lance_pipeline_demo.py   # writes lance-demo/
```

## The idea: one table is the lake, the index and the training set

Lance stores a table as columnar fragments on object storage. Raw sensor payloads,
metadata, captions, embeddings and indexes all live in **one versioned table**, so the
pipeline never copies data between a lake, a vector database and a training store.
A retrieval job produces a list of row IDs, and training reads the blobs behind them
directly.

```
 drive logs ──► 1 ingest (Ray/Spark workers write fragments, one commit)
               segments.lance: ids · time · GPS · weather · tags · caption · trajectory
                               · camera blob · lidar blob · radar/BEV blob
 models ──────► 2 evolve: add emb_video / emb_bev / emb_traj / captions / scores as new
                 columns (batch UDF or merge), no rewrite of blobs
               3 index: IVF_RQ per embedding (shared centroids, per-worker segments)
                 + BTREE(ts) · BITMAP(weather) · LABEL_LIST(tags) · RTREE(GPS) · FTS(caption)
 curation job ► 4 retrieve: prefiltered vector search (large k, nprobes, refine) + FTS
               5 curate: dedup near-identical segments, balance by drive or scenario
               6 publish: manifest table (row IDs, scores), tags on both versions
 trainer ─────► 7 load: lance.torch / take_blobs by row ID → camera, LiDAR, BEV tensors
```

## Stage by stage

### 1. Ingest: distributed fragment writes
Workers (Ray, Spark or Kubernetes jobs) call `lance.fragment.write_fragments` on their
share of the drive logs. One coordinator commits all fragment metadata with
`LanceDataset.commit(uri, LanceOperation.Overwrite|Append(...))`. That is one atomic
version. Lance ships `lance-ray` and `lance-spark` for this (`guide/distributed_write.md`).

- **Row unit:** one row per clip segment (a few seconds), not per frame.
- **Sensor payloads:** stored as **Blob v2** columns (`lance.blob_field`), either inline,
  packed, in dedicated files, or as references to existing objects
  (`external_blob_mode="reference"`), so the existing EB-scale lake need not be copied
  (`guide/blob.md`).
- **Write order:** fragments are written in time order, so time filters prune them.

### 2. Evolve: embeddings and labels as new columns
Embedding models change often. **Data evolution** adds columns without rewriting
existing data files (`guide/data_evolution.md`):
- `add_columns(batch_udf)` runs a model over batches (with optional checkpointing for
  long GPU jobs). Use it to add `emb_video`, `emb_bev`, `emb_traj`, VLM captions, or
  detector tags.
- `merge(table, key)` joins columns computed elsewhere, such as uncertainty from the
  driving model or auto-labels.
- `alter_columns` can cast embeddings to fp16 to halve their size.

This is how one segment ends up with several embeddings (§2b of RETRIEVAL_DESIGN.md)
without duplicating its sensor data.

### 3. Index: vector, scalar and full-text
- **Vector (`format/index/vector/index.md`, `guide/vector_search.md`).** An IVF index
  is k-means partitions, an optional sub-index (FLAT or HNSW), and a quantizer.
  `IVF_RQ` (RaBitQ, 1–5 bits) is Lance's recommended default. The build trains centroids
  on a sample, then makes one streaming pass that assigns, quantizes and writes vectors
  partition by partition (`guide/performance.md`).
- **Distributed vector build (`guide/distributed_indexing.md`).** Train IVF centroids
  once, then have each worker call `create_index_uncommitted(..., ivf_centroids=C,
  fragment_ids=[...])`. Commit the worker outputs with
  `commit_existing_index_segments` (optionally `merge_existing_index_segments` first).
  Shared centroids keep partition IDs meaning the same thing in every segment.
- **Scalar.** `create_scalar_index` with BTREE (time), BITMAP (weather, rig), LABEL_LIST
  (scenario tags), RTREE (GPS), ZONEMAP or BLOOMFILTER. These make prefiltered vector
  search cheap.
- **Full text.** An `INVERTED` (BM25) index on captions, for CARIM-style text matching.

### 4. Retrieve: hybrid queries with large k
`dataset.to_table(nearest={"column", "q", "k", "nprobes", "refine_factor"},
filter="<SQL>", prefilter=True)`:
- `prefilter=True` applies the metadata filter first, so filtered queries still return k rows.
- `nprobes` sets how many partitions are scanned.
- `refine_factor` re-ranks k × factor candidates with full vectors.
- `full_text_query=` (or a combined filter) adds caption search.
- `with_row_id=True` returns row IDs for the manifest.
- Rows appended since the last index update are still searched, by a flat scan.

### 5. Curate: dedup and balance
Lance has no curation operator, so this stage runs in the job (Ray or NumPy):
1. union the candidates across queries;
2. keep each segment once, at its best distance;
3. drop near-duplicates, such as adjacent segments of the same drive;
4. cap or reweight per drive or scenario (Alpamayo-R1, Vo et al.);
5. prioritize by uncertainty (Mining the Long Tail).

Optionally, add VLM scoring of the top candidates (the WOD-E2E funnel).

### 6. Publish: manifest plus versions
Write the result (row ID, segment ID, query, score, weight) as its own Lance table.
Tag the source version and the manifest version (`tags.create`), so the dataset is
reproducible even as new drives arrive or columns change. Tags and branches are
zero-copy (`format/table/branch_tag.md`).

### 7. Load: training reads by row ID
`lance.dataset(uri, version="job-...-source")` pins the snapshot. Then:
- `take` reads metadata and embeddings;
- `take_blobs` / `read_blobs` read camera, LiDAR and BEV bytes;
- `lance.torch.data.LanceDataset` with `ShardedFragmentSampler` or `ShardedBatchSampler`
  handles distributed training reads (`integrations/pytorch.md`).

Random access is what makes manifests of scattered row IDs practical. Lance claims
it is up to 100× faster than Parquet for random reads (vendor claim).

## Demo results (`lance_pipeline_demo.py`, defaults)

80,000 segments from 40 synthetic drives, 64-d video embedding, 4 index workers,
128 shared partitions, 20 seed queries, k = 1,000. The filter is
`weather IN ('rain','night') AND array_has_any(tags, ['pedestrian','cyclist'])`, and
recall is measured against exact filtered search.

| Stage | Result |
| --- | --- |
| ingest | 80,000 rows, 40 fragments, one commit, 3.4 s |
| evolve | `emb_traj` added by UDF, `uncertainty` merged, blobs not rewritten |
| index | 4 IVF_RQ segments with shared centroids (1.1 s) committed as one index, plus BTREE, BITMAP, LABEL_LIST and FTS indexes |
| retrieve, nprobes 8 | recall@1000 = 0.483, 15 ms/query |
| retrieve, nprobes 32 | 0.975, 15 ms/query |
| retrieve, nprobes 32, refine 4 | 0.980, 35 ms/query |
| retrieve, nprobes 64, refine 4 | 0.9999, 44 ms/query |
| FTS "construction fog" | caption matches returned |
| curate | 7,072 unique candidates → 2,000 kept (dedup, at most 50 per drive, by uncertainty) |
| publish | manifest table, source tagged `job-0001-source` |
| load | camera and LiDAR blobs for 32 manifest rows read by row ID |

Synthetic data and one machine, so the numbers show that the pipeline works, not
production performance. The `list_indices` output shows the vector index as four
segments under one name: Lance searches them independently and merges by distance.
That is the fan-out issue in RETRIEVAL_DESIGN.md §7.2.

## What production deployments report (vendor sources, not independently verified)

- **Volcano Engine LAS:** Lance is the core format of its AI data lake for text, image,
  audio and video. Reported for its AV annotation and training lake: up to 70% less
  storage, GPU utilization from 60% to 96%, and 10 PB of label processing from 4 days
  to 1. Distributed IVF-PQ training across fragments cut a 7-day pipeline to 1 day. A
  hybrid-search deployment serves 100K+ QPS.
- **WeRide:** long-tail scenario mining went from about 1 week to 1 hour after moving
  to LanceDB.
- **LanceDB:** claims flat latency at 10B+ vectors (HNSW over centroids, RaBitQ,
  segment-level routing), without published recall numbers.

## What to add for 10^12 vectors and training-size k

These come from RETRIEVAL_DESIGN.md §7.2.
1. **Route once, read everywhere.** With shared centroids, route each query once and
   read partition p from every segment. Do not let each segment re-rank its own
   partitions.
2. **A router for many partitions.** A PiPNN graph over the centroids, if you go
   beyond about 10^6 partitions.
3. **A batch engine.** Plan a whole job, read each `(segment, partition)` range once,
   score it against all its queries on GPU, and fall back to a full scan above the
   crossover.
4. **Optional spilled assignment** (SOAR or closure), and **SPFresh-style rebalancing**
   of partitions as new drives arrive.

## Sources

- Lance: [repository](https://github.com/lance-format/lance) (docs under `docs/src/`:
  `guide/distributed_write.md`, `guide/data_evolution.md`, `guide/blob.md`,
  `format/index/vector/index.md`, `guide/vector_search.md`, `guide/distributed_indexing.md`,
  `guide/performance.md`, `integrations/pytorch.md`)
- Volcano Engine LAS: [PB-scale AV data lake](https://www.lancedb.com/blog/volcano-engine-autonomous-driving-data-lake-solution),
  [AI stack on Lance](https://www.lancedb.com/blog/volcano-engine-lance-agent-memory)
- WeRide: [case study](https://www.lancedb.com/blog/werides-data-platform-transformation-how-lancedb-fuels-model-development-velocity)
- LanceDB at 10B: [blog](https://www.lancedb.com/blog/how-lancedb-accelerates-vector-search-at-10-billion-scale)
