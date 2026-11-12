# Lance Indexing and Cache Architecture Research

**Research date:** 11 November 2026\
**Revision:** v2, 11 November 2026. Re-verified against `lance-format/lance`
`main` @ `04e619963`; corrections are listed in
[Revision notes (v2)](#revision-notes-v2) and the proposals are split into
two research tracks for ADS targets ④ and ⑥ (section 11).\
**Scope:** The Lance Rust engine and its indexing, storage, query, and
cache implementations. This report deliberately excludes LanceDB APIs
and database-layer behavior.\
**Primary repository:**
[lance-format/lance](https://github.com/lance-format/lance)

> **Evidence labels**
>
> -   **Verified** --- directly supported by current Lance source or
>     official Lance documentation.
> -   **Research finding** --- a result or design reported by an
>     external paper; performance figures are author-reported and
>     workload-specific.
> -   **Proposal** --- a possible extension for Lance, not a description
>     of existing behavior.

------------------------------------------------------------------------

## Executive summary

Lance's current documented vector-index architecture is **IVF-first**.
It divides a vector index into three cooperating parts:

1.  **Clustering:** IVF partitions vectors using k-means centroids.
2.  **Sub-index:** each partition uses either a flat scan or an HNSW
    graph.
3.  **Quantization:** partition vectors are stored as full precision,
    PQ, SQ, or RaBitQ representations.

This distinction matters: in the current documented architecture,
`IVF_HNSW_*` means an HNSW sub-index inside IVF partitions. It is not a
standalone global HNSW index.

The current documented physical vector-index format is **V3**. It stores
the search structure in `index.idx` and quantized vector storage in
`auxiliary.idx`. Both are regular Lance files. The V3 row-reference
representation uses physical row addresses (`_rowaddr`) for the relevant
storage formats.

Lance already has a substantial Rust cache implementation:

-   Session-level index and metadata caches.
-   A typed `LanceCache` / `CacheBackend` abstraction.
-   A default byte-weighted `QuickCacheBackend` for index data, built on
    the `quick_cache` crate, whose eviction policy is **S3-FIFO**.
-   An optional built-in Moka backend (**TinyLFU** admission) and
    custom-backend registration.
-   An `OriginLatencyClass` (low / high) that already adapts RaBitQ
    reader policies to the latency of the index origin.
-   Dataset-, index-, and fragment-reuse-aware key namespaces.
-   IVF partition-entry caching and specialized RaBitQ code-only / plane
    caching paths.
-   Cache codecs, priority-aware admission, explicit prewarming, and
    detailed diagnostics.

The largest opportunities are therefore not "add a cache" or "add IVF
partition caching"; those already exist. More promising directions are:

1.  **Cost-aware admission for IVF partitions** --- decide whether a
    miss should be admitted, based on reuse probability, partition size,
    and remote load cost. Frequency-based admission alone is *not* new:
    the default backend already uses S3-FIFO and the Moka backend
    TinyLFU. The new ingredients are load cost and, for filtered
    queries, how many rows of the partition can pass the filter.
2.  **A real local-NVMe tier** --- the backend interface has extension
    points for local/persistent storage, but I found no built-in
    persistent disk backend in the core registry.
3.  **Fine-grained HNSW graph caching** --- separate graph adjacency
    from vector payload and consider record/page-level caching for
    `IVF_HNSW_*`.
4.  **Optional similarity-aware query-result caching** --- QVCache and
    Aker are recent research directions, but they introduce
    approximate-result semantics and need explicit recall and freshness
    contracts.
5.  **Query-aware scheduling** --- group or schedule concurrent queries
    that are likely to touch overlapping IVF partitions.

For the ADS pipeline these directions are now split into **two separate
research tracks** (section 11):

> **Track ④ --- vector + tag hybrid retrieval:** predicate-result
> caching, filter-aware partition skipping, and batched mining queries
> over `IVF_RQ`.
>
> **Track ⑥ --- multimodal data cache:** a tiered byte cache (RAM → UB
> pooled memory → local NVMe → OBS) under the Lance reader, prefetch
> driven by the training sampler, and cost-aware admission and
> placement for objects, vectors, tags and index partitions.

Both build on existing Lance primitives and neither requires changing
the on-disk vector-index format.

------------------------------------------------------------------------

## 1. Scope and source-verification method

### 1.1 What is in scope

This report covers:

-   `rust/lance`: dataset-level index build, open, and query execution.
-   `rust/lance-index`: vector index abstractions, IVF, HNSW,
    quantizers, and vector storage.
-   `rust/lance-core`: typed cache abstractions and cache backends.
-   Lance's official vector-index format and performance documentation.

It does **not** use LanceDB's Python connection, table, or database APIs
as evidence for Lance's Rust internals.

### 1.2 Primary source set

The principal source files inspected are:

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Area                                Source
  ----------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------
  Vector index construction           [`rust/lance/src/index/vector/builder.rs`](https://github.com/lance-format/lance/blob/main/rust/lance/src/index/vector/builder.rs)

  IVF runtime implementation          [`rust/lance/src/index/vector/ivf/v2.rs`](https://github.com/lance-format/lance/blob/main/rust/lance/src/index/vector/ivf/v2.rs)

  IVF partition serialization         [`rust/lance/src/index/vector/ivf/partition_serde.rs`](https://github.com/lance-format/lance/blob/main/rust/lance/src/index/vector/ivf/partition_serde.rs)

  IVF sub-index interface             [`rust/lance-index/src/vector/v3/subindex.rs`](https://github.com/lance-format/lance/blob/main/rust/lance-index/src/vector/v3/subindex.rs)

  IVF shuffle and partition layout    [`rust/lance-index/src/vector/v3/shuffler.rs`](https://github.com/lance-format/lance/blob/main/rust/lance-index/src/vector/v3/shuffler.rs)

  Vector storage and RaBitQ planes    [`rust/lance-index/src/vector/storage.rs`](https://github.com/lance-format/lance/blob/main/rust/lance-index/src/vector/storage.rs)

  Session cache configuration         [`rust/lance/src/session.rs`](https://github.com/lance-format/lance/blob/main/rust/lance/src/session.rs)

  Index cache namespaces              [`rust/lance/src/session/index_caches.rs`](https://github.com/lance-format/lance/blob/main/rust/lance/src/session/index_caches.rs)

  Cache backend contract              [`rust/lance-core/src/cache/backend.rs`](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/backend.rs)

  Cache facade and typed keys         [`rust/lance-core/src/cache/mod.rs`](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/mod.rs)

  Quick cache implementation          [`rust/lance-core/src/cache/quick.rs`](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/quick.rs)

  Priority-aware entries              [`rust/lance-core/src/cache/priority.rs`](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/priority.rs)

  Cache codecs                        [`rust/lance-core/src/cache/codec.rs`](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/codec.rs)

  Cache metrics                       [`rust/lance/src/metrics.md`](https://github.com/lance-format/lance/blob/main/rust/lance/src/metrics.md)
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Official documentation:

-   [Vector Search guide](https://lance.org/guide/vector_search/)
-   [Vector Index Format --- V3](https://lance.org/format/index/vector/)
-   [Performance Guide](https://lance.org/guide/performance/)

### 1.3 Important version-naming clarification

The runtime implementation file is named
`rust/lance/src/index/vector/ivf/v2.rs`, while the current documented
physical vector-index storage layout is V3.

These names refer to different things. **Do not infer the on-disk index
format version from the runtime module filename.**

------------------------------------------------------------------------

## 2. Lance indexing architecture

### 2.1 Indexes are secondary structures associated with a Lance dataset

**Verified:** Lance's index system is a secondary-index system. The
index metadata is associated with a dataset version, while the index
data is stored in regular Lance files.

At a high level:

``` mermaid
flowchart TD
    A["Lance dataset"] --> B["Dataset manifest / version"]
    B --> C["Index metadata"]
    C --> D["Index identity and file locations"]
    D --> E["Vector index files"]
    D --> F["Scalar / other index files"]
    E --> G["Vector search execution"]
    F --> H["Filtering / other query operators"]
    G --> I["Top-k candidates"]
    H --> I
    I --> J["Rows returned from Lance data"]
```

This means that the vector index is not the canonical table data. It is
an auxiliary search structure that points back to rows in the dataset.

This distinction matters for caching and correctness:

-   Index entries must be scoped to the correct dataset and index
    identity.
-   A cached partition must not be reused for a different index
    generation merely because its partition number is the same.
-   Query-result caches must account for dataset snapshots, deletions,
    index coverage, and search parameters.

### 2.2 Vector index composition: IVF + sub-index + quantization

**Verified:** The current vector-index format documentation describes
three components.

  -----------------------------------------------------------------------
  Component               Role                    Current options
  ----------------------- ----------------------- -----------------------
  Clustering              Narrows the search to   IVF
                          relevant partitions     

  Sub-index               Organizes candidates    FLAT or HNSW
                          within a partition      

  Quantization            Determines how vector   FLAT, PQ, SQ, RaBitQ
                          data is represented     
  -----------------------------------------------------------------------

#### IVF clustering

IVF (Inverted File) uses k-means centroids to divide the dataset into
disjoint partitions.

At query time:

1.  Compute the query's distance to the IVF centroids.
2.  Select the most relevant partitions.
3.  Search only those partitions.

The number of partitions and number of probes trade work against recall.
More partitions reduce the work per partition but generally require more
probes to reach similar recall.

#### Sub-index

-   **FLAT:** scans the vectors in the selected partition.
-   **HNSW:** traverses a hierarchical proximity graph inside the
    selected partition.

#### Quantization

-   **FLAT:** retains original full-precision vectors.
-   **PQ:** splits vectors into sub-vectors and encodes each using
    product quantization.
-   **SQ:** quantizes each vector dimension independently.
-   **RaBitQ / RQ:** uses random rotation and binary quantization, with
    correction information used for distance estimation.

Common documented index combinations:

  -----------------------------------------------------------------------
  Index type              Structure               Main trade-off
  ----------------------- ----------------------- -----------------------
  `IVF_FLAT`              IVF + flat scan +       Exact scoring inside
                          original vectors        probed partitions, but
                                                  larger candidate I/O

  `IVF_PQ`                IVF + PQ                Compact and efficient,
                                                  but quantization can
                                                  reduce recall

  `IVF_SQ`                IVF + scalar            Moderate compression
                          quantization            

  `IVF_RQ`                IVF + RaBitQ            Compact storage and
                                                  efficient approximate
                                                  distance estimates

  `IVF_HNSW_FLAT`         IVF + HNSW + original   Graph search with exact
                          vectors                 candidate distances

  `IVF_HNSW_PQ`           IVF + HNSW + PQ         Graph search with
                                                  compressed candidate
                                                  storage

  `IVF_HNSW_SQ`           IVF + HNSW + SQ         Graph search with
                                                  scalar-quantized
                                                  storage
  -----------------------------------------------------------------------

The current official guide recommends starting with `IVF_RQ` for many
workloads, but stresses benchmarking recall and latency on
representative data.

**Important exactness distinction:**

-   Searching without a usable vector index is exhaustive over the
    search scope and is exact.
-   `IVF_FLAT` computes exact distances for vectors in the selected IVF
    partitions, but it is not globally exact unless the search examines
    every relevant partition.

Sources: [Vector index format](https://lance.org/format/index/vector/) ·
[Vector search guide](https://lance.org/guide/vector_search/)

### 2.3 Vector-index composition diagram

``` mermaid
flowchart TB
    Q["Query vector"] --> C["IVF centroid routing"]
    C --> P0["Selected partition 0"]
    C --> P1["Selected partition 1"]
    C --> PN["Selected partition N"]

    subgraph P0S["Partition 0"]
      P0 --> S0["FLAT or HNSW sub-index"]
      S0 --> V0["FLAT / PQ / SQ / RQ storage"]
    end

    subgraph P1S["Partition 1"]
      P1 --> S1["FLAT or HNSW sub-index"]
      S1 --> V1["FLAT / PQ / SQ / RQ storage"]
    end

    subgraph PNS["Partition N"]
      PN --> SN["FLAT or HNSW sub-index"]
      SN --> VN["FLAT / PQ / SQ / RQ storage"]
    end

    V0 --> M["Merge candidates / global top-k"]
    V1 --> M
    VN --> M
    M --> R["Optional refinement and row fetch"]
```

The HNSW graph is therefore a **per-partition sub-index** in the current
documented IVF architecture.

### 2.4 V3 physical storage layout

**Verified:** The current vector-index format specification describes
two regular Lance files per vector index:

``` text
_indices/
  <index-uuid>/
    index.idx
    auxiliary.idx
```

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `index.idx`                         Search structure: IVF metadata and
                                      the sub-index structure, including
                                      HNSW graph data where applicable

  `auxiliary.idx`                     Vector storage: full-precision or
                                      quantized vector data and
                                      associated row references /
                                      correction data
  -----------------------------------------------------------------------

The index file stores all partitions in one file, in partition order.
The auxiliary file also stores partitions in one file, in partition
order.

#### What is in `index.idx`?

-   Sub-index-specific Arrow data.
-   HNSW graph data for HNSW variants.
-   Per-partition HNSW metadata, such as entry points and level offsets.
-   IVF metadata in a Lance global buffer, including centroids,
    partition offsets, and partition lengths.

For HNSW, the documented structure includes vector identifiers, neighbor
lists, and distances.

#### What is in `auxiliary.idx`?

The schema depends on the quantizer:

  Quantizer   Representative storage columns
  ----------- ----------------------------------------------------
  FLAT        Original vector values
  PQ          `__pq_code`
  SQ          `__sq_code`
  RQ          Packed RaBitQ codes and correction / error factors

The V3 storage specification uses `_rowaddr` for the relevant
vector-storage row references. Older index formats may use `_rowid`.

A row address is a physical address in the dataset's fragment/row
layout. It must not be treated as a stable logical identity across
compaction or index replacement.

#### Storage diagram

``` mermaid
flowchart LR
    I["Index directory"] --> A["index.idx"]
    I --> B["auxiliary.idx"]

    A --> A1["IVF centroids"]
    A --> A2["Partition offsets / lengths"]
    A --> A3["FLAT marker or HNSW graph"]

    B --> B1["Row addresses"]
    B --> B2["Original vectors or quantized codes"]
    B --> B3["Quantizer correction data"]

    A3 -. "partition-local vector IDs" .-> B2
    B1 --> D["Lance dataset rows"]
```

Source: [Lance Vector Index Format --- Storage Layout
V3](https://lance.org/format/index/vector/)

### 2.5 Vector-index build pipeline

**Verified:** The current `IvfIndexBuilder` implementation builds the
vector index in three main stages:

1.  Train the IVF model and quantizer.
2.  Shuffle the dataset into IVF partitions.
3.  Build and merge the partition sub-indexes and vector storage.

More detail:

#### Stage 1 --- Train IVF and quantizer

-   Train or load the IVF centroids.
-   Sample training vectors for the quantizer.
-   If the quantizer uses residuals, compute residual vectors relative
    to the IVF centroids before quantizer training.
-   Normalize cosine training vectors as required by the implementation.

#### Stage 2 --- Transform and shuffle

-   Assign vectors to IVF partitions.
-   Apply the selected vector transform and quantizer.
-   Shuffle rows by partition ID.
-   Preserve row references so the index can map search results back to
    the dataset.

The current V3 shuffler includes a `TwoFileShuffler` that writes
shuffled data and partition offsets rather than opening one file per
partition. This avoids file-descriptor pressure for large partition
counts.

This is **build-time scratch data**, not the final two-file index
format.

#### Stage 3 --- Build each partition and write the index

-   Read each shuffled partition.
-   Build the selected sub-index (FLAT or HNSW).
-   Build the quantized vector storage.
-   Merge/write partitions in order.
-   Write `index.idx` and `auxiliary.idx`.
-   Commit the resulting index metadata to the dataset.

``` mermaid
flowchart TD
    A["Dataset vector column + row references"] --> B["Sample training vectors"]
    B --> C["Train IVF k-means centroids"]
    B --> D["Train quantizer"]
    C --> E["Assign vectors to IVF partitions"]
    D --> E
    E --> F["Transform / quantize vectors"]
    F --> G["Shuffle rows by partition"]
    G --> H["Read one partition at a time"]
    H --> I["Build FLAT or HNSW sub-index"]
    H --> J["Build vector storage"]
    I --> K["Write index.idx"]
    J --> L["Write auxiliary.idx"]
    K --> M["Commit index metadata"]
    L --> M
```

The builder also supports merging existing indexes, incremental
optimization, partition splitting/joining, and retraining. These are
important because index data can evolve independently of the underlying
dataset's append operations.

Source: [Vector index
builder](https://github.com/lance-format/lance/blob/main/rust/lance/src/index/vector/builder.rs)

### 2.6 Query execution pipeline

A vector search is not simply "load an index and run HNSW."

The main stages are:

1.  Open the dataset snapshot and index metadata.
2.  Open one or more vector-index segments.
3.  Route the query to IVF partitions.
4.  Prepare each selected partition, including loading its sub-index and
    storage.
5.  Apply prefiltering where configured.
6.  Search each partition using FLAT or HNSW and the selected quantizer.
7.  Merge candidates into global top-k.
8.  Optionally refine candidates using more precise/original vector
    data.
9.  Resolve result row references and fetch requested columns.

``` mermaid
flowchart TD
    Q["Query vector + k + search options"] --> D["Dataset snapshot / index metadata"]
    D --> S["One or more index segments"]
    S --> R["IVF centroid routing"]
    R --> P["Select partitions by distance / probe budget"]

    P --> L["Load partition entry"]
    L --> C{"Cache hit?"}
    C -->|Yes| E["Reuse cached partition"]
    C -->|No| F["Read index.idx and auxiliary.idx"]
    F --> E

    E --> PF["Prepare prefilter / row mapping"]
    PF --> SI["FLAT scan or HNSW traversal"]
    SI --> QT["Quantized distance calculation"]
    QT --> TOP["Merge global top-k"]
    TOP --> REF["Optional refinement"]
    REF --> ROW["Resolve row references / fetch columns"]
    ROW --> OUT["Results"]
```

#### Query parameters and recall

The public guide documents these controls:

-   `nprobes`: explicit IVF partition probe count.
-   `search_effort`: a calibrated partition-routing target.
-   `minimum_nprobes` and `maximum_nprobes`: bounds on probing.
-   `refine_factor`: over-fetch/refine trade-off.
-   HNSW build/search controls for HNSW variants.

The current guide reports a default `search_effort` of `0.5`, associated
with a mean routing-recall target of 95% on its calibration setup for
supported metrics and `k` ranges. This is **not a guarantee of final
Recall@k**. Quantization, filters, refinement, HNSW search, and dataset
distribution can all change final recall.

#### Filtering

Lance supports prefilter and postfilter behavior. A prefilter can reduce
the candidate set before vector search, while postfiltering may search
first and filter afterward. Scalar indexes can accelerate relevant
filters.

Deletions and explicit fragment scopes constrain candidates regardless
of the prefilter setting.

#### Multiple index segments and unindexed rows

Lance's index maintenance documentation makes two important points:

-   Appending rows does not automatically add them to existing vector
    indexes.
-   By default, Lance combines indexed results with exact search over
    unindexed data, so appended rows remain visible.

`optimize_indices` can incrementally add new data using existing IVF and
quantizer models. Retraining is available when the data distribution has
changed enough that existing centroids/codebooks no longer represent it
well.

This matters for cache correctness: a query-result cache must account
for the dataset version, index generation/segments, unindexed rows, and
deletion state.

Sources: [Vector Search guide](https://lance.org/guide/vector_search/) ·
[Index
maintenance](https://lance.org/guide/vector_search/#index-maintenance)

------------------------------------------------------------------------

## 3. Existing Lance cache architecture

### 3.1 Two session cache families

**Verified:** A Lance `Session` maintains two logical cache families:

  ------------------------------------------------------------------------
  Cache                 Contents                              Default size
  --------------------- --------------------- ----------------------------
  Index cache           Opened vector/scalar                         6 GiB
                        index data and index  
                        payloads              

  Metadata cache        Manifests,                                   1 GiB
                        transactions,         
                        deletion files,       
                        row-id structures,    
                        index metadata, and   
                        file metadata         
  ------------------------------------------------------------------------

The Rust `Session::new` implementation builds both using
`QuickCacheBackend`. The index cache uses a single-shard policy because
it commonly contains a small number of large, unequal partition entries.
The metadata cache uses the recommended shard policy.

The performance guide describes the caches as byte-sized LRU caches.
**Correction (v2):** neither built-in backend is plain LRU.
`QuickCacheBackend` wraps `quick_cache` 0.6, which uses **S3-FIFO**
(scan-resistant, with a small probationary queue), and
`MokaCacheBackend` wraps `moka` 0.12, which uses **TinyLFU** admission.
Both weigh entries by accounted bytes through the same weigher. The
current Rust source also has a priority-aware path for entries whose
codecs declare a nonzero memory priority.

**Capacity is not an RSS limit.** Active queries, loaders, temporary
batches, shared allocations, and other live objects can retain memory
outside the cache budget.

A nuance about table sharing:

-   Dataset URI prefixes prevent index entries for different datasets
    from colliding or being reused as if they were the same index.
-   If several datasets share one `Session`, the session's index-cache
    capacity is still a shared budget across those indices.
-   The public performance guide's statement that index contents are not
    shared between tables should not be read as a guarantee of separate
    physical cache budgets per dataset.

Sources: [Session
source](https://github.com/lance-format/lance/blob/main/rust/lance/src/session.rs)
· [Performance guide](https://lance.org/guide/performance/)

### 3.2 Index-cache key hierarchy

The current index-cache namespace is hierarchical:

``` mermaid
flowchart TD
    G["GlobalIndexCache"] --> D["Dataset cache"]
    D -->|Dataset URI prefix| I["Index-specific cache"]
    I -->|Index UUID prefix| F["Optional fragment-reuse UUID prefix"]
    F --> K["Typed cache keys"]
    K --> P["IVF partition / row-address / other index entries"]
```

This is important for correctness:

-   Dataset URI isolates datasets.
-   Index UUID isolates index generations.
-   Fragment-reuse identity can further isolate row-address translation
    contexts.
-   Individual typed keys identify partition IDs and other cached
    values.

The IVF partition key is not globally identified by partition number
alone. It is used inside an index-specific cache namespace.

Sources: [Index cache
namespaces](https://github.com/lance-format/lance/blob/main/rust/lance/src/session/index_caches.rs)
· [IVF
runtime](https://github.com/lance-format/lance/blob/main/rust/lance/src/index/vector/ivf/v2.rs)

### 3.3 IVF partition cache

**Verified:** Lance has a typed `IVFPartitionKey<S, Q>` for IVF
partition entries.

Its identity includes:

-   The IVF partition ID.
-   The sub-index type.
-   The quantization type.

Its value is a `PartitionEntry<S, Q>`, which contains the partition's
sub-index and quantized storage.

For the normal whole-partition path, a cache miss loads the partition's
sub-index batch and vector storage. A successful load can be inserted
into the index cache.

The partition cache therefore already covers the common IVF partition
materialization problem.

### 3.4 Specialized RaBitQ caching paths

The current implementation has more than one RaBitQ path. It is
important not to assume that every RaBitQ index uses the same cache
representation.

#### Code-only partition entries

For a native RaBitQ index with a flat sub-index and suitable
resident-column configuration, Lance can cache partition codes
separately.

On each read:

-   Reuse the cached code-only entry if present.
-   Attach the resident columns needed to reconstruct the partition's
    storage.
-   Build the read-specific storage object.
-   Avoid caching a reconstructed partition object that would keep a
    larger resident store alive through shared views.

#### Plane caching

For supported native multi-bit packed RaBitQ formats, a scan strategy
can use separate sign and ex planes.

The implementation includes:

-   Separate plane cache keys.
-   Sign-plane dependency checks.
-   Plane admission gating for backends that support it.
-   Priority-aware admission so important sign data is not crowded out
    by lower-priority plane data.

This is already a specialized, index-aware caching design.

#### Resident columns

The storage implementation also supports resident columns for certain
RaBitQ paths. The plane/code-only entries can be combined with resident
data when a query needs to construct the partition's search
representation.

#### File-scoped raw cache

The IVF runtime also creates a `raw_cache` keyed by the object-store
prefix and auxiliary-file path. The code comments describe it as holding
immutable planes and code-only entries independently of fragment-reuse
translation.

This is distinct from the normal logical index/partition cache.

#### Origin latency class (added in v2)

**Verified:** `OriginLatencyClass` (`lance-index/src/vector/storage.rs`)
records whether index origin reads are slow (`High`, the default for a
cloud object store) or fast (`Low`, local files and memory). It is
resolved when an index opens, from `LANCE_RQ_ORIGIN_LATENCY`, from
`Session::with_index_origin_latency`, or from the object store type.
Reader policies that trade extra bytes or background work for fewer
origin requests default to on only for `High`. The source comment says
that a process whose object store is wrapped in a local cache should
declare `Low` on its session.

This is the closest existing mechanism to cost-aware caching, but it is
a per-index reader-policy switch for RaBitQ, not an admission or
eviction input. A cost-aware policy should extend it rather than add a
parallel concept.

Sources: [IVF
runtime](https://github.com/lance-format/lance/blob/main/rust/lance/src/index/vector/ivf/v2.rs)
· [Vector
storage](https://github.com/lance-format/lance/blob/main/rust/lance-index/src/vector/storage.rs)
· [RaBitQ
prewarming](https://github.com/lance-format/lance/blob/main/rust/lance-index/src/vector/storage/prewarm.rs)

### 3.5 Prewarming already exists

Lance has explicit prewarming paths and cache activity attribution for
warm loads.

The current RaBitQ plane prewarm implementation:

-   Warms sign and ex planes.
-   Bounds concurrent work.
-   Uses an estimated 64 MiB value-staging target.
-   Respects plane-admission dependencies for gated backends.

The IVF implementation also has bounded partition-window prewarming and
byte-based staging controls.

Therefore, "add prewarming" is not a sufficiently novel proposal. A
better question is whether Lance can make prewarming more
**query-aware**, workload-aware, and selective.

### 3.6 Cache backend abstraction

The cache system is built around:

-   `LanceCache`: typed cache facade.
-   `CacheKey`: typed key and value mapping.
-   `CacheBackend`: pluggable storage and eviction interface.
-   `CacheCodec`: optional serialization support.
-   `CacheLoadOrigin`: distinguishes demand loads from explicit warm
    loads.
-   Cache diagnostics and per-type activity.

#### Current built-in backend picture

  -----------------------------------------------------------------------
  Backend                             Role
  ----------------------------------- -----------------------------------
  `QuickCacheBackend`                 Default Rust Session backend for
                                      index and metadata caches;
                                      S3-FIFO eviction (`quick_cache`)

  `MokaCacheBackend`                  Built-in alternative backend;
                                      TinyLFU admission (`moka`)

  Custom `CacheBackend`               Extension point for
                                      application-specific storage
                                      policies
  -----------------------------------------------------------------------

I found no built-in local-NVMe or persistent disk cache backend in the
current `lance-core` backend registry. The registry automatically
installs Moka; the default Session constructor directly creates Quick
cache instances.

This is an important gap, but it is not an absence of an extension
interface.

#### Persistent-tier extension points already in the trait

The backend interface contains concepts that could support a tiered
backend:

-   `CacheTier::Resident`
-   `CacheTier::Local`
-   `CacheTier::Absent`
-   `peek_tier`
-   `get_without_promotion`
-   `get_rows`
-   `CacheCodec` serialization
-   `CacheRangeReader` for bounded range reads

A custom backend could implement a local persistent tier and use codecs
to serialize cache entries.

However, not every cache entry type necessarily has a codec. The backend
contract says that an entry with no codec must remain in memory rather
than be persisted.

The current cache codec implementation also labels the serialized entry
format as experimental. Decode failures are intentionally converted into
normal cache misses so the entry can be regenerated.

**Implication:** A persistent cache is technically plausible, but it
needs a codec-coverage audit, versioning policy, and recovery behavior
before being treated as a production-safe cross-version cache.

Sources: [Cache backend
trait](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/backend.rs)
· [Cache
facade](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/mod.rs)
· [Cache
codecs](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/codec.rs)
· [Backend
registry](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/registry.rs)

### 3.7 Current cache topology

``` mermaid
flowchart TB
    Q["Vector query"] --> S["Lance Session"]

    S --> IC["Index cache: QuickCacheBackend by default"]
    S --> MC["Metadata cache: QuickCacheBackend by default"]

    IC --> DS["Dataset namespace"]
    DS --> IX["Index UUID / optional FRI namespace"]
    IX --> IVF["IVFPartitionKey"]
    IVF --> PE["PartitionEntry: sub-index + vector storage"]

    IX --> RAW["File-scoped raw cache"]
    RAW --> RQ["RaBitQ plane / code-only entries"]

    MC --> MM["Manifest / index metadata / file metadata / row-id metadata"]

    PE --> SEARCH["FLAT or HNSW partition search"]
    RQ --> SEARCH
    MM --> SEARCH
```

The diagram is conceptual: the file-scoped raw cache can use a
file-cache handle when supplied, otherwise it falls back to the index
cache with a physical-file namespace.

### 3.8 What is not currently a separate cache layer?

The following distinctions are useful:

  -----------------------------------------------------------------------
  Potential cache                     Current status in the code examined
  ----------------------------------- -----------------------------------
  Session index cache                 Exists

  Session metadata cache              Exists

  IVF partition-entry cache           Exists

  RaBitQ code-only / plane cache      Exists for supported paths

  Explicit prewarm                    Exists

  Independent local NVMe persistent   No built-in implementation found
  tier                                

  General semantic query-result cache No such core cache layer found

  Independent HNSW adjacency-page     No separate general page-cache
  cache                               abstraction found in the IVF HNSW
                                      path examined; graph data is loaded
                                      through partition sub-index
                                      structures

  Frequency-aware admission           Exists: S3-FIFO (Quick, default)
                                      and TinyLFU (Moka)

  Cost-aware admission for IVF        Not found; `OriginLatencyClass`
  partitions                          only switches RaBitQ reader policies

  Deletion / fragment mask            Exists: `RowAddrMaskKey` in the
                                      metadata cache, per dataset version
                                      and manifest e-tag

  Scalar-index pages (BTree, bitmap,  Exists: per-page cache keys
  n-gram, ...)

  Filter allow-list (evaluated        Not cached: rebuilt per query by the
  predicate result)                   filter loader

  Column / blob data bytes            Not cached: `PageDataCacheKey` holds
                                      page metadata only; `BlobFile` has a
                                      per-handle 512 KiB read-ahead buffer

  Build-time spill store              Exists (`LocalSpillStore`), but it
                                      is scratch space, not a cache
  -----------------------------------------------------------------------

### 3.9 Filtered (prefiltered) IVF search path (added in v2)

**Verified** in `rust/lance/src/index/vector/ivf/v2.rs` and
`rust/lance/src/index/prefilter.rs`:

1.  `DatasetPreFilter` combines a deletion mask (cached, see 3.8) with
    `filtered_ids`, the allow-list produced by the filter loader. The
    allow-list is computed for each query; only the scalar-index pages
    it reads are cached.
2.  `prepare_partition` loads the partition and waits for the prefilter
    **concurrently** (`tokio::try_join!`). A probed partition is
    therefore read and, on the normal path, admitted to the index cache
    before the engine knows whether any of its rows pass the filter.
3.  `selects_nothing()` is true only when the **whole** mask is an empty
    allow-list. There is no per-partition "no qualifying row" test
    before loading; rows are filtered during the partition scan.
4.  Partitions are chosen by centroid distance only, so `nprobes` (or
    `search_effort`) is spent on partitions that may contain no
    qualifying rows.

These are the integration points for track ④ (section 11).

------------------------------------------------------------------------

## 4. Diagnostics and a source-level caveat

### 4.1 Use session diagnostics before designing a new policy

The performance guide documents:

-   `Session::cache_diagnostics()`
-   Per-type cache activity and occupancy.
-   Cache hits and misses.
-   Backend occupancy and capacity.
-   Warm-load activity.
-   Loader duration and size-removal metrics.

Lance also exposes OpenTelemetry metrics for cache lookups, loads, warm
loads, capacity, occupancy, and removals.

The diagnostics are useful, but byte counts are **policy weights**, not
process RSS or guaranteed freed heap bytes. Shared allocations may be
counted through more than one entry.

A full cache is not automatically a problem. A full cache with strong
warm-hit rates may be healthy. Repeated misses, loader time, and
size-removal churn are more useful together.

### 4.2 `write_cache=false` metrics/documentation discrepancy

This is a concrete issue found during the source audit.

The execution-summary documentation (doc comments on
`index_cache_hits` and `index_cache_hit_ratio` in
`rust/lance-datafusion-exec/src/exec.rs`, not `metrics.md`) says that IVF
v2 streaming scans using `write_cache=false` always execute the loader
and never write the result back, so they are counted as misses.

However, the current `IVFIndex::load_partition_attached` source does
this in the `write_cache == false` branch:

1.  Call `index_cache.get_with_key(&cache_key)`.
2.  If an entry exists, record a hit and return it.
3.  If no entry exists, record a miss, load the partition, and do not
    insert it.

That is a **read-only cache lookup** on a hit, not a strict cache
bypass.

The code and execution-summary documentation therefore appear
inconsistent in their description of this path. The exact query call
path should be tested, and the metrics documentation should be
reconciled with the implementation.

The same `exec.rs` doc comment also states that HNSW graph pages and
quantizer codebooks are not yet instrumented. (`metrics.md` documents the
OpenTelemetry cache metrics only.)

**Recommendation:** Use session-level cache diagnostics and loader/I/O
measurements as the primary evidence. Treat the execution-summary hit
ratio as a partial view until the call paths and documentation are
reconciled.

Sources:
[`v2.rs`](https://github.com/lance-format/lance/blob/main/rust/lance/src/index/vector/ivf/v2.rs)
·
[`exec.rs`](https://github.com/lance-format/lance/blob/main/rust/lance-datafusion-exec/src/exec.rs)
·
[`metrics.md`](https://github.com/lance-format/lance/blob/main/rust/lance/src/metrics.md)

------------------------------------------------------------------------

## 5. Recent research on vector-database caching and related systems

The papers below are grouped by the type of optimization they introduce.
Their reported performance numbers are not directly comparable because
the datasets, index implementations, hardware, and recall targets
differ.

### 5.1 Similarity-aware query-result caching

#### Aker --- Density-Aware Approximate Caching for Vector Search (2026)

-   **Status:** Published in Proceedings of the VLDB Endowment, June
    2026; extended version on arXiv in September 2026.
-   **Link:** [Microsoft Research
    publication](https://www.microsoft.com/en-us/research/publication/aker-density-aware-approximate-caching-for-vector-search/)
    · [arXiv extended version](https://arxiv.org/abs/2609.03712)
-   **Core idea:** Cache results of past ANN queries and reuse them for
    sufficiently similar future queries.
-   **Mechanisms:**
    -   Each cached query has its own similarity threshold.
    -   Thresholds adapt based on observed cache-hit behavior and local
        neighborhood density.
    -   A del-consistency model applies deletions eagerly and insertions
        lazily.
-   **Reported results:** Up to 64 percentage points of recall
    improvement over prior approximate-cache approaches, up to 3.2× QPS,
    and 0.6× the memory of pgvector shared buffers in the authors'
    workloads.
-   **Lance applicability:** High for workloads with repeated or
    semantically similar queries, but it is a new query-level layer
    rather than a replacement for `LanceCache`.

**Potential design for Lance:**

Put an optional similarity-aware cache in front of the normal ANN
execution path. Cache the query embedding and result IDs/distances. Use
a per-entry threshold and an explicit fallback to the ordinary Lance
search when the cache cannot confidently serve the query.

Lance's snapshot/version model offers a useful alternative to blindly
applying Aker's update model: a cache can be scoped to a dataset version
and index-generation identity, then invalidated or refreshed when the
snapshot changes.

#### QVCache --- A Query-Aware Vector Cache (2026)

-   **Status:** arXiv preprint, February 2026. The arXiv record still
    contains placeholder publication metadata; treat it as a preprint
    rather than assuming a finalized proceedings publication.
-   **Link:** [arXiv:2602.02057](https://arxiv.org/abs/2602.02057)
-   **Core idea:** Query repetition in vector search is semantic rather
    than exact. Two slightly different query vectors may have
    sufficiently similar top-k neighbors.
-   **Mechanisms:**
    -   Maintains multiple small in-memory mini-indexes.
    -   Uses FreshVamana graph indexes for the cached vectors.
    -   Learns region-specific similarity thresholds online.
    -   Uses cache mini-indexes as independently evictable units.
    -   Can bypass backend ANN execution when its hit predicate accepts
        a cached result.
-   **Reported results:** Sub-millisecond cache-hit latency and up to
    40--1000× lower query latency for cache hits in the authors'
    experiments.
-   **Lance applicability:** High for RAG/recommendation workloads with
    temporal-semantic query locality. It does not require changing
    Lance's index builder, but it does require a new layer above vector
    search.

**Important distinction:** QVCache caches query-level results/vectors,
not IVF partition entries. It should be evaluated as a separate cache
with a separate memory budget.

#### Aker vs QVCache

  -----------------------------------------------------------------------
  Dimension               Aker                    QVCache
  ----------------------- ----------------------- -----------------------
  Cache target            Past query results      Past query results and
                                                  cached vectors

  Similarity logic        Per-entry adaptive      Region-specific learned
                          threshold               thresholds

  Main freshness idea     Eager deletions, lazy   Backend feedback and
                          insertions              threshold learning

  Cache organization      Query entries with      Multiple FreshVamana
                          adaptive thresholds     mini-indexes

  Best fit for Lance      Similar query reuse     Semantic query locality
                          with snapshot-aware     and bounded query-cache
                          freshness               lookup
  -----------------------------------------------------------------------

### 5.2 Graph-structure and node-level caching

These papers are most relevant to Lance's `IVF_HNSW_*` variants.

#### GPS --- Revisiting the Data Layout for Disk-based High-Dimensional Vector Search (2026)

-   **Status:** Published in Proceedings of the ACM on Management of
    Data, June 2026.
-   **Link:** [Publication record and
    DOI](https://research.cuhk.edu.hk/en/publications/gps-revisiting-the-data-layout-for-disk-based-high-dimensional-ve/)
    · [DOI 10.1145/3802069](https://doi.org/10.1145/3802069)
-   **Core idea:** Graph adjacency information is accessed more
    frequently than vector payloads during graph traversal.
-   **Mechanisms:**
    -   Keep graph adjacency lists in a dedicated memory cache.
    -   Co-locate neighboring adjacency lists with vectors in disk
        blocks.
    -   Treat graph structure and vector data as distinct access
        classes.
-   **Reported results:** The arXiv predecessor *Gorgeous*
    ([arXiv:2508.15290](https://arxiv.org/abs/2508.15290)) reports more
    than 60% higher throughput and more than 35% lower latency than
    DiskANN and Starling. The 52% / 32% figures versus PipeANN in v1 of
    this report came from a secondary summary of the SIGMOD version and
    have **not** been checked against the paper.
-   **Lance applicability:** High for `IVF_HNSW_*`.

Lance already separates the physical graph/search structure
(`index.idx`) from vector storage (`auxiliary.idx`). The more immediate
opportunity is to separate their **cache budgets and admission
policies**, rather than assuming a new file format is required.

A finer-grained graph-record cache would require changing how HNSW
sub-index data is read and retained, because the current IVF partition
path materializes the partition sub-index as part of a partition entry.

#### SplitPGV --- Separating Neighbors from Element for Efficient HNSW Indexing (2026)

-   **Status:** IEEE CLOUD 2026.
-   **Link:** [IEEE Xplore
    record](https://ieeexplore.ieee.org/abstract/document/11662388) ·
    [DOI
    10.1109/CLOUD72782.2026.00061](https://doi.org/10.1109/CLOUD72782.2026.00061)
-   **Core idea:** Neighbor/graph tuples and vector element tuples have
    different access patterns and should not be forced to share the same
    physical cache behavior.
-   **Reported results:** Up to 3.5× query throughput and a 20%
    increase in buffer-cache hit ratio over vanilla pgvector.
    **Verification (v2):** the DOI resolves, but the abstract could not
    be retrieved and no other index lists the paper; treat the numbers
    as unverified.
-   **Lance applicability:** High as a cache-design principle. Lance's
    two-file format already distinguishes graph structure and vector
    payload; a logical split of the cache can build on that separation.

#### GoVector --- An I/O-Efficient Caching Strategy for High-Dimensional Vector Nearest Neighbor Search (2025/2026)

-   **Status:** arXiv version submitted August 2025; described as the
    English version of a paper accepted for the *Journal of Software*,
    Vol. 37, No. 3, 2026.
-   **Link:** [arXiv:2508.15694](https://arxiv.org/abs/2508.15694)
-   **Core idea:** Static caching of likely entry nodes is not enough
    because actual graph traversal paths are query-dependent.
-   **Mechanisms:**
    -   Static caching for likely starting nodes and nearby graph
        regions.
    -   Dynamic caching for nodes encountered during real searches.
    -   Locality-aware data layout.
-   **Lance applicability:** Medium-high for HNSW partitions. A query
    can warm graph records that are actually traversed, rather than only
    relying on static prewarming.

#### VeloANN --- Optimizing SSD-Resident Graph Indexing for High-Throughput Vector Search (2026)

-   **Status:** arXiv preprint, February 2026; the paper's arXiv text
    contains placeholder PVLDB metadata and says artifacts are still
    undergoing review.
-   **Link:** [arXiv:2602.22805](https://arxiv.org/abs/2602.22805)
-   **Core idea:** SSD-resident graph search needs coordinated
    improvements to data layout, buffering, and I/O scheduling.
-   **Mechanisms:**
    -   Record-level buffer pool rather than only page-level caching.
    -   A lightweight clock/second-chance policy.
    -   Neighbor-aware record layout.
    -   Asynchronous prefetching and cache-aware beam search.
    -   Coroutine-based scheduling to overlap I/O and computation.
-   **Reported results:** Up to 5.8× higher throughput and 3.25× lower
    latency than the compared disk-based ANN systems in the authors'
    evaluation.
-   **Lance applicability:** High as an architectural reference for
    local-SSD `IVF_HNSW_*` and future out-of-memory graph partitions. It
    is not a drop-in replacement for Lance's current object-store
    scheduler.

#### Graph Reordering for Cache-Efficient Near Neighbor Search (NeurIPS 2022)

-   **Status:** NeurIPS 2022 main conference.
-   **Link:** [NeurIPS
    paper](https://papers.neurips.cc/paper_files/paper/2022/hash/fb44a668c2d4bc984e9d6ca261262cbb-Abstract-Conference.html)
-   **Core idea:** Reorder graph nodes so nodes likely to be visited
    together are closer in memory.
-   **Reported result:** Up to 40% query-time improvement in the
    reported HNSW experiments.
-   **Lance applicability:** Medium. HNSW node IDs are partition-local;
    a build-time graph permutation could improve locality, but must
    preserve the mapping from graph IDs to the vector-storage row
    offsets and dataset row addresses.

#### Lossless Compression of Proximity Graph Indexes for Efficient Approximate Nearest Neighbors Search (2026)

-   **Status:** CIKM 2026 paper; arXiv version submitted 27 October
    2026.
-   **Link:** [arXiv:2610.37176](https://arxiv.org/abs/2610.37176) ·
    [DOI
    10.1145/3799682.3840116](https://doi.org/10.1145/3799682.3840116)
-   **Core idea:** In compressed-vector indexes, graph edge storage can
    become a large share of the total index footprint.
-   **Mechanisms:** Offline node permutations create structural
    locality; SIMD-friendly integer compression (such as Stream VByte)
    compresses adjacency lists while keeping query-time decode overhead
    low.
-   **Reported results:** Up to 52.7% graph-edge compression; the paper
    reports a 44.3% edge-storage reduction in an eBay production
    deployment without observed latency degradation.
-   **Lance applicability:** Medium-high for `IVF_HNSW_*`. Smaller graph
    entries can increase effective cache capacity, even without adding
    more RAM.

### 5.3 Storage layout, SSD I/O, and locality

#### Turbocharging Vector Databases Using Modern SSDs (PVLDB 2025)

-   **Status:** Proceedings of the VLDB Endowment, Volume 18, Issue 11,
    July 2025.
-   **Link:** [PVLDB article](https://doi.org/10.14778/3749646.3749724)
    · [PDF](https://www.vldb.org/pvldb/vol18/p4710-do.pdf)
-   **Core idea:** Disk-resident graph ANN performance depends on SSD
    concurrency and data locality, not just cache capacity.
-   **Mechanisms:**
    -   Parallel I/O using `io_uring`.
    -   Spatially-aware insertion ordering.
    -   Locality-preserving co-location of graph records.
-   **Reported results:** Up to 11.1× throughput, 3.23× cache hit ratio,
    and 98.4% lower index-build time in the authors' pgvector-based
    evaluation.
-   **Lance applicability:** Medium-high for local SSD deployments.
    Lance already ships io_uring readers for high-IOPS NVMe
    (`lance-io/src/uring.rs`), so the transferable ideas are the
    locality and co-location ones. They transfer less directly to remote
    S3/GCS/Azure/OBS access, where request latency and range-read
    behavior dominate.

#### Efficient Vector Search on Disaggregated Memory with d-HNSW (2025)

-   **Status:** arXiv preprint, May 2025.
-   **Link:** [arXiv:2505.11783](https://arxiv.org/abs/2505.11783)
-   **Core idea:** Separate compute from a shared remote memory pool and
    avoid excessive remote round trips during graph search.
-   **Mechanisms:**
    -   Representative index caching at the compute node.
    -   RDMA-friendly layout.
    -   Batched query-aware data loading.
-   **Lance applicability:** Medium/low for current single-process Lance
    use. It is more relevant to a future distributed Lance serving
    architecture or shared-memory tier than to the current local Session
    cache.

#### CD-ANN --- Scalable Approximate Nearest Neighbor Search on Client-Side Devices (Journal of Systems Architecture, 2026)

-   **Status:** Journal of Systems Architecture, June 2026.
-   **Link:** [ScienceDirect
    article](https://www.sciencedirect.com/science/article/pii/S1383762126000895)
    · [DOI
    10.1016/j.sysarc.2026.103771](https://doi.org/10.1016/j.sysarc.2026.103771)
-   **Core idea:** Segment HNSW and load frequently accessed segments on
    demand, reducing memory pressure on constrained clients.
-   **Verification (v2):** the DOI resolves; the abstract was not
    retrieved.
-   **Lance applicability:** Medium. It reinforces the value of
    independently cacheable index segments, but Lance's existing IVF
    partition boundary is already a natural segment boundary. The useful
    question is whether HNSW partitions need an even finer graph-record
    boundary.

#### Query grouping: CaGR-RAG (2025) and CALL (2025)

-   **v2 correction:** v1 cited "CARGO" (SSRN 7422835). The SSRN page
    could not be retrieved and no index lists the paper, so it has been
    replaced by two verifiable preprints on the same idea.
-   **Links:** [CaGR-RAG, arXiv:2505.01164](https://arxiv.org/abs/2505.01164)
    · [CALL, arXiv:2509.18670](https://arxiv.org/abs/2509.18670)
-   **Core idea:** With many concurrent queries, the combined working
    set may exceed cache capacity. Instead of only changing eviction,
    group or schedule queries whose retrieval clusters overlap.
-   **Reported results:** CaGR-RAG reports up to 51.55% lower p99 latency
    with a higher cache hit ratio; CALL reports up to 33% lower p99.
-   **Lance applicability:** Medium-high for high-concurrency or batch
    workloads. Lance identifies the IVF partitions of each query before
    loading them, so overlap estimation is natural.

### 5.4 General cache admission and eviction research

These are not vector-database papers, but they are directly useful for
the policy inside Lance's existing `CacheBackend`.

#### TinyLFU --- A Highly Efficient Cache Admission Policy (ACM TOS 2017)

-   **Status:** ACM Transactions on Storage, 2017.
-   **Link:** [ACM DOI](https://doi.org/10.1145/3149371) · [arXiv
    version](https://arxiv.org/abs/1512.00727)
-   **Core idea:** Do not automatically admit every miss. Estimate
    recent frequency with a compact approximate frequency sketch and
    admit a new object only when it is more valuable than the eviction
    candidate.
-   **Lance status (v2):** already available. The built-in
    `MokaCacheBackend` uses TinyLFU admission.

#### S3-FIFO --- FIFO Queues Are All You Need for Cache Eviction (SOSP 2023)

-   **Status:** SOSP 2023.
-   **Link:** [SOSP
    proceedings](https://sigops.org/s/conferences/sosp/2023/toc.html) ·
    [Author
    explanation](https://s3fifo.com/blog/2023/08/01/fifo-queues-are-all-you-need-for-cache-eviction/)
-   **Core idea:** Three FIFO queues, including a small probationary
    queue, quickly demote one-hit objects and avoid the synchronization
    overhead of a heavily updated LRU list.
-   **Reported result:** The authors report lower miss ratios across a
    large cache-trace collection and up to 6× higher throughput than
    optimized LRU at 16 threads.
-   **Lance status (v2):** already the default. `QuickCacheBackend`
    wraps `quick_cache`, which implements byte-weighted S3-FIFO.
    Comparing Quick and Moka is therefore a configuration baseline, not
    a research contribution.

#### Cost-aware admission

-   **Reference:** [Boosting Cache Performance by Access Time
    Measurements](https://doi.org/10.1145/3572778)
-   **Core idea:** Combine frequency with the actual cost of bringing an
    item into cache.
-   **Lance application:** A partition loaded from a high-latency object
    store should not necessarily be treated the same as a similarly
    sized partition read from local storage.

A useful candidate utility score for evaluation is:

\[ U(p) = `\frac{
\widehat{F}(p)\cdot \widehat{C}(p)\cdot \widehat{R}(p)
}{
S(p)
}`{=tex} \]

where:

-   (`\widehat{F}`{=tex}(p)) is estimated recent access frequency.
-   (`\widehat{C}`{=tex}(p)) is observed load cost, such as latency or
    I/O bytes.
-   (`\widehat{R}`{=tex}(p)) is an estimated reuse probability over the
    admission horizon.
-   (S(p)) is the cache-accounted size.

This is a **proposed heuristic**, not an algorithm already implemented
in Lance. Its baselines are the two existing backends (S3-FIFO and
TinyLFU) at the same byte budget.

### 5.5 Filtered vector search (added in v2, track ④)

#### SieveIVF --- Threshold-Aware IVF Execution (2026)

-   **Status:** arXiv preprint, August 2026; no venue listed.
-   **Link:** [arXiv:2608.03199](https://arxiv.org/abs/2608.03199)
-   **Core idea:** Stop probing after W consecutive partitions yield no
    qualifying candidate; group ready queries by partition (continuous
    batching) with a lookahead scheduler.
-   **Reported results:** 4.1--7.6× faster than fixed-probe IVF on 10M
    workloads (0.03--1.13 pp filtered recall loss) and 6.1--8.4× on 100M
    workloads (1.43--2.29 pp). **Implemented in Lance.**
-   **Lance applicability:** High; it is the closest prior art for
    track ④.

#### VADER --- Filtered Vector Search with Declarative Recall (VLDB 2027)

-   **Link:** [arXiv:2609.38053](https://arxiv.org/abs/2609.38053)
-   **Core idea:** A filter-aware recall predictor (selectivity and
    filter--vector correlation) ends the search when the predicted
    recall reaches a user target, on top of IVF or graph indexes.
-   **Reported results:** Up to 53% faster and 28% better result quality
    than the best baseline.

#### Filtered vector search in a disaggregated lakehouse (2026)

-   **Link:** [arXiv:2608.05441](https://arxiv.org/abs/2608.05441)
    (IBM Research; Iceberg + Parquet, per-file IVF in the footer)
-   **Core idea:** Prune data files with the predicate before ANN; a
    rendezvous-hashed per-file cache hides object-store latency.
-   **Reported results:** 355 of 444 files pruned for a selective
    predicate; about 94× faster once a join-derived filter is
    materialized into a partitioned layout. Residual predicates are only
    safe to push into search over a partitioned column.

#### Predicate Caching (SIGMOD 2024)

-   **Reference:** Schmidt, Kipf, Horn, Saxena, Kraska, *Predicate
    Caching: Query-Driven Secondary Indexing for Cloud Data Warehouses*,
    SIGMOD 2024 ([slides](https://www.db.in.tum.de/~schmidt/papers/predicate-caching-slides.pdf)).
-   **Core idea:** Cache the qualifying row ranges of base-table
    predicates as a by-product of query execution; inserts do not
    invalidate entries.
-   **Reported results:** Up to 10× on selected queries in Amazon
    Redshift; best for skewed data and selective repeated predicates.
-   **Lance applicability:** High; Lance evaluates the tag allow-list
    again on every query (3.9).

### 5.6 Training data loading (added in v2, track ⑥)

#### Seneca (USENIX FAST 2026)

-   **Link:** [FAST '26](https://www.usenix.org/conference/fast26/presentation/desai)
    · [arXiv:2511.13724](https://arxiv.org/abs/2511.13724)
-   **Core idea:** A performance model partitions the cache between
    encoded, decoded and augmented samples; batch sampling
    opportunistically prefers cached samples so concurrent jobs help
    each other.
-   **Reported results:** 45.23% lower makespan than PyTorch and up to
    3.45× higher throughput than the next best data loader.
-   **Lance applicability:** High for ⑥ and ⑦ (encoded vs decoded
    video).

#### Lance random access (2025)

-   **Link:** [arXiv:2504.15247](https://arxiv.org/abs/2504.15247)
-   **Relevance:** Lance file format 2.1 is designed for NVMe random
    access, so a local or pooled tier under Lance can serve row-level
    and range reads instead of whole files.

------------------------------------------------------------------------

## 6. Lessons from other vector databases

These systems are useful engineering references, but their local-storage
assumptions should not be copied blindly into Lance.

### Qdrant

-   [Qdrant storage
    documentation](https://qdrant.tech/documentation/manage-data/storage/)
-   Qdrant uses memory-mapped vector storage.
-   Its `cached` memory tier preloads data into the OS disk cache;
    `cold` relies on the operating system to cache pages as accessed.
-   This provides a clear distinction between application-managed RAM
    and OS-managed page caching.

**Lance lesson:** An OS page-cache strategy is most relevant when Lance
reads local files. For cloud object stores, an explicit local
read-through tier may be more useful than relying on OS page caching.

### Milvus

-   [Milvus mmap documentation](https://milvus.io/docs/v2.5.x/mmap.md)
-   [Milvus eviction documentation](https://milvus.io/docs/eviction.md)
-   Milvus exposes mmap controls for vector fields and vector indexes.
-   Its QueryNode eviction documentation describes LRU-based resource
    reclamation and metadata that remains cached.

**Lance lesson:** Keep small, frequently required metadata distinct from
large, evictable index payloads. Lance already has separate metadata and
index cache families; per-type budgets or reservations could build on
this.

### Weaviate

-   [Weaviate vector index
    documentation](https://docs.weaviate.io/weaviate/concepts/vector-index)
-   Weaviate exposes vector-cache limits and discusses tuning cache size
    for query-heavy workloads.
-   Its HFresh documentation also illustrates a compressed,
    centroid/posting-list-oriented index architecture.

**Caution:** The documented vector-cache behavior includes a whole-cache
drop when the configured cache fills. That is not a policy to copy into
Lance; it is an example of why admission and eviction behavior need
workload-specific evaluation.

------------------------------------------------------------------------

## 7. What the research suggests for Lance

### 7.1 Opportunity ranking

  ------------------------------------------------------------------------------
  Priority          Opportunity              Lance change      Expected value
                                             required          
  ----------------- ------------------------ ----------------- -----------------
  1                 Cost-aware IVF partition Cache policy /    High
                    admission                admission hook    

  2                 Local NVMe persistent    Custom backend +  High for
                    tier                     codec coverage    remote-storage
                                                               workloads

  3                 Separate HNSW graph and  Partition cache   Low (v2): Lance
                    vector-storage budgets   refactor          loads each
                                                               per-partition
                                                               graph whole;
                                                               ADS uses IVF_RQ

  4                 Query-aware prefetch /   Query planner /   Medium-high
                    concurrent query         scheduler         
                    grouping                 integration       

  5                 Similarity-aware         New optional      High for semantic
                    query-result cache       query layer       query locality

  6                 Graph                    Index builder and Medium-high for
                    reordering/compression   format/runtime    HNSW
                                             work              

  7                 Disaggregated memory     New distributed   Longer-term
                    cache                    serving           
                                             architecture      
  ------------------------------------------------------------------------------

### 7.2 Proposal A --- Cost-aware admission for IVF partitions

**Problem:** The current cache has byte accounting and priority support,
but the normal IVF partition get-or-load path does not implement a
general frequency- and load-cost-aware admission policy.

**Proposed behavior:**

1.  On a cache miss, load the partition.
2.  Record its size, load latency, bytes read, and source latency class.
3.  Track a compact recent-frequency estimate for partition keys.
4.  Admit only if the estimated benefit is higher than the current
    victim's benefit.
5.  Allow oversized or one-hit partitions to be used for the current
    query without displacing a more valuable resident entry.

A first implementation could compare:

-   Current default behavior (S3-FIFO, `QuickCacheBackend`).
-   The built-in TinyLFU backend (`MokaCacheBackend`).
-   Cost-aware frequency admission using load latency and size.
-   Filter-aware admission for filtered queries (track ④).

**Why this is the best first project:** It works with existing IVF
partition boundaries, avoids a new index format, and can be measured
using real query traces.

### 7.3 Proposal B --- RAM + local NVMe + object-store tiering

A possible future architecture:

``` mermaid
flowchart TD
    Q["Lance query"] --> L1["L1: RAM cache"]
    L1 -->|Hit| R["Use cached partition"]
    L1 -->|Miss| L2["L2: Local NVMe cache"]
    L2 -->|Hit| P["Deserialize / promote if useful"]
    L2 -->|Miss| L3["L3: Canonical Lance index files"]
    L3 --> W["Read from object store or local source"]
    W --> A["Admit to NVMe / RAM according to policy"]
    A --> R
    P --> R
```

**Suggested tier roles:**

  -----------------------------------------------------------------------
  Tier                    Contents                Policy
  ----------------------- ----------------------- -----------------------
  RAM                     Hot partition entries,  Fast admission, strict
                          hot graph records,      memory budget
                          frequently used RQ      
                          planes                  

  Local NVMe              Reusable serialized     Cost-aware retention,
                          partition entries or    bounded disk quota
                          graph records           

  Origin                  `index.idx`,            Canonical source of
                          `auxiliary.idx`,        truth
                          object-store files      
  -----------------------------------------------------------------------

Implementation notes:

-   Implement a custom `CacheBackend`.
-   Use `CacheCodec` for serializable entry types.
-   Add an explicit disk quota and background write policy.
-   Use the opaque cache key plus cache-key format and codec
    identity/version.
-   Use atomic writes and checksums for persistent entries.
-   Treat decode failures as cache misses and reload from origin.
-   Audit codec coverage before expecting all Lance cache types to
    persist.
-   Consider `get_rows` and `CacheRangeReader` for partial reads.

The local cache must remain disposable. It must never become the
authoritative copy of the index.

### 7.4 Proposal C --- Separate HNSW graph cache from vector storage cache

GPS, SplitPGV, GoVector, and VeloANN all reinforce the idea that graph
structure and vector payloads have different access patterns.

Lance's V3 files already separate graph structure and vector storage
physically. But the common IVF partition entry combines the sub-index
and quantized storage.

A possible next design:

``` mermaid
flowchart TB
    P["IVF partition ID"] --> G["HNSW graph cache"]
    P --> V["Vector storage cache"]
    G --> GP["Adjacency / graph records"]
    V --> VP["FLAT / PQ / SQ / RQ data"]
    GP --> S["HNSW search"]
    VP --> S
```

Potential policy:

-   Keep graph entry points and frequently traversed adjacency records
    in RAM.
-   Keep vector storage under a separate byte budget.
-   Dynamically promote graph records observed in actual traversal.
-   Prefetch likely neighbors in batches where storage layout allows it.
-   Avoid admitting a large vector payload if the graph metadata is much
    more valuable to retain.

This is most relevant to `IVF_HNSW_*`. It is less directly relevant to
`IVF_PQ` or `IVF_RQ` variants that use flat partition scoring rather
than graph traversal. **v2 note:** Lance loads each per-partition HNSW
graph whole (`HNSW::load` from one record batch), and IVF partitions are
usually small, so this proposal is deprioritized for the ADS tracks.

### 7.5 Proposal D --- Similarity-aware query-result caching

QVCache and Aker suggest a new layer above Lance's current index cache.

``` mermaid
flowchart TD
    Q["Query embedding + search context"] --> C["Similarity-aware query cache"]
    C --> H{"Confidence / threshold accepts?"}
    H -->|Yes| R["Return cached candidate IDs and distances"]
    H -->|No| L["Run normal Lance ANN search"]
    L --> U["Update query cache and thresholds"]
    U --> R
```

This must be an explicit approximate-search feature, not a transparent
replacement for exact semantics.

**Required cache identity/context:**

-   Dataset snapshot/version.
-   Index UUIDs and segment coverage.
-   Query metric and vector transform.
-   `k`.
-   Filter and prefilter identity.
-   `nprobes` / `search_effort`.
-   `refine_factor` and relevant HNSW parameters.
-   Whether unindexed rows are included.
-   Deletion and update visibility.

A safe first prototype should:

1.  Cache over-fetched candidate IDs and distances.
2.  Re-score cached candidates against the incoming query where
    possible.
3.  Apply the correct snapshot's deletion and filter semantics.
4.  Fall back to ordinary ANN search when the learned threshold is not
    met.
5.  Expose a recall/approximation contract and benchmark Recall@k
    against exact ground truth.

Do not use physical row addresses as durable cross-version query-cache
identities. Prefer the appropriate logical row identity and scope it to
the relevant dataset snapshot.

### 7.6 Proposal E --- Query-aware prefetch and concurrent query grouping

Lance already computes IVF partition candidates before loading them.
That provides a natural signal for query-aware prefetch.

Potential approach:

-   Compute each query's top candidate partition IDs.
-   Estimate overlap between in-flight queries.
-   Batch partition reads or prioritize partitions needed by several
    queries.
-   Limit batching windows to protect tail latency.
-   Maintain fairness so a large query batch cannot starve single-query
    requests.

This direction is supported by CaGR-RAG, CALL and the continuous
batching in SieveIVF (5.3, 5.5).

------------------------------------------------------------------------

## 8. Recommended research and implementation plan

### Phase 0 --- Establish a trustworthy baseline

Before changing cache behavior:

-   Collect representative query traces.
-   Record partition IDs, partition sizes, loader latency, bytes read,
    and cache outcomes.
-   Separate local-storage and remote-object-store workloads.
-   Measure cold-cache and warm-cache performance.
-   Capture p50, p95, and p99 latency.
-   Measure Recall@k against exact search.
-   Measure RSS separately from cache-accounted bytes.

Use session cache diagnostics and object-store I/O metrics. Avoid
relying only on the execution-summary hit ratio until the
`write_cache=false` discrepancy is resolved.

### Phase 1 --- Partition admission experiments

Compare:

1.  Current Quick cache behavior (S3-FIFO).
2.  The Moka backend (TinyLFU).
3.  Cost-aware admission using frequency × load cost / size.
4.  A hybrid: S3-FIFO or TinyLFU plus cost-aware victim selection.

The first success criterion should be fewer expensive partition reloads
at the same memory budget, without degrading p99 latency or increasing
CPU overhead excessively.

### Phase 2 --- Persistent local tier

Prototype a custom backend that supports:

-   RAM resident lookup.
-   Local NVMe lookup.
-   No-promotion reads.
-   Selective row/range reads where supported.
-   Codec version validation.
-   Bounded disk usage.
-   Background persistence.

Test restart behavior, cache corruption, codec upgrades, and concurrent
writes.

### Phase 3 --- HNSW graph/storage separation

Only after confirming that `IVF_HNSW_*` workloads are limited by
graph-cache misses:

-   Split graph and vector-storage cache entries.
-   Add graph-record or graph-page admission.
-   Add neighbor-aware prefetch.
-   Evaluate graph reordering/compression.

### Phase 4 --- Optional query-result cache

Prototype QVCache/Aker-like behavior behind an explicit
approximate-search option. Keep it separate from the
correctness-sensitive index-data cache.

### Phase 5 --- Query scheduling

Evaluate grouping in-flight queries by overlapping IVF partition sets.
Include fairness and tail-latency constraints.

------------------------------------------------------------------------

## 9. Benchmark design

A credible cache evaluation needs to test more than one dataset and one
hit ratio.

### 9.1 Workload dimensions

  Dimension          Suggested cases
  ------------------ --------------------------------------------------
  Index family       IVF_RQ, IVF_PQ, IVF_FLAT, IVF_HNSW_PQ
  Dataset size       10M, 100M, and larger representative scale
  Query locality     Uniform, Zipf/skewed, bursty, semantic clusters
  Query repetition   Exact repeat, near-repeat, no repeat
  Filters            No filter, selective prefilter, broad filter
  Storage            Local SSD, local filesystem, remote object store
  Cache state        Cold, warm, partially warm
  Concurrency        Single query, moderate concurrency, saturation
  Dataset changes    Append, delete, optimize, retrain, compaction
  Memory budget      Small, medium, and large fractions of the index

### 9.2 Metrics

  -----------------------------------------------------------------------
  Metric                              Why
  ----------------------------------- -----------------------------------
  Recall@k                            Correctness/quality of approximate
                                      search

  p50 / p95 / p99 latency             User-facing performance

  QPS                                 Throughput

  Object-store bytes and request      Remote I/O avoided
  count                               

  Partition cache hit/miss and loader Partition cache effectiveness
  duration                            

  Cache size-removal churn            Admission/eviction pressure

  Cache occupancy by type             Which objects consume memory

  Process RSS / cgroup memory         Actual memory pressure

  CPU utilization and I/O wait        Whether caching moves the
                                      bottleneck

  Prewarm bytes and warm-hit rate     Whether prewarming pays off
  -----------------------------------------------------------------------

### 9.3 Correctness and concurrency tests

Include tests for:

-   Two datasets with the same partition numbers.
-   Different index UUIDs at the same dataset URI.
-   Fragment-reuse identity changes.
-   Dataset append and unindexed rows.
-   Deletions and filtered searches.
-   Index replacement/retraining.
-   Compaction and row-address remapping.
-   Cache codec version mismatch and corrupted entries.
-   Concurrent misses for the same partition.
-   Cancellation during a load.
-   `write_cache=false` with an already resident entry and with a cold
    entry.

### 9.4 Essential ablations

For each policy, test:

-   Same memory budget.
-   Same recall target.
-   Same query set and ordering.
-   Same storage source.
-   Same concurrency.
-   Same prewarm policy.

Otherwise a higher cache hit ratio may simply reflect a different memory
budget or query distribution.

------------------------------------------------------------------------

## 10. Final recommendation

### What is already solved in Lance

-   IVF partition caching.
-   Session index and metadata caches.
-   Typed, namespaced cache keys.
-   Byte-weighted capacity.
-   Priority-aware caching.
-   Specialized RaBitQ plane and code-only caching.
-   Explicit bounded prewarming.
-   Cache codecs and custom backend extension points.
-   Cache diagnostics and per-type occupancy.

### What remains promising

1.  **Adaptive, cost-aware admission for partition entries.**
2.  **A production-ready local persistent cache tier.**
3.  **A separate graph-structure cache for HNSW partitions.**
4.  **A query-aware prefetch/scheduling layer.**
5.  **An opt-in similarity-aware query-result cache.**

### Best first project

**Adaptive and cost-aware admission for Lance IVF partition caching.**

A good first paper/engineering contribution could compare:

-   Existing cache behavior (S3-FIFO, default).
-   The built-in TinyLFU backend.
-   Cost-aware admission.
-   A hybrid admission/eviction design.

**v2:** for the ADS pipeline this recommendation is superseded by the
two tracks in section 11.

Use actual partition access traces and remote load costs, rather than
synthetic object requests alone.

This has the clearest connection to Lance's current code, avoids a
format migration, and gives a strong experimental foundation for later
NVMe tiering and HNSW-specific work.

------------------------------------------------------------------------

## 11. Two research tracks for the ADS pipeline (added in v2)

The ADS framework (`main-framework.pptx`) names two targets that this
cache research serves. They are run as **separate research tracks** with
separate proposal decks.

### Track ④ --- vector + tag hybrid retrieval

ADS problem: "join computation of vector and scalar data is
inefficient." Baseline: `IVF_RQ` with tag filters.

  --------------------------------------------------------------------------
  Key technology           Lance integration point     Prior art
  ------------------------ --------------------------- ---------------------
  KT1 predicate-result     `DatasetPreFilter`          Predicate Caching
  cache (allow-list per    `filtered_ids`; key follows (SIGMOD 2024)
  predicate and version)   `RowAddrMaskKey`

  KT2 filter-aware         `prepare_partition`         SieveIVF, VADER,
  partition skipping       (load before mask today)    lakehouse pruning
  (conservative per-
  partition tag summary,
  optional adaptive stop)

  KT3 batched mining       knn exec; partition-major   SieveIVF batching,
  (partition-major         schedule                    CaGR-RAG, CALL
  schedule)
  --------------------------------------------------------------------------

**Worked model:** with tags independent of the embedding, the share of
probed partitions with no qualifying row is about e^(−s·n) for
selectivity s and n rows per partition. For 10M rows and K = 16,384
(n ≈ 610) that is 94% at s = 0.01%, 54% at 0.1%, 4.7% at 0.5% and 0.2%
at 1%. The benefit of KT2 is concentrated on selective tags. This is a
model, not a measurement.

Deck: `presentations/lance-ivf-rq-proposal/ads-lance-hybrid-retrieval-v1.pptx`.

### Track ⑥ --- multimodal data cache

ADS problem: "pulling a single clip (4 GB) from OBS must drop to ms
level." Served objects are the original data (video, point clouds),
vectors, tags and index partitions in Lance format. UB / Lingqu pooled
memory is assumed to be available.

  --------------------------------------------------------------------------
  Key technology           Lance integration point     Prior art
  ------------------------ --------------------------- ---------------------
  KT1 tiered byte cache    `WrappingObjectStore`;      Seneca, Hoard,
  (RAM → UB pool → NVMe →  `CacheTier::Local`,         lakehouse per-file
  OBS)                     `get_without_promotion`;    cache
                           `OriginLatencyClass::Low`

  KT2 sampler-driven       `take` / `take_blobs` read  Seneca, CoorDL,
  prefetch (plus           plan → byte ranges          SHADE
  retrieval results from
  ④ as hints)

  KT3 cost-aware admission `CacheBackend` policy;      access-time
  and shared placement     rendezvous hashing over the admission (ACM
  (U = F·C·R / S)          UB pool                     TOS), S3-FIFO /
                                                       TinyLFU baselines
  --------------------------------------------------------------------------

**Worked model:** moving a whole 4 GB clip takes 4,000 / B ms at an
aggregate bandwidth of B GB/s (400 ms at 10 GB/s, 40 ms at 100 GB/s).
"ms level" is therefore reachable only as time-to-first-batch: read the
range a step needs and serve it from a close, high-bandwidth tier.

Deck: `presentations/lance-ivf-rq-proposal/ads-lance-multimodal-cache-v1.pptx`.

------------------------------------------------------------------------

## Revision notes (v2)

  -----------------------------------------------------------------------
  Section     Change
  ----------- -----------------------------------------------------------
  Summary,    The built-in backends are not LRU: Quick = S3-FIFO, Moka =
  3.1, 3.6,   TinyLFU. TinyLFU / S3-FIFO are baselines, not proposals.
  5.4, 7, 8,
  10

  3.4         Added `OriginLatencyClass`.

  3.8, 3.9    Added mask, scalar-page, allow-list and data-byte cache
              rows; added the filtered IVF search path.

  4.2         The "write_cache=false is always a miss" and "HNSW not
              instrumented" statements are in `exec.rs` doc comments,
              not `metrics.md`.

  5.2         GPS numbers now from the arXiv predecessor; SplitPGV and
              CD-ANN marked unverified (DOIs resolve, abstracts not
              retrieved).

  5.3         Lance already has io_uring readers. CARGO (unverifiable)
              replaced by CaGR-RAG and CALL.

  5.5, 5.6    Added filtered-search and data-loading research.

  7.1, 7.4    HNSW graph cache deprioritized.

  11          Split into tracks ④ and ⑥.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Appendix A --- Source map

### Lance documentation

-   [Vector Search guide](https://lance.org/guide/vector_search/)
-   [Vector Index Format](https://lance.org/format/index/vector/)
-   [Performance Guide](https://lance.org/guide/performance/)

### Lance Rust source

-   [Vector index
    builder](https://github.com/lance-format/lance/blob/main/rust/lance/src/index/vector/builder.rs)
-   [IVF v2 runtime
    implementation](https://github.com/lance-format/lance/blob/main/rust/lance/src/index/vector/ivf/v2.rs)
-   [Partition cache
    serialization](https://github.com/lance-format/lance/blob/main/rust/lance/src/index/vector/ivf/partition_serde.rs)
-   [IVF sub-index
    interface](https://github.com/lance-format/lance/blob/main/rust/lance-index/src/vector/v3/subindex.rs)
-   [IVF
    shuffler](https://github.com/lance-format/lance/blob/main/rust/lance-index/src/vector/v3/shuffler.rs)
-   [Vector
    storage](https://github.com/lance-format/lance/blob/main/rust/lance-index/src/vector/storage.rs)
-   [Session](https://github.com/lance-format/lance/blob/main/rust/lance/src/session.rs)
-   [Index cache
    namespaces](https://github.com/lance-format/lance/blob/main/rust/lance/src/session/index_caches.rs)
-   [Cache backend
    trait](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/backend.rs)
-   [Cache
    facade](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/mod.rs)
-   [Quick cache
    backend](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/quick.rs)
-   [Cache priority
    entries](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/priority.rs)
-   [Cache
    codecs](https://github.com/lance-format/lance/blob/main/rust/lance-core/src/cache/codec.rs)
-   [Cache metrics
    documentation](https://github.com/lance-format/lance/blob/main/rust/lance/src/metrics.md)
-   [Execution
    metrics](https://github.com/lance-format/lance/blob/main/rust/lance-datafusion-exec/src/exec.rs)

### Vector-cache and index research

-   [Aker --- Microsoft
    Research](https://www.microsoft.com/en-us/research/publication/aker-density-aware-approximate-caching-for-vector-search/)
-   [Aker --- arXiv extended version](https://arxiv.org/abs/2609.03712)
-   [QVCache --- arXiv](https://arxiv.org/abs/2602.02057)
-   [GPS --- PACMMOD 2026](https://doi.org/10.1145/3802069)
-   [SplitPGV --- IEEE CLOUD
    2026](https://doi.org/10.1109/CLOUD72782.2026.00061)
-   [GoVector --- arXiv](https://arxiv.org/abs/2508.15694)
-   [VeloANN --- arXiv](https://arxiv.org/abs/2602.22805)
-   [Turbocharging Vector Databases Using Modern SSDs --- PVLDB
    2025](https://doi.org/10.14778/3749646.3749724)
-   [d-HNSW --- arXiv](https://arxiv.org/abs/2505.11783)
-   [CaGR-RAG --- arXiv](https://arxiv.org/abs/2505.01164) (replaces
    the unverifiable CARGO citation)
-   [CALL --- arXiv](https://arxiv.org/abs/2509.18670)
-   [SieveIVF --- arXiv](https://arxiv.org/abs/2608.03199)
-   [VADER --- arXiv / VLDB 2027](https://arxiv.org/abs/2609.38053)
-   [Filtered vector search in a disaggregated lakehouse ---
    arXiv](https://arxiv.org/abs/2608.05441)
-   [Predicate Caching --- SIGMOD 2024
    slides](https://www.db.in.tum.de/~schmidt/papers/predicate-caching-slides.pdf)
-   [Seneca --- FAST 2026](https://www.usenix.org/conference/fast26/presentation/desai)
-   [Lance random access --- arXiv](https://arxiv.org/abs/2504.15247)
-   [Gorgeous (GPS predecessor) --- arXiv](https://arxiv.org/abs/2508.15290)
-   [CD-ANN --- Journal of Systems
    Architecture](https://doi.org/10.1016/j.sysarc.2026.103771)
-   [Lossless Compression of Proximity Graph Indexes --- CIKM
    2026](https://doi.org/10.1145/3799682.3840116)
-   [Graph Reordering for Cache-Efficient Near Neighbor Search ---
    NeurIPS
    2022](https://papers.neurips.cc/paper_files/paper/2022/hash/fb44a668c2d4bc984e9d6ca261262cbb-Abstract-Conference.html)
-   [TinyLFU --- ACM TOS](https://doi.org/10.1145/3149371)
-   [S3-FIFO --- SOSP
    2023](https://sigops.org/s/conferences/sosp/2023/toc.html)
-   [Cost-aware admission --- Boosting Cache Performance by Access Time
    Measurements](https://doi.org/10.1145/3572778)

### Engineering references from other vector databases

-   [Qdrant storage and memory
    tiers](https://qdrant.tech/documentation/manage-data/storage/)
-   [Milvus mmap](https://milvus.io/docs/v2.5.x/mmap.md)
-   [Milvus eviction](https://milvus.io/docs/eviction.md)
-   [Weaviate vector index and cache
    considerations](https://docs.weaviate.io/weaviate/concepts/vector-index)
