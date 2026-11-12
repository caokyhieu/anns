# Sources · track ⑥ multimodal cache v1

Lance source (lance-format/lance main @ 04e619963, read 11 Nov 2026)
- rust/lance/src/dataset/blob/file.rs: BlobFile read_range / read_ranges, DEFAULT_READ_BUFFER_SIZE = 512 KiB
- rust/lance-encoding/src/encodings/logical/primitive.rs: PageDataCacheKey (page metadata only)
- rust/lance-io/src/object_store.rs L292 WrappingObjectStore; spill.rs LocalSpillStore (scratch); uring.rs
- rust/lance-core/src/cache/backend.rs CacheTier::Local, get_without_promotion; codec.rs (experimental); registry.rs
- rust/lance-index/src/vector/storage.rs OriginLatencyClass; rust/lance/src/session.rs with_index_origin_latency
- rust/lance-core/src/cache/quick.rs (S3-FIFO), moka.rs (TinyLFU); rust/lance/src/dataset.rs L242/L246

Papers
- Desai et al., Seneca, USENIX FAST 2026 — https://www.usenix.org/conference/fast26/presentation/desai
- Pace et al., Lance random access, arXiv:2504.15247 — https://arxiv.org/abs/2504.15247
- Jain, Griffin, Zawad, arXiv:2608.05441 (per-file rendezvous cache) — https://arxiv.org/abs/2608.05441
- Mohan et al., CoorDL, PVLDB 14(5) 2021; Khan et al., SHADE, FAST 2023; Hoard, arXiv:1812.00669 (ideas only)
- Boosting Cache Performance by Access Time Measurements, ACM TOS — https://doi.org/10.1145/3572778
- TinyLFU, ACM TOS 2017 — https://doi.org/10.1145/3149371 ; S3-FIFO, SOSP 2023
- d-HNSW, arXiv:2505.11783 — https://arxiv.org/abs/2505.11783

Company context: main-framework.pptx (ADS target ⑥: 4 GB clip from OBS at ms level, Lingqu acceleration).
Model: transfer ms = 4000 / B (whole clip), 62.5 / B (64 MiB window).
