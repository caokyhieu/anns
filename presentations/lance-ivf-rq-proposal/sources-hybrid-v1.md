# Sources · track ④ hybrid retrieval v1

Lance source (lance-format/lance main @ 04e619963, read 11 Nov 2026)
- rust/lance/src/index/vector/ivf/v2.rs: prepare_partition L1331 (try_join of partition load and prefilter wait), prefilter_for_partition L1241, selects_nothing check L1399, write_cache admission L2179
- rust/lance/src/index/prefilter.rs: DatasetPreFilter (deleted_ids, filtered_ids), selects_nothing (whole mask), create_deletion_mask cached
- rust/lance/src/session/caches.rs L167: RowAddrMaskKey (version, e-tag, restrict hash)
- rust/lance-core/src/cache/quick.rs (quick_cache 0.6.24, S3-FIFO), moka.rs (TinyLFU); rust/lance/src/dataset.rs L242/L246

Papers
- Schmidt et al., Predicate Caching, SIGMOD 2024 — https://www.db.in.tum.de/~schmidt/papers/predicate-caching-slides.pdf
- Hu et al., SieveIVF, arXiv:2608.03199 — https://arxiv.org/abs/2608.03199
- Chatzakis et al., VADER, arXiv:2609.38053 (VLDB 2027) — https://arxiv.org/abs/2609.38053
- Jain, Griffin, Zawad, Filtered Vector Search in a Disaggregated Lakehouse, arXiv:2608.05441 — https://arxiv.org/abs/2608.05441
- CaGR-RAG, arXiv:2505.01164 — https://arxiv.org/abs/2505.01164 ; CALL, arXiv:2509.18670 — https://arxiv.org/abs/2509.18670

Company context: main-framework.pptx (ADS target ④); ADS-PLAN slide 1 via ads-lance-ivf-rq-proposal-v5.1.pptx slide 2.
Model: P(empty) = exp(-s · 10M / K); see build_deck_hybrid_v1.py `empty_share`.
