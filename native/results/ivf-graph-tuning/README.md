Tuning runs for the fastpipnn sub-index in `ivf_graph_bench` (1M x 128 synthetic, 300 queries).

- `sub-leaf-sweep.jsonl`: C = 1024 and 64, flat assignment; records in order: hnsw, then fastpipnn
  leaf128-lk8-lpl2, leaf128-lk8-lpl4, leaf64-lk8-lpl2, leaf128-lk4-lpl4, leaf256-lk8-lpl4
  (this run predates the `tag` field).
- `c64-leaders-alpha.jsonl`: C = 64, leaf / local_k / leaders-per-leaf / alpha / degree sweep (tagged).
- `c64-reachability-seeds.jsonl`: C = 64, reachability from the entry and searches seeded with 0/16/64 rows.
- `auto-leaf.jsonl`: leaf = rows/8 within [64, 1024], local_k 2 vs 4, at C = 64 and 1024.
