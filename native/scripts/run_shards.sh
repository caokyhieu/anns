#!/usr/bin/env bash
# Step 1 of the sharded-design study: D1 (fastpipnn / HNSW shards), D6 (LanceDB-style IVF:
# k-means + router + RaBitQ segments) and D5 (PiPNN-carved centroids), plus the router study.
# Usage: native/scripts/run_shards.sh [data_dir] [out_dir] [threads]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-results/synthetic1m}; OUT=${2:-results/shards-synthetic1m}; T=${3:-4}
mkdir -p "$OUT"; R="$OUT/results.jsonl"; : > "$R"
B="./build/shard_bench --base $DATA/base.u8bin --query $DATA/query.u8bin --gt $DATA/gt.ibin --dataset $(basename $DATA) --threads $T --output $R"
for index in fastpipnn hnsw; do
  for s in 1 8 16 32; do $B --design graph --index $index --shards $s --beams 10,16,32,64,128,256 2>>"$OUT/log.txt"; done
done
$B --design ivf --partition kmeans --lists 1024 --routers exact,hnsw --shards 1,8,16,32 --nprobes 1,2,4,8,16,32,64 2>>"$OUT/log.txt"
$B --design ivf --partition carve  --lists 1024 --routers exact,fastpipnn --shards 1,8,16,32 --nprobes 1,2,4,8,16,32,64 2>>"$OUT/log.txt"
$B --design ivf --partition kmeans --lists 4096 --routers exact,hnsw,fastpipnn --shards 1,8,32 --nprobes 2,4,8,16,32,64,128,256 2>>"$OUT/log.txt"
$B --design ivf --partition carve  --lists 4096 --routers exact,hnsw,fastpipnn --shards 1,8,32 --nprobes 2,4,8,16,32,64,128,256 2>>"$OUT/log.txt"
# Rerank budget at S=1 (separates the refine effect from sharding).
$B --design ivf --partition kmeans --lists 1024 --routers exact --shards 1 --refines 40 --nprobes 4,8,16,32 2>>"$OUT/log.txt"
$B --design ivf --partition carve  --lists 1024 --routers exact --shards 1 --refines 40 --nprobes 4,8,16,32 2>>"$OUT/log.txt"
for c in 1024 4096 16384 65536; do
  $B --design router --partition carve --lists $c --nprobes 8,32 --router-beams 16,32,64,128,256 2>>"$OUT/log.txt"
done
$B --design router --partition kmeans --lists 4096 --nprobes 8,32 --router-beams 16,32,64,128,256 2>>"$OUT/log.txt"
echo done >> "$OUT/log.txt"
