#!/usr/bin/env bash
# Step 1 of the sharded-design study: D1 (fastpipnn / HNSW shards), D6 (LanceDB-style IVF:
# k-means + router + RaBitQ segments) and D5 (PiPNN-carved centroids), plus the router study.
# Usage: native/scripts/run_shards.sh [data_dir] [out_dir] [threads] [sections]
#   sections: any of "graph ivf router" (default: all three)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-results/synthetic1m}; OUT=${2:-results/shards-synthetic1m}; T=${3:-4}; SECTIONS=${4:-"graph ivf router"}
mkdir -p "$OUT"; R="$OUT/results.jsonl"; LOG="$OUT/log.txt"
B="./build/shard_bench --base $DATA/base.u8bin --query $DATA/query.u8bin --gt $DATA/gt.ibin --dataset $(basename $DATA) --threads $T"
if [[ $SECTIONS == *graph* ]]; then
  for index in fastpipnn hnsw; do
    for s in 1 8 16 32; do $B --output $R --design graph --index $index --shards $s --beams 10,16,32,64,128,256 2>>$LOG; done
  done
fi
if [[ $SECTIONS == *ivf* ]]; then
  for p in kmeans carve; do
    $B --output $R --design ivf --partition $p --lists 1024 --routers exact,hnsw,fastpipnn --shards 1,8,16,32 \
       --nprobes 1,2,4,8,16,32,64 --refines 10,40,100 2>>$LOG
    $B --output $R --design ivf --partition $p --lists 4096 --routers exact,hnsw,fastpipnn --shards 1,8,32 \
       --nprobes 2,4,8,16,32,64,128,256 --refines 10,40,100 2>>$LOG
  done
  # Per-segment rerank (k*refine reranked in every segment): recall then grows with S.
  $B --output $OUT/segment-rerank.jsonl --design ivf --partition kmeans --lists 1024 --routers exact --rerank segment \
     --shards 1,8,32 --nprobes 4,8,16,32 --refines 10 2>>$LOG
fi
if [[ $SECTIONS == *router* ]]; then
  for c in 1024 4096 16384 65536; do
    $B --output $R --design router --partition carve --lists $c --nprobes 8,32 --router-beams 16,32,64,128,256 2>>$LOG
  done
  $B --output $R --design router --partition kmeans --lists 4096 --nprobes 8,32 --router-beams 16,32,64,128,256 2>>$LOG
fi
echo done >> $LOG
