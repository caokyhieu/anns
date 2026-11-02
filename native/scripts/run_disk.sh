#!/usr/bin/env bash
# On-disk graph study: PiPNN (fastpipnn) vs HNSW level 0, same DiskANN-style layout and search.
# Usage: native/scripts/run_disk.sh [data_dir] [out_dir] [work_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-results/synthetic1m}; OUT=${2:-results/disk-synthetic1m}; WORK=${3:-../data/disk-synthetic1m}
mkdir -p "$OUT" "$WORK"; R="$OUT/results.jsonl"; LOG="$OUT/log.txt"
B="./build/disk_bench --base $DATA/base.u8bin --query $DATA/query.u8bin --gt $DATA/gt.ibin --dataset $(basename $DATA) --work $WORK --output $R --query-threads 4"
BEAMS=16,32,64,128,256
for g in fastpipnn hnsw; do
  # Layouts x page expansion, PQ 64 B, O_DIRECT reads. The first run builds and saves the graph.
  $B --graph $g --pq-m 64 --orders none,bfs,carve --io direct --page-expand 0,1 --beams $BEAMS 2>>$LOG
  # Same engine with the node file in RAM (no I/O): separates I/O time from search work.
  $B --graph $g --pq-m 64 --orders bfs --io ram --page-expand 0,1 --beams $BEAMS --ram-reference 0 2>>$LOG
  # Smaller PQ codes (32 B) and a 1% hot-node cache.
  $B --graph $g --pq-m 32 --orders bfs --io direct --page-expand 0,1 --beams $BEAMS --ram-reference 0 2>>$LOG
  $B --graph $g --pq-m 64 --orders bfs --io direct --page-expand 1 --cache-nodes 10000 --beams $BEAMS --ram-reference 0 2>>$LOG
done
echo done >> $LOG
