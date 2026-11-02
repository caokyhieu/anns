#!/usr/bin/env bash
# HNSW on disk with its upper layers in RAM choosing the level-0 entry (fair HNSW-on-disk),
# plus the medoid-entry baseline on the same graph. Rebuilds the HNSW graph.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-results/synthetic1m}; OUT=${2:-results/disk-synthetic1m}; WORK=${3:-../data/disk-synthetic1m}
R="$OUT/results-entry.jsonl"; LOG="$OUT/log-entry.txt"
B="./build/disk_bench --base $DATA/base.u8bin --query $DATA/query.u8bin --gt $DATA/gt.ibin --dataset $(basename $DATA) --work $WORK --output $R --query-threads 4 --beams 16,32,64,128,256 --pq-m 64 --io direct"
rm -f $WORK/hnsw.graph $WORK/hnsw.upper
$B --graph hnsw --entry upper --orders none,bfs,carve --page-expand 0,1 2>>$LOG
$B --graph hnsw --entry medoid --orders bfs --page-expand 0,1 --known-build-seconds 0 2>>$LOG
echo done >> $LOG
