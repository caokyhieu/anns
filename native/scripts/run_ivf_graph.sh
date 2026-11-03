#!/usr/bin/env bash
# IVF with graphs for assignment and inside partitions: LanceDB's HNSW vs fastpipnn (IVF_GRAPH.md).
# Usage: native/scripts/run_ivf_graph.sh [data_dir] [out_dir] [threads]
#   Few partitions (64, 256): the sub-index matters, Lance would assign flat (C*d < 10^6).
#   Many partitions (1024, 4096): both the assignment graph and the sub-index matter.
#   16384 partitions: assignment only (partitions of ~60 rows get no graph).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-results/synthetic1m}; OUT=${2:-results/ivf-graph-synthetic1m}; T=${3:-4}
mkdir -p "$OUT"; R="$OUT/results.jsonl"; LOG="$OUT/log.txt"
B="./build/ivf_graph_bench --base $DATA/base.u8bin --query $DATA/query.u8bin --gt $DATA/gt.ibin --dataset $(basename $DATA) --threads $T --output $R"
# fastpipnn inside partitions: leaf 128, local_k 8, depth-0 leaders = 4 * rows / leaf (tuned in IVF_GRAPH.md §3).
F="--leaf-size 128 --local-k 8 --leaders-per-leaf 4"
EFS=10,20,40,80,160
$B $F --lists 64   --assign flat --sub flat,hnsw,fastpipnn --nprobes 1,2,4,8 --efs $EFS 2>>$LOG
$B $F --lists 256  --assign flat --sub flat,hnsw,fastpipnn --nprobes 1,2,4,8,16 --efs $EFS 2>>$LOG
$B $F --lists 1024 --assign flat,hnsw:15,fastpipnn:15,fastpipnn:32 --sub flat,hnsw,fastpipnn --nprobes 1,2,4,8,16,32,64 --efs $EFS 2>>$LOG
$B $F --lists 4096 --assign flat,hnsw:15,fastpipnn:15,fastpipnn:32 --sub flat,hnsw,fastpipnn --nprobes 2,4,8,16,32,64,128 --efs $EFS 2>>$LOG
$B $F --lists 16384 --assign flat,hnsw:15,fastpipnn:15,fastpipnn:32 --sub flat --nprobes 4,8,16,32,64,128,256 2>>$LOG
echo done >> $LOG
