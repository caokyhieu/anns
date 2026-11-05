#!/usr/bin/env bash
# Baseline vs optimized Lance IVF_RQ builds on GIST1M, using one binary.
#   baseline      : LANCE_DISABLE_BLOCKED_L2=1 (the unmodified upstream code path)
#   blocked       : blocked f32 L2 nearest-centroid kernel (default on with the patch)
#   blocked+const : also LANCE_RQ_CONST_EX_SCALE=1 (multi-bit codes only)
#
# Required: LANCE_DIR (patched Lance checkout), DATA_DIR (output of convert_gist.py)
# Optional: OUT_DIR (default ./results), REPS (2), BITS ("1 5 9"),
#           NUM_PARTITIONS (1000), NPROBES ("10,20,50,100")
set -eu
: "${LANCE_DIR:?set LANCE_DIR to the patched Lance checkout}"
: "${DATA_DIR:?set DATA_DIR to the directory written by convert_gist.py}"
OUT_DIR=${OUT_DIR:-$PWD/results}
mkdir -p "$OUT_DIR"
BIN=$(ls -t "$LANCE_DIR"/target/release/deps/ivf_rq_gist-* | grep -v '\.d$' | head -1)
export GIST_DIR=$DATA_DIR DS_PATH=$DATA_DIR/gist.lance OUT_JSON=$OUT_DIR/results.jsonl
export NUM_PARTITIONS=${NUM_PARTITIONS:-1000} NPROBES=${NPROBES:-10,20,50,100}
REPS=${REPS:-2}
BITS=${BITS:-"1 5 9"}
for rep in $(seq 1 "$REPS"); do
  for b in $BITS; do
    echo "=== rep $rep bits $b baseline"
    LABEL=baseline NUM_BITS=$b LANCE_DISABLE_BLOCKED_L2=1 "$BIN"
    echo "=== rep $rep bits $b blocked"
    LABEL=blocked NUM_BITS=$b "$BIN"
    if [ "$b" -gt 1 ]; then
      echo "=== rep $rep bits $b blocked+const"
      LABEL=blocked+const NUM_BITS=$b LANCE_RQ_CONST_EX_SCALE=1 "$BIN"
    fi
  done
done
