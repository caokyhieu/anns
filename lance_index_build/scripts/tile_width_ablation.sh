#!/usr/bin/env bash
# Blocked (4x4 tile) vs per-vector exact assignment at EQUAL SIMD width.
#
# The earlier GIST result compared an AVX-512 tile with Lance's 256-bit per-vector
# batch kernel, so tiling and vector width were confounded. This runs a 2x2 design:
#
#   arm   assignment   width  environment
#   P256  per-vector   256    LANCE_SIMD_MAX_WIDTH=256 LANCE_DISABLE_BLOCKED_L2=1
#   B256  blocked      256    LANCE_SIMD_MAX_WIDTH=256
#   P512  per-vector   512    LANCE_L2_BATCH_AVX512=1 LANCE_DISABLE_BLOCKED_L2=1
#   B512  blocked      512    LANCE_L2_BATCH_AVX512=1
#   REF   per-vector   mixed  LANCE_DISABLE_BLOCKED_L2=1   (upstream default; micro only)
#
# Valid comparisons: B256 vs P256 and B512 vs P512. REF vs B512 reproduces the old,
# confounded comparison. Arms are interleaved and rotated across rounds.
#
# Usage: scripts/tile_width_ablation.sh micro|e2e [ROUNDS]
#   micro  kernel-only, Lance's compute_partitions_arrow_array on real rows
#          BIGANN (D=128) K = 1024, 4096, 16384; GIST (D=960) K = 64, 128, 256
#   e2e    full IVF_RQ builds through scripts/run.sh
#          BIGANN10M K=16384 (fixed centroids, exact assignment); GIST1M K=256 (complete)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WT="$HERE/worktrees/lance"
MODE="${1:?micro|e2e}"
ROUNDS="${2:-}"
: "${ANNS_DATA:=$ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$HERE/results/$(uname -m)-$(hostname -s)}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

declare -A ARM_ENV=(
  [P256]="LANCE_SIMD_MAX_WIDTH=256 LANCE_DISABLE_BLOCKED_L2=1"
  [B256]="LANCE_SIMD_MAX_WIDTH=256"
  [P512]="LANCE_L2_BATCH_AVX512=1 LANCE_DISABLE_BLOCKED_L2=1"
  [B512]="LANCE_L2_BATCH_AVX512=1"
  [REF]="LANCE_DISABLE_BLOCKED_L2=1"
)
rotate() {  # rotate <round> <arms...>: start at a different arm each round
  local r=$1; shift; local a=("$@") n=$#
  for ((i = 0; i < n; i++)); do echo "${a[$(((i + r) % n))]}"; done
}

if [ "$MODE" = micro ]; then
  ROUNDS="${ROUNDS:-2}"
  OUT="$RESULTS_DIR/tilewidth-micro-$STAMP"
  mkdir -p "$OUT/assign"
  BIN="$WT/target/release-with-debug/assign_micro"
  [ -x "$BIN" ] || { echo "build first: cargo build --profile release-with-debug -p lance-index-bench" >&2; exit 1; }
  { lscpu; echo; uptime; git -C "$WT" rev-parse HEAD; git -C "$WT" status --short; } > "$OUT/machine.txt"
  # name threads cpu-pin dataset data_dir rows ks reps
  CASES=(
    "bigann-t32 32 - bigann $ANNS_DATA/bigann10m 500000 1024,4096,16384 3"
    "gist-t32   32 - gist   $ANNS_DATA           200000 64,128,256     3"
    "bigann-t1  1  2 bigann $ANNS_DATA/bigann10m 20000  1024,4096,16384 3"
    "gist-t1    1  2 gist   $ANNS_DATA           10000  64,128,256     3"
  )
  for ((round = 0; round < ROUNDS; round++)); do
    for c in "${CASES[@]}"; do
      read -r name threads pin ds dir rows ks reps <<< "$c"
      for arm in $(rotate "$round" P256 B256 P512 B512 REF); do
        echo "[round $round] $name $arm" >&2
        pincmd=(); [ "$pin" != - ] && pincmd=(taskset -c "$pin")
        env ${ARM_ENV[$arm]} RAYON_NUM_THREADS="$threads" "${pincmd[@]}" \
          "$BIN" "$ds" "$dir" "$rows" "$ks" "$reps" "$OUT/assign/$name-$arm-r$round" \
          | sed "s/^{/{\"arm\":\"$arm\",\"case\":\"$name\",\"round\":$round,\"load\":\"$(cut -d' ' -f1 /proc/loadavg)\",/" \
          >> "$OUT/micro.jsonl"
      done
    done
  done
  echo "wrote $OUT/micro.jsonl"
elif [ "$MODE" = e2e ]; then
  ROUNDS="${ROUNDS:-3}"
  for ((round = 0; round < ROUNDS; round++)); do
    for arm in $(rotate "$round" P256 B256 P512 B512); do
      # shellcheck disable=SC2086
      "$HERE/scripts/run.sh" "$HERE/configs/suiteB-bigann10m-k16384-fixed.env" "tw-$STAMP-bigann-k16384-$arm-r$round" \
        ASSIGN=exact REPEATS=1 NPROBES=16,64 KS=10 ${ARM_ENV[$arm]}
      "$HERE/scripts/run.sh" "$HERE/configs/suiteA-gist-k1000.env" "tw-$STAMP-gist-k256-$arm-r$round" \
        NUM_PARTITIONS=256 ASSIGN=exact REPEATS=1 NPROBES=4,16 KS=10 ${ARM_ENV[$arm]}
    done
  done
  echo "runs under $RESULTS_DIR/tw-$STAMP-*"
else
  echo "usage: $0 micro|e2e [ROUNDS]" >&2; exit 1
fi
