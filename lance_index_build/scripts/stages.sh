#!/usr/bin/env bash
# Standalone IVF_RQ stage benchmark (lance-index-bench `stages` binary) for one kernel arm.
# Usage: scripts/stages.sh configs/<name>.env OUT_DIR [KEY=VALUE ...]
# The config is the one scripts/run.sh uses for the end-to-end build; STAGE_ASSIGN
# and STAGE_REPEATS (default 3) apply here. KEY=VALUE overrides come last, e.g. the
# arm settings from scripts/arms.sh plus ARM=<name>, CENTROIDS_FROM, REF_DIR.
# HW_CPUS pins the run to that many CPUs and sizes rayon to match.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WT="$HERE/worktrees/lance"
CONFIG="$1"
OUT="$2"
shift 2
: "${ANNS_DATA:=$ROOT/data}"
: "${DATASETS_DIR:=$HERE/datasets}"
export ANNS_DATA DATASETS_DIR
"$HERE/scripts/setup_lance.sh" >/dev/null
set -a
# shellcheck disable=SC1090
source "$CONFIG"
for kv in "$@"; do export "$kv"; done
set +a
if [ -e "$OUT/stages.jsonl" ]; then
  echo "$OUT already holds a stage run; choose another directory" >&2
  exit 1
fi
mkdir -p "$OUT/logs"
cp "$CONFIG" "$OUT/config.env"
printf '%s\n' "$@" > "$OUT/overrides.env"
export ASSIGN="${STAGE_ASSIGN:-auto,exact}"
export GIT_REV="$(git -C "$WT" rev-parse HEAD)"
export PATCH_SHA256="$(cat "$HERE"/patches/*.patch | sha256sum | cut -d' ' -f1)"
: "${EIGEN_INCLUDE_DIR:=$ROOT/native/third_party/eigen}"
export EIGEN_INCLUDE_DIR
FEATURES="${FEATURES-pipnn}"   # FEATURES= builds without PiPNN (no Eigen needed)
if [ -n "${LANCE_DISABLE_SVE_BUILD:-}" ]; then
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WT/target-nosve}"
fi
TARGET="${CARGO_TARGET_DIR:-$WT/target}"
(cd "$WT" && cargo build --profile release-with-debug -p lance-index-bench --features "$FEATURES" --bin stages) \
  2>&1 | tail -3 | tee "$OUT/logs/build.log"
BIN="$TARGET/release-with-debug/stages"
{ lscpu; echo; free -b; echo; uname -a; } > "$OUT/logs/machine.txt" 2>/dev/null || true
# SVE vector length in bytes (the "equal width" in blocked-vs-pairwise comparisons).
cat /proc/sys/abi/sve_default_vector_length > "$OUT/logs/sve_vector_bytes.txt" 2>/dev/null || true
LIMIT=()
if [ -n "${HW_CPUS:-}" ]; then
  CPUSET="$(python3 -c 'import os,sys; c=sorted(os.sched_getaffinity(0)); n=int(sys.argv[1])
if n > len(c): sys.exit(f"HW_CPUS={n} but only {len(c)} CPUs are available")
print(",".join(map(str, c[:n])))' "$HW_CPUS")"
  export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$HW_CPUS}" PIPNN_THREADS="${PIPNN_THREADS:-$HW_CPUS}"
  LIMIT=(taskset -c "$CPUSET")
fi
export OUT_DIR="$OUT"
for ((r = 0; r < ${STAGE_REPEATS:-3}; r++)); do
  echo "== ${ARM:-unlabeled} repeat $r -> $OUT"
  REPEAT=$r ${LIMIT[@]+"${LIMIT[@]}"} "$BIN" 2>&1 | tee -a "$OUT/logs/run.log"
done
