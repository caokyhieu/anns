#!/usr/bin/env bash
# Run one versioned experiment config with revision/patch provenance.
# Usage: scripts/run.sh configs/<name>.env [RUN_ID] [KEY=VALUE ...]
# KEY=VALUE overrides are applied after the config and recorded in overrides.env.
#
# Paths (override in the environment):
#   ANNS_DATA     raw datasets (default <repo>/data)
#   DATASETS_DIR  ingested Lance datasets, reused across runs (default lance_index_build/datasets)
#   RESULTS_DIR   run outputs (default lance_index_build/results/<arch>-<host>)
# Results of different machines never share a directory.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WT="$HERE/worktrees/lance"
CONFIG="$1"
ARCH="$(uname -m)"
HOST="$(hostname -s)"
RUN_ID="${2:-$(basename "$CONFIG" .env)-$(date -u +%Y%m%dT%H%M%SZ)}"
: "${ANNS_DATA:=$ROOT/data}"
: "${DATASETS_DIR:=$HERE/datasets}"
: "${RESULTS_DIR:=$HERE/results/$ARCH-$HOST}"
export ANNS_DATA DATASETS_DIR
"$HERE/scripts/setup_lance.sh" >/dev/null
set -a
# shellcheck disable=SC1090
source "$CONFIG"
shift; [ $# -gt 0 ] && shift
OVERRIDES=("$@")
for kv in "${OVERRIDES[@]}"; do export "$kv"; done
set +a
# Lance keeps f32 L2/dot on its 16-lane scalar kernels by default (other code
# reproduces them bit for bit); experiments measure the SVE kernels unless a
# config or override sets LANCE_F32_SVE=0. The manifest records the value.
if [ "$ARCH" = aarch64 ]; then
  export LANCE_F32_SVE="${LANCE_F32_SVE:-1}"
fi
: "${EIGEN_INCLUDE_DIR:=$ROOT/native/third_party/eigen}"
if [ ! -f "$EIGEN_INCLUDE_DIR/Eigen/Dense" ]; then
  echo "Eigen headers missing at $EIGEN_INCLUDE_DIR; run: ./lance_index_build/bench.sh setup" >&2
  exit 1
fi
export EIGEN_INCLUDE_DIR
export RUN_DIR="${RUN_DIR:-$RESULTS_DIR/$RUN_ID}"
if [ -e "$RUN_DIR/builds.jsonl" ]; then
  echo "$RUN_DIR already holds a run; choose another RUN_ID" >&2
  exit 1
fi
export GIT_REV="$(git -C "$WT" rev-parse HEAD)"
export LANCE_TREE="$(git -C "$WT" rev-parse 'HEAD^{tree}')"
export PATCH_SHA256="$(cat "$HERE"/patches/*.patch | sha256sum | cut -d' ' -f1)"
FEATURES="${FEATURES:-pipnn}"
# Build-time ablation: objects without SVE need their own target directory.
if [ -n "${LANCE_DISABLE_SVE_BUILD:-}" ]; then
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WT/target-nosve}"
fi
TARGET="${CARGO_TARGET_DIR:-$WT/target}"
mkdir -p "$RUN_DIR/logs"
cp "$CONFIG" "$RUN_DIR/config.env"
printf '%s\n' "${OVERRIDES[@]}" > "$RUN_DIR/overrides.env"
{ date -u +%FT%TZ; uptime; who | wc -l; } > "$RUN_DIR/logs/load-start.txt"
(cd "$WT" && cargo build --profile release-with-debug -p lance-index-bench --features "$FEATURES") \
  2>&1 | tail -3 | tee "$RUN_DIR/logs/build.log"
BIN="$TARGET/release-with-debug/lance-index-bench"
{ lscpu; echo; free -b; echo; uname -a; echo; numactl -H 2>/dev/null || true; \
  echo; (cd "$WT" && rustc --version); cc --version | head -1; } > "$RUN_DIR/logs/machine.txt"
"$BIN" 2>&1 | tee -a "$RUN_DIR/logs/run.log"
