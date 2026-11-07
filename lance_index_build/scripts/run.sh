#!/usr/bin/env bash
# Run one versioned experiment config with revision/patch provenance.
# Usage: scripts/run.sh configs/<name>.env [RUN_ID]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WT="$HERE/worktrees/lance"
CONFIG="$1"
RUN_ID="${2:-$(basename "$CONFIG" .env)-$(date -u +%Y%m%dT%H%M%SZ)}"
set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a
: "${EIGEN_INCLUDE_DIR:=$HERE/../native/third_party/eigen}"
export EIGEN_INCLUDE_DIR
export RUN_DIR="${RUN_DIR:-$HERE/results/$RUN_ID}"
export GIT_REV="$(git -C "$WT" rev-parse HEAD)"
export PATCH_SHA256="$( (git -C "$WT" diff HEAD; git -C "$WT" ls-files --others --exclude-standard -z | xargs -0 cat) | sha256sum | cut -d' ' -f1)"
FEATURES="${FEATURES:-pipnn}"
mkdir -p "$RUN_DIR/logs"
cp "$CONFIG" "$RUN_DIR/config.env"
(cd "$WT" && cargo build --profile release-with-debug -p lance-index-bench --features "$FEATURES")
BIN="$WT/target/release-with-debug/lance-index-bench"
{ lscpu; echo; free -b; echo; uname -a; echo; numactl -H 2>/dev/null || true; } > "$RUN_DIR/logs/machine.txt"
"$BIN" 2>&1 | tee -a "$RUN_DIR/logs/run.log"
