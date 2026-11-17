#!/usr/bin/env bash
# Every Kunpeng ws2/5/6 session, detached: GIST1M (all steps), then BIGANN10M (doctor,
# stage, search: test and micro do not depend on the dataset), then summary.md for both
# and one results tarball. Survives SSH logout.
#
#   ./lance_index_build/kunpeng_ws256_all.sh            start in the background
#   tail -f lance_index_build/results/ws256-all.log     follow progress
#
# GIST_TAG / BIGANN_TAG resume earlier sessions (finished steps are skipped). Other
# environment (HW_CPUS, KML_DIR, KRL_DIR, ...) is passed through to kunpeng_ws256.sh.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/results/ws256-all.log"
mkdir -p "$HERE/results"

if [ "${WS256_FG:-}" != 1 ]; then
  WS256_FG=1 nohup setsid "$0" "$@" >>"$LOG" 2>&1 </dev/null &
  echo "started in background (pid $!); follow with: tail -f $LOG"
  exit 0
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
GIST_TAG="${GIST_TAG:-gist-$STAMP}"
BIGANN_TAG="${BIGANN_TAG:-bigann-$STAMP}"
BASE="$HERE/results/$(uname -m)-$(hostname -s)"
echo "==> [$(date -u +%FT%TZ)] ws256-all: GIST_TAG=$GIST_TAG BIGANN_TAG=$BIGANN_TAG"

rc=0
CASE=gist TAG="$GIST_TAG" "$HERE/kunpeng_ws256.sh" || rc=1
if [ $rc = 0 ]; then
  CASE=bigann10m TAG="$BIGANN_TAG" "$HERE/kunpeng_ws256.sh" doctor stage search || rc=1
fi

dirs=()
for t in "$GIST_TAG" "$BIGANN_TAG"; do
  d="$BASE/kunpeng-ws256-$t"
  [ -d "$d" ] || continue
  python3 "$HERE/scripts/ws256_report.py" "$d" >/dev/null && dirs+=("kunpeng-ws256-$t")
done
if [ ${#dirs[@]} -gt 0 ]; then
  (cd "$BASE" && tar -czf "$HERE/results/ws256-results-$STAMP.tar.gz" \
    $(for d in "${dirs[@]}"; do ls -d "$d"/summary.md "$d"/doctor.txt "$d"/*.jsonl "$d"/*.log "$d"/*.txt 2>/dev/null; done))
  echo "==> results: $HERE/results/ws256-results-$STAMP.tar.gz"
fi
echo "==> [$(date -u +%FT%TZ)] ws256-all finished (rc=$rc)"
exit $rc
