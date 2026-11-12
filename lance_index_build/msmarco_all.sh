#!/usr/bin/env bash
# Every MS MARCO v2.1 benchmark in one go: kernel tests and microbenchmarks, the 2M
# prefix smoke runs, the full-scale tree-assignment runs, the kernel ablations at scale,
# graph-based (PiPNN / HNSW) assignment, and the reports. A step that fails is recorded
# and the next one starts; the exit status is non-zero if any step failed.
#
#   ./lance_index_build/msmarco_all.sh                 all steps, new session
#   ./lance_index_build/msmarco_all.sh tree4 nosve     only the named steps
#   TAG=20261111T120000Z ./lance_index_build/msmarco_all.sh   resume a session:
#        steps that passed are skipped, failed or unfinished ones run again
#   DRY_RUN=1 ./lance_index_build/msmarco_all.sh       print the plan, run nothing
#   STEP_TIMEOUT=6h ...                                kill and fail a step after 6 h
#
# Run as root (the 6 GB limit needs systemd-run scopes), inside tmux or nohup. Needs the
# data first: HF_ENDPOINT=https://hf-mirror.com ./lance_index_build/bench.sh --data msmarco data
# Session state: $RESULTS_DIR/msmarco-all-$TAG/ (status.tsv, one log per step, summary).
# Run directories: $RESULTS_DIR/msm-<step>-$TAG/ (each with its own summary.md report).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
ARCH="$(uname -m)"
HOST="$(hostname -s)"
: "${ANNS_DATA:=$ROOT/data}"
: "${RESULTS_DIR:=$HERE/results/$ARCH-$HOST}"
: "${TAG:=$(date -u +%Y%m%dT%H%M%SZ)}"
: "${STEP_TIMEOUT:=0}"
: "${DRY_RUN:=0}"
export ANNS_DATA RESULTS_DIR
STATE="$RESULTS_DIR/msmarco-all-$TAG"
SCALE=configs/scale-msmarco.env
SMOKE=configs/scale-msmarco2m-smoke.env

# name | where | what
#   bench: a bench.sh command (not dataset specific)
#   run:   <config> [KEY=VALUE ...] through scripts/run.sh, run id msm-<name>-$TAG
#   arm:   like run, aarch64 only (kernel switches that are no-ops elsewhere)
# Order: cheap checks first, then the headline scale runs, then the slower extras.
STEPS=$(cat <<EOF
test            | bench | test
kernels         | bench | kernels
smoke-tree4     | run   | $SMOKE
smoke-pipnn     | run   | $SMOKE INDEX=ivf_pipnn_flat HW_MEM_GB=
smoke-pipnn-scalar | run | $SMOKE INDEX=ivf_pipnn_flat HW_MEM_GB= LANCE_PIPNN_SCALAR=1
smoke-hnsw      | run   | $SMOKE INDEX=ivf_hnsw_flat HW_MEM_GB=
tree4           | run   | $SCALE ASSIGN=tree:4
tree16          | run   | $SCALE ASSIGN=tree:16
tree4-rq6       | run   | $SCALE ASSIGN=tree:4 NUM_BITS=6
noblocked       | run   | $SCALE ASSIGN=tree:4 LANCE_DISABLE_BLOCKED_L2=1
nosve           | arm   | $SCALE ASSIGN=tree:4 LANCE_DISABLE_SVE_BUILD=1
f32scalar       | arm   | $SCALE ASSIGN=tree:4 LANCE_F32_SVE=0
tree4-mem32     | run   | $SCALE ASSIGN=tree:4 HW_MEM_GB=32
pipnn-assign    | run   | $SCALE ASSIGN=pipnn:15:64:256 PIPNN_LOCAL_K=4
pipnn-assign-scalar | run | $SCALE ASSIGN=pipnn:15:64:256 PIPNN_LOCAL_K=4 LANCE_PIPNN_SCALAR=1
hnsw-assign     | run   | $SCALE ASSIGN=hnsw:12:15:32
EOF
)

log() { printf '\n==> [%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
status_of() { [ -f "$STATE/status.tsv" ] && awk -F'\t' -v n="$1" '$1==n{s=$2} END{print s}' "$STATE/status.tsv"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$STATE/status.tsv"; }

mapfile -t ALL < <(printf '%s\n' "$STEPS" | sed 's/[[:space:]]*|[[:space:]]*/|/g')
names=()
for line in "${ALL[@]}"; do names+=("${line%%|*}"); done
for want in "$@"; do
  [[ " ${names[*]} " == *" $want "* ]] || { echo "unknown step: $want (steps: ${names[*]})" >&2; exit 2; }
done

if [ "$DRY_RUN" != 1 ]; then
  for f in base.fbin query.fbin gt.ibin; do
    [ -f "$ANNS_DATA/msmarco-v2.1/$f" ] || {
      echo "missing $ANNS_DATA/msmarco-v2.1/$f; run: ./lance_index_build/bench.sh --data msmarco data" >&2; exit 1; }
  done
  # Without the cgroup limit every scale step would fail at once in run.sh.
  if [ "${HW_MEM_STRICT:-1}" = 1 ] &&
     ! { command -v systemd-run >/dev/null &&
         systemd-run --scope --quiet -p MemoryMax=1G -p MemorySwapMax=0 true 2>/dev/null; }; then
    echo "the 6 GB memory limit needs systemd-run scopes: run as root (or HW_MEM_STRICT=0 to only record it)" >&2
    exit 1
  fi
  mkdir -p "$STATE"
fi
log "MS MARCO session $TAG -> $STATE"

for line in "${ALL[@]}"; do
  IFS='|' read -r name kind what <<<"$line"
  if [ $# -gt 0 ] && [[ " $* " != *" $name "* ]]; then continue; fi
  if [ "$kind" = arm ] && [ "$ARCH" != aarch64 ]; then
    log "$name: skipped (aarch64 only)"; continue
  fi
  if [ "$DRY_RUN" != 1 ] && [ "$(status_of "$name")" = ok ]; then
    log "$name: passed earlier in this session, skipping"; continue
  fi
  run_id="msm-$name-$TAG"
  # shellcheck disable=SC2206
  args=($what)
  if [ "$kind" = bench ]; then
    cmd=("$HERE/bench.sh" "${args[@]}")
  else
    cmd=("$HERE/scripts/run.sh" "${args[0]}" "$run_id" "${args[@]:1}")
  fi
  if [ "$STEP_TIMEOUT" != 0 ]; then cmd=(timeout --kill-after=120 "$STEP_TIMEOUT" "${cmd[@]}"); fi
  log "$name: ${cmd[*]}"
  [ "$DRY_RUN" = 1 ] && continue
  # A retried run gets a fresh directory; the failed attempt is kept beside it.
  if [ "$kind" != bench ] && [ -e "$RESULTS_DIR/$run_id" ]; then
    mv "$RESULTS_DIR/$run_id" "$RESULTS_DIR/$run_id.failed-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  record "$name" running "$(date -u +%FT%TZ)" -
  start=$SECONDS
  (cd "$HERE" && "${cmd[@]}") > >(tee "$STATE/$name.log") 2>&1
  rc=$?
  secs=$((SECONDS - start))
  # The benchmark records a failed build in builds.jsonl and still exits 0.
  why=""
  if [ $rc -eq 0 ] && [ "$kind" != bench ]; then
    builds="$RESULTS_DIR/$run_id/builds.jsonl"
    if [ ! -s "$builds" ]; then
      rc=1; why="no build recorded"
    elif grep -q '"status": *"error"' "$builds"; then
      rc=1; why="build error: $(grep -m1 -o '"error": *"[^"]*' "$builds" | sed 's/"error": *"//' | cut -c1-160)"
    fi
  fi
  if [ $rc -eq 0 ]; then
    record "$name" ok "$(date -u +%FT%TZ)" "${secs}s"
    log "$name: ok in $((secs / 60)) min"
    if [ "$kind" != bench ]; then
      python3 "$HERE/scripts/report.py" "$RESULTS_DIR/$run_id" -o "$RESULTS_DIR/$run_id" \
        >> "$STATE/$name.log" 2>&1 || log "$name: report failed (see $STATE/$name.log)"
    fi
  else
    if [ -z "$why" ]; then
      case $rc in
        124) why="timeout after $STEP_TIMEOUT" ;;
        137) why="killed, likely out of memory (exit 137)" ;;
        *) why="exit $rc" ;;
      esac
    fi
    record "$name" "failed ($why)" "$(date -u +%FT%TZ)" "${secs}s"
    log "$name: FAILED ($why) after $((secs / 60)) min; continuing. Log: $STATE/$name.log"
  fi
done
[ "$DRY_RUN" = 1 ] && exit 0

# Final state per step (last record wins).
{
  printf '# MS MARCO session %s on %s (%s)\n\n| step | status | duration | finished |\n|---|---|---|---|\n' "$TAG" "$HOST" "$ARCH"
  awk -F'\t' '{s[$1]=$2; t[$1]=$3; d[$1]=$4; if (!($1 in o)) {o[$1]=++n; k[n]=$1}}
    END {for (i=1; i<=n; i++) printf "| %s | %s | %s | %s |\n", k[i], s[k[i]], d[k[i]], t[k[i]]}' "$STATE/status.tsv"
  printf '\nRun reports: %s/msm-<step>-%s/summary.md\n' "$RESULTS_DIR" "$TAG"
} > "$STATE/summary.md"
log "summary: $STATE/summary.md"
cat "$STATE/summary.md"
if awk -F'\t' '{s[$1]=$2} END {for (k in s) if (s[k] != "ok") bad=1; exit !bad}' "$STATE/status.tsv"; then
  echo "some steps failed; rerun the failed ones with: TAG=$TAG $0" >&2
  exit 1
fi
