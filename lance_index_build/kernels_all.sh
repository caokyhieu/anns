#!/usr/bin/env bash
# ARM kernels: legacy Lance vs the new kernels, measured at three levels, one session.
#
#   1. micro   criterion kernel microbenchmarks, SVE build vs build without SVE objects
#   2. stage   standalone IVF_RQ stages (k-means, quantizer, assignment, residual + RQ
#              encode, query probe) calling Lance's own stage functions, once per arm
#              (scripts/arms.sh: legacy, blocked, sve, new); no index is written
#   3. e2e     the full Lance IVF_RQ build + search, legacy vs new: Lance's own stage
#              breakdown (train_ivf, train_quantizer, shuffle, merge_partitions, ...)
#
# Then scripts/kernel_report.py writes report.md: speedups per kernel, per stage and end
# to end, accuracy against legacy, and the stage speedups projected onto the legacy
# build's breakdown.
#
#   ./lance_index_build/kernels_all.sh                       everything, new session
#   CASES=gist ./lance_index_build/kernels_all.sh            only GIST1M
#   ./lance_index_build/kernels_all.sh stage-gist-legacy stage-gist-new   named steps
#   TAG=20261112T090000Z ./lance_index_build/kernels_all.sh  resume: passed steps skipped
#   DRY_RUN=1 ./lance_index_build/kernels_all.sh             print the plan only
#   STEP_TIMEOUT=3h ...                                      fail a step after 3 h
#
# CASES (default "gist bigann10m") picks configs/kernels-<case>.env; a case whose data
# is missing is skipped. E2E_ARMS (default "legacy new") picks the end-to-end arms.
# Data first: ./lance_index_build/bench.sh --data gist,bigann data
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WT="$HERE/worktrees/lance"
ARCH="$(uname -m)"
HOST="$(hostname -s)"
: "${ANNS_DATA:=$ROOT/data}"
: "${RESULTS_DIR:=$HERE/results/$ARCH-$HOST}"
: "${TAG:=$(date -u +%Y%m%dT%H%M%SZ)}"
: "${STEP_TIMEOUT:=0}"
: "${DRY_RUN:=0}"
: "${CASES:=gist bigann10m}"
: "${E2E_ARMS:=legacy new}"
export ANNS_DATA RESULTS_DIR
# shellcheck source=scripts/arms.sh
source "$HERE/scripts/arms.sh"
STATE="$RESULTS_DIR/armk-$TAG"
ARMS="$(arms_for_host)"

case_data() {
  case "$1" in
    gist) echo "$ANNS_DATA/base.f32 $ANNS_DATA/query.f32 $ANNS_DATA/gt.i32 $ANNS_DATA/meta.txt" ;;
    bigann10m) echo "$ANNS_DATA/bigann10m/base.10M.u8bin $ANNS_DATA/bigann10m/query.u8bin $ANNS_DATA/bigann10m/gt.10M.ibin" ;;
  esac
}

# Steps: name|kind|case|arm. Stage runs first (legacy before the other arms of a
# case: they reuse its centroids), then the end-to-end builds.
STEPS=("test|bench||" "micro|bench||")
for c in $CASES; do
  [ -f "$HERE/configs/kernels-$c.env" ] || { echo "no config configs/kernels-$c.env" >&2; exit 2; }
  for a in $ARMS; do STEPS+=("stage-$c-$a|stage|$c|$a"); done
done
for c in $CASES; do
  for a in $E2E_ARMS; do STEPS+=("e2e-$c-$a|e2e|$c|$a"); done
done
names=()
for s in "${STEPS[@]}"; do names+=("${s%%|*}"); done
for want in "$@"; do
  [[ " ${names[*]} " == *" $want "* ]] || { echo "unknown step: $want (steps: ${names[*]})" >&2; exit 2; }
done

log() { printf '\n==> [%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
status_of() { [ -f "$STATE/status.tsv" ] && awk -F'\t' -v n="$1" '$1==n{s=$2} END{print s}' "$STATE/status.tsv"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$STATE/status.tsv"; }

[ "$DRY_RUN" = 1 ] || mkdir -p "$STATE"
log "ARM kernel session $TAG on $ARCH ($ARMS) -> $STATE"

for s in "${STEPS[@]}"; do
  IFS='|' read -r name kind case arm <<<"$s"
  if [ $# -gt 0 ] && [[ " $* " != *" $name "* ]]; then continue; fi
  if [ "$DRY_RUN" != 1 ] && [ "$(status_of "$name")" = ok ]; then
    log "$name: passed earlier in this session, skipping"; continue
  fi
  if [ -n "$case" ]; then
    missing=""
    for f in $(case_data "$case"); do [ -f "$f" ] || missing="$f"; done
    if [ -n "$missing" ]; then
      log "$name: skipped, missing $missing (./lance_index_build/bench.sh data)"
      [ "$DRY_RUN" = 1 ] || record "$name" "skipped (no data)" "$(date -u +%FT%TZ)" -
      continue
    fi
  fi
  config="configs/kernels-$case.env"
  out=""
  case $kind in
    bench)
      cmd=("$HERE/bench.sh" "$([ "$name" = micro ] && echo kernels || echo test)") ;;
    stage)
      out="$STATE/stages/$case/$arm"
      # shellcheck disable=SC2207
      kv=($(arm_env "$arm") "ARM=$arm")
      legacy="$STATE/stages/$case/legacy"
      if [ "$arm" != legacy ] && [ -f "$legacy/centroids.f32" ]; then
        kv+=("CENTROIDS_FROM=$legacy/centroids.f32" "REF_DIR=$legacy")
      fi
      cmd=("$HERE/scripts/stages.sh" "$config" "$out" "${kv[@]}") ;;
    e2e)
      out="$RESULTS_DIR/krn-$case-$arm-$TAG"
      # shellcheck disable=SC2207
      kv=($(arm_env "$arm"))
      cmd=("$HERE/scripts/run.sh" "$config" "krn-$case-$arm-$TAG" ${kv[@]+"${kv[@]}"}) ;;
  esac
  if [ "$STEP_TIMEOUT" != 0 ]; then cmd=(timeout --kill-after=120 "$STEP_TIMEOUT" "${cmd[@]}"); fi
  log "$name: ${cmd[*]}"
  [ "$DRY_RUN" = 1 ] && continue
  # A retried run gets a fresh directory; the failed attempt is kept beside it.
  if [ -n "$out" ] && [ -e "$out" ]; then
    mv "$out" "$out.failed-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  record "$name" running "$(date -u +%FT%TZ)" -
  start=$SECONDS
  (cd "$HERE" && "${cmd[@]}") > >(tee "$STATE/$name.log") 2>&1
  rc=$?
  secs=$((SECONDS - start))
  why=""
  if [ $rc -eq 0 ] && [ "$kind" = e2e ]; then
    # The benchmark records a failed build in builds.jsonl and still exits 0.
    if [ ! -s "$out/builds.jsonl" ]; then
      rc=1; why="no build recorded"
    elif grep -q '"status": *"error"' "$out/builds.jsonl"; then
      rc=1; why="build error: $(grep -m1 -o '"error": *"[^"]*' "$out/builds.jsonl" | sed 's/"error": *"//' | cut -c1-160)"
    fi
    mkdir -p "$STATE/e2e/$case"
    ln -sfn "$out" "$STATE/e2e/$case/$arm"
  fi
  if [ $rc -eq 0 ] && [ "$kind" = stage ] && grep -q '"status":"error"' "$out/stages.jsonl"; then
    rc=1; why="stage error: $(grep -m1 -o '"error":"[^"]*' "$out/stages.jsonl" | cut -c10-170)"
  fi
  if [ $rc -eq 0 ] && [ "$name" = micro ]; then
    for t in target target-nosve; do
      [ -d "$WT/$t/criterion" ] && rm -rf "$STATE/criterion-$t" && cp -r "$WT/$t/criterion" "$STATE/criterion-$t"
    done
  fi
  if [ $rc -eq 0 ]; then
    record "$name" ok "$(date -u +%FT%TZ)" "${secs}s"
    log "$name: ok in $((secs / 60)) min"
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

python3 "$HERE/scripts/kernel_report.py" "$STATE" > "$STATE/report.log" 2>&1 \
  && log "report: $STATE/report.md" || log "report failed (see $STATE/report.log)"
for d in "$STATE"/e2e/*/*; do
  [ -d "$d" ] && python3 "$HERE/scripts/report.py" "$d" -o "$d" >/dev/null 2>&1
done
awk -F'\t' '{s[$1]=$2; if (!($1 in o)) {o[$1]=++n; k[n]=$1}} END {for (i=1; i<=n; i++) printf "  %-24s %s\n", k[i], s[k[i]]}' "$STATE/status.tsv"
if awk -F'\t' '{s[$1]=$2} END {for (k in s) if (s[k] != "ok" && s[k] !~ /^skipped/) bad=1; exit !bad}' "$STATE/status.tsv"; then
  echo "some steps failed; rerun the failed ones with: TAG=$TAG $0" >&2
  exit 1
fi
