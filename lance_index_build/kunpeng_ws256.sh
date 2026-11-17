#!/usr/bin/env bash
# Kunpeng porting workstreams 2, 5 and 6 (lance_ivf_rq_kunpeng_porting_2_5_6.md), one session.
#
#   ws2  shared FP32 centroid assignment: blocked_l2 backends tile (NEON / SVE 4x4),
#        f32mm (SVE FMMLA 8x4), kml (KML cblas_sgemm), krl / krl_ny (KRL fp32 L2)
#   ws5  SUM_HACC accurate-mode scan: scalar (old ARM path), NEON, SVE, SVE2 (not on 920 V200)
#   ws6  SUM_U16 normal scan: NEON, SVE, KRL FastScan; ex_dot survivor scoring: NEON, SVE
#
# Steps (all by default; name some to run only those):
#   doctor   host flags (sve / sve2 / svef32mm), SVE vector length, KML/KRL installs
#   test     correctness on this host: lance-core dispatch, lance-linalg (every scan
#            variant vs scalar, exact), lance-index blocked_l2 (every backend vs the
#            per-vector reference) and ex_dot (SVE bit-identical to NEON)
#   micro    kunpeng_kernels: every arm side by side in one process (JSON + table)
#   stage    stages.sh train/assign/encode on GIST1M, once per assignment backend; every
#            arm reuses the tile-neon arm's centroids and records agreement with it
#   search   run.sh build + search on GIST1M per scan arm, normal and accurate mode
#
# Environment:
#   KML_DIR (default /usr/local/kml), KRL_DIR (default /usr/local/sra_krl): installs to
#     build the vendor adapters against; missing ones are skipped. KML_LIBDIR / KML_LIBS /
#     KRL_LIBDIR / KRL_LIBS override library discovery (see lance-linalg/build.rs).
#   CASE (default gist) picks configs/kernels-$CASE.env for stage/search.
#   HW_CPUS limits stage/search to that many CPUs (e.g. 160 = no SMT on 2 x 80 cores).
#   TAG resumes a session; finished steps are not rerun.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$HERE/worktrees/lance"
TAG="${TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARCH="$(uname -m)"
HOST="$(hostname -s)"
OUT="${OUT:-$HERE/results/$ARCH-$HOST/kunpeng-ws256-$TAG}"
CASE="${CASE:-gist}"
KML_DIR="${KML_DIR:-/usr/local/kml}"
KRL_DIR="${KRL_DIR:-/usr/local/sra_krl}"
mkdir -p "$OUT"
log() { printf '\n==> [%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/session.log"; }
done_step() { [ -f "$OUT/.done-$1" ]; }
mark() { touch "$OUT/.done-$1"; }

# Vendor adapters: only for installs that exist. Their libraries must be on the loader
# path at run time too.
VENDOR=()
if [ -f "$KML_DIR/include/kblas.h" ]; then
  VENDOR+=("LANCE_KML_DIR=$KML_DIR")
  [ -n "${KML_LIBDIR:-}" ] && VENDOR+=("LANCE_KML_LIBDIR=$KML_LIBDIR")
  [ -n "${KML_LIBS:-}" ] && VENDOR+=("LANCE_KML_LIBS=$KML_LIBS")
fi
if [ -f "$KRL_DIR/include/krl.h" ]; then
  VENDOR+=("LANCE_KRL_DIR=$KRL_DIR")
  [ -n "${KRL_LIBDIR:-}" ] && VENDOR+=("LANCE_KRL_LIBDIR=$KRL_LIBDIR")
  [ -n "${KRL_LIBS:-}" ] && VENDOR+=("LANCE_KRL_LIBS=$KRL_LIBS")
fi
for d in "${KML_LIBDIR:-}" "$KML_DIR/lib/kblas/nolocking" "$KML_DIR/lib/neon/kblas/nolocking" "$KML_DIR/lib" "${KRL_LIBDIR:-}" "$KRL_DIR/lib"; do
  [ -n "$d" ] && [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
[ ${#VENDOR[@]} -gt 0 ] && export "${VENDOR[@]}"
# KML runs inside rayon workers: one BLAS thread each (KML: overrides OMP_NUM_THREADS).
export BLAS_NUM_THREADS="${BLAS_NUM_THREADS:-1}"
# Exhaustive assignment comparisons: no centroid-HNSW shortcut (production policy is
# measured separately by the existing kernels_all.sh "auto" spec).
export LANCE_USE_HNSW_SPEEDUP_INDEXING="${LANCE_USE_HNSW_SPEEDUP_INDEXING:-disabled}"

# Assignment backend arms (ws2): KEY=VALUE settings for stages.sh.
assign_arm_env() {
  case $1 in
    tile-neon) echo "LANCE_KERNELS=-dot_tile_sve" ;;
    tile-sve)  echo "LANCE_KERNELS=+dot_tile_sve" ;;
    f32mm|kml|krl|krl_ny) echo "LANCE_ASSIGN_BACKEND=$1" ;;
  esac
}
assign_arms() {
  local arms="tile-neon tile-sve"
  grep -qw svef32mm /proc/cpuinfo && arms+=" f32mm"
  [[ " ${VENDOR[*]} " == *LANCE_KML_DIR* ]] && arms+=" kml"
  [[ " ${VENDOR[*]} " == *LANCE_KRL_DIR* ]] && arms+=" krl krl_ny"
  echo "$arms"
}
# Scan arms (ws5 / ws6): LANCE_KERNELS for run.sh.
NEON_SCAN="-sum_hacc_sve2,-sum_hacc_sve,-sum_u16_sve,-ex_dot_sve"
scan_arm_env() {
  case $1 in
    scalar-hacc) echo "LANCE_KERNELS=-sum_hacc_sve2,-sum_hacc_sve,-sum_hacc_neon,-sum_u16_sve,-ex_dot_sve" ;;
    neon) echo "LANCE_KERNELS=$NEON_SCAN" ;;
    sve)  echo "LANCE_KERNELS=-sum_hacc_sve2" ;;
    krl)  echo "LANCE_KERNELS=-sum_hacc_sve2,-sum_u16_sve,+sum_u16_krl" ;;
  esac
}
scan_arms() {
  local arms="neon sve"
  [[ " ${VENDOR[*]} " == *LANCE_KRL_DIR* ]] && arms+=" krl"
  echo "$arms"
}

step_doctor() {
  { lscpu; echo; grep -m1 -i features /proc/cpuinfo; echo
    for f in asimd sve sve2 svef32mm i8mm bf16; do
      printf '%-9s %s\n' "$f" "$(grep -m1 -i features /proc/cpuinfo | grep -qw "$f" && echo yes || echo no)"
    done
    echo "sve default vector length: $(cat /proc/sys/abi/sve_default_vector_length 2>/dev/null || echo n/a) bytes"
    echo "KML: ${KML_DIR} $([ -f "$KML_DIR/include/kblas.h" ] && echo found || echo 'not found')"
    echo "KRL: ${KRL_DIR} $([ -f "$KRL_DIR/include/krl.h" ] && echo found || echo 'not found')"
    echo "vendor build env: ${VENDOR[*]:-none}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    echo "assign arms: $(assign_arms)"; echo "scan arms: $(scan_arms)"
    (cd "$WT" && git log --oneline -1 && rustc --version && ${CC:-cc} --version | head -1)
  } 2>&1 | tee "$OUT/doctor.txt"
}

step_test() {
  "$HERE/scripts/setup_lance.sh" || return 1
  (cd "$WT/rust" &&
    cargo test -p lance-core --lib utils::cpu &&
    cargo test -p lance-linalg --lib &&
    cargo test -p lance-index --lib -- blocked_l2 bq::ex_dot &&
    # The same, with every SVE scan kernel disabled (NEON arms), so both paths run.
    LANCE_KERNELS="$NEON_SCAN" cargo test -p lance-index --lib -- bq::ex_dot) 2>&1 | tee "$OUT/test.log"
  local rc=${PIPESTATUS[0]}
  return "$rc"
}

step_micro() {
  "$HERE/scripts/setup_lance.sh" >/dev/null || return 1
  (cd "$WT" && cargo build --release -p lance-index-bench --bin kunpeng_kernels) > "$OUT/micro-build.log" 2>&1 ||
    { tail -20 "$OUT/micro-build.log"; return 1; }
  local bin="$WT/target/release/kunpeng_kernels"
  # Scans and ex_dot are single-thread kernels; assignment uses every CPU (and once
  # pinned to one core, for per-core throughput).
  OUT="$OUT/micro.jsonl" SECTIONS=hacc,scan,ex_dot "$bin" 2>&1 | tee "$OUT/micro.txt"
  # tile = NEON 4x4 (the measured-best existing path) plus every other backend; then
  # the SVE 4x4 tile alone (dispatch picks it by default on SVE hosts).
  OUT="$OUT/micro-assign.jsonl" SECTIONS=assign LANCE_KERNELS=-dot_tile_sve "$bin" 2>&1 | tee "$OUT/micro-assign.txt"
  OUT="$OUT/micro-assign-svetile.jsonl" SECTIONS=assign ASSIGN_ARMS=tile "$bin" 2>&1 | tee "$OUT/micro-assign-svetile.txt"
  OUT="$OUT/micro-assign-1t.jsonl" SECTIONS=assign RAYON_NUM_THREADS=1 ASSIGN_ROWS=8192 \
    LANCE_KERNELS=-dot_tile_sve taskset -c 0 "$bin" 2>&1 | tee "$OUT/micro-assign-1t.txt"
}

step_stage() {
  local config="$HERE/configs/kernels-$CASE.env" ref="$OUT/stages/tile-neon"
  for arm in $(assign_arms); do
    local dir="$OUT/stages/$arm"
    [ -f "$dir/stages.jsonl" ] && { log "stage $arm: done"; continue; }
    rm -rf "$dir"
    # shellcheck disable=SC2207
    local kv=($(assign_arm_env "$arm") "ARM=$arm" "STAGES=train,assign,encode" "STAGE_ASSIGN=exact")
    if [ "$arm" != tile-neon ] && [ -f "$ref/centroids.f32" ]; then
      kv+=("CENTROIDS_FROM=$ref/centroids.f32" "REF_DIR=$ref")
    fi
    log "stage $arm: ${kv[*]}"
    "$HERE/scripts/stages.sh" "$config" "$dir" "${kv[@]}" || return 1
  done
}

step_search() {
  local config="$HERE/configs/kernels-$CASE.env"
  for mode in normal accurate; do
    for arm in $(scan_arms); do
      local id="ws256-$CASE-$arm-$mode-$TAG"
      [ -f "$OUT/.done-search-$arm-$mode" ] && continue
      # shellcheck disable=SC2207
      local kv=($(scan_arm_env "$arm") "APPROX_MODE=$mode" "REPEATS=1")
      log "search $arm/$mode: ${kv[*]}"
      RESULTS_DIR="$OUT/search" "$HERE/scripts/run.sh" "$config" "$id" "${kv[@]}" || return 1
      touch "$OUT/.done-search-$arm-$mode"
    done
  done
}

STEPS=(doctor test micro stage search)
[ $# -gt 0 ] && STEPS=("$@")
log "Kunpeng ws2/5/6 session $TAG -> $OUT (vendors: ${VENDOR[*]:-none})"
for s in "${STEPS[@]}"; do
  if done_step "$s"; then log "$s: done earlier, skipping"; continue; fi
  log "$s"
  if "step_$s"; then mark "$s"; else log "$s FAILED (see $OUT)"; exit 1; fi
done
log "finished; summarize with: python3 $HERE/scripts/ws256_report.py $OUT"
