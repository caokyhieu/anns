#!/usr/bin/env bash
# Single entry point for every index-build experiment, identical on x86_64 and aarch64.
#
#   ./lance_index_build/bench.sh doctor            host, toolchain, kernels, data status
#   ./lance_index_build/bench.sh setup             Lance checkout (base + patches), C++ deps
#   ./lance_index_build/bench.sh data [bigann|gist|all]   download + convert + verify hashes
#   ./lance_index_build/bench.sh test              kernel / tree / PiPNN correctness tests
#   ./lance_index_build/bench.sh kernels           criterion kernel microbenchmarks (+ no-SVE on ARM)
#   ./lance_index_build/bench.sh suite <name>...   run suites/<name>.txt (smoke, A, B, C, ablation)
#   ./lance_index_build/bench.sh native            standalone reference PiPNN vs hnswlib (BIGANN10M)
#   ./lance_index_build/bench.sh python            mutable PiPNN Python demo
#   ./lance_index_build/bench.sh report [dir...]   summary.md + plots per run (default: this host)
#   ./lance_index_build/bench.sh all               setup data test kernels smoke A B C ablation report
#
# Environment: ANNS_DATA, DATASETS_DIR, RESULTS_DIR (see scripts/run.sh), JOBS (cargo -j).
# Results land in results/<arch>-<host>/<run>-<UTC stamp>/; nothing is overwritten.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WT="$HERE/worktrees/lance"
ARCH="$(uname -m)"
HOST="$(hostname -s)"
: "${ANNS_DATA:=$ROOT/data}"
: "${RESULTS_DIR:=$HERE/results/$ARCH-$HOST}"
: "${EIGEN_INCLUDE_DIR:=$ROOT/native/third_party/eigen}"
export ANNS_DATA RESULTS_DIR EIGEN_INCLUDE_DIR
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CRATES=(-p lance-linalg -p lance-pipnn -p lance-index)

log() { printf '\n==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null || { echo "missing required tool: $1" >&2; exit 1; }; }

cmd_doctor() {
  log "host"
  echo "arch=$ARCH host=$HOST cpus=$(nproc) mem=$(free -g | awk '/Mem:/{print $2}')G"
  lscpu 2>/dev/null | grep -E "Model name|Vendor ID|NUMA node\(s\)" || true
  if [ "$ARCH" = aarch64 ]; then
    flags="$(grep -m1 -E '^Features' /proc/cpuinfo || true)"
    for f in asimd asimddp sve sve2 bf16 i8mm sme; do
      printf '  %-7s %s\n' "$f" "$(grep -qw "$f" <<<"$flags" && echo yes || echo no)"
    done
    [ -r /proc/sys/abi/sve_default_vector_length ] &&
      echo "  SVE default vector length: $(cat /proc/sys/abi/sve_default_vector_length) bytes"
  else
    flags="$(grep -m1 -E '^flags' /proc/cpuinfo || true)"
    for f in avx2 fma avx512f avx512bw avx512_bf16; do
      printf '  %-11s %s\n' "$f" "$(grep -qw "$f" <<<"$flags" && echo yes || echo no)"
    done
  fi
  log "toolchain"
  for t in git cargo rustup cc c++ cmake python3 uv curl; do
    printf '  %-8s %s\n' "$t" "$(command -v "$t" >/dev/null && ("$t" --version 2>&1 | head -1) || echo MISSING)"
  done
  if [ -d "$WT" ]; then
    echo "  pinned rust: $(cd "$WT" && rustc --version 2>&1)"
    echo "  lance tree:  $(git -C "$WT" rev-parse 'HEAD^{tree}') (expected $(sed -n 's/^LANCE_TREE=//p' "$HERE/patches/SERIES.env"))"
  else
    echo "  lance: not set up (run: bench.sh setup)"
  fi
  printf '  eigen:   %s\n' "$([ -f "$EIGEN_INCLUDE_DIR/Eigen/Dense" ] && echo "$EIGEN_INCLUDE_DIR" || echo MISSING)"
  log "data ($ANNS_DATA)"
  for f in bigann10m/base.10M.u8bin bigann10m/query.u8bin bigann10m/gt.10M.ibin base.f32 query.f32 gt.i32 meta.txt; do
    printf '  %-26s %s\n' "$f" "$([ -f "$ANNS_DATA/$f" ] && du -h "$ANNS_DATA/$f" | cut -f1 || echo missing)"
  done
}

cmd_setup() {
  need git; need cargo; need cc; need c++; need python3
  log "Lance checkout"
  "$HERE/scripts/setup_lance.sh"
  log "C++ dependencies (Eigen, reference PiPNN, hnswlib, ...; pinned in native/sources.lock.json)"
  python3 "$ROOT/native/scripts/bootstrap.py"
  log "Rust toolchain pinned by Lance"
  (cd "$WT" && rustup show active-toolchain || rustup toolchain install "$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)")
}

cmd_data() { "$HERE/scripts/prepare_data.sh" "${1:-all}"; }

cmd_test() {
  log "correctness tests on $ARCH (kernels: tile, L2 batch, tree, blocked L2, PiPNN, k-means, IVF)"
  (cd "$WT" && cargo test "${CRATES[@]}" --features lance-index/pipnn --lib -- \
      tile distance::l2 sve_ tests:: blocked_l2 assignment_tree vector::pipnn kmeans vector::ivf)
  if [ "$ARCH" = aarch64 ]; then
    log "same tests built without SVE objects (NEON paths)"
    (cd "$WT" && LANCE_DISABLE_SVE_BUILD=1 CARGO_TARGET_DIR="$WT/target-nosve" \
      cargo test "${CRATES[@]}" --features lance-index/pipnn --lib -- \
      tile distance::l2 tests:: blocked_l2 assignment_tree vector::pipnn kmeans)
  fi
}

cmd_kernels() {
  local out="$RESULTS_DIR/kernels-$STAMP"
  mkdir -p "$out"
  log "criterion microbenchmarks -> $out"
  (cd "$WT" && cargo bench -p lance-linalg --bench arm_kernels -- --save-baseline default) \
    2>&1 | tee "$out/default.log"
  if [ "$ARCH" = aarch64 ]; then
    log "same benchmarks without SVE objects (NEON / upstream fallbacks)"
    (cd "$WT" && LANCE_DISABLE_SVE_BUILD=1 CARGO_TARGET_DIR="$WT/target-nosve" \
      cargo bench -p lance-linalg --bench arm_kernels -- --save-baseline nosve) 2>&1 | tee "$out/nosve.log"
  fi
  lscpu > "$out/machine.txt" 2>/dev/null || true
}

cmd_suite() {
  [ $# -gt 0 ] || { echo "usage: bench.sh suite <smoke|A|B|C|ablation> ..." >&2; exit 1; }
  for name in "$@"; do
    local list="$HERE/suites/$name.txt"
    [ -f "$list" ] || { echo "no such suite: $list" >&2; exit 1; }
    log "suite $name"
    while read -r config run args; do
      [ -z "${config:-}" ] && continue
      case "$config" in \#*) continue ;; esac
      log "$name: $run ${args:-}"
      # shellcheck disable=SC2086
      (cd "$HERE" && scripts/run.sh "$config" "$run-$STAMP" ${args:-})
    done < "$list"
  done
}

cmd_native() {
  need cmake
  log "reference PiPNN vs hnswlib (native/, -O3 -march=native)"
  cmake -S "$ROOT/native" -B "$ROOT/native/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$ROOT/native/build" -j "${JOBS:-$(nproc)}" --target ann_bench
  (cd "$ROOT" && python3 native/scripts/run_benchmark.py --threads "$(nproc)" \
     --base "$ANNS_DATA/bigann10m/base.10M.u8bin" --queries "$ANNS_DATA/bigann10m/query.u8bin" \
     --gt "$ANNS_DATA/bigann10m/gt.10M.ibin" --output "$RESULTS_DIR/native-bigann10m-$STAMP")
}

cmd_python() {
  need uv
  log "mutable PiPNN demo"
  (cd "$ROOT" && uv run --locked python -m unittest -v &&
     uv run --locked python experiment.py --output "$RESULTS_DIR/python-demo-$STAMP.json")
}

cmd_report() {
  local dirs=("$@")
  if [ ${#dirs[@]} -eq 0 ]; then
    mapfile -t dirs < <(find "$RESULTS_DIR" -mindepth 2 -maxdepth 2 -name builds.jsonl -printf '%h\n' | sort)
  fi
  [ ${#dirs[@]} -gt 0 ] || { echo "no runs under $RESULTS_DIR" >&2; exit 1; }
  for d in "${dirs[@]}"; do
    python3 "$HERE/scripts/report.py" "$d" -o "$d"
  done
  log "reports written next to each run (summary.md, *.svg)"
}

cmd_all() {
  cmd_doctor
  cmd_setup
  cmd_data all
  cmd_test
  cmd_kernels
  cmd_suite smoke A B C ablation
  cmd_report
}

case "${1:-help}" in
  doctor|setup|data|test|kernels|suite|native|python|report|all)
    c="$1"; shift; "cmd_$c" "$@" ;;
  *) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
