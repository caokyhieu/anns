# Kernel arms: which distance kernels a run uses, as KEY=VALUE settings for
# scripts/run.sh and scripts/stages.sh. Sourced; `arm_env <arm>` prints them.
#
#   legacy       Lance's own kernels: built without the SVE objects (every new SVE/SVE2
#                kernel is compiled out), per-vector exact assignment, scalar f32 kernels
#   blocked      legacy + the blocked 4x4 L2 assignment (NEON tile)          [algorithm]
#   sve          legacy + the SVE/SVE2 kernels (f32 pair/batch, u8, lookup)   [kernels]
#   new          both: every new kernel, the proposal as a whole
#
# On x86_64 there are no SVE builds: legacy and new differ by the blocked path only,
# and the sve / blocked arms are skipped.
ARMS_AARCH64="legacy blocked sve new"
ARMS_X86="legacy new"

arm_env() {
  case "$(uname -m):$1" in
    aarch64:legacy)  echo "LANCE_DISABLE_SVE_BUILD=1 LANCE_DISABLE_BLOCKED_L2=1 LANCE_F32_SVE=0" ;;
    aarch64:blocked) echo "LANCE_DISABLE_SVE_BUILD=1 LANCE_F32_SVE=0" ;;
    aarch64:sve)     echo "LANCE_F32_SVE=1 LANCE_DISABLE_BLOCKED_L2=1" ;;
    aarch64:new)     echo "LANCE_F32_SVE=1" ;;
    *:legacy)        echo "LANCE_DISABLE_BLOCKED_L2=1" ;;
    *:new)           echo "" ;;
    *) echo "unknown arm $1 on $(uname -m)" >&2; return 1 ;;
  esac
}

arms_for_host() {
  if [ "$(uname -m)" = aarch64 ]; then echo "$ARMS_AARCH64"; else echo "$ARMS_X86"; fi
}
