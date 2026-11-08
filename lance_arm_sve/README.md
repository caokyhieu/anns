# ARM SVE/SVE2 kernels for Lance

> **Superseded.** These changes are part of the Lance patch series in
> [`lance_index_build/patches/`](../lance_index_build/patches/); run experiments with
> [`lance_index_build/bench.sh`](../lance_index_build/bench.sh). Kept for history.

This patch adds ARM implementations of existing Lance operations and selects
them through Lance's CPU dispatcher. It does not introduce blocked-L2 IVF,
change vector-search algorithms, or change file/index layouts.

Base: `lance-format/lance` commit
[`6e18b21e4efaec197208ec212e7b03c3fbecc7d8`](https://github.com/lance-format/lance/tree/6e18b21e4efaec197208ec212e7b03c3fbecc7d8).
The patch is independent of this repository's `lance_ivf_rq` patch.

## Implemented operations

| Operation | Existing x86 counterpart | New ARM path |
| --- | --- | --- |
| FP32 dot product, pair and batch | AVX / AVX+FMA / AVX-512 pair kernels | SVE loads, FMA and reduction |
| FP32 squared L2, pair and batch | AVX / AVX+FMA / AVX-512 pair kernels | SVE subtraction, FMA and reduction |
| Unsigned-byte dot product | AVX2 / AVX-512 VNNI | SVE `UDOT` |
| Unsigned-byte squared L2 | AVX2 / AVX-512 VNNI | SVE unsigned absolute difference and `UDOT` |
| Selected-row FP32 4-bit LUT accumulation | AVX2 / AVX-512 | SVE byte gathers and FP32 table gathers |
| High-accuracy 4-bit LUT accumulation | AVX2 | SVE2 table lookup and widening even/odd addition |

There are six C entry points and eight Rust dispatch candidates (pair/batch
distance entry points share C kernels). The batch paths retain upstream's
per-vector algorithm; they do not introduce a tiled or matrix algorithm.
This is a focused implementation, not complete parity with every x86 kernel.
FP16/BF16, cosine, full-scan LUT and bitpacking paths retain their existing
implementations.

## Apply

From this repository, with a Lance checkout at the pinned revision:

```bash
rtk proxy python3 lance_arm_sve/scripts/apply.py /path/to/lance --check
rtk proxy python3 lance_arm_sve/scripts/apply.py /path/to/lance
```

`apply.py` verifies HEAD and runs `git apply --check` before making changes.
It preserves unrelated files; overlapping changes fail the patch check.
To inspect the implementation, open [lance-arm-sve.patch](lance-arm-sve.patch).

## Automatic selection and packaging

On AArch64 Linux, `lance-linalg/build.rs` probes the configured C compiler for
SVE and SVE2 separately. Successful objects enable corresponding Rust
candidates. At runtime, `CpuFeatures` checks OS-visible SVE/SVE2 capabilities;
Lance's existing cached dispatcher selects the supported candidate. A compiler
that cannot build an object produces a warning and retains existing fallbacks.
No external Huawei library or application-level registration is required.

Compile the rest of Lance for its normal baseline. Do not globally enable
SVE, SVE2, or `target-cpu=native` in a wheel intended for multiple ARM machines.
The C objects use `-msve-vector-bits=scalable` and query the calling thread's
vector length on every invocation. No vector length is cached process-wide.
The ARM kernels are currently built on Linux; macOS and other targets retain
their existing paths.

Example on the ARM server, from the patched Lance root, with Lance's documented
build prerequisites and pinned Rust toolchain installed:

```bash
rtk proxy env CC=clang-18 cargo test -p lance-core utils::cpu
rtk proxy env CC=clang-18 cargo test -p lance-linalg --lib
```

For a Python wheel, use Lance's normal Python build environment. From its
`python/` directory, initialize a fresh checkout and build the wheel:

```bash
rtk proxy env CC=clang-18 make install
rtk proxy env CC=clang-18 uv run maturin build --release --features fp16kernels
```

The existing `fp16kernels` feature preserves half-precision support; it is not
required for these new SVE objects. Install the resulting AArch64 wheel with
your environment's package manager. Existing `import lance` operations reach
the new kernels through the same Rust APIs; datasets do not need rewriting.
Rebuild other bindings or consumers that embed Lance separately.

`CpuFeatures::host().names()` reports SVE/SVE2 availability to Rust callers.
The existing Python `simd_info()` remains unchanged and is not an
operation-specific backend indicator. Added tests assert selection of each compiled candidate
when its features are available; the existing candidate tests validate results.

## Correctness validation

The self-contained test runner needs Clang 18, LLD 18 and QEMU user-mode 7.2
or newer. It uses Linux syscalls directly, so no AArch64 libc/sysroot is needed:

```bash
rtk proxy python3 lance_arm_sve/scripts/validate_emulated.py /path/to/patched/lance \
  --clang clang-18 --linker ld.lld-18 --qemu qemu-aarch64-static
```

It compiles the actual patched C sources and checks:

- All six kernels at 128-, 256-, 512-, 1024- and 2048-bit SVE widths, changing
  the vector length between calls in one process.
- Empty and irregular lengths, unsigned-byte overflow modulo 2^32, and FP32
  error against double-precision references.
- Exact FP32 LUT addition order, signed zero, repeated and unsorted row IDs.
- Lance's PERM0 packed-row layout, multiple batches, high-accuracy byte carries
  and sums exceeding 16 bits.
- Guard pages immediately after inputs and outputs, exposing tail overreads,
  oversized stores and invalid gather addresses.

FP32 distance reductions may round differently from the existing paths,
as do the existing x86 kernels. Integer outputs and FP32 LUT sums retain their
reference contracts. Native Rust tests remain necessary to validate the entire
Rust/C integration and higher-level behavior.

## Fair native benchmark

Run this on the Kunpeng server with SVE/SVE2 visible to Linux:

```bash
rtk proxy env CC=clang-18 python3 lance_arm_sve/scripts/benchmark_native.py /path/to/patched/lance
```

The script first checks that the compiler actually enabled both kernel families
and runs the CPU/linalg unit tests. It then benchmarks identical operations and
deterministic inputs twice using the repository's `release-with-debug` profile:

1. `LANCE_DISABLE_SVE_BUILD=1`: existing upstream dispatch and fallbacks.
2. Variable unset: new automatic SVE/SVE2 dispatch.

Criterion saves the first run as `without-sve` and compares the second run
against it. Neither run introduces blocked-L2, changes algorithms, changes
global target flags, nor allocates output buffers inside timed iterations.
There are pair/batch cases, small PQ dimensions, embedding dimensions, and
irregular tails. Keep CPU affinity, NUMA placement and frequency policy fixed;
repeat measurements to assess order and thermal effects. Emulation timings
must not be used to claim a speedup. Validate application Recall@k and latency
before selecting this patch for a production deployment.

## Validation performed here

- Clang 18 cross-compilation: both objects passed `-Wall -Wextra -Werror`.
- QEMU 7.2: all six C kernels passed the complete multi-width and guard-page suite.
- Independent Rust 1.97 harness: the actual modified CPU/dispatch modules passed
  feature-implication and x86 fallback checks. This is not a Lance workspace build.
- Changed Rust files were syntax-parsed/formatted with the installed formatter.
- Full Lance Cargo tests, native ARM Rust linking and native performance remain
  unverified: this environment is x86 and downloading the pinned Rust 1.98.1
  toolchain failed certificate validation. No Kunpeng speedup is claimed.

SME is not implemented in this patch. It requires independently confirmed
hardware/OS support and streaming/ZA state handling; it is not a wider-register
replacement for these SVE operations. A separate SME implementation should target
an existing matrix/batched operation with a measured benefit.

References: [Linux SVE ABI](https://docs.kernel.org/arch/arm64/sve.html),
[Arm ACLE](https://arm-software.github.io/acle/main/acle.html).
