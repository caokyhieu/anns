#!/usr/bin/env python3
"""Compile the patched C sources and run them under QEMU; never a speed test."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--clang", default="clang-18")
    parser.add_argument("--linker", default="ld.lld-18")
    parser.add_argument("--qemu", default="qemu-aarch64-static")
    args = parser.parse_args()
    linker = shutil.which(args.linker)
    if linker is None:
        parser.error(f"linker not found: {args.linker}")
    source = args.checkout.resolve() / "rust/lance-linalg/src/simd"
    test = Path(__file__).resolve().parents[1] / "tests/kernels.c"
    common = [args.clang, "--target=aarch64-linux-gnu", "-ffreestanding",
              "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]
    with tempfile.TemporaryDirectory(prefix="lance-sve-test-") as temp:
        root = Path(temp)
        objects = []
        for isa in ("sve", "sve2"):
            obj = root / f"{isa}.o"
            subprocess.run(common + [f"-march=armv8-a+{isa}",
                           "-msve-vector-bits=scalable", "-O3", "-c",
                           str(source / f"{isa}.c"), "-o", str(obj)], check=True)
            objects.append(str(obj))
        executable = root / "test-kernels"
        subprocess.run(common + ["-march=armv8-a", "-fno-builtin",
                       "-fno-stack-protector", "-O2", "-nostdlib", "-static",
                       f"--ld-path={linker}", "-Wl,-e,_start", str(test)]
                       + objects + ["-o", str(executable)], check=True)
        subprocess.run([args.qemu, "-cpu", "max", str(executable)], check=True)


if __name__ == "__main__":
    main()
