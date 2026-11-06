#!/usr/bin/env python3
"""Compare existing Lance dispatch with SVE enabled on the same ARM host."""
import argparse
import os
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--filter", default="arm_", help="Criterion benchmark filter")
    args = parser.parse_args()
    if platform.machine() not in ("aarch64", "arm64"):
        parser.error("run this benchmark on the ARM server, not under QEMU")
    cpuinfo = Path("/proc/cpuinfo").read_text()
    features = next((line for line in cpuinfo.splitlines() if line.startswith("Features")), "")
    if not {"sve", "sve2"}.issubset(features.split()):
        parser.error("this full-suite comparison requires OS-visible SVE and SVE2")
    checkout = args.checkout.resolve()
    env = os.environ.copy()
    env.pop("LANCE_DISABLE_SVE_BUILD", None)
    print(features, flush=True)
    for command in (["git", "rev-parse", "HEAD"], ["rustc", "--version"], ["cargo", "--version"]):
        subprocess.run(command, cwd=checkout, env=env, check=True)
    # Verify that compile probing enabled the ARM candidates. Otherwise a
    # compiler without ACLE could produce two identical fallback builds.
    listing = subprocess.check_output(
        ["cargo", "test", "-p", "lance-linalg", "--lib", "--", "--list"],
        cwd=checkout, env=env, text=True,
    )
    for name in ("sve_candidate_is_selected_when_supported", "sve2_candidate_is_selected_when_supported"):
        if name not in listing:
            parser.error(f"missing {name}; inspect the C compiler's SVE build warnings")
    subprocess.run(["cargo", "test", "-p", "lance-core", "utils::cpu"], cwd=checkout, env=env, check=True)
    subprocess.run(["cargo", "test", "-p", "lance-linalg", "--lib"], cwd=checkout, env=env, check=True)
    bench = ["cargo", "bench", "--profile", "release-with-debug", "-p", "lance-linalg",
             "--bench", "arm_kernels", "--", args.filter]
    baseline_env = env | {"LANCE_DISABLE_SVE_BUILD": "1"}
    subprocess.run(bench + ["--save-baseline", "without-sve"], cwd=checkout, env=baseline_env, check=True)
    subprocess.run(bench + ["--baseline", "without-sve"], cwd=checkout, env=env, check=True)
    print("Criterion reports: " + str(checkout / "target/criterion"))


if __name__ == "__main__":
    main()
