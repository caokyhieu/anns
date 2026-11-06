#!/usr/bin/env python3
"""Apply the standalone ARM patch to its pinned, unmodified Lance revision."""
import argparse
from pathlib import Path
import subprocess

BASE = "6e18b21e4efaec197208ec212e7b03c3fbecc7d8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    patch = Path(__file__).resolve().parents[1] / "lance-arm-sve.patch"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if head != BASE:
        parser.error(f"expected Lance {BASE}, found {head}; rebase the patch explicitly")
    command = ["git", "apply", "--check", str(patch)]
    subprocess.run(command, cwd=checkout, check=True)
    if not args.check:
        subprocess.run(["git", "apply", str(patch)], cwd=checkout, check=True)
    print("Patch checked" if args.check else "ARM kernels and runtime dispatch applied")


if __name__ == "__main__":
    main()
