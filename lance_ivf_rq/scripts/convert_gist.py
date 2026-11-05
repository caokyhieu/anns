"""Convert ann-benchmarks gist-960-euclidean.hdf5 to raw little-endian files.

Writes base.f32 (N x 960 float32), query.f32 (Q x 960 float32),
gt.i32 (Q x 100 int32 neighbor ids) and meta.txt next to the input.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

src = Path(sys.argv[1])
out = src.parent
with h5py.File(src, "r") as f:
    train = f["train"]
    n, d = train.shape
    with open(out / "base.f32", "wb") as fh:
        for start in range(0, n, 100_000):
            np.ascontiguousarray(train[start : start + 100_000], dtype="<f4").tofile(fh)
    test = np.ascontiguousarray(f["test"][:], dtype="<f4")
    gt = np.ascontiguousarray(f["neighbors"][:], dtype="<i4")
    test.tofile(out / "query.f32")
    gt.tofile(out / "gt.i32")
    (out / "meta.txt").write_text(f"{n} {d} {test.shape[0]} {gt.shape[1]}\n")
    print("base", (n, d), "query", test.shape, "gt", gt.shape, "metric", f.attrs.get("distance"))
