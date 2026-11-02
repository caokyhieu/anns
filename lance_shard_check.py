"""Real-Lance check for the D6 baseline in native/src/shard_bench.cpp.

Builds Lance IVF_RQ indexes on the same uint8 data (stored as float32) two ways:
  S = 1   one create_index call (Lance trains its own k-means, assigns and encodes);
  S > 1   centroids trained once with lance.util.KMeans on a sample, then one
          create_index_uncommitted per fragment group (one "worker" each, run one after
          another and timed separately), committed as one logical index.
Then sweeps nprobes with refine_factor and reports recall@k against the exact ground truth.

Usage:
  uv run --locked --extra lance python lance_shard_check.py \
      --data native/results/synthetic1m --shards 1 8 --output native/results/shards-synthetic1m/lance.jsonl
"""
import argparse
import json
import shutil
import struct
import time
from pathlib import Path

import lance
import lance.util
import numpy as np
import pyarrow as pa


def read_bin(path, dtype):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        return np.fromfile(f, dtype=dtype, count=n * d).reshape(n, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="native/results/synthetic1m")
    ap.add_argument("--work", default="data/lance-shards")
    ap.add_argument("--shards", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--partitions", type=int, default=1024)
    ap.add_argument("--num-bits", type=int, default=1)
    ap.add_argument("--nprobes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--refine", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--output", default="native/results/shards-synthetic1m/lance.jsonl")
    a = ap.parse_args()

    base = read_bin(Path(a.data) / "base.u8bin", np.uint8).astype(np.float32)
    queries = read_bin(Path(a.data) / "query.u8bin", np.uint8).astype(np.float32)[: a.queries]
    gt = read_bin(Path(a.data) / "gt.ibin", np.uint32)[: a.queries, : a.k]
    n, d = base.shape
    out = open(a.output, "a")

    def record(row):
        row.update(dataset=Path(a.data).name, n=n, dim=d, k=a.k, lists=a.partitions, num_bits=a.num_bits)
        print(json.dumps(row), flush=True)
        out.write(json.dumps(row) + "\n")
        out.flush()

    for shards in a.shards:
        uri = Path(a.work) / f"s{shards}.lance"
        shutil.rmtree(uri, ignore_errors=True)
        vec = pa.FixedSizeListArray.from_arrays(pa.array(base.ravel()), d)
        table = pa.table({"id": pa.array(np.arange(n, dtype=np.uint32)), "vec": vec})
        ds = lance.write_dataset(table, uri, max_rows_per_file=-(-n // shards))
        fids = [f.fragment_id for f in ds.get_fragments()]
        assert len(fids) == shards, (len(fids), shards)
        if shards == 1:
            t = time.perf_counter()
            ds.create_index("vec", "IVF_RQ", num_partitions=a.partitions, num_bits=a.num_bits)
            build = {"fixed_seconds": 0.0, "segment_seconds": [time.perf_counter() - t]}
        else:
            t = time.perf_counter()
            rng = np.random.default_rng(7)
            sample = base[rng.choice(n, min(n, 256 * a.partitions), replace=False)]
            km = lance.util.KMeans(a.partitions, max_iters=50)
            km.fit(sample)
            centroids = np.stack(km.centroids.to_numpy(zero_copy_only=False)).astype(np.float32)
            fixed = time.perf_counter() - t
            segs, seconds = [], []
            for fid in fids:
                t = time.perf_counter()
                segs.append(ds.create_index_uncommitted("vec", "IVF_RQ", name="vec_idx", num_partitions=a.partitions,
                                                        ivf_centroids=centroids, num_bits=a.num_bits, fragment_ids=[fid]))
                seconds.append(time.perf_counter() - t)
            t = time.perf_counter()
            ds = ds.commit_existing_index_segments("vec_idx", "vec", segs)
            build = {"fixed_seconds": fixed, "merge_seconds": time.perf_counter() - t, "segment_seconds": seconds}
        ds = lance.dataset(uri)
        seg = build["segment_seconds"]
        record({"kind": "build", "design": "lance", "shards": shards, **build,
                "shard_seconds_max": max(seg), "shard_seconds_sum": sum(seg),
                "wall_seconds": build["fixed_seconds"] + max(seg) + build.get("merge_seconds", 0.0)})
        for nprobes in a.nprobes:
            hits, lat = 0, []
            for qi, q in enumerate(queries):
                t = time.perf_counter()
                res = ds.to_table(columns=["id"], nearest={"column": "vec", "q": q, "k": a.k, "nprobes": nprobes,
                                                           "refine_factor": a.refine})
                lat.append((time.perf_counter() - t) * 1e3)
                hits += len(set(res.column("id").to_numpy().tolist()) & set(gt[qi].tolist()))
            lat.sort()
            record({"kind": "query", "design": "lance", "shards": shards, "nprobe": nprobes, "refine": a.refine,
                    "recall": hits / (len(queries) * a.k), "latency_ms_mean": float(np.mean(lat)),
                    "latency_ms_p50": lat[len(lat) // 2], "latency_ms_p99": lat[int(len(lat) * 0.99)]})


if __name__ == "__main__":
    main()
