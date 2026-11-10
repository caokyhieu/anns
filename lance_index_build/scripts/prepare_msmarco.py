"""Download MS MARCO v2.1 (Cohere Embed English v3) and convert it for the benchmark runner.

Source: huggingface.co/datasets/CohereLabs/msmarco-v2.1-embed-english-v3 at the
revision pinned in datasets.lock.json (key "msmarco-v2.1"): 113,520,750 passages
x 1024 float16 in 60 .npy files, plus 1677 TREC-DL 2021-2023 queries with their
published brute-force top-1000 (cosine).

Writes to --output:
  base.fbin   unit-normalized float32 passages, rows in the order the .npy files
              stack (the order the published neighbour offsets refer to)
  query.fbin  unit-normalized float32 queries
  gt.ibin     published top-1000 per query: ids are row numbers in base.fbin,
              distances are squared L2 between unit vectors (2 - 2 cos)
  meta.json   revision, counts, and the vector norms before normalization

The runner ranks by L2. On unit vectors L2 ranks exactly like cosine, so the
published cosine neighbours are the L2 ground truth after normalization.

Every download resumes after interruption and is checked against its SHA-256.
Each .npy is converted as soon as it is verified and then deleted (--keep-npy
keeps them), so peak disk is base.fbin (465 GB) plus one .npy (~4 GB).
A re-run continues where it stopped.

Faster sources (the SHA-256 check makes any mirror safe):
  HF_ENDPOINT=https://hf-mirror.com   download through a mirror instead of huggingface.co
  DOWNLOAD_JOBS=8                     .npy files fetched in parallel (default 4; each
                                      in-flight file adds ~4 GB of peak disk)
  .npy files copied into DIR/passages_npy/ by other means (rsync, aria2c) are
  verified and used instead of downloading them.

Run with: uv run --no-project --with numpy --with pyarrow python prepare_msmarco.py --output DIR
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
LOCK_KEY = "msmarco-v2.1"
CHUNK_ROWS = 1 << 16
# Published cosines are computed from float16 embeddings; recomputed ones agree to
# about 1e-3. A misaligned row order gives unrelated vectors (cosine error ~0.5).
SPOT_CHECK_TOLERANCE = 1e-2


def log(message: str) -> None:
    print(message, flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, size: int, sha256: str, token: str | None) -> None:
    """Fetch `url` to `path`, resuming a `.partial`; a `.verified` marker skips re-hashing."""
    marker = path.with_name(path.name + ".verified")
    if path.is_file() and marker.is_file() and marker.read_text() == sha256:
        return
    partial = path.with_name(path.name + ".partial")
    if path.is_file() and not partial.exists():
        path.rename(partial)
    have = partial.stat().st_size if partial.exists() else 0
    if have > size:
        partial.unlink()
        have = 0
    if have < size:
        headers = {"Range": f"bytes={have}-{size - 1}"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("ab") as out:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
                f"bytes {have}-{size - 1}/"
            ):
                raise SystemExit(f"{url}: server did not honour the byte range (status {response.status})")
            done, step = have, size // 20 or 1
            while block := response.read(1 << 23):
                out.write(block)
                if (done + len(block)) // step > done // step:
                    log(f"  {path.name}: {(done + len(block)) / size:.0%}")
                done += len(block)
        if partial.stat().st_size != size:
            raise SystemExit(f"{path.name}: download truncated at {partial.stat().st_size} of {size} bytes")
    got = sha256_of(partial)
    if got != sha256:
        partial.unlink()
        raise SystemExit(f"{path.name}: SHA-256 {got} != {sha256}; removed, re-run to fetch again")
    partial.rename(path)
    marker.write_text(sha256)


def write_header(f, rows: int, dim: int) -> None:
    f.seek(0)
    f.write(struct.pack("<II", rows, dim))


def norm_stats(norms: np.ndarray, into: dict) -> None:
    into["min"] = min(into.get("min", np.inf), float(norms.min()))
    into["max"] = max(into.get("max", 0.0), float(norms.max()))
    into["sum"] = into.get("sum", 0.0) + float(norms.sum())
    into["count"] = into.get("count", 0) + int(norms.size)


def normalized(chunk: np.ndarray, stats: dict, what: str) -> np.ndarray:
    vectors = np.asarray(chunk, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise SystemExit(f"{what}: non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms == 0).any():
        raise SystemExit(f"{what}: zero vector at row {int(np.argmin(norms))}")
    norm_stats(norms, stats)
    return vectors / norms[:, None]


def convert_base(spec: dict, out_dir: Path, base_url: str, token: str | None, keep_npy: bool) -> dict | None:
    """Writes base.fbin; returns the norm summary, or the recorded one (None if
    unknown) when base.fbin was already complete."""
    sources = sorted(p for p in spec["sources"] if p.startswith("passages_npy/"))
    rows_expected, dim = spec["rows"], spec["dim"]
    final, partial = out_dir / "base.fbin", out_dir / "base.fbin.partial"
    progress_path = out_dir / "base.progress.json"
    if final.is_file():
        log(f"{final} exists; skipping passages")
        meta = out_dir / "meta.json"
        return json.loads(meta.read_text()).get("base_norms") if meta.is_file() else None
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {"files": 0, "rows": 0, "norms": {}}
    npy_dir = out_dir / "passages_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    todo = [(index, source) for index, source in enumerate(sources) if index >= progress["files"]]
    jobs = max(1, int(os.environ.get("DOWNLOAD_JOBS", "4")))

    def fetch(source: str) -> None:
        want = spec["sources"][source]
        download(f"{base_url}/{source}", out_dir / source, want["bytes"], want["sha256"], token)

    # Downloads run up to `jobs` files ahead; conversion stays in file order.
    pool = ThreadPoolExecutor(max_workers=jobs)
    pending = {}
    def top_up(position: int) -> None:
        for _, source in todo[position : position + jobs]:
            if source not in pending:
                pending[source] = pool.submit(fetch, source)
    with partial.open("r+b" if partial.exists() else "w+b") as out:
        # Drop anything written after the last completed file.
        out.truncate(8 + progress["rows"] * dim * 4)
        write_header(out, progress["rows"], dim)
        out.seek(0, os.SEEK_END)
        try:
            for position, (index, source) in enumerate(todo):
                want = spec["sources"][source]
                local = out_dir / source
                top_up(position)
                log(f"[{index + 1}/{len(sources)}] {source} ({want['bytes'] / 1e9:.1f} GB)")
                pending.pop(source).result()
                matrix = np.load(local, mmap_mode="r")
                if matrix.ndim != 2 or matrix.shape[1] != dim:
                    raise SystemExit(f"{source}: shape {matrix.shape}, expected (*, {dim})")
                for lo in range(0, matrix.shape[0], CHUNK_ROWS):
                    chunk = matrix[lo : lo + CHUNK_ROWS]
                    out.write(normalized(chunk, progress["norms"], f"{source} rows {lo}..").tobytes())
                out.flush()
                os.fsync(out.fileno())
                progress["files"] = index + 1
                progress["rows"] += matrix.shape[0]
                del matrix
                progress_path.write_text(json.dumps(progress))
                if not keep_npy:
                    local.unlink()
                    local.with_name(local.name + ".verified").unlink()
        finally:
            # On failure, stop queued downloads; in-flight ones finish into their
            # .partial and a re-run resumes them.
            pool.shutdown(wait=True, cancel_futures=True)
        if progress["rows"] != rows_expected:
            raise SystemExit(f"passages: {progress['rows']} rows, expected {rows_expected}")
        write_header(out, progress["rows"], dim)
    partial.rename(final)
    progress_path.unlink()
    log(f"wrote {final}: {progress['rows']} x {dim} float32")
    return summary(progress["norms"])


def convert_queries(spec: dict, out_dir: Path, base_url: str, token: str | None) -> tuple[dict, int]:
    import pyarrow.parquet as pq

    source = "queries_parquet/queries.parquet"
    want = spec["sources"][source]
    local = out_dir / source
    local.parent.mkdir(parents=True, exist_ok=True)
    download(f"{base_url}/{source}", local, want["bytes"], want["sha256"], token)
    table = pq.read_table(local, columns=["emb", "top1k_offsets", "top1k_cossim"])
    emb = np.array(table["emb"].to_pylist(), dtype=np.float32)
    offsets = table["top1k_offsets"].to_pylist()
    cossim = table["top1k_cossim"].to_pylist()
    nq, k, dim = len(offsets), spec["gt_k"], spec["dim"]
    if nq != spec["queries"] or emb.shape != (nq, dim):
        raise SystemExit(f"queries: {nq} rows, embeddings {emb.shape}; expected {spec['queries']} x {dim}")
    if any(len(o) != k or len(c) != k for o, c in zip(offsets, cossim)):
        raise SystemExit(f"queries: every query must list exactly {k} neighbours")
    ids = np.array(offsets, dtype=np.int64)
    sims = np.array(cossim, dtype=np.float64)
    if ids.min() < 0 or ids.max() >= spec["rows"]:
        raise SystemExit(f"queries: neighbour offsets outside 0..{spec['rows']}")
    # Nearest first; a stable sort keeps the published order among equal cosines.
    order = np.argsort(-sims, axis=1, kind="stable")
    ids = np.take_along_axis(ids, order, axis=1)
    sims = np.take_along_axis(sims, order, axis=1)
    stats: dict = {}
    queries = normalized(emb, stats, "queries")
    with (out_dir / "query.fbin").open("wb") as f:
        f.write(struct.pack("<II", nq, dim))
        f.write(queries.tobytes())
    with (out_dir / "gt.ibin").open("wb") as f:
        f.write(struct.pack("<II", nq, k))
        f.write(ids.astype("<u4").tobytes())
        f.write(np.clip(2.0 - 2.0 * sims, 0.0, None).astype("<f4").tobytes())
    log(f"wrote query.fbin ({nq} x {dim}) and gt.ibin (top-{k})")
    return stats, nq


def spot_check(out_dir: Path, dim: int, samples: int = 50) -> float:
    """Recompute the published cosine of the first and last listed neighbour of
    `samples` queries from base.fbin; catches a misaligned row order."""
    base = np.memmap(out_dir / "base.fbin", dtype="<f4", mode="r", offset=8)
    base = base.reshape(-1, dim)
    raw = (out_dir / "query.fbin").read_bytes()
    nq, _ = struct.unpack("<II", raw[:8])
    queries = np.frombuffer(raw[8:], dtype="<f4").reshape(nq, dim)
    gt = (out_dir / "gt.ibin").read_bytes()
    _, k = struct.unpack("<II", gt[:8])
    ids = np.frombuffer(gt[8 : 8 + nq * k * 4], dtype="<u4").reshape(nq, k)
    dists = np.frombuffer(gt[8 + nq * k * 4 :], dtype="<f4").reshape(nq, k)
    worst = 0.0
    for q in np.linspace(0, nq - 1, min(samples, nq)).astype(int):
        for j in (0, k - 1):
            cos = float(queries[q] @ base[ids[q, j]])
            worst = max(worst, abs(cos - (1.0 - dists[q, j] / 2.0)))
    if worst > SPOT_CHECK_TOLERANCE:
        raise SystemExit(f"spot check failed: recomputed cosine differs by {worst:.4f} from the published one")
    log(f"spot check ok: published vs recomputed cosine within {worst:.2e}")
    return worst


def summary(stats: dict) -> dict:
    return {"min": stats["min"], "max": stats["max"], "mean": stats["sum"] / stats["count"], "count": stats["count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-npy", action="store_true", help="keep the downloaded .npy files (+232 GB)")
    args = parser.parse_args()
    spec = json.loads((HERE / "datasets.lock.json").read_text())[LOCK_KEY]
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    base_url = f"{endpoint}/datasets/{spec['repo']}/resolve/{spec['revision']}"
    log(f"source: {endpoint} ({os.environ.get('DOWNLOAD_JOBS', '4')} parallel downloads)")
    token = os.environ.get("HF_TOKEN")
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    free = os.statvfs(out_dir)
    free_gb = free.f_bavail * free.f_frsize / 1e9
    need_gb = spec["rows"] * spec["dim"] * 4 / 1e9 + 10
    if not (out_dir / "base.fbin").is_file() and free_gb < need_gb:
        log(f"warning: {free_gb:.0f} GB free in {out_dir}, conversion needs ~{need_gb:.0f} GB")

    query_norms, nq = convert_queries(spec, out_dir, base_url, token)
    base_norms = convert_base(spec, out_dir, base_url, token, args.keep_npy)
    worst = spot_check(out_dir, spec["dim"])
    meta = {
        "dataset": "MS MARCO v2.1 (TREC-RAG 2024), Cohere Embed English v3",
        "repo": spec["repo"],
        "revision": spec["revision"],
        "rows": spec["rows"],
        "dim": spec["dim"],
        "queries": nq,
        "gt_k": spec["gt_k"],
        "normalized": True,
        "base_norms": base_norms,
        "query_norms": summary(query_norms),
        "spot_check_max_cosine_error": worst,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"done: {out_dir} (norms before normalization: base {meta['base_norms']}, queries {meta['query_norms']})")


if __name__ == "__main__":
    sys.exit(main())
