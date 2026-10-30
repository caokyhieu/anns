"""End-to-end training-data retrieval pipeline on Lance, on synthetic AV-style data.

Stages (each maps to a Lance feature; see LANCE_PIPELINE.md):
  1. ingest    parallel fragment writes per drive, one commit (distributed write)
  2. evolve    add an embedding column with a batch UDF and a score column by merge
  3. index     shared IVF centroids, per-worker IVF_RQ index segments committed as one
               logical index; scalar indexes (BTREE, BITMAP, LABEL_LIST) and FTS
  4. retrieve  a job of seed queries: prefiltered vector search with large k, plus
               caption full-text search; recall measured against exact search
  5. curate    near-duplicate removal and per-drive balancing on the candidates
  6. publish   manifest table of row IDs; source and manifest versions tagged
  7. load      training reader fetches camera/LiDAR blobs by row ID from the manifest

Synthetic data stands in for real embeddings and sensor payloads.
"""
import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter
import numpy as np
import pyarrow as pa
import lance
from lance.fragment import write_fragments
import dpipnn as D

WEATHER = ["clear", "rain", "fog", "snow", "night"]
TAGS = ["pedestrian", "cyclist", "cut_in", "construction", "emergency_vehicle", "unprotected_left"]


def make_drive(rng, drive, rows, dim, centers):
    """Clip segments of one drive: metadata, caption, embedding, fake sensor blobs."""
    c = rng.integers(0, len(centers), rows)
    # Consecutive segments of a drive are correlated (near-duplicates), like real logs.
    base = centers[c] + 0.6 * rng.normal(size=(rows, dim))
    emb = (base + 0.15 * np.cumsum(rng.normal(size=(rows, dim)), 0) / np.sqrt(np.arange(1, rows + 1))[:, None])
    emb = emb.astype(np.float32)
    weather = rng.choice(WEATHER, rows)
    tags = [list(rng.choice(TAGS, rng.integers(0, 3), replace=False)) for _ in range(rows)]
    captions = [f"{w} scene with {' and '.join(t) if t else 'normal traffic'}" for w, t in zip(weather, tags)]
    traj = rng.normal(size=(rows, 16)).astype(np.float32)
    return pa.table({
        "segment_id": pa.array(drive * 1_000_000 + np.arange(rows), pa.int64()),
        "drive_id": pa.array(np.full(rows, drive), pa.int32()),
        "ts": pa.array(1.7e9 + drive * 86400 + np.arange(rows) * 2.0, pa.float64()),
        "lat": pa.array(37.0 + rng.normal(size=rows) * 0.1, pa.float64()),
        "lon": pa.array(-122.0 + rng.normal(size=rows) * 0.1, pa.float64()),
        "weather": pa.array(weather),
        "tags": pa.array(tags, pa.list_(pa.string())),
        "caption": pa.array(captions),
        "trajectory": pa.FixedSizeListArray.from_arrays(traj.ravel(), 16),
        "emb_video": pa.FixedSizeListArray.from_arrays(emb.ravel(), dim),
        "camera": lance.blob_array([rng.bytes(256) for _ in range(rows)]),
        "lidar": lance.blob_array([rng.bytes(512) for _ in range(rows)]),
    }, schema=SCHEMA(dim))


def SCHEMA(dim):
    return pa.schema([
        ("segment_id", pa.int64()), ("drive_id", pa.int32()), ("ts", pa.float64()),
        ("lat", pa.float64()), ("lon", pa.float64()), ("weather", pa.string()),
        ("tags", pa.list_(pa.string())), ("caption", pa.string()),
        ("trajectory", pa.list_(pa.float32(), 16)), ("emb_video", pa.list_(pa.float32(), dim)),
        lance.blob_field("camera"), lance.blob_field("lidar"),
    ])


def log(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw), default=str), flush=True)
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="lance-demo")
    ap.add_argument("--drives", type=int, default=40)
    ap.add_argument("--rows-per-drive", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--partitions", type=int, default=128)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.root)
    shutil.rmtree(root, ignore_errors=True)
    uri = str(root / "segments.lance")
    report = {}

    # 1. Ingest: each worker writes fragments for its drives; one commit publishes all.
    centers = rng.normal(size=(64, a.dim)) * 2
    start = perf_counter()
    frags = []
    for drive in range(a.drives):
        frags += write_fragments(make_drive(rng, drive, a.rows_per_drive, a.dim, centers), uri,
                                 schema=SCHEMA(a.dim), data_storage_version="2.2")
    ds = lance.LanceDataset.commit(uri, lance.LanceOperation.Overwrite(SCHEMA(a.dim), frags))
    report["ingest"] = log("ingest", rows=ds.count_rows(), fragments=len(ds.get_fragments()),
                           version=ds.version, seconds=perf_counter() - start)

    # 2. Data evolution: new embedding column from a UDF, model scores merged by key.
    @lance.batch_udf()
    def traj_embedding(batch):
        t = np.stack(batch.column("trajectory").to_numpy(zero_copy_only=False))
        e = np.tanh(t @ np.random.default_rng(0).normal(size=(16, 8))).astype(np.float32)
        return pa.RecordBatch.from_arrays([pa.FixedSizeListArray.from_arrays(e.ravel(), 8)], ["emb_traj"])
    ds.add_columns(traj_embedding, read_columns=["trajectory"])
    ids = ds.to_table(columns=["segment_id"]).column(0).to_numpy()
    ds.merge(pa.table({"segment_id": ids, "uncertainty": rng.random(len(ids)).astype(np.float32)}), "segment_id")
    ds = lance.dataset(uri)
    report["evolve"] = log("evolve", columns=ds.schema.names, version=ds.version)

    # 3. Index: train shared centroids once on a sample, then each worker indexes its
    #    fragments with them; the segments are committed as one logical index.
    start = perf_counter()
    sample = np.stack(ds.sample(min(20000, ds.count_rows()), columns=["emb_video"])
                      .column(0).to_numpy(zero_copy_only=False)).astype(np.float32)
    centroids = D.kmeans(sample, a.partitions, rng).astype(np.float32)
    fids = [f.fragment_id for f in ds.get_fragments()]
    segments = [ds.create_index_uncommitted("emb_video", "IVF_RQ", name="emb_video_idx",
                                            num_partitions=a.partitions, ivf_centroids=centroids,
                                            fragment_ids=list(group))
                for group in np.array_split(fids, a.workers)]
    ds = ds.commit_existing_index_segments("emb_video_idx", "emb_video", segments)
    vector_seconds = perf_counter() - start
    ds.create_scalar_index("ts", "BTREE")
    ds.create_scalar_index("weather", "BITMAP")
    ds.create_scalar_index("tags", "LABEL_LIST")
    ds.create_scalar_index("caption", "INVERTED")
    ds = lance.dataset(uri)
    report["index"] = log("index", vector_segments=len(segments), vector_seconds=vector_seconds,
                          indices=[i["name"] for i in ds.list_indices()])

    # 4. Retrieve: a job of seed queries with metadata filters and large k.
    flt = "weather IN ('rain', 'night') AND array_has_any(tags, ['pedestrian', 'cyclist'])"
    queries = sample[rng.choice(len(sample), a.queries, replace=False)] + 0.1 * rng.normal(size=(a.queries, a.dim)).astype(np.float32)
    rows = []
    for nprobes, refine in ((8, None), (32, None), (32, 4), (64, 4)):
        recall, seconds = [], 0.0
        for q in queries:
            near = {"column": "emb_video", "q": q, "k": a.k, "nprobes": nprobes}
            if refine:
                near["refine_factor"] = refine
            start = perf_counter()
            got = ds.to_table(columns=["segment_id", "_distance"], nearest=near, filter=flt, prefilter=True)
            seconds += perf_counter() - start
            exact = ds.to_table(columns=["segment_id", "_distance"], filter=flt, prefilter=True,
                                nearest={"column": "emb_video", "q": q, "k": a.k, "use_index": False})
            want = set(exact.column("segment_id").to_pylist())
            recall.append(len(want & set(got.column("segment_id").to_pylist())) / max(1, len(want)))
        rows.append(log("retrieve", k=a.k, nprobes=nprobes, refine_factor=refine, filter=flt,
                        recall=float(np.mean(recall)), ms_per_query=1000 * seconds / len(queries)))
    report["retrieve"] = rows
    fts = ds.to_table(columns=["segment_id", "caption"], full_text_query="construction fog", limit=5)
    report["fts"] = log("fts", query="construction fog", hits=fts.column("caption").to_pylist())

    # 5. Curate: union candidates, drop near-duplicates, cap segments per drive.
    near = {"column": "emb_video", "k": a.k, "nprobes": 64, "refine_factor": 4}
    cand = pa.concat_tables([
        ds.to_table(columns=["segment_id", "drive_id", "emb_video", "uncertainty", "_distance"],
                    nearest=dict(near, q=q), filter=flt, prefilter=True, with_row_id=True)
        for q in queries])
    # One row per segment, keeping its best distance across queries.
    by = np.lexsort((cand.column("_distance").to_numpy(), cand.column("_rowid").to_numpy()))
    cand = cand.take(by)
    _, first = np.unique(cand.column("_rowid").to_numpy(), return_index=True)
    cand = cand.take(first)
    emb = np.stack(cand.column("emb_video").to_numpy(zero_copy_only=False))
    drive = cand.column("drive_id").to_numpy()
    order = np.argsort(-cand.column("uncertainty").to_numpy())  # most informative first
    per_drive, chosen = {}, []
    for i in order:
        d = int(drive[i])
        if per_drive.get(d, 0) >= 50:
            continue
        if chosen and D.sqdist(emb[i:i+1], emb[chosen]).min() < 0.5:
            continue  # near-duplicate of a kept segment (e.g. adjacent frames)
        chosen.append(i)
        per_drive[d] = per_drive.get(d, 0) + 1
    manifest = cand.take(chosen).select(["_rowid", "segment_id", "drive_id", "_distance"]) \
                   .rename_columns(["row_id", "segment_id", "drive_id", "distance"])
    report["curate"] = log("curate", candidates=cand.num_rows, kept=manifest.num_rows,
                           drives=len(per_drive))

    # 6. Publish: manifest as its own Lance table; tag both versions for reproducibility.
    ds.tags.create("job-0001-source", ds.version)
    mds = lance.write_dataset(manifest, str(root / "manifests.lance"))
    mds.tags.create("job-0001", mds.version)
    report["publish"] = log("publish", source_version=ds.version, manifest_rows=mds.count_rows(),
                            tags=list(ds.tags.list()))

    # 7. Load: a training reader resolves row IDs to blobs (camera and LiDAR bytes).
    src = lance.dataset(uri, version="job-0001-source")
    row_ids = lance.dataset(str(root / "manifests.lance"), version="job-0001").to_table(columns=["row_id"]).column(0).to_pylist()
    batch = row_ids[:32]
    cams = src.take_blobs("camera", ids=batch)
    lidars = src.take_blobs("lidar", ids=batch)
    nbytes = sum(len(b.read()) for b in cams) + sum(len(b.read()) for b in lidars)
    report["load"] = log("load", batch=len(batch), blob_bytes=nbytes)
    (root / "report.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
