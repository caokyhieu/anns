// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
#![allow(clippy::print_stdout)]

//! IVF_RQ build + recall benchmark on raw GIST1M-style files.
//!
//! Environment:
//! - `GIST_DIR`: directory with base.f32, query.f32, gt.i32, meta.txt
//! - `DS_PATH`: Lance dataset path (written once if missing)
//! - `NUM_PARTITIONS` (default 1000), `NUM_BITS` (default 5)
//! - `NPROBES` comma list (default "10,20,50,100"), `K` (default 100)
//! - `SKIP_BUILD=1` to only query the existing index
//! - `OUT_JSON`: append one JSON line with results

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow_array::{
    FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
    cast::AsArray, types::Int32Type,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_arrow::FixedSizeListArrayExt;
use lance_index::IndexType;
use lance_index::progress::IndexBuildProgress;
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_linalg::distance::MetricType;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn read_f32(path: &Path) -> Vec<f32> {
    let mut bytes = Vec::new();
    BufReader::new(File::open(path).unwrap())
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn read_i32(path: &Path) -> Vec<i32> {
    let mut bytes = Vec::new();
    BufReader::new(File::open(path).unwrap())
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[derive(Debug, Default)]
struct StageTimer {
    started: Mutex<HashMap<String, Instant>>,
    done: Mutex<Vec<(String, f64)>>,
}

#[async_trait]
impl IndexBuildProgress for StageTimer {
    async fn stage_start(&self, stage: &str, _: Option<u64>, _: &str) -> lance_core::Result<()> {
        self.started
            .lock()
            .unwrap()
            .insert(stage.to_string(), Instant::now());
        Ok(())
    }
    async fn stage_progress(&self, _: &str, _: u64) -> lance_core::Result<()> {
        Ok(())
    }
    async fn stage_complete(&self, stage: &str) -> lance_core::Result<()> {
        if let Some(t) = self.started.lock().unwrap().remove(stage) {
            let secs = t.elapsed().as_secs_f64();
            println!("  stage {stage:<18} {secs:8.2}s");
            self.done.lock().unwrap().push((stage.to_string(), secs));
        }
        Ok(())
    }
}

async fn write_dataset(dir: &Path, uri: &str, n: usize, dim: usize) {
    let base = read_f32(&dir.join("base.f32"));
    assert_eq!(base.len(), n * dim);
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]));
    let rows_per_batch = 50_000;
    let batches: Vec<RecordBatch> = (0..n)
        .step_by(rows_per_batch)
        .map(|start| {
            let end = (start + rows_per_batch).min(n);
            let ids = Int32Array::from_iter_values(start as i32..end as i32);
            let values = Float32Array::from(base[start * dim..end * dim].to_vec());
            let fsl = FixedSizeListArray::try_new_from_values(values, dim as i32).unwrap();
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(fsl)]).unwrap()
        })
        .collect();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let params = WriteParams {
        max_rows_per_file: n,
        max_rows_per_group: 8192,
        mode: WriteMode::Create,
        ..Default::default()
    };
    Dataset::write(reader, uri, Some(params)).await.unwrap();
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let dir = std::path::PathBuf::from(std::env::var("GIST_DIR").expect("GIST_DIR"));
    let uri = std::env::var("DS_PATH").expect("DS_PATH");
    let meta = std::fs::read_to_string(dir.join("meta.txt")).unwrap();
    let meta: Vec<usize> = meta.split_whitespace().map(|v| v.parse().unwrap()).collect();
    let (n, dim, nq, gt_k) = (meta[0], meta[1], meta[2], meta[3]);

    let num_partitions: usize = env_or("NUM_PARTITIONS", 1000);
    let num_bits: u8 = env_or("NUM_BITS", 5);
    let k: usize = env_or("K", 100);
    let nprobes: Vec<usize> = std::env::var("NPROBES")
        .unwrap_or_else(|_| "10,20,50,100".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let label = std::env::var("LABEL").unwrap_or_else(|_| "run".into());

    if !Path::new(&uri).exists() {
        let t = Instant::now();
        write_dataset(&dir, &uri, n, dim).await;
        println!("wrote dataset in {:.1}s", t.elapsed().as_secs_f64());
    }
    let mut dataset = Dataset::open(&uri).await.unwrap();

    let mut stages = Vec::new();
    let mut build_secs = f64::NAN;
    if std::env::var("SKIP_BUILD").is_err() {
        let mut ivf = IvfBuildParams::new(num_partitions);
        ivf.max_iters = env_or("MAX_ITERS", 50);
        let rq = RQBuildParams::new(num_bits);
        let params = VectorIndexParams::with_ivf_rq_params(MetricType::L2, ivf, rq);
        let timer = Arc::new(StageTimer::default());
        println!("[{label}] building IVF{num_partitions}_RQ{num_bits} on {n}x{dim}");
        let t = Instant::now();
        dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vec_idx".into())
            .replace(true)
            .progress(timer.clone())
            .await
            .unwrap();
        build_secs = t.elapsed().as_secs_f64();
        println!("  total build        {build_secs:8.2}s");
        stages = timer.done.lock().unwrap().clone();
        dataset = Dataset::open(&uri).await.unwrap();
    }

    let queries = read_f32(&dir.join("query.f32"));
    let gt = read_i32(&dir.join("gt.i32"));
    let mut recall_rows = Vec::new();
    for &np in &nprobes {
        let mut hits10 = 0usize;
        let mut hits100 = 0usize;
        let t = Instant::now();
        for qi in 0..nq {
            let q = Float32Array::from(queries[qi * dim..(qi + 1) * dim].to_vec());
            let batches: Vec<RecordBatch> = dataset
                .scan()
                .project(&["id"])
                .unwrap()
                .nearest("vector", &q, k)
                .unwrap()
                .minimum_nprobes(np)
                .maximum_nprobes(np)
                .try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let ids: Vec<i32> = batches
                .iter()
                .flat_map(|b| {
                    b.column_by_name("id")
                        .unwrap()
                        .as_primitive::<Int32Type>()
                        .values()
                        .to_vec()
                })
                .collect();
            let truth = &gt[qi * gt_k..qi * gt_k + gt_k];
            let t10: std::collections::HashSet<i32> = truth[..10].iter().copied().collect();
            let t100: std::collections::HashSet<i32> = truth[..k.min(gt_k)].iter().copied().collect();
            hits10 += ids.iter().take(10).filter(|id| t10.contains(id)).count();
            hits100 += ids.iter().filter(|id| t100.contains(id)).count();
        }
        let secs = t.elapsed().as_secs_f64();
        let r10 = hits10 as f64 / (nq * 10) as f64;
        let r100 = hits100 as f64 / (nq * k.min(gt_k)) as f64;
        let ms = secs * 1000.0 / nq as f64;
        println!("  nprobes={np:<4} recall@10={r10:.4} recall@100={r100:.4} avg_latency={ms:.2}ms");
        recall_rows.push(format!(
            "{{\"nprobes\":{np},\"recall10\":{r10:.4},\"recall100\":{r100:.4},\"latency_ms\":{ms:.3}}}"
        ));
    }

    if let Ok(out) = std::env::var("OUT_JSON") {
        let stage_json: Vec<String> = stages
            .iter()
            .map(|(s, t)| format!("\"{s}\":{t:.3}"))
            .collect();
        let line = format!(
            "{{\"label\":\"{label}\",\"num_partitions\":{num_partitions},\"num_bits\":{num_bits},\"build_secs\":{build_secs:.3},\"stages\":{{{}}},\"search\":[{}]}}\n",
            stage_json.join(","),
            recall_rows.join(",")
        );
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(out)
            .unwrap()
            .write_all(line.as_bytes())
            .unwrap();
    }
    let _ = dataset.count_rows(None).await.unwrap();
}
