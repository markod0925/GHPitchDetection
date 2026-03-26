use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Context;
use clap::Parser;
use ndarray_npy::NpzWriter;
use serde::{Deserialize, Serialize};
use serde_json::json;

use fretnet_runtime::{
    default_model_path, load_audio_for_frontend, FeatureExtractor, FretNetRuntime, FrontendConfig,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    manifest_path: PathBuf,
    #[arg(long, default_value_os_t = default_model_path())]
    model_path: PathBuf,
    #[arg(long)]
    summary_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct BatchManifest {
    jobs: Vec<BatchJob>,
}

#[derive(Debug, Deserialize)]
struct BatchJob {
    file_id: String,
    audio_path: PathBuf,
    output_prefix: PathBuf,
    max_frames: Option<usize>,
}

#[derive(Debug, Serialize)]
struct BatchTimings {
    audio_load_seconds: f64,
    frontend_seconds: f64,
    inference_seconds: f64,
    write_output_seconds: f64,
    total_seconds: f64,
}

#[derive(Debug, Serialize)]
struct BatchJobResult {
    file_id: String,
    audio_path: PathBuf,
    output_prefix: PathBuf,
    status: String,
    npz_path: PathBuf,
    json_path: PathBuf,
    hcqt_shape: Option<Vec<usize>>,
    batch_shape: Option<Vec<usize>>,
    tensor_count: Option<usize>,
    timings: Option<BatchTimings>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BatchSummary {
    model_path: PathBuf,
    manifest_path: PathBuf,
    summary_path: PathBuf,
    frontend_sample_rate: u32,
    extractor_create_seconds: f64,
    runtime_load_seconds: f64,
    jobs: Vec<BatchJobResult>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let manifest_file = File::open(&args.manifest_path)
        .with_context(|| format!("failed to open {}", args.manifest_path.display()))?;
    let manifest: BatchManifest =
        serde_json::from_reader(manifest_file).context("failed to parse batch manifest JSON")?;

    let frontend_config = FrontendConfig::default();
    let frontend_sample_rate = frontend_config.sample_rate;

    let extractor_started = Instant::now();
    let extractor = FeatureExtractor::new(frontend_config)
        .context("failed to create feature extractor for batch runner")?;
    let extractor_create_seconds = extractor_started.elapsed().as_secs_f64();

    let runtime = {
        let started = Instant::now();
        let runtime = FretNetRuntime::load(&args.model_path)?;
        (runtime, started.elapsed().as_secs_f64())
    };
    let (mut runtime, runtime_load_seconds) = runtime;

    let mut job_results = Vec::with_capacity(manifest.jobs.len());
    for job in manifest.jobs {
        job_results.push(process_job(&mut runtime, &extractor, job));
    }

    let summary = BatchSummary {
        model_path: args.model_path,
        manifest_path: args.manifest_path,
        summary_path: args.summary_path.clone(),
        frontend_sample_rate,
        extractor_create_seconds,
        runtime_load_seconds,
        jobs: job_results,
    };

    if let Some(parent) = args.summary_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(&args.summary_path, serde_json::to_vec_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", args.summary_path.display()))?;

    println!("Processed {} jobs", summary.jobs.len());
    println!("Summary path: {}", args.summary_path.display());
    Ok(())
}

fn process_job(
    runtime: &mut FretNetRuntime,
    extractor: &FeatureExtractor,
    job: BatchJob,
) -> BatchJobResult {
    let npz_path = job.output_prefix.with_extension("npz");
    let json_path = job.output_prefix.with_extension("json");
    let total_started = Instant::now();

    let result = || -> anyhow::Result<(Vec<usize>, Vec<usize>, usize, BatchTimings)> {
        let audio_started = Instant::now();
        let audio =
            load_audio_for_frontend(&job.audio_path, FrontendConfig::default().sample_rate)?;
        let audio_load_seconds = audio_started.elapsed().as_secs_f64();

        let frontend_started = Instant::now();
        let hcqt = extractor.extract_hcqt(&audio.samples, audio.sample_rate)?;
        let hcqt_shape = hcqt.shape().to_vec();
        let batch = extractor.hcqt_to_batch(&hcqt, job.max_frames)?;
        let batch_shape = batch.shape().to_vec();
        let frontend_seconds = frontend_started.elapsed().as_secs_f64();

        let inference_started = Instant::now();
        let output = runtime.infer_features(&batch)?;
        let inference_seconds = inference_started.elapsed().as_secs_f64();

        let write_started = Instant::now();
        write_outputs(
            &npz_path,
            &json_path,
            &job.audio_path,
            runtime.model_path(),
            &hcqt_shape,
            &batch_shape,
            &output,
        )?;
        let write_output_seconds = write_started.elapsed().as_secs_f64();

        let timings = BatchTimings {
            audio_load_seconds,
            frontend_seconds,
            inference_seconds,
            write_output_seconds,
            total_seconds: total_started.elapsed().as_secs_f64(),
        };

        Ok((hcqt_shape, batch_shape, output.tensors.len(), timings))
    }();

    match result {
        Ok((hcqt_shape, batch_shape, tensor_count, timings)) => BatchJobResult {
            file_id: job.file_id,
            audio_path: job.audio_path,
            output_prefix: job.output_prefix,
            status: "ok".to_owned(),
            npz_path,
            json_path,
            hcqt_shape: Some(hcqt_shape),
            batch_shape: Some(batch_shape),
            tensor_count: Some(tensor_count),
            timings: Some(timings),
            error: None,
        },
        Err(err) => BatchJobResult {
            file_id: job.file_id,
            audio_path: job.audio_path,
            output_prefix: job.output_prefix,
            status: "failed".to_owned(),
            npz_path,
            json_path,
            hcqt_shape: None,
            batch_shape: None,
            tensor_count: None,
            timings: None,
            error: Some(format!("{err:#}")),
        },
    }
}

fn write_outputs(
    npz_path: &Path,
    json_path: &Path,
    audio_path: &Path,
    model_path: &Path,
    hcqt_shape: &[usize],
    batch_shape: &[usize],
    output: &fretnet_runtime::ModelOutput,
) -> anyhow::Result<()> {
    if let Some(parent) = npz_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let file = File::create(npz_path)
        .with_context(|| format!("failed to create {}", npz_path.display()))?;
    let mut archive = NpzWriter::new(file);
    for tensor in &output.tensors {
        archive
            .add_array(&tensor.name, &tensor.data)
            .with_context(|| format!("failed to write tensor {} to NPZ", tensor.name))?;
    }
    archive.finish().context("failed to finalize NPZ archive")?;

    let tensor_metadata: Vec<_> = output
        .tensors
        .iter()
        .map(|tensor| {
            json!({
                "name": tensor.name,
                "shape": tensor.data.shape(),
                "dtype": "f32",
            })
        })
        .collect();

    let metadata = json!({
        "audio_path": audio_path,
        "model_path": model_path,
        "npz_path": npz_path,
        "hcqt_shape": hcqt_shape,
        "batch_shape": batch_shape,
        "tensor_count": output.tensors.len(),
        "tensors": tensor_metadata,
        "notes": "Raw Rust ONNX outputs only. Semantic decoding is intentionally performed in Python for consistency benchmarking.",
    });

    std::fs::write(json_path, serde_json::to_vec_pretty(&metadata)?)
        .with_context(|| format!("failed to write {}", json_path.display()))?;

    Ok(())
}
