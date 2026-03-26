use std::{fs::File, path::PathBuf};

use anyhow::Context;
use clap::Parser;
use ndarray_npy::NpzWriter;
use serde_json::json;

use fretnet_runtime::{
    default_model_path, load_audio_for_frontend, FeatureExtractor, FretNetRuntime, FrontendConfig,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    audio_path: PathBuf,
    #[arg(long, default_value_os_t = default_model_path())]
    model_path: PathBuf,
    #[arg(long)]
    output_prefix: PathBuf,
    #[arg(long)]
    max_frames: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let frontend_config = FrontendConfig::default();
    let audio = load_audio_for_frontend(&args.audio_path, frontend_config.sample_rate)?;
    let extractor = FeatureExtractor::new(frontend_config)?;
    let hcqt = extractor.extract_hcqt(&audio.samples, audio.sample_rate)?;
    let batch = extractor.hcqt_to_batch(&hcqt, args.max_frames)?;

    let mut runtime = FretNetRuntime::load(&args.model_path)?;
    let output = runtime.infer_features(&batch)?;

    let npz_path = args.output_prefix.with_extension("npz");
    let json_path = args.output_prefix.with_extension("json");
    if let Some(parent) = npz_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let file = File::create(&npz_path)
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
        "audio_path": args.audio_path,
        "model_path": args.model_path,
        "npz_path": npz_path,
        "hcqt_shape": hcqt.shape(),
        "batch_shape": batch.shape(),
        "tensor_count": output.tensors.len(),
        "tensors": tensor_metadata,
        "notes": "Raw Rust ONNX outputs only. Semantic decoding is intentionally performed in Python for consistency benchmarking.",
    });

    std::fs::write(&json_path, serde_json::to_vec_pretty(&metadata)?)
        .with_context(|| format!("failed to write {}", json_path.display()))?;

    println!("Audio path: {}", args.audio_path.display());
    println!("Model path: {}", args.model_path.display());
    println!("NPZ path: {}", npz_path.display());
    println!("JSON path: {}", json_path.display());

    Ok(())
}
