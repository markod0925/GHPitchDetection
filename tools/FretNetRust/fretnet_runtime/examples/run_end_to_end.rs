use clap::Parser;
use fretnet_runtime::{default_model_path, load_audio_for_frontend, FeatureExtractor, FrontendConfig, FretNetRuntime};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    audio_path: std::path::PathBuf,
    #[arg(long, default_value_os_t = default_model_path())]
    model_path: std::path::PathBuf,
    #[arg(long, default_value_t = 130)]
    max_frames: usize,
}

fn main() -> anyhow::Result<()> {
    let args = args();
    let frontend_config = FrontendConfig::default();
    let audio = load_audio_for_frontend(&args.audio_path, frontend_config.sample_rate)?;
    let extractor = FeatureExtractor::new(frontend_config)?;
    let hcqt = extractor.extract_hcqt(&audio.samples, audio.sample_rate)?;
    let batch = extractor.hcqt_to_batch(&hcqt, Some(args.max_frames))?;

    let mut runtime = FretNetRuntime::load(&args.model_path)?;
    let output = runtime.infer_features(&batch)?;
    let decoded = runtime.decode_output(&output)?;

    println!("Audio path: {}", args.audio_path.display());
    println!("Model path: {}", args.model_path.display());
    println!("HCQT shape: {:?}", hcqt.shape());
    println!("Model input shape: {:?}", batch.shape());
    println!("Outputs:");
    for tensor in &output.tensors {
        println!("  {} -> {:?}", tensor.name, tensor.data.shape());
    }
    println!("Decoded:");
    for tensor in &decoded.tensors {
        println!(
            "  {} -> shape={:?} min={:.6} max={:.6} mean={:.6}",
            tensor.name, tensor.shape, tensor.min, tensor.max, tensor.mean
        );
    }

    Ok(())
}

fn args() -> Args {
    Args::parse()
}
