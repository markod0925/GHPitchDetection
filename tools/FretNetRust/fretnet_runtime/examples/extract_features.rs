use clap::Parser;
use fretnet_runtime::{load_audio_for_frontend, FeatureExtractor, FrontendConfig};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    audio_path: std::path::PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let audio = load_audio_for_frontend(&args.audio_path, FrontendConfig::default().sample_rate)?;
    let extractor = FeatureExtractor::new(FrontendConfig::default())?;
    let hcqt = extractor.extract_hcqt(&audio.samples, audio.sample_rate)?;

    let data = hcqt.data();
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().copied().map(f64::from).sum::<f64>() / data.len() as f64;

    println!("Audio path: {}", args.audio_path.display());
    println!("Sample rate: {}", audio.sample_rate);
    println!("Audio samples: {}", audio.samples.len());
    println!("HCQT shape: {:?}", hcqt.shape());
    println!("HCQT stats: min={:.6} max={:.6} mean={:.6}", min, max, mean);

    Ok(())
}
