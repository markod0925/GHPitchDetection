use clap::Parser;
use fretnet_runtime::{
    default_fixture_path, default_model_path, load_regression_fixture, FeatureBatch, FretNetRuntime,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_os_t = default_model_path())]
    model_path: std::path::PathBuf,
    #[arg(long)]
    fixture_path: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = 1)]
    batch: usize,
    #[arg(long, default_value_t = 130)]
    frames: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut runtime = FretNetRuntime::load(&args.model_path)?;
    let features = if let Some(path) = args.fixture_path.as_ref() {
        let fixture = load_regression_fixture(path)?;
        println!("Feature source: fixture ({})", path.display());
        fixture.features
    } else if default_fixture_path().exists() {
        let path = default_fixture_path();
        let fixture = load_regression_fixture(&path)?;
        println!("Feature source: default fixture ({})", path.display());
        fixture.features
    } else {
        println!("Feature source: synthetic");
        FeatureBatch::synthetic(args.batch, args.frames, 6, 144, 9)
    };
    let output = runtime.infer_features(&features)?;
    let decoded = runtime.decode_output(&output)?;

    println!("Input shape: {:?}", features.shape());
    println!("Raw outputs:");
    for tensor in &output.tensors {
        println!("  {} -> {:?}", tensor.name, tensor.data.shape());
    }

    println!("Decoded tensor stats:");
    for tensor in &decoded.tensors {
        println!(
            "  {} -> shape={:?} min={:.6} max={:.6} mean={:.6}",
            tensor.name, tensor.shape, tensor.min, tensor.max, tensor.mean
        );
    }

    if let Some(notes) = decoded.notes {
        println!("Decode note: {notes}");
    }

    Ok(())
}
