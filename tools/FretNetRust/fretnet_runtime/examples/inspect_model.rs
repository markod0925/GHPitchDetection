use clap::Parser;
use fretnet_runtime::{default_model_path, FretNetRuntime};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_os_t = default_model_path())]
    model_path: std::path::PathBuf,
}

fn main() -> anyhow::Result<()> {
    let runtime = FretNetRuntime::load(&args().model_path)?;
    println!("Model: {}", runtime.model_path().display());
    println!("Load time: {:?}", runtime.load_time());

    println!("Inputs:");
    for input in &runtime.metadata().inputs {
        println!(
            "  {} | {} | {:?}",
            input.name, input.element_type, input.dimensions
        );
    }

    println!("Outputs:");
    for output in &runtime.metadata().outputs {
        println!(
            "  {} | {} | {:?}",
            output.name, output.element_type, output.dimensions
        );
    }

    Ok(())
}

fn args() -> Args {
    Args::parse()
}
