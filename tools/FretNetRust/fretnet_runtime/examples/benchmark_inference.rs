use std::{path::Path, time::Instant};

use clap::Parser;
use fretnet_runtime::{
    benchmark_runtime, default_fixture_paths, default_model_path, load_regression_fixture,
    FeatureBatch, FretNetRuntime,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_os_t = default_model_path())]
    model_path: std::path::PathBuf,
    #[arg(long, default_value_t = 130)]
    frames: usize,
    #[arg(long, default_value_t = 50)]
    iterations: usize,
}

#[derive(Debug)]
struct FixtureBenchmarkResult {
    fixture_name: String,
    summary: fretnet_runtime::BenchmarkSummary,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut runtime = FretNetRuntime::load(&args.model_path)?;
    println!("Model: {}", runtime.model_path().display());
    println!("Model load time: {:?}", runtime.load_time());
    let model_load_time = runtime.load_time();

    let fixture_paths: Vec<_> = default_fixture_paths()
        .into_iter()
        .filter(|path| path.exists())
        .collect();

    if fixture_paths.is_empty() {
        println!("Benchmark mode: synthetic fallback (no default real fixtures found)");
        let features = FeatureBatch::synthetic(1, args.frames, 6, 144, 9);
        let summary = benchmark_runtime(&mut runtime, &features, args.iterations, model_load_time)?;

        println!("Input shape: {:?}", features.shape());
        println!("Iterations: {}", summary.iterations);
        println!("First inference: {:?}", summary.first_run);
        println!("Average inference: {:?}", summary.average_run);
        println!("Min inference: {:?}", summary.min_run);
        println!("Max inference: {:?}", summary.max_run);
        if let Some(p95) = summary.p95_run {
            println!("P95 inference: {:?}", p95);
        }
        println!("Last output tensors: {}", summary.last_output.tensors.len());

        return Ok(());
    }

    println!("Benchmark mode: real fixtures");

    let mut results = Vec::with_capacity(fixture_paths.len());
    for fixture_path in fixture_paths {
        let fixture_name = fixture_name(&fixture_path);
        let fixture_load_start = Instant::now();
        let fixture = load_regression_fixture(&fixture_path)?;
        let fixture_load_time = fixture_load_start.elapsed();
        let input_shape = fixture.features.shape();
        let summary = benchmark_runtime(
            &mut runtime,
            &fixture.features,
            args.iterations,
            model_load_time,
        )?;

        println!();
        println!("Fixture: {fixture_name}");
        println!("  Path: {}", fixture_path.display());
        println!("  Input shape: {:?}", input_shape);
        println!("  Fixture load time: {:?}", fixture_load_time);
        println!("  Runs: {}", summary.iterations);
        println!("  First inference: {:?}", summary.first_run);
        println!("  Average inference: {:?}", summary.average_run);
        println!("  Min inference: {:?}", summary.min_run);
        println!("  Max inference: {:?}", summary.max_run);
        if let Some(p95) = summary.p95_run {
            println!("  P95 inference: {:?}", p95);
        }

        results.push(FixtureBenchmarkResult {
            fixture_name,
            summary,
        });
    }

    let fastest = results
        .iter()
        .min_by_key(|result| result.summary.average_run)
        .expect("at least one fixture result must exist");
    let slowest = results
        .iter()
        .max_by_key(|result| result.summary.average_run)
        .expect("at least one fixture result must exist");
    let global_mean = results
        .iter()
        .map(|result| result.summary.average_run)
        .fold(std::time::Duration::ZERO, |acc, next| acc + next)
        / results.len() as u32;
    let global_max_latency = results
        .iter()
        .map(|result| result.summary.max_run)
        .max()
        .unwrap_or(std::time::Duration::ZERO);

    println!();
    println!("Summary:");
    println!("  Fixtures benchmarked: {}", results.len());
    println!(
        "  Fastest by mean latency: {} ({:?})",
        fastest.fixture_name, fastest.summary.average_run
    );
    println!(
        "  Slowest by mean latency: {} ({:?})",
        slowest.fixture_name, slowest.summary.average_run
    );
    println!("  Global mean of per-fixture means: {:?}", global_mean);
    println!("  Global max observed latency: {:?}", global_max_latency);

    Ok(())
}

fn fixture_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown_fixture")
        .to_owned()
}
