use std::{path::PathBuf, time::Duration};

use anyhow::{anyhow, bail};
use clap::Parser;
use fretnet_runtime::{
    benchmark_end_to_end_audio, default_existing_stage_audio_inputs, default_model_path,
    FeatureExtractor, FretNetRuntime, FrontendConfig,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_os_t = default_model_path())]
    model_path: PathBuf,
    #[arg(long)]
    audio_path: Vec<PathBuf>,
    #[arg(long, default_value_t = 10)]
    iterations: usize,
    #[arg(long)]
    max_frames: Option<usize>,
}

#[derive(Debug, Clone)]
struct AudioBenchmarkTarget {
    display_name: String,
    audio_path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let targets = benchmark_targets(&args.audio_path)?;
    if targets.is_empty() {
        bail!("no benchmark audio inputs available");
    }

    let frontend_config = FrontendConfig::default();
    let extractor = FeatureExtractor::new(frontend_config)?;
    let mut runtime = FretNetRuntime::load(&args.model_path)?;

    println!("Model: {}", runtime.model_path().display());
    println!("Model load time: {:?}", runtime.load_time());
    println!("Iterations per input: {}", args.iterations.max(1));
    if let Some(max_frames) = args.max_frames {
        println!("Max frames: {}", max_frames);
    } else {
        println!("Max frames: full audio");
    }

    let mut summaries = Vec::with_capacity(targets.len());

    for target in targets {
        let summary = benchmark_end_to_end_audio(
            &mut runtime,
            &extractor,
            &target.audio_path,
            args.iterations,
            args.max_frames,
        )?;

        println!();
        println!("Input: {}", target.display_name);
        println!("  Audio path: {}", summary.audio_path.display());
        println!(
            "  Audio metadata: duration={:.3}s original_sr={} original_samples={} prepared_sr={} prepared_samples={}",
            summary.duration_seconds,
            summary.original_sample_rate,
            summary.original_num_samples,
            summary.prepared_sample_rate,
            summary.prepared_num_samples
        );
        println!("  HCQT shape: {:?}", summary.hcqt_shape);
        println!("  Model input shape: {:?}", summary.batch_shape);
        println!(
            "  Output tensors: {}",
            summary
                .output_tensors
                .iter()
                .map(|tensor| format!("{} {:?}", tensor.name, tensor.shape))
                .collect::<Vec<_>>()
                .join(", ")
        );
        print_stage("Stage A load", &summary.stage_load);
        print_stage("Stage B prep", &summary.stage_prepare);
        print_stage("  resample", &summary.stage_resample);
        print_stage("  normalize", &summary.stage_normalize);
        print_stage("Stage C frontend", &summary.stage_frontend);
        print_stage("Stage D inference", &summary.stage_inference);
        print_stage("Stage E total", &summary.stage_total);

        summaries.push(summary);
    }

    let fastest = summaries
        .iter()
        .min_by_key(|summary| summary.stage_total.average_run)
        .ok_or_else(|| anyhow!("no benchmark summaries collected"))?;
    let slowest = summaries
        .iter()
        .max_by_key(|summary| summary.stage_total.average_run)
        .ok_or_else(|| anyhow!("no benchmark summaries collected"))?;
    let global_mean_total = summaries
        .iter()
        .map(|summary| summary.stage_total.average_run)
        .fold(Duration::ZERO, |acc, next| acc + next)
        / summaries.len() as u32;
    let global_max_total = summaries
        .iter()
        .map(|summary| summary.stage_total.max_run)
        .max()
        .unwrap_or(Duration::ZERO);

    let average_load_pct =
        average_stage_fraction(&summaries, |summary| summary.stage_load.average_run);
    let average_prep_pct =
        average_stage_fraction(&summaries, |summary| summary.stage_prepare.average_run);
    let average_frontend_pct =
        average_stage_fraction(&summaries, |summary| summary.stage_frontend.average_run);
    let average_inference_pct =
        average_stage_fraction(&summaries, |summary| summary.stage_inference.average_run);

    println!();
    println!("Summary:");
    println!("  Inputs benchmarked: {}", summaries.len());
    println!(
        "  Fastest by total mean latency: {} ({:?})",
        fastest.audio_name, fastest.stage_total.average_run
    );
    println!(
        "  Slowest by total mean latency: {} ({:?})",
        slowest.audio_name, slowest.stage_total.average_run
    );
    println!("  Global mean of total means: {:?}", global_mean_total);
    println!(
        "  Global max observed total latency: {:?}",
        global_max_total
    );
    println!(
        "  Average stage contribution: load={:.1}% prep={:.1}% frontend={:.1}% inference={:.1}%",
        average_load_pct * 100.0,
        average_prep_pct * 100.0,
        average_frontend_pct * 100.0,
        average_inference_pct * 100.0
    );

    Ok(())
}

fn benchmark_targets(
    explicit_audio_paths: &[PathBuf],
) -> anyhow::Result<Vec<AudioBenchmarkTarget>> {
    if !explicit_audio_paths.is_empty() {
        return Ok(explicit_audio_paths
            .iter()
            .map(|path| AudioBenchmarkTarget {
                display_name: path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("explicit_audio")
                    .to_owned(),
                audio_path: path.clone(),
            })
            .collect());
    }

    let discovered = default_existing_stage_audio_inputs()?;
    if discovered.is_empty() {
        bail!(
            "no default real-audio benchmark inputs found; generate the stage fixtures or pass --audio-path explicitly"
        );
    }

    Ok(discovered
        .into_iter()
        .map(|input| AudioBenchmarkTarget {
            display_name: input.fixture_name,
            audio_path: input.audio_path,
        })
        .collect())
}

fn print_stage(label: &str, stats: &fretnet_runtime::StageLatencySummary) {
    println!(
        "  {label}: first={:?} mean={:?} min={:?} max={:?} p95={:?} runs={}",
        stats.first_run,
        stats.average_run,
        stats.min_run,
        stats.max_run,
        stats.p95_run,
        stats.iterations
    );
}

fn average_stage_fraction(
    summaries: &[fretnet_runtime::EndToEndBenchmarkSummary],
    stage_duration: impl Fn(&fretnet_runtime::EndToEndBenchmarkSummary) -> Duration,
) -> f64 {
    let mut total_fraction = 0.0f64;
    for summary in summaries {
        let total = summary.stage_total.average_run.as_secs_f64();
        if total > 0.0 {
            total_fraction += stage_duration(summary).as_secs_f64() / total;
        }
    }

    if summaries.is_empty() {
        0.0
    } else {
        total_fraction / summaries.len() as f64
    }
}
