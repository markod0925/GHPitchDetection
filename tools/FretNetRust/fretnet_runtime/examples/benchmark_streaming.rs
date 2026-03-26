use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail};
use clap::{Parser, ValueEnum};
use fretnet_runtime::{
    benchmark_streaming_audio, default_existing_stage_audio_inputs, default_model_path,
    FeatureExtractor, FretNetRuntime, FrontendConfig, IncrementalStreamingPreset,
    InferenceScheduling, StreamingBenchmarkConfig, StreamingBenchmarkResult, StreamingFrontendMode,
    StreamingTimingMode,
};
use serde::Serialize;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_os_t = default_model_path())]
    model_path: PathBuf,
    #[arg(long)]
    audio_path: Vec<PathBuf>,
    #[arg(long)]
    audio_dir: Vec<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = "128,256,512,1024")]
    callback_sizes: Vec<usize>,
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    #[arg(long, default_value_t = 3.0)]
    analysis_window_seconds: f64,
    #[arg(long, default_value_t = 1.0)]
    incremental_context_seconds: f64,
    #[arg(long, default_value_t = 1)]
    incremental_inference_hops: usize,
    #[arg(long, value_enum)]
    preset: Option<PresetArg>,
    #[arg(long)]
    inference_every: Option<usize>,
    #[arg(long)]
    max_frames: Option<usize>,
    #[arg(long, value_delimiter = ',', value_enum, default_value = "virtual")]
    mode: Vec<TimingModeArg>,
    #[arg(long)]
    sleep_real_time: bool,
    #[arg(long, visible_alias = "no-sleep")]
    virtual_time: bool,
    #[arg(
        long,
        value_delimiter = ',',
        value_enum,
        default_value = "legacy,incremental"
    )]
    frontend_mode: Vec<FrontendModeArg>,
    #[arg(long)]
    output_json: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TimingModeArg {
    Virtual,
    Sleep,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FrontendModeArg {
    Legacy,
    Incremental,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PresetArg {
    ProductionSafe,
    Balanced,
}

#[derive(Debug, Clone)]
struct AudioBenchmarkTarget {
    display_name: String,
    audio_path: PathBuf,
}

#[derive(Debug, Serialize)]
struct StreamingBenchmarkJsonReport {
    model_path: PathBuf,
    model_load_time_seconds: f64,
    generated_unix_seconds: u64,
    results: Vec<StreamingBenchmarkResult>,
    aggregate: StreamingAggregateSummary,
}

#[derive(Debug, Serialize)]
struct StreamingAggregateSummary {
    configuration_count: usize,
    deadline_met_count: usize,
    deadline_met_rate: f64,
    worst_miss_rate: f64,
    worst_configuration: Option<String>,
    highest_p99_callback_ms: f64,
    highest_p99_configuration: Option<String>,
    minimum_callback_size_without_misses: Option<usize>,
    dominant_stage_histogram: BTreeMap<String, usize>,
    by_frontend_mode: Vec<FrontendModeAggregate>,
}

#[derive(Debug, Serialize)]
struct FrontendModeAggregate {
    frontend_mode: String,
    configuration_count: usize,
    mean_callback_ms: f64,
    mean_miss_rate: f64,
    mean_decision_latency_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ComparisonKey {
    audio_name: String,
    callback_size: usize,
    timing_mode: StreamingTimingMode,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let targets = benchmark_targets(&args.audio_path, &args.audio_dir)?;
    if targets.is_empty() {
        bail!("no streaming benchmark audio inputs available");
    }

    let requested_modes = requested_timing_modes(&args);
    let requested_frontend_modes = requested_frontend_modes(&args);
    let selected_preset = requested_preset(args.preset);
    let callback_sizes = selected_preset
        .map(|preset| vec![preset.config().callback_size])
        .unwrap_or_else(|| args.callback_sizes.clone());
    let incremental_context_seconds = selected_preset
        .map(|preset| preset.config().incremental_context_seconds)
        .unwrap_or(args.incremental_context_seconds);
    let incremental_inference_hops = selected_preset
        .map(|preset| preset.config().incremental_inference_hops)
        .unwrap_or(args.incremental_inference_hops.max(1));
    let frontend_config = FrontendConfig::default();
    let extractor = FeatureExtractor::new(frontend_config)?;
    let mut runtime = FretNetRuntime::load(&args.model_path)?;

    println!("Streaming realtime benchmark");
    println!("Model: {}", runtime.model_path().display());
    println!("Model load time: {:?}", runtime.load_time());
    println!("Iterations per configuration: {}", args.iterations);
    println!(
        "Timing modes: {}",
        requested_modes
            .iter()
            .map(timing_mode_label)
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Frontend modes: {}",
        requested_frontend_modes
            .iter()
            .map(frontend_mode_label)
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Callback sizes: {}",
        callback_sizes
            .iter()
            .map(|size| size.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if let Some(preset) = selected_preset {
        println!("Frozen incremental preset: {}", preset.config().name);
    }
    println!(
        "Legacy schedule: {}",
        scheduling_description(args.inference_every)
    );
    println!(
        "Legacy rolling window: {:.3}s | Incremental context: {:.3}s | Incremental inference hops: {}",
        args.analysis_window_seconds,
        incremental_context_seconds,
        incremental_inference_hops
    );
    if let Some(max_frames) = args.max_frames {
        println!("Max frames for batch conversion: {}", max_frames);
    } else {
        println!("Max frames for batch conversion: full window");
    }

    let mut results = Vec::new();
    for target in &targets {
        for callback_size in &callback_sizes {
            for timing_mode in &requested_modes {
                for frontend_mode in &requested_frontend_modes {
                    let mut config =
                        StreamingBenchmarkConfig::new(&target.audio_path, *callback_size);
                    config.iterations = args.iterations;
                    config.timing_mode = *timing_mode;
                    config.frontend_mode = *frontend_mode;
                    config.analysis_window_seconds = args.analysis_window_seconds;
                    config.incremental_context_seconds = incremental_context_seconds;
                    config.incremental_inference_hops = incremental_inference_hops;
                    config.max_frames = args.max_frames;
                    config.inference_schedule = match args.inference_every {
                        Some(callbacks) => InferenceScheduling::EveryNCallbacks {
                            callbacks: callbacks.max(1),
                        },
                        None => InferenceScheduling::HopCadence,
                    };

                    let result = benchmark_streaming_audio(&mut runtime, &extractor, &config)?;
                    print_result_summary(target, &result);
                    results.push(result);
                }
            }
        }
    }

    print_frontend_comparison(&results);
    let aggregate = build_aggregate_summary(&results)?;
    print_aggregate_summary(&aggregate);

    if let Some(output_json) = args.output_json {
        if let Some(parent) = output_json.parent() {
            fs::create_dir_all(parent)?;
        }
        let generated_unix_seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let report = StreamingBenchmarkJsonReport {
            model_path: args.model_path,
            model_load_time_seconds: runtime.load_time().as_secs_f64(),
            generated_unix_seconds,
            results,
            aggregate,
        };
        fs::write(&output_json, serde_json::to_vec_pretty(&report)?)?;
        println!();
        println!("JSON report: {}", output_json.display());
    }

    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    if args.iterations == 0 {
        bail!("--iterations must be positive");
    }
    if args.preset.is_none() {
        if args.callback_sizes.is_empty() {
            bail!("--callback-sizes must contain at least one value");
        }
        if args.callback_sizes.iter().any(|size| *size == 0) {
            bail!("--callback-sizes cannot include 0");
        }
    }
    if !(args.analysis_window_seconds.is_finite() && args.analysis_window_seconds > 0.0) {
        bail!("--analysis-window-seconds must be > 0");
    }
    if !(args.incremental_context_seconds.is_finite() && args.incremental_context_seconds > 0.0) {
        bail!("--incremental-context-seconds must be > 0");
    }
    if args.incremental_inference_hops == 0 {
        bail!("--incremental-inference-hops must be positive");
    }
    if let Some(inference_every) = args.inference_every {
        if inference_every == 0 {
            bail!("--inference-every must be positive");
        }
    }
    Ok(())
}

fn requested_timing_modes(args: &Args) -> Vec<StreamingTimingMode> {
    if args.virtual_time || args.sleep_real_time {
        let mut modes = Vec::new();
        if args.virtual_time {
            modes.push(StreamingTimingMode::VirtualNoSleep);
        }
        if args.sleep_real_time {
            modes.push(StreamingTimingMode::SleepRealTime);
        }
        return modes;
    }

    let mut modes = Vec::new();
    for mode in &args.mode {
        match mode {
            TimingModeArg::Virtual => modes.push(StreamingTimingMode::VirtualNoSleep),
            TimingModeArg::Sleep => modes.push(StreamingTimingMode::SleepRealTime),
        }
    }
    if modes.is_empty() {
        vec![StreamingTimingMode::VirtualNoSleep]
    } else {
        modes
    }
}

fn requested_frontend_modes(args: &Args) -> Vec<StreamingFrontendMode> {
    let mut modes = Vec::new();
    for mode in &args.frontend_mode {
        match mode {
            FrontendModeArg::Legacy => modes.push(StreamingFrontendMode::LegacyRollingWindow),
            FrontendModeArg::Incremental => modes.push(StreamingFrontendMode::IncrementalPrototype),
        }
    }
    if modes.is_empty() {
        vec![StreamingFrontendMode::LegacyRollingWindow]
    } else {
        modes
    }
}

fn requested_preset(value: Option<PresetArg>) -> Option<IncrementalStreamingPreset> {
    value.map(|value| match value {
        PresetArg::ProductionSafe => IncrementalStreamingPreset::ProductionSafe,
        PresetArg::Balanced => IncrementalStreamingPreset::Balanced,
    })
}

fn benchmark_targets(
    explicit_audio_paths: &[PathBuf],
    audio_dirs: &[PathBuf],
) -> anyhow::Result<Vec<AudioBenchmarkTarget>> {
    let mut paths = BTreeSet::<PathBuf>::new();
    for path in explicit_audio_paths {
        if !path.exists() {
            bail!("audio file does not exist: {}", path.display());
        }
        paths.insert(path.clone());
    }

    for dir in audio_dirs {
        if !dir.exists() {
            bail!("audio directory does not exist: {}", dir.display());
        }
        if !dir.is_dir() {
            bail!("audio directory is not a directory: {}", dir.display());
        }
        collect_wavs_recursive(dir, &mut paths)?;
    }

    if paths.is_empty() {
        let discovered = default_existing_stage_audio_inputs()?;
        for input in discovered {
            paths.insert(input.audio_path);
        }
    }

    if paths.is_empty() {
        bail!(
            "no benchmark audio inputs found; pass --audio-path, --audio-dir, or generate stage fixtures"
        );
    }

    Ok(paths
        .into_iter()
        .map(|audio_path| AudioBenchmarkTarget {
            display_name: audio_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("audio")
                .to_owned(),
            audio_path,
        })
        .collect())
}

fn collect_wavs_recursive(root: &Path, out: &mut BTreeSet<PathBuf>) -> anyhow::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_wavs_recursive(&path, out)?;
            continue;
        }
        let is_wav = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("wav"))
            .unwrap_or(false);
        if is_wav {
            out.insert(path);
        }
    }
    Ok(())
}

fn print_result_summary(target: &AudioBenchmarkTarget, result: &StreamingBenchmarkResult) {
    println!();
    println!("Input: {}", target.display_name);
    println!("  Audio path: {}", result.audio_path.display());
    println!(
        "  Frontend mode: {} | timing mode: {}",
        frontend_mode_label(&result.frontend_mode),
        timing_mode_label(&result.mode)
    );
    println!(
        "  Callback: {} samples (budget {:.3} ms)",
        result.callback_size, result.callback_budget_ms
    );
    println!("  Schedule: {}", result.inference_schedule);
    println!(
        "  Context window: {:.3}s ({} samples @ stream SR)",
        result.context_window_seconds, result.context_window_samples
    );
    println!(
        "  Stream span: {:.3}s (source {:.3}s @ {} Hz -> stream @ {} Hz)",
        result.simulated_stream_duration_seconds,
        result.source_audio_duration_seconds,
        result.source_audio_sample_rate,
        result.stream_input_sample_rate
    );
    println!(
        "  Inference invocations: {} / callbacks {}",
        result.inference_invocations, result.total_callbacks
    );
    print_distribution("  Callback processing", &result.callbacks.processing_ms);
    print_distribution("  Decision latency", &result.decision_latency_ms);
    print_distribution("  Trigger latency", &result.trigger_latency_ms);
    println!(
        "  Deadlines: misses={} ({:.2}%), backlog mean={:.3} ms max={:.3} ms",
        result.deadline.miss_count,
        result.deadline.miss_rate * 100.0,
        result.deadline.mean_backlog_ms,
        result.deadline.max_backlog_ms
    );
    println!(
        "  Stage means/callback: prep={:.3} ms frontend={:.3} ms inference={:.3} ms other={:.3} ms",
        result.stage_breakdown_ms.audio_prep_mean_ms_per_callback,
        result.stage_breakdown_ms.frontend_mean_ms_per_callback,
        result.stage_breakdown_ms.inference_mean_ms_per_callback,
        result.stage_breakdown_ms.other_mean_ms_per_callback
    );
    println!(
        "  Stage fractions: prep={:.1}% frontend={:.1}% inference={:.1}% other={:.1}% (dominant: {})",
        result.stage_breakdown_ms.audio_prep_fraction_of_processing * 100.0,
        result.stage_breakdown_ms.frontend_fraction_of_processing * 100.0,
        result.stage_breakdown_ms.inference_fraction_of_processing * 100.0,
        result.stage_breakdown_ms.other_fraction_of_processing * 100.0,
        result.dominant_stage
    );
    println!("  Incrementality: {}", result.incrementality_note);
    println!("  Conclusion: {}", result.conclusion);
}

fn print_frontend_comparison(results: &[StreamingBenchmarkResult]) {
    let mut legacy = BTreeMap::<ComparisonKey, &StreamingBenchmarkResult>::new();
    let mut incremental = BTreeMap::<ComparisonKey, &StreamingBenchmarkResult>::new();

    for result in results {
        let key = ComparisonKey {
            audio_name: result.audio_name.clone(),
            callback_size: result.callback_size,
            timing_mode: result.mode,
        };
        match result.frontend_mode {
            StreamingFrontendMode::LegacyRollingWindow => {
                legacy.insert(key, result);
            }
            StreamingFrontendMode::IncrementalPrototype => {
                incremental.insert(key, result);
            }
        }
    }

    let mut printed_header = false;
    for (key, old) in legacy {
        let Some(new) = incremental.get(&key) else {
            continue;
        };
        if !printed_header {
            println!();
            println!("Legacy vs incremental comparison:");
            printed_header = true;
        }
        let callback_gain = relative_improvement(
            old.callbacks.processing_ms.mean_ms,
            new.callbacks.processing_ms.mean_ms,
        );
        let miss_gain = relative_improvement(old.deadline.miss_rate, new.deadline.miss_rate);
        println!(
            "  {} | mode={} | cb={}: callback_mean {:.3} -> {:.3} ms ({:+.1}%), miss_rate {:.2}% -> {:.2}% ({:+.1}%)",
            key.audio_name,
            timing_mode_label(&key.timing_mode),
            key.callback_size,
            old.callbacks.processing_ms.mean_ms,
            new.callbacks.processing_ms.mean_ms,
            callback_gain * 100.0,
            old.deadline.miss_rate * 100.0,
            new.deadline.miss_rate * 100.0,
            miss_gain * 100.0
        );
    }
}

fn print_distribution(label: &str, stats: &fretnet_runtime::DistributionStats) {
    println!(
        "{}: mean={:.3} ms p95={} p99={} max={} samples={}",
        label,
        stats.mean_ms,
        fmt_opt_ms(stats.p95_ms),
        fmt_opt_ms(stats.p99_ms),
        fmt_opt_ms(stats.max_ms),
        stats.count
    );
}

fn fmt_opt_ms(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3} ms"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn timing_mode_label(mode: &StreamingTimingMode) -> &'static str {
    match mode {
        StreamingTimingMode::VirtualNoSleep => "virtual/no-sleep",
        StreamingTimingMode::SleepRealTime => "sleep/real-time",
    }
}

fn frontend_mode_label(mode: &StreamingFrontendMode) -> &'static str {
    match mode {
        StreamingFrontendMode::LegacyRollingWindow => "legacy-rolling",
        StreamingFrontendMode::IncrementalPrototype => "incremental-prototype",
    }
}

fn scheduling_description(inference_every: Option<usize>) -> String {
    match inference_every {
        Some(callbacks) => format!("every {callbacks} callbacks"),
        None => "hop cadence".to_owned(),
    }
}

fn build_aggregate_summary(
    results: &[StreamingBenchmarkResult],
) -> anyhow::Result<StreamingAggregateSummary> {
    if results.is_empty() {
        return Err(anyhow!("no streaming benchmark results collected"));
    }

    let configuration_count = results.len();
    let deadline_met_count = results.iter().filter(|result| result.deadline_met).count();
    let deadline_met_rate = deadline_met_count as f64 / configuration_count as f64;

    let worst = results
        .iter()
        .max_by(|left, right| left.deadline.miss_rate.total_cmp(&right.deadline.miss_rate));
    let worst_miss_rate = worst.map(|result| result.deadline.miss_rate).unwrap_or(0.0);
    let worst_configuration = worst.map(configuration_label);

    let highest_p99 = results
        .iter()
        .filter_map(|result| {
            result
                .callbacks
                .processing_ms
                .p99_ms
                .map(|p99| (result, p99))
        })
        .max_by(|left, right| left.1.total_cmp(&right.1));
    let highest_p99_callback_ms = highest_p99.map(|(_, p99)| p99).unwrap_or(0.0);
    let highest_p99_configuration = highest_p99.map(|(result, _)| configuration_label(result));

    let minimum_callback_size_without_misses = results
        .iter()
        .filter(|result| result.deadline_met)
        .map(|result| result.callback_size)
        .min();

    let mut dominant_stage_histogram = BTreeMap::<String, usize>::new();
    for result in results {
        *dominant_stage_histogram
            .entry(result.dominant_stage.clone())
            .or_default() += 1;
    }

    let by_frontend_mode = summarize_frontend_modes(results);

    Ok(StreamingAggregateSummary {
        configuration_count,
        deadline_met_count,
        deadline_met_rate,
        worst_miss_rate,
        worst_configuration,
        highest_p99_callback_ms,
        highest_p99_configuration,
        minimum_callback_size_without_misses,
        dominant_stage_histogram,
        by_frontend_mode,
    })
}

fn summarize_frontend_modes(results: &[StreamingBenchmarkResult]) -> Vec<FrontendModeAggregate> {
    let mut groups = BTreeMap::<String, Vec<&StreamingBenchmarkResult>>::new();
    for result in results {
        groups
            .entry(frontend_mode_label(&result.frontend_mode).to_owned())
            .or_default()
            .push(result);
    }

    let mut out = Vec::new();
    for (mode, items) in groups {
        let count = items.len();
        let mean_callback_ms = items
            .iter()
            .map(|item| item.callbacks.processing_ms.mean_ms)
            .sum::<f64>()
            / count as f64;
        let mean_miss_rate = items
            .iter()
            .map(|item| item.deadline.miss_rate)
            .sum::<f64>()
            / count as f64;
        let mean_decision_latency_ms = items
            .iter()
            .map(|item| item.decision_latency_ms.mean_ms)
            .sum::<f64>()
            / count as f64;

        out.push(FrontendModeAggregate {
            frontend_mode: mode,
            configuration_count: count,
            mean_callback_ms,
            mean_miss_rate,
            mean_decision_latency_ms,
        });
    }
    out
}

fn print_aggregate_summary(summary: &StreamingAggregateSummary) {
    println!();
    println!("Overall summary:");
    println!("  Configurations tested: {}", summary.configuration_count);
    println!(
        "  Deadline met: {} ({:.1}%)",
        summary.deadline_met_count,
        summary.deadline_met_rate * 100.0
    );
    println!(
        "  Worst miss rate: {:.2}% ({})",
        summary.worst_miss_rate * 100.0,
        summary.worst_configuration.as_deref().unwrap_or("n/a")
    );
    println!(
        "  Highest callback p99: {:.3} ms ({})",
        summary.highest_p99_callback_ms,
        summary
            .highest_p99_configuration
            .as_deref()
            .unwrap_or("n/a")
    );
    println!(
        "  Minimum callback size with zero misses: {}",
        summary
            .minimum_callback_size_without_misses
            .map(|size| size.to_string())
            .unwrap_or_else(|| "none".to_owned())
    );
    println!(
        "  Dominant stages: {}",
        summary
            .dominant_stage_histogram
            .iter()
            .map(|(stage, count)| format!("{stage}={count}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for mode in &summary.by_frontend_mode {
        println!(
            "  Mode {}: configs={} mean_callback={:.3} ms mean_miss_rate={:.2}% mean_decision_latency={:.3} ms",
            mode.frontend_mode,
            mode.configuration_count,
            mode.mean_callback_ms,
            mode.mean_miss_rate * 100.0,
            mode.mean_decision_latency_ms
        );
    }
}

fn relative_improvement(old: f64, new: f64) -> f64 {
    if old.abs() <= f64::EPSILON {
        0.0
    } else {
        (old - new) / old
    }
}

fn configuration_label(result: &StreamingBenchmarkResult) -> String {
    format!(
        "{} | frontend={} | mode={} | cb={} | schedule={}",
        result.audio_name,
        frontend_mode_label(&result.frontend_mode),
        timing_mode_label(&result.mode),
        result.callback_size,
        result.inference_schedule
    )
}
