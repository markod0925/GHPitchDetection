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
    StreamingBenchmarkConfig, StreamingBenchmarkResult, StreamingFrontendMode, StreamingTimingMode,
};
use serde::Serialize;

const DEFAULT_CALLBACK_SIZES: &str = "128,256,512,1024";
const DEFAULT_CONTEXT_SECONDS: &str = "0.5,0.75,1.0,1.25,1.5";
const DEFAULT_INFERENCE_HOPS: &str = "1,2,4,8";

// Practical realtime classification rule for incremental sweep configurations.
const GO_MAX_MISS_RATE: f64 = 0.01;
const GO_MAX_P95_BUDGET_UTILIZATION: f64 = 0.85;
const GO_MAX_BACKLOG_BUDGET_MULTIPLIER: f64 = 1.0;
const GO_MAX_DECISION_P95_MS: f64 = 120.0;

const BORDERLINE_MAX_MISS_RATE: f64 = 0.10;
const BORDERLINE_MAX_P95_BUDGET_UTILIZATION: f64 = 1.20;
const BORDERLINE_MAX_BACKLOG_BUDGET_MULTIPLIER: f64 = 4.0;
const BORDERLINE_MAX_DECISION_P95_MS: f64 = 350.0;

// Lower score is better.
// score = 500*miss_rate + 0.35*callback_p95_ms + 0.05*decision_p95_ms + 0.03*max_backlog_ms
const SCORE_MISS_WEIGHT: f64 = 500.0;
const SCORE_CALLBACK_P95_WEIGHT: f64 = 0.35;
const SCORE_DECISION_P95_WEIGHT: f64 = 0.05;
const SCORE_BACKLOG_MAX_WEIGHT: f64 = 0.03;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_os_t = default_model_path())]
    model_path: PathBuf,
    #[arg(long)]
    audio_path: Vec<PathBuf>,
    #[arg(long)]
    audio_dir: Vec<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_CALLBACK_SIZES)]
    callback_sizes: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_CONTEXT_SECONDS)]
    context_seconds: Vec<f64>,
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_INFERENCE_HOPS)]
    inference_hops: Vec<usize>,
    #[arg(long, value_delimiter = ',', value_enum)]
    preset: Vec<PresetArg>,
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    #[arg(long)]
    max_frames: Option<usize>,
    #[arg(long, value_delimiter = ',', value_enum, default_value = "virtual")]
    mode: Vec<TimingModeArg>,
    #[arg(long, default_value_t = 0)]
    sleep_validate_top_k: usize,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long)]
    output_csv: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TimingModeArg {
    Virtual,
    Sleep,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PresetArg {
    ProductionSafe,
    Balanced,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RealtimeClassification {
    Go,
    Borderline,
    NoGo,
}

#[derive(Debug, Clone)]
struct AudioBenchmarkTarget {
    audio_path: PathBuf,
    display_name: String,
}

#[derive(Debug, Clone, Serialize)]
struct SweepParameterConfig {
    callback_sizes: Vec<usize>,
    context_seconds: Vec<f64>,
    inference_hops: Vec<usize>,
    frozen_presets: Vec<String>,
    modes: Vec<String>,
    iterations: usize,
    max_frames: Option<usize>,
    sleep_validate_top_k: usize,
    top_k: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SweepConfigResult {
    mode: StreamingTimingMode,
    callback_size: usize,
    incremental_context_seconds: f64,
    incremental_inference_hops: usize,
    files_processed: usize,
    file_audio_names: Vec<String>,
    total_callbacks: usize,
    total_inference_invocations: usize,
    callback_budget_ms: f64,
    callback_mean_ms: f64,
    callback_p95_ms: Option<f64>,
    callback_p99_ms: Option<f64>,
    callback_max_ms: Option<f64>,
    deadline_miss_count: usize,
    deadline_miss_rate: f64,
    backlog_mean_ms: f64,
    backlog_max_ms: f64,
    decision_latency_mean_ms: f64,
    decision_latency_p95_ms: Option<f64>,
    decision_latency_max_ms: Option<f64>,
    trigger_latency_mean_ms: f64,
    trigger_latency_p95_ms: Option<f64>,
    trigger_latency_max_ms: Option<f64>,
    stage_audio_prep_mean_ms_per_callback: f64,
    stage_frontend_mean_ms_per_callback: f64,
    stage_inference_mean_ms_per_callback: f64,
    stage_other_mean_ms_per_callback: f64,
    stage_audio_prep_fraction: f64,
    stage_frontend_fraction: f64,
    stage_inference_fraction: f64,
    stage_other_fraction: f64,
    dominant_stage: String,
    realtime_classification: RealtimeClassification,
    score: f64,
}

#[derive(Debug, Clone, Serialize)]
struct RankedConfigRef {
    rank: usize,
    mode: String,
    callback_size: usize,
    incremental_context_seconds: f64,
    incremental_inference_hops: usize,
    realtime_classification: RealtimeClassification,
    score: f64,
    deadline_miss_rate: f64,
    callback_p95_ms: Option<f64>,
    decision_latency_p95_ms: Option<f64>,
    backlog_max_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SweepRankingSummary {
    best_overall: Vec<RankedConfigRef>,
    best_go: Vec<RankedConfigRef>,
    best_low_latency: Vec<RankedConfigRef>,
    best_throughput_safe: Vec<RankedConfigRef>,
    worst: Vec<RankedConfigRef>,
}

#[derive(Debug, Clone, Serialize)]
struct MetricSensitivitySummary {
    metric: String,
    strongest_parameter: String,
    callback_size_effect: f64,
    context_seconds_effect: f64,
    inference_hops_effect: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CallbackSizeSummary {
    callback_size: usize,
    configuration_count: usize,
    go_count: usize,
    borderline_count: usize,
    no_go_count: usize,
    go_rate: f64,
    mean_miss_rate: f64,
    mean_callback_p95_ms: f64,
    mean_decision_p95_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OperatingEnvelopeAnalysis {
    strongest_effects: Vec<MetricSensitivitySummary>,
    callback_size_summary: Vec<CallbackSizeSummary>,
    workable_region_1024: bool,
    callback_512_salvageable: bool,
    callbacks_128_256_outside_practical_range: bool,
    robust_safe_region: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SweepReport {
    model_path: PathBuf,
    model_load_time_seconds: f64,
    generated_unix_seconds: u64,
    input_audio_paths: Vec<PathBuf>,
    parameters: SweepParameterConfig,
    classification_rule: String,
    score_formula: String,
    stage1_virtual_configuration_count: usize,
    stage2_sleep_validation_count: usize,
    results: Vec<SweepConfigResult>,
    ranking: SweepRankingSummary,
    analysis: OperatingEnvelopeAnalysis,
}

#[derive(Debug, Clone)]
struct SweepPoint {
    callback_size: usize,
    context_seconds: f64,
    inference_hops: usize,
    mode: StreamingTimingMode,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let targets = benchmark_targets(&args.audio_path, &args.audio_dir)?;
    if targets.is_empty() {
        bail!("no benchmark audio inputs available");
    }

    let modes = requested_timing_modes(&args.mode);
    let presets = requested_presets(&args.preset);
    let sweep_points = if presets.is_empty() {
        build_sweep_points(
            &args.callback_sizes,
            &args.context_seconds,
            &args.inference_hops,
            &modes,
        )
    } else {
        build_sweep_points_from_presets(&presets, &modes)
    };

    let extractor = FeatureExtractor::new(FrontendConfig::default())?;
    let mut runtime = FretNetRuntime::load(&args.model_path)?;

    println!("Streaming incremental sweep");
    println!("Model: {}", runtime.model_path().display());
    println!("Model load time: {:?}", runtime.load_time());
    println!(
        "Inputs: {}",
        targets
            .iter()
            .map(|target| target.display_name.clone())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if presets.is_empty() {
        println!(
            "Sweep sizes: callbacks={} context-values={} hops-values={} modes={}",
            args.callback_sizes.len(),
            args.context_seconds.len(),
            args.inference_hops.len(),
            modes.len()
        );
    } else {
        println!(
            "Frozen presets: {}",
            presets
                .iter()
                .map(|preset| preset.config().name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "Modes: {}",
            modes
                .iter()
                .map(|mode| mode_label(*mode))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("Total stage-1 sweep configurations: {}", sweep_points.len());
    println!("Iterations per config: {}", args.iterations);

    let mut results = run_sweep(
        &mut runtime,
        &extractor,
        &targets,
        &sweep_points,
        args.iterations,
        args.max_frames,
    )?;
    let stage1_virtual_count = results
        .iter()
        .filter(|result| result.mode == StreamingTimingMode::VirtualNoSleep)
        .count();

    let mut stage2_sleep_count = 0usize;
    if args.sleep_validate_top_k > 0 && modes.contains(&StreamingTimingMode::VirtualNoSleep) {
        let sleep_points = top_virtual_sleep_validation_points(&results, args.sleep_validate_top_k);
        if !sleep_points.is_empty() {
            println!();
            println!(
                "Stage-2 sleep validation on top {} virtual configurations",
                sleep_points.len()
            );
            let sleep_results = run_sweep(
                &mut runtime,
                &extractor,
                &targets,
                &sleep_points,
                args.iterations,
                args.max_frames,
            )?;
            stage2_sleep_count = sleep_results.len();
            results.extend(sleep_results);
        }
    }

    results.sort_by(config_sort_key);
    let ranking = build_rankings(&results, args.top_k.max(1));
    let analysis = analyze_operating_envelope(&results);
    print_summary(&results, &ranking, &analysis, args.top_k.max(1));

    let report = SweepReport {
        model_path: args.model_path.clone(),
        model_load_time_seconds: runtime.load_time().as_secs_f64(),
        generated_unix_seconds: now_unix_seconds(),
        input_audio_paths: targets
            .iter()
            .map(|target| target.audio_path.clone())
            .collect(),
        parameters: SweepParameterConfig {
            callback_sizes: sorted_unique_usize(
                sweep_points
                    .iter()
                    .map(|point| point.callback_size)
                    .collect(),
            ),
            context_seconds: sorted_unique_f64(
                sweep_points
                    .iter()
                    .map(|point| point.context_seconds)
                    .collect(),
            ),
            inference_hops: sorted_unique_usize(
                sweep_points
                    .iter()
                    .map(|point| point.inference_hops)
                    .collect(),
            ),
            frozen_presets: presets
                .iter()
                .map(|preset| preset.config().name.to_owned())
                .collect(),
            modes: modes.iter().map(mode_label_owned).collect(),
            iterations: args.iterations,
            max_frames: args.max_frames,
            sleep_validate_top_k: args.sleep_validate_top_k,
            top_k: args.top_k.max(1),
        },
        classification_rule: classification_rule_text(),
        score_formula: score_formula_text(),
        stage1_virtual_configuration_count: stage1_virtual_count,
        stage2_sleep_validation_count: stage2_sleep_count,
        results: results.clone(),
        ranking,
        analysis,
    };

    if let Some(path) = args.output_json {
        write_json(&path, &report)?;
        println!();
        println!("JSON report written: {}", path.display());
    }
    if let Some(path) = args.output_csv {
        write_csv(&path, &results)?;
        println!("CSV table written: {}", path.display());
    }

    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    if args.iterations == 0 {
        bail!("--iterations must be positive");
    }
    if args.preset.is_empty() {
        if args.callback_sizes.is_empty() || args.callback_sizes.iter().any(|value| *value == 0) {
            bail!("--callback-sizes must contain positive values");
        }
        if args.context_seconds.is_empty()
            || args
                .context_seconds
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            bail!("--context-seconds must contain finite values > 0");
        }
        if args.inference_hops.is_empty() || args.inference_hops.iter().any(|value| *value == 0) {
            bail!("--inference-hops must contain positive values");
        }
    }
    Ok(())
}

fn requested_timing_modes(values: &[TimingModeArg]) -> Vec<StreamingTimingMode> {
    let mut out = Vec::new();
    for value in values {
        match value {
            TimingModeArg::Virtual => out.push(StreamingTimingMode::VirtualNoSleep),
            TimingModeArg::Sleep => out.push(StreamingTimingMode::SleepRealTime),
        }
    }
    if out.is_empty() {
        vec![StreamingTimingMode::VirtualNoSleep]
    } else {
        out
    }
}

fn requested_presets(values: &[PresetArg]) -> Vec<IncrementalStreamingPreset> {
    let mut out = Vec::new();
    for value in values {
        let preset = match value {
            PresetArg::ProductionSafe => IncrementalStreamingPreset::ProductionSafe,
            PresetArg::Balanced => IncrementalStreamingPreset::Balanced,
        };
        if !out.contains(&preset) {
            out.push(preset);
        }
    }
    out
}

fn build_sweep_points(
    callback_sizes: &[usize],
    context_seconds: &[f64],
    inference_hops: &[usize],
    modes: &[StreamingTimingMode],
) -> Vec<SweepPoint> {
    let mut out = Vec::new();
    let callback_sizes = sorted_unique_usize(callback_sizes.to_vec());
    let context_seconds = sorted_unique_f64(context_seconds.to_vec());
    let inference_hops = sorted_unique_usize(inference_hops.to_vec());

    for &mode in modes {
        for &callback_size in &callback_sizes {
            for &context_seconds in &context_seconds {
                for &inference_hops in &inference_hops {
                    out.push(SweepPoint {
                        callback_size,
                        context_seconds,
                        inference_hops,
                        mode,
                    });
                }
            }
        }
    }
    out
}

fn build_sweep_points_from_presets(
    presets: &[IncrementalStreamingPreset],
    modes: &[StreamingTimingMode],
) -> Vec<SweepPoint> {
    let mut out = Vec::new();
    for &mode in modes {
        for &preset in presets {
            let config = preset.config();
            out.push(SweepPoint {
                callback_size: config.callback_size,
                context_seconds: config.incremental_context_seconds,
                inference_hops: config.incremental_inference_hops,
                mode,
            });
        }
    }
    out
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
        for discovered in default_existing_stage_audio_inputs()? {
            paths.insert(discovered.audio_path);
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

fn run_sweep(
    runtime: &mut FretNetRuntime,
    extractor: &FeatureExtractor,
    targets: &[AudioBenchmarkTarget],
    sweep_points: &[SweepPoint],
    iterations: usize,
    max_frames: Option<usize>,
) -> anyhow::Result<Vec<SweepConfigResult>> {
    let mut out = Vec::with_capacity(sweep_points.len());
    let total = sweep_points.len();

    for (index, point) in sweep_points.iter().enumerate() {
        println!();
        println!(
            "[{}/{}] mode={} callback={} context={:.3}s hops={}",
            index + 1,
            total,
            mode_label(point.mode),
            point.callback_size,
            point.context_seconds,
            point.inference_hops
        );

        let mut per_file = Vec::with_capacity(targets.len());
        for target in targets {
            let mut config = StreamingBenchmarkConfig::new(&target.audio_path, point.callback_size);
            config.iterations = iterations;
            config.timing_mode = point.mode;
            config.frontend_mode = StreamingFrontendMode::IncrementalPrototype;
            config.incremental_context_seconds = point.context_seconds;
            config.incremental_inference_hops = point.inference_hops;
            config.max_frames = max_frames;

            let result = benchmark_streaming_audio(runtime, extractor, &config)?;
            print_file_result(&result);
            per_file.push(result);
        }

        let aggregated = aggregate_results(point, &per_file)?;
        print_config_result(&aggregated);
        out.push(aggregated);
    }

    Ok(out)
}

fn aggregate_results(
    point: &SweepPoint,
    per_file: &[StreamingBenchmarkResult],
) -> anyhow::Result<SweepConfigResult> {
    if per_file.is_empty() {
        return Err(anyhow!("cannot aggregate empty per-file result set"));
    }

    let files_processed = per_file.len();
    let file_audio_names = per_file
        .iter()
        .map(|result| result.audio_name.clone())
        .collect::<Vec<_>>();

    let total_callbacks = per_file
        .iter()
        .map(|result| result.total_callbacks)
        .sum::<usize>();
    let total_inference_invocations = per_file
        .iter()
        .map(|result| result.inference_invocations)
        .sum::<usize>();
    let callback_budget_ms = per_file[0].callback_budget_ms;

    let callback_mean_ms = weighted_mean(per_file.iter().map(|result| {
        (
            result.callbacks.processing_ms.mean_ms,
            result.total_callbacks,
        )
    }));
    let callback_p95_ms = weighted_optional_mean(per_file.iter().map(|result| {
        (
            result.callbacks.processing_ms.p95_ms,
            result.total_callbacks.max(1),
        )
    }));
    let callback_p99_ms = weighted_optional_mean(per_file.iter().map(|result| {
        (
            result.callbacks.processing_ms.p99_ms,
            result.total_callbacks.max(1),
        )
    }));
    let callback_max_ms = per_file
        .iter()
        .filter_map(|result| result.callbacks.processing_ms.max_ms)
        .max_by(|left, right| left.total_cmp(right));

    let deadline_miss_count = per_file
        .iter()
        .map(|result| result.deadline.miss_count)
        .sum::<usize>();
    let deadline_miss_rate = safe_ratio(deadline_miss_count as f64, total_callbacks as f64);

    let backlog_mean_ms = weighted_mean(
        per_file
            .iter()
            .map(|result| (result.backlog_ms.mean_ms, result.total_callbacks)),
    );
    let backlog_max_ms = per_file
        .iter()
        .map(|result| result.backlog_ms.max_ms.unwrap_or(0.0))
        .max_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.0);

    let decision_latency_mean_ms = weighted_mean(per_file.iter().map(|result| {
        (
            result.decision_latency_ms.mean_ms,
            result.inference_invocations.max(1),
        )
    }));
    let decision_latency_p95_ms = weighted_optional_mean(per_file.iter().map(|result| {
        (
            result.decision_latency_ms.p95_ms,
            result.inference_invocations.max(1),
        )
    }));
    let decision_latency_max_ms = per_file
        .iter()
        .filter_map(|result| result.decision_latency_ms.max_ms)
        .max_by(|left, right| left.total_cmp(right));

    let trigger_latency_mean_ms = weighted_mean(per_file.iter().map(|result| {
        (
            result.trigger_latency_ms.mean_ms,
            result.inference_invocations.max(1),
        )
    }));
    let trigger_latency_p95_ms = weighted_optional_mean(per_file.iter().map(|result| {
        (
            result.trigger_latency_ms.p95_ms,
            result.inference_invocations.max(1),
        )
    }));
    let trigger_latency_max_ms = per_file
        .iter()
        .filter_map(|result| result.trigger_latency_ms.max_ms)
        .max_by(|left, right| left.total_cmp(right));

    let stage_audio_prep_mean_ms_per_callback = weighted_mean(per_file.iter().map(|result| {
        (
            result.stage_breakdown_ms.audio_prep_mean_ms_per_callback,
            result.total_callbacks,
        )
    }));
    let stage_frontend_mean_ms_per_callback = weighted_mean(per_file.iter().map(|result| {
        (
            result.stage_breakdown_ms.frontend_mean_ms_per_callback,
            result.total_callbacks,
        )
    }));
    let stage_inference_mean_ms_per_callback = weighted_mean(per_file.iter().map(|result| {
        (
            result.stage_breakdown_ms.inference_mean_ms_per_callback,
            result.total_callbacks,
        )
    }));
    let stage_other_mean_ms_per_callback = weighted_mean(per_file.iter().map(|result| {
        (
            result.stage_breakdown_ms.other_mean_ms_per_callback,
            result.total_callbacks,
        )
    }));
    let stage_audio_prep_fraction = weighted_mean(per_file.iter().map(|result| {
        (
            result.stage_breakdown_ms.audio_prep_fraction_of_processing,
            result.total_callbacks,
        )
    }));
    let stage_frontend_fraction = weighted_mean(per_file.iter().map(|result| {
        (
            result.stage_breakdown_ms.frontend_fraction_of_processing,
            result.total_callbacks,
        )
    }));
    let stage_inference_fraction = weighted_mean(per_file.iter().map(|result| {
        (
            result.stage_breakdown_ms.inference_fraction_of_processing,
            result.total_callbacks,
        )
    }));
    let stage_other_fraction = weighted_mean(per_file.iter().map(|result| {
        (
            result.stage_breakdown_ms.other_fraction_of_processing,
            result.total_callbacks,
        )
    }));

    let dominant_stage = [
        ("audio_prep", stage_audio_prep_fraction),
        ("frontend", stage_frontend_fraction),
        ("inference", stage_inference_fraction),
        ("other", stage_other_fraction),
    ]
    .into_iter()
    .max_by(|left, right| left.1.total_cmp(&right.1))
    .map(|(name, _)| name.to_owned())
    .unwrap_or_else(|| "other".to_owned());

    let realtime_classification = classify_configuration(
        deadline_miss_rate,
        callback_p95_ms,
        callback_budget_ms,
        backlog_max_ms,
        decision_latency_p95_ms,
    );
    let score = configuration_score(
        deadline_miss_rate,
        callback_p95_ms.unwrap_or(callback_mean_ms),
        decision_latency_p95_ms.unwrap_or(decision_latency_mean_ms),
        backlog_max_ms,
    );

    Ok(SweepConfigResult {
        mode: point.mode,
        callback_size: point.callback_size,
        incremental_context_seconds: point.context_seconds,
        incremental_inference_hops: point.inference_hops,
        files_processed,
        file_audio_names,
        total_callbacks,
        total_inference_invocations,
        callback_budget_ms,
        callback_mean_ms,
        callback_p95_ms,
        callback_p99_ms,
        callback_max_ms,
        deadline_miss_count,
        deadline_miss_rate,
        backlog_mean_ms,
        backlog_max_ms,
        decision_latency_mean_ms,
        decision_latency_p95_ms,
        decision_latency_max_ms,
        trigger_latency_mean_ms,
        trigger_latency_p95_ms,
        trigger_latency_max_ms,
        stage_audio_prep_mean_ms_per_callback,
        stage_frontend_mean_ms_per_callback,
        stage_inference_mean_ms_per_callback,
        stage_other_mean_ms_per_callback,
        stage_audio_prep_fraction,
        stage_frontend_fraction,
        stage_inference_fraction,
        stage_other_fraction,
        dominant_stage,
        realtime_classification,
        score,
    })
}

fn classify_configuration(
    miss_rate: f64,
    callback_p95_ms: Option<f64>,
    callback_budget_ms: f64,
    backlog_max_ms: f64,
    decision_p95_ms: Option<f64>,
) -> RealtimeClassification {
    let callback_p95_ms = callback_p95_ms.unwrap_or(f64::INFINITY);
    let decision_p95_ms = decision_p95_ms.unwrap_or(f64::INFINITY);
    let p95_budget_util = safe_ratio(callback_p95_ms, callback_budget_ms);
    let backlog_budget_multiple = safe_ratio(backlog_max_ms, callback_budget_ms);

    if miss_rate <= GO_MAX_MISS_RATE
        && p95_budget_util <= GO_MAX_P95_BUDGET_UTILIZATION
        && backlog_budget_multiple <= GO_MAX_BACKLOG_BUDGET_MULTIPLIER
        && decision_p95_ms <= GO_MAX_DECISION_P95_MS
    {
        return RealtimeClassification::Go;
    }

    if miss_rate <= BORDERLINE_MAX_MISS_RATE
        && p95_budget_util <= BORDERLINE_MAX_P95_BUDGET_UTILIZATION
        && backlog_budget_multiple <= BORDERLINE_MAX_BACKLOG_BUDGET_MULTIPLIER
        && decision_p95_ms <= BORDERLINE_MAX_DECISION_P95_MS
    {
        return RealtimeClassification::Borderline;
    }

    RealtimeClassification::NoGo
}

fn configuration_score(
    miss_rate: f64,
    callback_p95_ms: f64,
    decision_p95_ms: f64,
    backlog_max_ms: f64,
) -> f64 {
    SCORE_MISS_WEIGHT * miss_rate
        + SCORE_CALLBACK_P95_WEIGHT * callback_p95_ms
        + SCORE_DECISION_P95_WEIGHT * decision_p95_ms
        + SCORE_BACKLOG_MAX_WEIGHT * backlog_max_ms
}

fn top_virtual_sleep_validation_points(results: &[SweepConfigResult], k: usize) -> Vec<SweepPoint> {
    let mut virtual_results = results
        .iter()
        .filter(|result| result.mode == StreamingTimingMode::VirtualNoSleep)
        .cloned()
        .collect::<Vec<_>>();
    virtual_results.sort_by(rank_overall_cmp);

    virtual_results
        .into_iter()
        .take(k)
        .map(|result| SweepPoint {
            callback_size: result.callback_size,
            context_seconds: result.incremental_context_seconds,
            inference_hops: result.incremental_inference_hops,
            mode: StreamingTimingMode::SleepRealTime,
        })
        .collect()
}

fn build_rankings(results: &[SweepConfigResult], top_k: usize) -> SweepRankingSummary {
    let mut best_overall = results.to_vec();
    best_overall.sort_by(rank_overall_cmp);

    let mut best_go = results
        .iter()
        .filter(|result| result.realtime_classification == RealtimeClassification::Go)
        .cloned()
        .collect::<Vec<_>>();
    best_go.sort_by(rank_overall_cmp);

    let mut best_low_latency = results.to_vec();
    best_low_latency.sort_by(rank_low_latency_cmp);

    let mut best_throughput_safe = results.to_vec();
    best_throughput_safe.sort_by(rank_throughput_cmp);

    let mut worst = results.to_vec();
    worst.sort_by(|left, right| rank_overall_cmp(right, left));

    SweepRankingSummary {
        best_overall: to_ranked_refs(&best_overall, top_k),
        best_go: to_ranked_refs(&best_go, top_k),
        best_low_latency: to_ranked_refs(&best_low_latency, top_k),
        best_throughput_safe: to_ranked_refs(&best_throughput_safe, top_k),
        worst: to_ranked_refs(&worst, top_k),
    }
}

fn to_ranked_refs(results: &[SweepConfigResult], top_k: usize) -> Vec<RankedConfigRef> {
    results
        .iter()
        .take(top_k)
        .enumerate()
        .map(|(index, result)| RankedConfigRef {
            rank: index + 1,
            mode: mode_label_owned(&result.mode),
            callback_size: result.callback_size,
            incremental_context_seconds: result.incremental_context_seconds,
            incremental_inference_hops: result.incremental_inference_hops,
            realtime_classification: result.realtime_classification,
            score: result.score,
            deadline_miss_rate: result.deadline_miss_rate,
            callback_p95_ms: result.callback_p95_ms,
            decision_latency_p95_ms: result.decision_latency_p95_ms,
            backlog_max_ms: result.backlog_max_ms,
        })
        .collect()
}

fn analyze_operating_envelope(results: &[SweepConfigResult]) -> OperatingEnvelopeAnalysis {
    let virtual_results = results
        .iter()
        .filter(|result| result.mode == StreamingTimingMode::VirtualNoSleep)
        .cloned()
        .collect::<Vec<_>>();
    let reference = if virtual_results.is_empty() {
        results.to_vec()
    } else {
        virtual_results
    };

    let strongest_effects = vec![
        metric_sensitivity(&reference, "deadline_miss_rate", |result| {
            result.deadline_miss_rate
        }),
        metric_sensitivity(&reference, "callback_mean_ms", |result| {
            result.callback_mean_ms
        }),
        metric_sensitivity(&reference, "decision_latency_mean_ms", |result| {
            result.decision_latency_mean_ms
        }),
        metric_sensitivity(&reference, "backlog_mean_ms", |result| {
            result.backlog_mean_ms
        }),
    ];

    let callback_size_summary = summarize_callback_sizes(&reference);
    let workable_region_1024 = reference.iter().any(|result| {
        result.callback_size == 1024 && result.realtime_classification == RealtimeClassification::Go
    });
    let callback_512_salvageable = reference.iter().any(|result| {
        result.callback_size == 512
            && result.realtime_classification != RealtimeClassification::NoGo
    });

    let callbacks_128_256_outside_practical_range = callback_size_summary
        .iter()
        .filter(|summary| summary.callback_size == 128 || summary.callback_size == 256)
        .all(|summary| {
            summary.go_count == 0 && summary.no_go_count * 10 >= summary.configuration_count * 8
        });

    let robust_safe_region = callback_size_summary
        .iter()
        .filter(|summary| summary.go_rate >= 0.70)
        .map(|summary| {
            format!(
                "callback={} (GO rate {:.1}%)",
                summary.callback_size,
                summary.go_rate * 100.0
            )
        })
        .collect::<Vec<_>>();

    OperatingEnvelopeAnalysis {
        strongest_effects,
        callback_size_summary,
        workable_region_1024,
        callback_512_salvageable,
        callbacks_128_256_outside_practical_range,
        robust_safe_region,
    }
}

fn summarize_callback_sizes(results: &[SweepConfigResult]) -> Vec<CallbackSizeSummary> {
    let mut grouped = BTreeMap::<usize, Vec<&SweepConfigResult>>::new();
    for result in results {
        grouped
            .entry(result.callback_size)
            .or_default()
            .push(result);
    }

    let mut out = Vec::new();
    for (callback_size, group) in grouped {
        let configuration_count = group.len();
        let go_count = group
            .iter()
            .filter(|result| result.realtime_classification == RealtimeClassification::Go)
            .count();
        let borderline_count = group
            .iter()
            .filter(|result| result.realtime_classification == RealtimeClassification::Borderline)
            .count();
        let no_go_count = group
            .iter()
            .filter(|result| result.realtime_classification == RealtimeClassification::NoGo)
            .count();
        let go_rate = safe_ratio(go_count as f64, configuration_count as f64);
        let mean_miss_rate = group
            .iter()
            .map(|result| result.deadline_miss_rate)
            .sum::<f64>()
            / configuration_count as f64;
        let mean_callback_p95_ms = group
            .iter()
            .map(|result| result.callback_p95_ms.unwrap_or(result.callback_mean_ms))
            .sum::<f64>()
            / configuration_count as f64;
        let mean_decision_p95_ms = group
            .iter()
            .map(|result| {
                result
                    .decision_latency_p95_ms
                    .unwrap_or(result.decision_latency_mean_ms)
            })
            .sum::<f64>()
            / configuration_count as f64;

        out.push(CallbackSizeSummary {
            callback_size,
            configuration_count,
            go_count,
            borderline_count,
            no_go_count,
            go_rate,
            mean_miss_rate,
            mean_callback_p95_ms,
            mean_decision_p95_ms,
        });
    }
    out
}

fn metric_sensitivity(
    results: &[SweepConfigResult],
    metric: &str,
    metric_value: impl Fn(&SweepConfigResult) -> f64,
) -> MetricSensitivitySummary {
    let callback_size_effect =
        grouped_effect(results, |result| result.callback_size as i64, &metric_value);
    let context_seconds_effect = grouped_effect(
        results,
        |result| (result.incremental_context_seconds * 1000.0).round() as i64,
        &metric_value,
    );
    let inference_hops_effect = grouped_effect(
        results,
        |result| result.incremental_inference_hops as i64,
        &metric_value,
    );

    let strongest_parameter = [
        ("callback_size", callback_size_effect),
        ("context_seconds", context_seconds_effect),
        ("inference_hops", inference_hops_effect),
    ]
    .into_iter()
    .max_by(|left, right| left.1.total_cmp(&right.1))
    .map(|(name, _)| name.to_owned())
    .unwrap_or_else(|| "callback_size".to_owned());

    MetricSensitivitySummary {
        metric: metric.to_owned(),
        strongest_parameter,
        callback_size_effect,
        context_seconds_effect,
        inference_hops_effect,
    }
}

fn grouped_effect(
    results: &[SweepConfigResult],
    key_fn: impl Fn(&SweepConfigResult) -> i64,
    value_fn: &impl Fn(&SweepConfigResult) -> f64,
) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let mut grouped = BTreeMap::<i64, Vec<f64>>::new();
    for result in results {
        grouped
            .entry(key_fn(result))
            .or_default()
            .push(value_fn(result));
    }
    if grouped.len() <= 1 {
        return 0.0;
    }

    let means = grouped
        .values()
        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
        .collect::<Vec<_>>();
    let min = means
        .iter()
        .copied()
        .min_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.0);
    let max = means
        .iter()
        .copied()
        .max_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.0);
    max - min
}

fn print_file_result(result: &StreamingBenchmarkResult) {
    println!(
        "  {}: miss={:.2}% callback_mean={:.3}ms p95={} decision_mean={:.3}ms backlog_max={:.3}ms",
        result.audio_name,
        result.deadline.miss_rate * 100.0,
        result.callbacks.processing_ms.mean_ms,
        fmt_opt(result.callbacks.processing_ms.p95_ms),
        result.decision_latency_ms.mean_ms,
        result.backlog_ms.max_ms.unwrap_or(0.0)
    );
}

fn print_config_result(result: &SweepConfigResult) {
    println!(
        "  => class={:?} score={:.3} miss={:.2}% callback_mean={:.3}ms p95={} backlog_max={:.3}ms decision_p95={}",
        result.realtime_classification,
        result.score,
        result.deadline_miss_rate * 100.0,
        result.callback_mean_ms,
        fmt_opt(result.callback_p95_ms),
        result.backlog_max_ms,
        fmt_opt(result.decision_latency_p95_ms)
    );
}

fn print_summary(
    results: &[SweepConfigResult],
    ranking: &SweepRankingSummary,
    analysis: &OperatingEnvelopeAnalysis,
    top_k: usize,
) {
    println!();
    println!("Classification rule:");
    println!("  {}", classification_rule_text());
    println!("Score formula:");
    println!("  {}", score_formula_text());

    let go = results
        .iter()
        .filter(|result| result.realtime_classification == RealtimeClassification::Go)
        .count();
    let borderline = results
        .iter()
        .filter(|result| result.realtime_classification == RealtimeClassification::Borderline)
        .count();
    let no_go = results
        .iter()
        .filter(|result| result.realtime_classification == RealtimeClassification::NoGo)
        .count();
    println!();
    println!(
        "Classification counts: GO={} BORDERLINE={} NO-GO={} (total={})",
        go,
        borderline,
        no_go,
        results.len()
    );

    println!();
    println!("Top {} overall:", top_k);
    print_ranked(&ranking.best_overall);

    println!();
    println!("Top {} GO configs:", top_k);
    print_ranked(&ranking.best_go);

    println!();
    println!("Top {} low-latency configs:", top_k);
    print_ranked(&ranking.best_low_latency);

    println!();
    println!("Top {} throughput-safe configs:", top_k);
    print_ranked(&ranking.best_throughput_safe);

    println!();
    println!("Worst {} configs:", top_k);
    print_ranked(&ranking.worst);

    println!();
    println!("Sensitivity (higher effect means stronger impact):");
    for effect in &analysis.strongest_effects {
        println!(
            "  {}: strongest={} | callback={:.4} context={:.4} hops={:.4}",
            effect.metric,
            effect.strongest_parameter,
            effect.callback_size_effect,
            effect.context_seconds_effect,
            effect.inference_hops_effect
        );
    }

    println!();
    println!("Callback-size operating summary:");
    for summary in &analysis.callback_size_summary {
        println!(
            "  cb={}: GO/BORDERLINE/NO-GO={}/{}/{} | GO rate {:.1}% | mean miss {:.2}% | mean callback p95 {:.3}ms | mean decision p95 {:.3}ms",
            summary.callback_size,
            summary.go_count,
            summary.borderline_count,
            summary.no_go_count,
            summary.go_rate * 100.0,
            summary.mean_miss_rate * 100.0,
            summary.mean_callback_p95_ms,
            summary.mean_decision_p95_ms
        );
    }

    println!();
    println!("Decision-oriented answers:");
    println!(
        "  Workable region at callback 1024: {}",
        yes_no(analysis.workable_region_1024)
    );
    println!(
        "  Callback 512 salvageable with tuning: {}",
        yes_no(analysis.callback_512_salvageable)
    );
    println!(
        "  Callback 128/256 outside practical range: {}",
        yes_no(analysis.callbacks_128_256_outside_practical_range)
    );
    if analysis.robust_safe_region.is_empty() {
        println!("  Robust safe region: none (GO rate < 70% for all callback sizes)");
    } else {
        println!(
            "  Robust safe region: {}",
            analysis.robust_safe_region.join(", ")
        );
    }
}

fn print_ranked(items: &[RankedConfigRef]) {
    if items.is_empty() {
        println!("  (none)");
        return;
    }
    for item in items {
        println!(
            "  #{} mode={} cb={} ctx={:.3}s hops={} class={:?} score={:.3} miss={:.2}% cb_p95={} dec_p95={} backlog_max={:.3}ms",
            item.rank,
            item.mode,
            item.callback_size,
            item.incremental_context_seconds,
            item.incremental_inference_hops,
            item.realtime_classification,
            item.score,
            item.deadline_miss_rate * 100.0,
            fmt_opt(item.callback_p95_ms),
            fmt_opt(item.decision_latency_p95_ms),
            item.backlog_max_ms
        );
    }
}

fn write_json(path: &Path, report: &SweepReport) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

fn write_csv(path: &Path, results: &[SweepConfigResult]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = String::new();
    out.push_str("mode,callback_size,incremental_context_seconds,incremental_inference_hops,files_processed,total_callbacks,total_inference_invocations,callback_budget_ms,callback_mean_ms,callback_p95_ms,callback_p99_ms,callback_max_ms,deadline_miss_count,deadline_miss_rate,backlog_mean_ms,backlog_max_ms,decision_latency_mean_ms,decision_latency_p95_ms,decision_latency_max_ms,trigger_latency_mean_ms,trigger_latency_p95_ms,trigger_latency_max_ms,stage_audio_prep_mean_ms_per_callback,stage_frontend_mean_ms_per_callback,stage_inference_mean_ms_per_callback,stage_other_mean_ms_per_callback,stage_audio_prep_fraction,stage_frontend_fraction,stage_inference_fraction,stage_other_fraction,dominant_stage,realtime_classification,score,file_audio_names\n");

    for result in results {
        out.push_str(&format!(
            "{},{},{:.6},{},{},{},{},{:.6},{:.6},{},{},{},{},{:.8},{:.6},{:.6},{:.6},{},{},{:.6},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:?},{:.6},{}\n",
            mode_label(result.mode),
            result.callback_size,
            result.incremental_context_seconds,
            result.incremental_inference_hops,
            result.files_processed,
            result.total_callbacks,
            result.total_inference_invocations,
            result.callback_budget_ms,
            result.callback_mean_ms,
            fmt_opt_csv(result.callback_p95_ms),
            fmt_opt_csv(result.callback_p99_ms),
            fmt_opt_csv(result.callback_max_ms),
            result.deadline_miss_count,
            result.deadline_miss_rate,
            result.backlog_mean_ms,
            result.backlog_max_ms,
            result.decision_latency_mean_ms,
            fmt_opt_csv(result.decision_latency_p95_ms),
            fmt_opt_csv(result.decision_latency_max_ms),
            result.trigger_latency_mean_ms,
            fmt_opt_csv(result.trigger_latency_p95_ms),
            fmt_opt_csv(result.trigger_latency_max_ms),
            result.stage_audio_prep_mean_ms_per_callback,
            result.stage_frontend_mean_ms_per_callback,
            result.stage_inference_mean_ms_per_callback,
            result.stage_other_mean_ms_per_callback,
            result.stage_audio_prep_fraction,
            result.stage_frontend_fraction,
            result.stage_inference_fraction,
            result.stage_other_fraction,
            result.dominant_stage,
            result.realtime_classification,
            result.score,
            csv_escape(&result.file_audio_names.join("|"))
        ));
    }

    fs::write(path, out)?;
    Ok(())
}

fn classification_rule_text() -> String {
    format!(
        "GO if miss_rate <= {:.2}%, callback_p95 <= {:.0}% budget, max_backlog <= {:.1}x budget, decision_p95 <= {:.0} ms; BORDERLINE if miss_rate <= {:.0}%, callback_p95 <= {:.0}% budget, max_backlog <= {:.1}x budget, decision_p95 <= {:.0} ms; else NO-GO.",
        GO_MAX_MISS_RATE * 100.0,
        GO_MAX_P95_BUDGET_UTILIZATION * 100.0,
        GO_MAX_BACKLOG_BUDGET_MULTIPLIER,
        GO_MAX_DECISION_P95_MS,
        BORDERLINE_MAX_MISS_RATE * 100.0,
        BORDERLINE_MAX_P95_BUDGET_UTILIZATION * 100.0,
        BORDERLINE_MAX_BACKLOG_BUDGET_MULTIPLIER,
        BORDERLINE_MAX_DECISION_P95_MS
    )
}

fn score_formula_text() -> String {
    format!(
        "score = {:.0}*miss_rate + {:.2}*callback_p95_ms + {:.2}*decision_p95_ms + {:.2}*max_backlog_ms (lower is better)",
        SCORE_MISS_WEIGHT,
        SCORE_CALLBACK_P95_WEIGHT,
        SCORE_DECISION_P95_WEIGHT,
        SCORE_BACKLOG_MAX_WEIGHT
    )
}

fn mode_label(mode: StreamingTimingMode) -> &'static str {
    match mode {
        StreamingTimingMode::VirtualNoSleep => "virtual",
        StreamingTimingMode::SleepRealTime => "sleep",
    }
}

fn mode_label_owned(mode: &StreamingTimingMode) -> String {
    mode_label(*mode).to_owned()
}

fn config_sort_key(left: &SweepConfigResult, right: &SweepConfigResult) -> std::cmp::Ordering {
    (
        mode_label(left.mode),
        left.callback_size,
        ordered_f64(left.incremental_context_seconds),
        left.incremental_inference_hops,
    )
        .cmp(&(
            mode_label(right.mode),
            right.callback_size,
            ordered_f64(right.incremental_context_seconds),
            right.incremental_inference_hops,
        ))
}

fn rank_overall_cmp(left: &SweepConfigResult, right: &SweepConfigResult) -> std::cmp::Ordering {
    left.score
        .total_cmp(&right.score)
        .then(left.deadline_miss_rate.total_cmp(&right.deadline_miss_rate))
        .then(
            left.callback_p95_ms
                .unwrap_or(left.callback_mean_ms)
                .total_cmp(&right.callback_p95_ms.unwrap_or(right.callback_mean_ms)),
        )
}

fn rank_low_latency_cmp(left: &SweepConfigResult, right: &SweepConfigResult) -> std::cmp::Ordering {
    left.decision_latency_p95_ms
        .unwrap_or(left.decision_latency_mean_ms)
        .total_cmp(
            &right
                .decision_latency_p95_ms
                .unwrap_or(right.decision_latency_mean_ms),
        )
        .then(left.deadline_miss_rate.total_cmp(&right.deadline_miss_rate))
        .then(
            left.callback_p95_ms
                .unwrap_or(left.callback_mean_ms)
                .total_cmp(&right.callback_p95_ms.unwrap_or(right.callback_mean_ms)),
        )
}

fn rank_throughput_cmp(left: &SweepConfigResult, right: &SweepConfigResult) -> std::cmp::Ordering {
    left.deadline_miss_rate
        .total_cmp(&right.deadline_miss_rate)
        .then(left.backlog_max_ms.total_cmp(&right.backlog_max_ms))
        .then(
            left.callback_p95_ms
                .unwrap_or(left.callback_mean_ms)
                .total_cmp(&right.callback_p95_ms.unwrap_or(right.callback_mean_ms)),
        )
}

fn sorted_unique_usize(values: Vec<usize>) -> Vec<usize> {
    let mut out = values;
    out.sort_unstable();
    out.dedup();
    out
}

fn sorted_unique_f64(values: Vec<f64>) -> Vec<f64> {
    let mut out = values;
    out.sort_by(|left, right| left.total_cmp(right));
    out.dedup_by(|left, right| (*left - *right).abs() < 1e-9);
    out
}

fn weighted_mean(values: impl Iterator<Item = (f64, usize)>) -> f64 {
    let mut weighted_total = 0.0f64;
    let mut total_weight = 0usize;
    for (value, weight) in values {
        weighted_total += value * weight as f64;
        total_weight += weight;
    }
    safe_ratio(weighted_total, total_weight as f64)
}

fn weighted_optional_mean(values: impl Iterator<Item = (Option<f64>, usize)>) -> Option<f64> {
    let mut weighted_total = 0.0f64;
    let mut total_weight = 0usize;
    for (value, weight) in values {
        if let Some(value) = value {
            weighted_total += value * weight as f64;
            total_weight += weight;
        }
    }
    if total_weight == 0 {
        None
    } else {
        Some(weighted_total / total_weight as f64)
    }
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator <= 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn ordered_f64(value: f64) -> i64 {
    (value * 1000.0).round() as i64
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}ms"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn fmt_opt_csv(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "YES"
    } else {
        "NO"
    }
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
