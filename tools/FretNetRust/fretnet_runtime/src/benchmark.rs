use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use serde::Serialize;

use crate::{
    audio::{load_audio_for_frontend, load_wav_mono, resample_kaiser_best, rms_normalize},
    error::RuntimeError,
    frontend::FeatureExtractor,
    model::FretNetRuntime,
    profile::FrontendRunProfile,
    types::{FeatureBatch, ModelOutput},
};

#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub load_time: Duration,
    pub first_run: Duration,
    pub average_run: Duration,
    pub min_run: Duration,
    pub max_run: Duration,
    pub p95_run: Option<Duration>,
    pub iterations: usize,
    pub last_output: ModelOutput,
}

#[derive(Debug, Clone)]
pub struct StageLatencySummary {
    pub first_run: Duration,
    pub average_run: Duration,
    pub min_run: Duration,
    pub max_run: Duration,
    pub p95_run: Option<Duration>,
    pub iterations: usize,
}

#[derive(Debug, Clone)]
pub struct OutputTensorSummary {
    pub name: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct EndToEndBenchmarkSummary {
    pub audio_path: PathBuf,
    pub audio_name: String,
    pub original_sample_rate: u32,
    pub original_num_samples: usize,
    pub duration_seconds: f32,
    pub prepared_sample_rate: u32,
    pub prepared_num_samples: usize,
    pub hcqt_shape: [usize; 3],
    pub batch_shape: [usize; 5],
    pub output_tensors: Vec<OutputTensorSummary>,
    pub stage_load: StageLatencySummary,
    pub stage_prepare: StageLatencySummary,
    pub stage_resample: StageLatencySummary,
    pub stage_normalize: StageLatencySummary,
    pub stage_frontend: StageLatencySummary,
    pub stage_inference: StageLatencySummary,
    pub stage_total: StageLatencySummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScalarProfileSummary {
    pub first_seconds: f64,
    pub mean_seconds: f64,
    pub min_seconds: f64,
    pub max_seconds: f64,
    pub p95_seconds: Option<f64>,
    pub iterations: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FrontendOctaveProfileSummary {
    pub octave_index: usize,
    pub hop_length: usize,
    pub input_samples: usize,
    pub output_frames: usize,
    pub total: ScalarProfileSummary,
    pub transform: ScalarProfileSummary,
    pub buffer_allocation: ScalarProfileSummary,
    pub mean_buffer_allocation_bytes: f64,
    pub frame_copy: ScalarProfileSummary,
    pub fft_execution: ScalarProfileSummary,
    pub dot_product: ScalarProfileSummary,
    pub array_materialization: ScalarProfileSummary,
    pub downsample: ScalarProfileSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct FrontendHarmonicProfileSummary {
    pub harmonic: f32,
    pub total: ScalarProfileSummary,
    pub audio_clone: ScalarProfileSummary,
    pub trim_stack: ScalarProfileSummary,
    pub normalization: ScalarProfileSummary,
    pub flatten: ScalarProfileSummary,
    pub post_process: ScalarProfileSummary,
    pub mean_temporary_allocation_bytes: f64,
    pub octaves: Vec<FrontendOctaveProfileSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FrontendProfileBenchmarkSummary {
    pub audio_path: PathBuf,
    pub audio_name: String,
    pub prepared_sample_rate: u32,
    pub prepared_num_samples: usize,
    pub duration_seconds: f32,
    pub audio_prepare_seconds: f64,
    pub frontend_total: ScalarProfileSummary,
    pub hcqt_assembly: ScalarProfileSummary,
    pub batch_conversion: ScalarProfileSummary,
    pub hcqt_shape: [usize; 3],
    pub batch_shape: [usize; 5],
    pub harmonics: Vec<FrontendHarmonicProfileSummary>,
}

pub fn benchmark_runtime(
    runtime: &mut FretNetRuntime,
    features: &FeatureBatch,
    iterations: usize,
    load_time: Duration,
) -> Result<BenchmarkSummary, RuntimeError> {
    let iterations = iterations.max(1);

    let first_start = Instant::now();
    let mut last_output = runtime.infer_features(features)?;
    let first_run = first_start.elapsed();

    let mut runs = Vec::with_capacity(iterations);
    runs.push(first_run);

    for _ in 1..iterations {
        let start = Instant::now();
        last_output = runtime.infer_features(features)?;
        runs.push(start.elapsed());
    }

    let total = runs
        .iter()
        .copied()
        .fold(Duration::ZERO, |acc, next| acc + next);
    let average_run = total / runs.len() as u32;
    let min_run = runs.iter().copied().min().unwrap_or(Duration::ZERO);
    let max_run = runs.iter().copied().max().unwrap_or(Duration::ZERO);

    let mut sorted = runs.clone();
    sorted.sort_unstable();
    let p95_index = ((sorted.len() as f32 * 0.95).ceil() as usize).saturating_sub(1);
    let p95_run = sorted.get(p95_index).copied();

    Ok(BenchmarkSummary {
        load_time,
        first_run,
        average_run,
        min_run,
        max_run,
        p95_run,
        iterations: runs.len(),
        last_output,
    })
}

pub fn benchmark_end_to_end_audio(
    runtime: &mut FretNetRuntime,
    extractor: &FeatureExtractor,
    audio_path: impl AsRef<Path>,
    iterations: usize,
    max_frames: Option<usize>,
) -> Result<EndToEndBenchmarkSummary, anyhow::Error> {
    let audio_path = audio_path.as_ref().to_path_buf();
    let iterations = iterations.max(1);
    let mut load_runs = Vec::with_capacity(iterations);
    let mut prepare_runs = Vec::with_capacity(iterations);
    let mut resample_runs = Vec::with_capacity(iterations);
    let mut normalize_runs = Vec::with_capacity(iterations);
    let mut frontend_runs = Vec::with_capacity(iterations);
    let mut inference_runs = Vec::with_capacity(iterations);
    let mut total_runs = Vec::with_capacity(iterations);

    let mut original_sample_rate = 0;
    let mut original_num_samples = 0usize;
    let mut duration_seconds = 0.0f32;
    let mut prepared_sample_rate = 0;
    let mut prepared_num_samples = 0usize;
    let mut hcqt_shape = [0usize; 3];
    let mut batch_shape = [0usize; 5];
    let mut output_tensors = Vec::new();

    for _ in 0..iterations {
        let total_start = Instant::now();

        let load_start = Instant::now();
        let raw_audio = load_wav_mono(&audio_path)?;
        let load_time = load_start.elapsed();

        let mut prepared_audio = raw_audio.clone();
        let prepare_start = Instant::now();

        let resample_start = Instant::now();
        if prepared_audio.sample_rate != extractor.config().sample_rate {
            prepared_audio.samples = resample_kaiser_best(
                &prepared_audio.samples,
                prepared_audio.sample_rate,
                extractor.config().sample_rate,
            )?;
            prepared_audio.sample_rate = extractor.config().sample_rate;
        }
        let resample_time = resample_start.elapsed();

        let normalize_start = Instant::now();
        rms_normalize(&mut prepared_audio.samples)?;
        let normalize_time = normalize_start.elapsed();
        let prepare_time = prepare_start.elapsed();

        let frontend_start = Instant::now();
        let hcqt = extractor.extract_hcqt(&prepared_audio.samples, prepared_audio.sample_rate)?;
        let batch = extractor.hcqt_to_batch(&hcqt, max_frames)?;
        let frontend_time = frontend_start.elapsed();

        let inference_start = Instant::now();
        let output = runtime.infer_features(&batch)?;
        let inference_time = inference_start.elapsed();
        let total_time = total_start.elapsed();

        original_sample_rate = raw_audio.sample_rate;
        original_num_samples = raw_audio.samples.len();
        duration_seconds = raw_audio.samples.len() as f32 / raw_audio.sample_rate as f32;
        prepared_sample_rate = prepared_audio.sample_rate;
        prepared_num_samples = prepared_audio.samples.len();
        hcqt_shape = hcqt.shape();
        batch_shape = batch.shape();
        output_tensors = summarize_outputs(&output);

        load_runs.push(load_time);
        prepare_runs.push(prepare_time);
        resample_runs.push(resample_time);
        normalize_runs.push(normalize_time);
        frontend_runs.push(frontend_time);
        inference_runs.push(inference_time);
        total_runs.push(total_time);
    }

    Ok(EndToEndBenchmarkSummary {
        audio_name: audio_name(&audio_path),
        audio_path,
        original_sample_rate,
        original_num_samples,
        duration_seconds,
        prepared_sample_rate,
        prepared_num_samples,
        hcqt_shape,
        batch_shape,
        output_tensors,
        stage_load: summarize_stage_runs(&load_runs),
        stage_prepare: summarize_stage_runs(&prepare_runs),
        stage_resample: summarize_stage_runs(&resample_runs),
        stage_normalize: summarize_stage_runs(&normalize_runs),
        stage_frontend: summarize_stage_runs(&frontend_runs),
        stage_inference: summarize_stage_runs(&inference_runs),
        stage_total: summarize_stage_runs(&total_runs),
    })
}

pub fn benchmark_frontend_profile_audio(
    extractor: &FeatureExtractor,
    audio_path: impl AsRef<Path>,
    iterations: usize,
    max_frames: Option<usize>,
) -> Result<FrontendProfileBenchmarkSummary, anyhow::Error> {
    let audio_path = audio_path.as_ref().to_path_buf();
    let iterations = iterations.max(1);

    let prepare_start = Instant::now();
    let prepared_audio = load_audio_for_frontend(&audio_path, extractor.config().sample_rate)?;
    let audio_prepare_seconds = prepare_start.elapsed().as_secs_f64();

    let mut run_profiles = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let (_, profile) = extractor.extract_profiled(
            &prepared_audio.samples,
            prepared_audio.sample_rate,
            max_frames,
        )?;
        run_profiles.push(profile);
    }

    let first_profile = run_profiles
        .first()
        .ok_or_else(|| anyhow::anyhow!("no frontend profiling runs collected"))?;
    let harmonics = summarize_harmonics(&run_profiles);

    Ok(FrontendProfileBenchmarkSummary {
        audio_name: audio_name(&audio_path),
        audio_path,
        prepared_sample_rate: prepared_audio.sample_rate,
        prepared_num_samples: prepared_audio.samples.len(),
        duration_seconds: prepared_audio.samples.len() as f32 / prepared_audio.sample_rate as f32,
        audio_prepare_seconds,
        frontend_total: summarize_seconds(
            &run_profiles
                .iter()
                .map(|profile| profile.total_seconds)
                .collect::<Vec<_>>(),
        ),
        hcqt_assembly: summarize_seconds(
            &run_profiles
                .iter()
                .map(|profile| profile.hcqt_assembly_seconds)
                .collect::<Vec<_>>(),
        ),
        batch_conversion: summarize_seconds(
            &run_profiles
                .iter()
                .map(|profile| profile.batch_conversion_seconds)
                .collect::<Vec<_>>(),
        ),
        hcqt_shape: first_profile.hcqt_shape,
        batch_shape: first_profile.batch_shape,
        harmonics,
    })
}

fn summarize_outputs(output: &ModelOutput) -> Vec<OutputTensorSummary> {
    output
        .tensors
        .iter()
        .map(|tensor| OutputTensorSummary {
            name: tensor.name.clone(),
            shape: tensor.data.shape().to_vec(),
        })
        .collect()
}

fn summarize_stage_runs(runs: &[Duration]) -> StageLatencySummary {
    let total = runs
        .iter()
        .copied()
        .fold(Duration::ZERO, |acc, next| acc + next);
    let average_run = if runs.is_empty() {
        Duration::ZERO
    } else {
        total / runs.len() as u32
    };
    let min_run = runs.iter().copied().min().unwrap_or(Duration::ZERO);
    let max_run = runs.iter().copied().max().unwrap_or(Duration::ZERO);

    let mut sorted = runs.to_vec();
    sorted.sort_unstable();
    let p95_index = ((sorted.len() as f32 * 0.95).ceil() as usize).saturating_sub(1);
    let p95_run = sorted.get(p95_index).copied();

    StageLatencySummary {
        first_run: runs.first().copied().unwrap_or(Duration::ZERO),
        average_run,
        min_run,
        max_run,
        p95_run,
        iterations: runs.len(),
    }
}

fn summarize_harmonics(run_profiles: &[FrontendRunProfile]) -> Vec<FrontendHarmonicProfileSummary> {
    let Some(first) = run_profiles.first() else {
        return Vec::new();
    };

    (0..first.harmonic_profiles.len())
        .map(|harmonic_index| {
            let harmonic = first.harmonic_profiles[harmonic_index].harmonic;
            let harmonic_runs: Vec<_> = run_profiles
                .iter()
                .map(|profile| &profile.harmonic_profiles[harmonic_index])
                .collect();
            let first_harmonic = harmonic_runs[0];
            let octaves = (0..first_harmonic.octave_profiles.len())
                .map(|octave_index| {
                    let octave_runs: Vec<_> = harmonic_runs
                        .iter()
                        .map(|harmonic_profile| &harmonic_profile.octave_profiles[octave_index])
                        .collect();
                    let first_octave = octave_runs[0];
                    FrontendOctaveProfileSummary {
                        octave_index: first_octave.octave_index,
                        hop_length: first_octave.hop_length,
                        input_samples: first_octave.input_samples,
                        output_frames: first_octave.output_frames,
                        total: summarize_seconds(
                            &octave_runs
                                .iter()
                                .map(|profile| profile.total_seconds)
                                .collect::<Vec<_>>(),
                        ),
                        transform: summarize_seconds(
                            &octave_runs
                                .iter()
                                .map(|profile| profile.transform_seconds)
                                .collect::<Vec<_>>(),
                        ),
                        buffer_allocation: summarize_seconds(
                            &octave_runs
                                .iter()
                                .map(|profile| profile.buffer_allocation_seconds)
                                .collect::<Vec<_>>(),
                        ),
                        mean_buffer_allocation_bytes: mean_usize(
                            &octave_runs
                                .iter()
                                .map(|profile| profile.buffer_allocation_bytes)
                                .collect::<Vec<_>>(),
                        ),
                        frame_copy: summarize_seconds(
                            &octave_runs
                                .iter()
                                .map(|profile| profile.frame_copy_seconds)
                                .collect::<Vec<_>>(),
                        ),
                        fft_execution: summarize_seconds(
                            &octave_runs
                                .iter()
                                .map(|profile| profile.fft_execution_seconds)
                                .collect::<Vec<_>>(),
                        ),
                        dot_product: summarize_seconds(
                            &octave_runs
                                .iter()
                                .map(|profile| profile.dot_product_seconds)
                                .collect::<Vec<_>>(),
                        ),
                        array_materialization: summarize_seconds(
                            &octave_runs
                                .iter()
                                .map(|profile| profile.array_materialization_seconds)
                                .collect::<Vec<_>>(),
                        ),
                        downsample: summarize_seconds(
                            &octave_runs
                                .iter()
                                .map(|profile| profile.downsample_seconds)
                                .collect::<Vec<_>>(),
                        ),
                    }
                })
                .collect();

            FrontendHarmonicProfileSummary {
                harmonic,
                total: summarize_seconds(
                    &harmonic_runs
                        .iter()
                        .map(|profile| profile.total_seconds)
                        .collect::<Vec<_>>(),
                ),
                audio_clone: summarize_seconds(
                    &harmonic_runs
                        .iter()
                        .map(|profile| profile.audio_clone_seconds)
                        .collect::<Vec<_>>(),
                ),
                trim_stack: summarize_seconds(
                    &harmonic_runs
                        .iter()
                        .map(|profile| profile.trim_stack_seconds)
                        .collect::<Vec<_>>(),
                ),
                normalization: summarize_seconds(
                    &harmonic_runs
                        .iter()
                        .map(|profile| profile.normalization_seconds)
                        .collect::<Vec<_>>(),
                ),
                flatten: summarize_seconds(
                    &harmonic_runs
                        .iter()
                        .map(|profile| profile.flatten_seconds)
                        .collect::<Vec<_>>(),
                ),
                post_process: summarize_seconds(
                    &harmonic_runs
                        .iter()
                        .map(|profile| profile.post_process_seconds)
                        .collect::<Vec<_>>(),
                ),
                mean_temporary_allocation_bytes: mean_usize(
                    &harmonic_runs
                        .iter()
                        .map(|profile| profile.temporary_allocation_bytes)
                        .collect::<Vec<_>>(),
                ),
                octaves,
            }
        })
        .collect()
}

fn summarize_seconds(runs: &[f64]) -> ScalarProfileSummary {
    if runs.is_empty() {
        return ScalarProfileSummary {
            first_seconds: 0.0,
            mean_seconds: 0.0,
            min_seconds: 0.0,
            max_seconds: 0.0,
            p95_seconds: None,
            iterations: 0,
        };
    }

    let total = runs.iter().sum::<f64>();
    let mean_seconds = total / runs.len() as f64;
    let min_seconds = runs.iter().copied().fold(f64::INFINITY, f64::min);
    let max_seconds = runs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sorted = runs.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let p95_index = ((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1);

    ScalarProfileSummary {
        first_seconds: runs[0],
        mean_seconds,
        min_seconds,
        max_seconds,
        p95_seconds: sorted.get(p95_index).copied(),
        iterations: runs.len(),
    }
}

fn mean_usize(values: &[usize]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<usize>() as f64 / values.len() as f64
    }
}

fn audio_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown_audio")
        .to_owned()
}
