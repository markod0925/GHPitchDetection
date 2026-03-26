use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use anyhow::{bail, Context};
use serde::Serialize;

use crate::{
    audio::{load_wav_mono, resample_kaiser_best, rms_normalize},
    error::FrontendError,
    frontend::FeatureExtractor,
    model::FretNetRuntime,
};

pub const STREAMING_INPUT_SAMPLE_RATE: u32 = 44_100;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IncrementalStreamingPreset {
    ProductionSafe,
    Balanced,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct IncrementalStreamingPresetConfig {
    pub name: &'static str,
    pub callback_size: usize,
    pub incremental_context_seconds: f64,
    pub incremental_inference_hops: usize,
}

pub const PRODUCTION_SAFE_INCREMENTAL_PRESET: IncrementalStreamingPresetConfig =
    IncrementalStreamingPresetConfig {
        name: "production-safe",
        callback_size: 1024,
        incremental_context_seconds: 0.5,
        incremental_inference_hops: 1,
    };

pub const BALANCED_INCREMENTAL_PRESET: IncrementalStreamingPresetConfig =
    IncrementalStreamingPresetConfig {
        name: "balanced",
        callback_size: 1024,
        incremental_context_seconds: 0.5,
        incremental_inference_hops: 2,
    };

impl IncrementalStreamingPreset {
    pub const fn config(self) -> IncrementalStreamingPresetConfig {
        match self {
            Self::ProductionSafe => PRODUCTION_SAFE_INCREMENTAL_PRESET,
            Self::Balanced => BALANCED_INCREMENTAL_PRESET,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum StreamingTimingMode {
    VirtualNoSleep,
    SleepRealTime,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum StreamingFrontendMode {
    LegacyRollingWindow,
    IncrementalPrototype,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InferenceScheduling {
    EveryCallback,
    EveryNCallbacks { callbacks: usize },
    HopCadence,
}

#[derive(Debug, Clone)]
pub struct StreamingBenchmarkConfig {
    pub audio_path: PathBuf,
    pub callback_size: usize,
    pub iterations: usize,
    pub timing_mode: StreamingTimingMode,
    pub frontend_mode: StreamingFrontendMode,
    pub inference_schedule: InferenceScheduling,
    pub analysis_window_seconds: f64,
    pub incremental_context_seconds: f64,
    pub incremental_inference_hops: usize,
    pub stream_input_sample_rate: u32,
    pub max_frames: Option<usize>,
}

impl StreamingBenchmarkConfig {
    pub fn new(audio_path: impl AsRef<Path>, callback_size: usize) -> Self {
        Self {
            audio_path: audio_path.as_ref().to_path_buf(),
            callback_size,
            iterations: 1,
            timing_mode: StreamingTimingMode::VirtualNoSleep,
            frontend_mode: StreamingFrontendMode::LegacyRollingWindow,
            inference_schedule: InferenceScheduling::HopCadence,
            analysis_window_seconds: 3.0,
            incremental_context_seconds: 1.0,
            incremental_inference_hops: 1,
            stream_input_sample_rate: STREAMING_INPUT_SAMPLE_RATE,
            max_frames: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DistributionStats {
    pub count: usize,
    pub mean_ms: f64,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
    pub max_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CallbackStats {
    pub callback_size: usize,
    pub callback_budget_ms: f64,
    pub processing_ms: DistributionStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeadlineStats {
    pub miss_count: usize,
    pub miss_rate: f64,
    pub mean_backlog_ms: f64,
    pub max_backlog_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamingStageBreakdown {
    pub audio_prep_total_ms: f64,
    pub frontend_total_ms: f64,
    pub inference_total_ms: f64,
    pub other_total_ms: f64,
    pub audio_prep_mean_ms_per_callback: f64,
    pub frontend_mean_ms_per_callback: f64,
    pub inference_mean_ms_per_callback: f64,
    pub other_mean_ms_per_callback: f64,
    pub audio_prep_fraction_of_processing: f64,
    pub frontend_fraction_of_processing: f64,
    pub inference_fraction_of_processing: f64,
    pub other_fraction_of_processing: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamingBenchmarkResult {
    pub audio_path: PathBuf,
    pub audio_name: String,
    pub mode: StreamingTimingMode,
    pub frontend_mode: StreamingFrontendMode,
    pub callback_size: usize,
    pub callback_budget_ms: f64,
    pub inference_schedule: String,
    pub iterations: usize,
    pub stream_input_sample_rate: u32,
    pub source_audio_sample_rate: u32,
    pub source_audio_duration_seconds: f64,
    pub simulated_stream_duration_seconds: f64,
    pub context_window_seconds: f64,
    pub context_window_samples: usize,
    pub max_frames: Option<usize>,
    pub total_callbacks: usize,
    pub inference_invocations: usize,
    pub callbacks: CallbackStats,
    pub deadline: DeadlineStats,
    pub backlog_ms: DistributionStats,
    pub decision_latency_ms: DistributionStats,
    pub trigger_latency_ms: DistributionStats,
    pub stage_breakdown_ms: StreamingStageBreakdown,
    pub dominant_stage: String,
    pub incrementality_note: String,
    pub deadline_met: bool,
    pub conclusion: String,
}

pub fn benchmark_streaming_audio(
    runtime: &mut FretNetRuntime,
    extractor: &FeatureExtractor,
    config: &StreamingBenchmarkConfig,
) -> Result<StreamingBenchmarkResult, anyhow::Error> {
    validate_config(config)?;
    match config.frontend_mode {
        StreamingFrontendMode::LegacyRollingWindow => {
            benchmark_legacy_rolling(runtime, extractor, config)
        }
        StreamingFrontendMode::IncrementalPrototype => {
            benchmark_incremental_prototype(runtime, extractor, config)
        }
    }
}

fn benchmark_legacy_rolling(
    runtime: &mut FretNetRuntime,
    extractor: &FeatureExtractor,
    config: &StreamingBenchmarkConfig,
) -> Result<StreamingBenchmarkResult, anyhow::Error> {
    let source_audio = load_stream_input_audio(config)?;
    let callback_interval_seconds =
        callback_interval_seconds(config.callback_size, config.stream_input_sample_rate);
    let callback_budget_ms = callback_interval_seconds * 1000.0;
    let callbacks_per_iteration = ceil_div(source_audio.stream_samples.len(), config.callback_size);
    let window_samples = ((config.analysis_window_seconds * config.stream_input_sample_rate as f64)
        .round() as usize)
        .max(config.callback_size);
    let schedule_description = legacy_schedule_description(
        &config.inference_schedule,
        extractor.config().hop_length,
        extractor.config().sample_rate,
        config.stream_input_sample_rate,
    );

    let mut metrics = MetricsCollector::new(config.iterations * callbacks_per_iteration);
    let mut callback_chunk = vec![0.0f32; config.callback_size];
    let mut ring_buffer = RollingAudioBuffer::new(window_samples);

    for _ in 0..config.iterations {
        metrics.begin_iteration();
        ring_buffer.clear();
        let mut schedule_state = LegacyScheduleState::new(
            &config.inference_schedule,
            extractor.config().hop_length,
            extractor.config().sample_rate,
            config.stream_input_sample_rate,
        )?;
        let iteration_start = Instant::now();

        for callback_index in 0..callbacks_per_iteration {
            let callback_arrival_sec =
                callback_arrival_time(callback_index, callback_interval_seconds);
            if config.timing_mode == StreamingTimingMode::SleepRealTime {
                sleep_until(iteration_start + Duration::from_secs_f64(callback_arrival_sec));
            }

            fill_callback_chunk(
                &source_audio.stream_samples,
                callback_index,
                config.callback_size,
                &mut callback_chunk,
            );

            let callback_start = Instant::now();
            ring_buffer.push_slice(&callback_chunk);

            let mut audio_prep_ms = 0.0f64;
            let mut frontend_ms = 0.0f64;
            let mut inference_ms = 0.0f64;
            let run_inference =
                schedule_state.should_run_inference(callback_index, config.callback_size);

            if run_inference {
                let prep_start = Instant::now();
                let mut analysis_audio = ring_buffer.as_vec();
                if config.stream_input_sample_rate != extractor.config().sample_rate {
                    analysis_audio = resample_kaiser_best(
                        &analysis_audio,
                        config.stream_input_sample_rate,
                        extractor.config().sample_rate,
                    )?;
                }
                rms_normalize_allow_silence(&mut analysis_audio)?;
                audio_prep_ms = elapsed_ms(prep_start.elapsed());

                let frontend_start = Instant::now();
                let hcqt =
                    extractor.extract_hcqt(&analysis_audio, extractor.config().sample_rate)?;
                let batch = extractor.hcqt_to_batch(&hcqt, config.max_frames)?;
                frontend_ms = elapsed_ms(frontend_start.elapsed());

                let inference_start = Instant::now();
                runtime.infer_features(&batch)?;
                inference_ms = elapsed_ms(inference_start.elapsed());
            }

            let callback_ms = elapsed_ms(callback_start.elapsed());
            metrics.record_callback(
                callback_arrival_sec,
                callback_interval_seconds,
                callback_ms,
                audio_prep_ms,
                frontend_ms,
                inference_ms,
                run_inference,
            );
        }
    }

    Ok(finalize_result(
        config,
        &source_audio,
        callback_interval_seconds,
        callback_budget_ms,
        callbacks_per_iteration,
        window_samples,
        schedule_description,
        "Legacy rolling-window path: callback-driven scheduling with frontend recomputation over the active analysis window at each scheduled inference.".to_owned(),
        metrics,
    ))
}

fn benchmark_incremental_prototype(
    runtime: &mut FretNetRuntime,
    extractor: &FeatureExtractor,
    config: &StreamingBenchmarkConfig,
) -> Result<StreamingBenchmarkResult, anyhow::Error> {
    let source_audio = load_stream_input_audio(config)?;
    let callback_interval_seconds =
        callback_interval_seconds(config.callback_size, config.stream_input_sample_rate);
    let callback_budget_ms = callback_interval_seconds * 1000.0;
    let callbacks_per_iteration = ceil_div(source_audio.stream_samples.len(), config.callback_size);

    let mut metrics = MetricsCollector::new(config.iterations * callbacks_per_iteration);
    let mut callback_chunk = vec![0.0f32; config.callback_size];
    let mut frontend_state = IncrementalStreamingFrontend::new(
        config.incremental_context_seconds,
        config.incremental_inference_hops.max(1),
        extractor.config().sample_rate,
        extractor.config().hop_length,
        config.stream_input_sample_rate,
    )?;

    let schedule_description = format!(
        "incremental hop scheduler (every {} completed frontend hops)",
        config.incremental_inference_hops.max(1)
    );

    for _ in 0..config.iterations {
        metrics.begin_iteration();
        frontend_state.reset();
        let iteration_start = Instant::now();

        for callback_index in 0..callbacks_per_iteration {
            let callback_arrival_sec =
                callback_arrival_time(callback_index, callback_interval_seconds);
            if config.timing_mode == StreamingTimingMode::SleepRealTime {
                sleep_until(iteration_start + Duration::from_secs_f64(callback_arrival_sec));
            }

            fill_callback_chunk(
                &source_audio.stream_samples,
                callback_index,
                config.callback_size,
                &mut callback_chunk,
            );

            let callback_start = Instant::now();
            let prep_start = Instant::now();
            frontend_state.push_callback(&callback_chunk);
            let mut audio_prep_ms = elapsed_ms(prep_start.elapsed());
            let run_inference = frontend_state.should_run_inference();

            let mut frontend_ms = 0.0f64;
            let mut inference_ms = 0.0f64;
            if run_inference {
                let normalize_start = Instant::now();
                let analysis_audio = frontend_state.normalized_frontend_window();
                audio_prep_ms += elapsed_ms(normalize_start.elapsed());

                let frontend_start = Instant::now();
                let hcqt =
                    extractor.extract_hcqt(&analysis_audio, extractor.config().sample_rate)?;
                let batch = extractor.hcqt_to_batch(&hcqt, config.max_frames)?;
                frontend_ms = elapsed_ms(frontend_start.elapsed());

                let inference_start = Instant::now();
                runtime.infer_features(&batch)?;
                inference_ms = elapsed_ms(inference_start.elapsed());
            }

            let callback_ms = elapsed_ms(callback_start.elapsed());
            metrics.record_callback(
                callback_arrival_sec,
                callback_interval_seconds,
                callback_ms,
                audio_prep_ms,
                frontend_ms,
                inference_ms,
                run_inference,
            );
        }
    }

    let context_window_samples_stream = frontend_state.context_samples()
        * (config.stream_input_sample_rate as usize / extractor.config().sample_rate as usize);

    Ok(finalize_result(
        config,
        &source_audio,
        callback_interval_seconds,
        callback_budget_ms,
        callbacks_per_iteration,
        context_window_samples_stream,
        schedule_description,
        "Incremental prototype: persistent callback-to-frontend downsampling state, persistent normalized frontend ring buffer, and hop-driven inference scheduling. Remaining limitation: HCQT extraction still recomputes within the bounded context window at inference time.".to_owned(),
        metrics,
    ))
}

struct StreamInputAudio {
    stream_samples: Vec<f32>,
    source_audio_sample_rate: u32,
    source_audio_duration_seconds: f64,
}

fn load_stream_input_audio(
    config: &StreamingBenchmarkConfig,
) -> Result<StreamInputAudio, anyhow::Error> {
    let mut source_audio = load_wav_mono(&config.audio_path).with_context(|| {
        format!(
            "failed to load streaming benchmark input '{}'",
            config.audio_path.display()
        )
    })?;
    let source_audio_sample_rate = source_audio.sample_rate;
    let source_audio_duration_seconds =
        source_audio.samples.len() as f64 / source_audio.sample_rate as f64;

    if source_audio.sample_rate != config.stream_input_sample_rate {
        source_audio.samples = resample_kaiser_best(
            &source_audio.samples,
            source_audio.sample_rate,
            config.stream_input_sample_rate,
        )?;
    }

    if source_audio.samples.is_empty() {
        bail!("audio input is empty: {}", config.audio_path.display());
    }

    Ok(StreamInputAudio {
        stream_samples: source_audio.samples,
        source_audio_sample_rate,
        source_audio_duration_seconds,
    })
}

fn validate_config(config: &StreamingBenchmarkConfig) -> Result<(), anyhow::Error> {
    if config.callback_size == 0 {
        bail!("callback_size must be positive");
    }
    if config.iterations == 0 {
        bail!("iterations must be positive");
    }
    if config.stream_input_sample_rate == 0 {
        bail!("stream_input_sample_rate must be positive");
    }
    if !(config.analysis_window_seconds.is_finite() && config.analysis_window_seconds > 0.0) {
        bail!("analysis_window_seconds must be > 0");
    }
    if !(config.incremental_context_seconds.is_finite() && config.incremental_context_seconds > 0.0)
    {
        bail!("incremental_context_seconds must be > 0");
    }
    if config.incremental_inference_hops == 0 {
        bail!("incremental_inference_hops must be positive");
    }
    Ok(())
}

#[derive(Default)]
struct MetricsCollector {
    callback_durations_ms: Vec<f64>,
    callback_backlog_ms: Vec<f64>,
    decision_latencies_ms: Vec<f64>,
    trigger_latencies_ms: Vec<f64>,
    stage_audio_prep_ms: Vec<f64>,
    stage_frontend_ms: Vec<f64>,
    stage_inference_ms: Vec<f64>,
    stage_other_ms: Vec<f64>,
    deadline_miss_count: usize,
    inference_invocations: usize,
    worker_available_sec: f64,
    pending_decision_start_sec: Option<f64>,
}

impl MetricsCollector {
    fn new(capacity: usize) -> Self {
        Self {
            callback_durations_ms: Vec::with_capacity(capacity),
            callback_backlog_ms: Vec::with_capacity(capacity),
            decision_latencies_ms: Vec::new(),
            trigger_latencies_ms: Vec::new(),
            stage_audio_prep_ms: Vec::with_capacity(capacity),
            stage_frontend_ms: Vec::with_capacity(capacity),
            stage_inference_ms: Vec::with_capacity(capacity),
            stage_other_ms: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    fn begin_iteration(&mut self) {
        self.worker_available_sec = 0.0;
        self.pending_decision_start_sec = None;
    }

    fn record_callback(
        &mut self,
        callback_arrival_sec: f64,
        callback_interval_seconds: f64,
        callback_ms: f64,
        audio_prep_ms: f64,
        frontend_ms: f64,
        inference_ms: f64,
        run_inference: bool,
    ) {
        if self.pending_decision_start_sec.is_none() {
            self.pending_decision_start_sec = Some(callback_arrival_sec);
        }

        let callback_start_sec = callback_arrival_sec.max(self.worker_available_sec);
        let callback_finish_sec = callback_start_sec + callback_ms / 1000.0;
        let callback_deadline_sec = callback_arrival_sec + callback_interval_seconds;
        let callback_backlog_sec = (callback_finish_sec - callback_deadline_sec).max(0.0);
        if callback_finish_sec > callback_deadline_sec {
            self.deadline_miss_count += 1;
        }
        self.worker_available_sec = callback_finish_sec;

        if run_inference {
            self.inference_invocations += 1;
            if let Some(decision_start_sec) = self.pending_decision_start_sec.take() {
                self.decision_latencies_ms
                    .push((callback_finish_sec - decision_start_sec).max(0.0) * 1000.0);
            }
            self.trigger_latencies_ms
                .push((callback_finish_sec - callback_arrival_sec).max(0.0) * 1000.0);
        }

        let other_ms = (callback_ms - audio_prep_ms - frontend_ms - inference_ms).max(0.0);
        self.callback_durations_ms.push(callback_ms);
        self.callback_backlog_ms.push(callback_backlog_sec * 1000.0);
        self.stage_audio_prep_ms.push(audio_prep_ms);
        self.stage_frontend_ms.push(frontend_ms);
        self.stage_inference_ms.push(inference_ms);
        self.stage_other_ms.push(other_ms);
    }
}

fn finalize_result(
    config: &StreamingBenchmarkConfig,
    source_audio: &StreamInputAudio,
    callback_interval_seconds: f64,
    callback_budget_ms: f64,
    callbacks_per_iteration: usize,
    context_window_samples: usize,
    schedule_description: String,
    incrementality_note: String,
    metrics: MetricsCollector,
) -> StreamingBenchmarkResult {
    let total_callbacks = metrics.callback_durations_ms.len();
    let callback_processing = summarize_distribution(&metrics.callback_durations_ms);
    let backlog_distribution = summarize_distribution(&metrics.callback_backlog_ms);
    let decision_latency = summarize_distribution(&metrics.decision_latencies_ms);
    let trigger_latency = summarize_distribution(&metrics.trigger_latencies_ms);

    let total_processing_ms = metrics.callback_durations_ms.iter().sum::<f64>();
    let total_audio_prep_ms = metrics.stage_audio_prep_ms.iter().sum::<f64>();
    let total_frontend_ms = metrics.stage_frontend_ms.iter().sum::<f64>();
    let total_inference_ms = metrics.stage_inference_ms.iter().sum::<f64>();
    let total_other_ms = metrics.stage_other_ms.iter().sum::<f64>();

    let stage_breakdown = StreamingStageBreakdown {
        audio_prep_total_ms: total_audio_prep_ms,
        frontend_total_ms: total_frontend_ms,
        inference_total_ms: total_inference_ms,
        other_total_ms: total_other_ms,
        audio_prep_mean_ms_per_callback: safe_mean(total_audio_prep_ms, total_callbacks),
        frontend_mean_ms_per_callback: safe_mean(total_frontend_ms, total_callbacks),
        inference_mean_ms_per_callback: safe_mean(total_inference_ms, total_callbacks),
        other_mean_ms_per_callback: safe_mean(total_other_ms, total_callbacks),
        audio_prep_fraction_of_processing: safe_fraction(total_audio_prep_ms, total_processing_ms),
        frontend_fraction_of_processing: safe_fraction(total_frontend_ms, total_processing_ms),
        inference_fraction_of_processing: safe_fraction(total_inference_ms, total_processing_ms),
        other_fraction_of_processing: safe_fraction(total_other_ms, total_processing_ms),
    };

    let deadline = DeadlineStats {
        miss_count: metrics.deadline_miss_count,
        miss_rate: safe_fraction(metrics.deadline_miss_count as f64, total_callbacks as f64),
        mean_backlog_ms: backlog_distribution.mean_ms,
        max_backlog_ms: backlog_distribution.max_ms.unwrap_or(0.0),
    };

    let dominant_stage = dominant_stage(
        total_audio_prep_ms,
        total_frontend_ms,
        total_inference_ms,
        total_other_ms,
    );
    let deadline_met = deadline.miss_count == 0;
    let conclusion = streaming_conclusion(
        deadline_met,
        deadline.miss_rate,
        deadline.max_backlog_ms,
        callback_budget_ms,
    );

    StreamingBenchmarkResult {
        audio_name: audio_name(&config.audio_path),
        audio_path: config.audio_path.clone(),
        mode: config.timing_mode,
        frontend_mode: config.frontend_mode,
        callback_size: config.callback_size,
        callback_budget_ms,
        inference_schedule: schedule_description,
        iterations: config.iterations,
        stream_input_sample_rate: config.stream_input_sample_rate,
        source_audio_sample_rate: source_audio.source_audio_sample_rate,
        source_audio_duration_seconds: source_audio.source_audio_duration_seconds,
        simulated_stream_duration_seconds: callbacks_per_iteration as f64
            * callback_interval_seconds
            * config.iterations as f64,
        context_window_seconds: context_window_samples as f64
            / config.stream_input_sample_rate as f64,
        context_window_samples,
        max_frames: config.max_frames,
        total_callbacks,
        inference_invocations: metrics.inference_invocations,
        callbacks: CallbackStats {
            callback_size: config.callback_size,
            callback_budget_ms,
            processing_ms: callback_processing,
        },
        deadline,
        backlog_ms: backlog_distribution,
        decision_latency_ms: decision_latency,
        trigger_latency_ms: trigger_latency,
        stage_breakdown_ms: stage_breakdown,
        dominant_stage,
        incrementality_note,
        deadline_met,
        conclusion,
    }
}

struct RollingAudioBuffer {
    samples: VecDeque<f32>,
    capacity: usize,
}

impl RollingAudioBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
        }
    }

    fn clear(&mut self) {
        self.samples.clear();
    }

    fn push_slice(&mut self, incoming: &[f32]) {
        if incoming.is_empty() {
            return;
        }

        if incoming.len() >= self.capacity {
            self.samples.clear();
            self.samples
                .extend(incoming[incoming.len() - self.capacity..].iter().copied());
            return;
        }

        let overflow = self
            .samples
            .len()
            .saturating_add(incoming.len())
            .saturating_sub(self.capacity);
        for _ in 0..overflow {
            let _ = self.samples.pop_front();
        }
        self.samples.extend(incoming.iter().copied());
    }

    fn as_vec(&self) -> Vec<f32> {
        self.samples.iter().copied().collect()
    }
}

struct HalfRateDownsamplerState {
    carry_sample: Option<f32>,
}

impl HalfRateDownsamplerState {
    fn new() -> Self {
        Self { carry_sample: None }
    }

    fn reset(&mut self) {
        self.carry_sample = None;
    }

    fn process_block(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.clear();
        output.reserve(input.len() / 2 + 1);
        for sample in input.iter().copied() {
            if let Some(previous) = self.carry_sample.take() {
                // First streaming-native prototype:
                // cheap persistent half-rate stage with state carried across callbacks.
                // This is intentionally approximate compared to the offline Kaiser path.
                output.push(0.5 * (previous + sample));
            } else {
                self.carry_sample = Some(sample);
            }
        }
    }
}

struct EnergyRingBuffer {
    samples: VecDeque<f32>,
    capacity: usize,
    energy_sum: f64,
}

impl EnergyRingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            energy_sum: 0.0,
        }
    }

    fn clear(&mut self) {
        self.samples.clear();
        self.energy_sum = 0.0;
    }

    fn push_slice(&mut self, incoming: &[f32]) {
        for sample in incoming.iter().copied() {
            if self.samples.len() == self.capacity {
                if let Some(removed) = self.samples.pop_front() {
                    self.energy_sum -= (removed as f64) * (removed as f64);
                }
            }
            self.samples.push_back(sample);
            self.energy_sum += (sample as f64) * (sample as f64);
        }
    }

    fn normalized_vec(&self) -> Vec<f32> {
        if self.samples.is_empty() {
            return vec![0.0];
        }
        let rms = (self.energy_sum / self.samples.len() as f64).sqrt() as f32;
        if !rms.is_finite() || rms <= f32::EPSILON {
            return self.samples.iter().copied().collect();
        }
        let gain = 1.0 / rms;
        self.samples.iter().map(|sample| *sample * gain).collect()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

struct IncrementalStreamingFrontend {
    downsampler: HalfRateDownsamplerState,
    frontend_ring: EnergyRingBuffer,
    scratch_downsampled: Vec<f32>,
    frontend_hop_length: usize,
    pending_frontend_samples: usize,
    completed_hops_pending: usize,
    inference_hops_per_run: usize,
}

impl IncrementalStreamingFrontend {
    fn new(
        context_seconds: f64,
        inference_hops_per_run: usize,
        frontend_sample_rate: u32,
        frontend_hop_length: usize,
        stream_sample_rate: u32,
    ) -> Result<Self, anyhow::Error> {
        if stream_sample_rate != frontend_sample_rate * 2 {
            bail!(
                "incremental prototype currently supports fixed 2:1 downsampling (stream_sample_rate={}, frontend_sample_rate={})",
                stream_sample_rate,
                frontend_sample_rate
            );
        }

        let context_samples = ((context_seconds * frontend_sample_rate as f64).round() as usize)
            .max(frontend_hop_length * 2);

        Ok(Self {
            downsampler: HalfRateDownsamplerState::new(),
            frontend_ring: EnergyRingBuffer::new(context_samples),
            scratch_downsampled: Vec::new(),
            frontend_hop_length,
            pending_frontend_samples: 0,
            completed_hops_pending: 0,
            inference_hops_per_run: inference_hops_per_run.max(1),
        })
    }

    fn reset(&mut self) {
        self.downsampler.reset();
        self.frontend_ring.clear();
        self.scratch_downsampled.clear();
        self.pending_frontend_samples = 0;
        self.completed_hops_pending = 0;
    }

    fn push_callback(&mut self, callback_samples: &[f32]) {
        self.downsampler
            .process_block(callback_samples, &mut self.scratch_downsampled);
        self.frontend_ring.push_slice(&self.scratch_downsampled);
        self.pending_frontend_samples += self.scratch_downsampled.len();
        let new_hops = self.pending_frontend_samples / self.frontend_hop_length;
        self.pending_frontend_samples %= self.frontend_hop_length;
        self.completed_hops_pending += new_hops;
    }

    fn should_run_inference(&mut self) -> bool {
        if self.completed_hops_pending >= self.inference_hops_per_run {
            self.completed_hops_pending -= self.inference_hops_per_run;
            true
        } else {
            false
        }
    }

    fn normalized_frontend_window(&self) -> Vec<f32> {
        self.frontend_ring.normalized_vec()
    }

    fn context_samples(&self) -> usize {
        self.frontend_ring.capacity()
    }
}

enum LegacyScheduleState {
    EveryCallback,
    EveryNCallbacks {
        callbacks: usize,
    },
    HopCadence {
        source_samples_per_hop: usize,
        source_samples_since_inference: usize,
    },
}

impl LegacyScheduleState {
    fn new(
        schedule: &InferenceScheduling,
        frontend_hop_length: usize,
        frontend_sample_rate: u32,
        stream_sample_rate: u32,
    ) -> Result<Self, anyhow::Error> {
        match schedule {
            InferenceScheduling::EveryCallback => Ok(Self::EveryCallback),
            InferenceScheduling::EveryNCallbacks { callbacks } => {
                if *callbacks == 0 {
                    bail!("inference schedule callbacks must be positive");
                }
                Ok(Self::EveryNCallbacks {
                    callbacks: *callbacks,
                })
            }
            InferenceScheduling::HopCadence => {
                let source_samples_per_hop = ((frontend_hop_length as f64
                    * stream_sample_rate as f64
                    / frontend_sample_rate as f64)
                    .round() as usize)
                    .max(1);
                Ok(Self::HopCadence {
                    source_samples_per_hop,
                    source_samples_since_inference: 0,
                })
            }
        }
    }

    fn should_run_inference(&mut self, callback_index: usize, callback_size: usize) -> bool {
        match self {
            LegacyScheduleState::EveryCallback => true,
            LegacyScheduleState::EveryNCallbacks { callbacks } => {
                (callback_index + 1) % *callbacks == 0
            }
            LegacyScheduleState::HopCadence {
                source_samples_per_hop,
                source_samples_since_inference,
            } => {
                *source_samples_since_inference += callback_size;
                if *source_samples_since_inference >= *source_samples_per_hop {
                    *source_samples_since_inference %= *source_samples_per_hop;
                    true
                } else {
                    false
                }
            }
        }
    }
}

fn fill_callback_chunk(
    stream_samples: &[f32],
    callback_index: usize,
    callback_size: usize,
    out: &mut [f32],
) {
    out.fill(0.0);
    let start = callback_index * callback_size;
    let end = (start + callback_size).min(stream_samples.len());
    if start >= end {
        return;
    }
    out[..end - start].copy_from_slice(&stream_samples[start..end]);
}

fn rms_normalize_allow_silence(samples: &mut [f32]) -> Result<(), FrontendError> {
    match rms_normalize(samples) {
        Ok(()) => Ok(()),
        Err(FrontendError::InvalidAudio(message))
            if message.contains("RMS is zero")
                || message.contains("cannot perform RMS normalization") =>
        {
            Ok(())
        }
        Err(err) => Err(err),
    }
}

fn summarize_distribution(values_ms: &[f64]) -> DistributionStats {
    if values_ms.is_empty() {
        return DistributionStats {
            count: 0,
            mean_ms: 0.0,
            p95_ms: None,
            p99_ms: None,
            max_ms: None,
        };
    }

    let mut sorted = values_ms.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let mean_ms = values_ms.iter().sum::<f64>() / values_ms.len() as f64;

    DistributionStats {
        count: values_ms.len(),
        mean_ms,
        p95_ms: percentile(&sorted, 0.95),
        p99_ms: percentile(&sorted, 0.99),
        max_ms: sorted.last().copied(),
    }
}

fn percentile(sorted_values: &[f64], percentile: f64) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    let index = ((sorted_values.len() as f64 * percentile).ceil() as usize).saturating_sub(1);
    sorted_values.get(index).copied()
}

fn legacy_schedule_description(
    schedule: &InferenceScheduling,
    frontend_hop_length: usize,
    frontend_sample_rate: u32,
    stream_sample_rate: u32,
) -> String {
    match schedule {
        InferenceScheduling::EveryCallback => "every callback".to_owned(),
        InferenceScheduling::EveryNCallbacks { callbacks } => {
            format!("every {callbacks} callbacks")
        }
        InferenceScheduling::HopCadence => {
            let source_samples_per_hop = ((frontend_hop_length as f64 * stream_sample_rate as f64
                / frontend_sample_rate as f64)
                .round() as usize)
                .max(1);
            format!(
                "hop cadence (~{} source samples per frontend hop of {} @ {} Hz)",
                source_samples_per_hop, frontend_hop_length, frontend_sample_rate
            )
        }
    }
}

fn dominant_stage(
    audio_prep_total_ms: f64,
    frontend_total_ms: f64,
    inference_total_ms: f64,
    other_total_ms: f64,
) -> String {
    [
        ("audio_prep", audio_prep_total_ms),
        ("frontend", frontend_total_ms),
        ("inference", inference_total_ms),
        ("other", other_total_ms),
    ]
    .into_iter()
    .max_by(|left, right| left.1.total_cmp(&right.1))
    .map(|(name, _)| name.to_owned())
    .unwrap_or_else(|| "other".to_owned())
}

fn streaming_conclusion(
    deadline_met: bool,
    miss_rate: f64,
    max_backlog_ms: f64,
    callback_budget_ms: f64,
) -> String {
    if deadline_met {
        return "GO: no deadline misses observed for this configuration.".to_owned();
    }
    if miss_rate <= 0.01 && max_backlog_ms <= callback_budget_ms {
        return "BORDERLINE: occasional overruns occurred, but backlog stayed bounded.".to_owned();
    }
    "NO-GO: callback overruns are significant for this configuration.".to_owned()
}

fn callback_arrival_time(callback_index: usize, callback_interval_seconds: f64) -> f64 {
    (callback_index + 1) as f64 * callback_interval_seconds
}

fn callback_interval_seconds(callback_size: usize, stream_sample_rate: u32) -> f64 {
    callback_size as f64 / stream_sample_rate as f64
}

fn safe_fraction(numerator: f64, denominator: f64) -> f64 {
    if denominator <= 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn safe_mean(total: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn ceil_div(value: usize, divisor: usize) -> usize {
    value / divisor + usize::from(value % divisor != 0)
}

fn audio_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("audio")
        .to_owned()
}

fn sleep_until(target: Instant) {
    let now = Instant::now();
    if target > now {
        thread::sleep(target - now);
    }
}
