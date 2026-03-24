use std::collections::{BTreeMap, BTreeSet};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::audio::{load_wav_mono, trim_audio_region};
use crate::config::AppConfig;
use crate::dataset::PolyEvalEvent;
use crate::full_eval::{evaluate_full_dataset, FullAggregateMetrics};
use crate::harmonic_map::build_harmonic_maps;
use crate::nms::select_pitch_candidates;
use crate::pitch::{harmonic_weights, score_pitch_frame};
use crate::stft::{build_stft_plan, compute_stft_frames, StftPlan};
use crate::templates::GuitarTemplates;
use crate::types::{MidiHarmonicMap, SpectrumFrame};
use crate::whitening::whiten_log_spectrum;

const NOTE_CONTEXT_MARGIN_SEC: f32 = 0.005;
const ONSET_POOL_WINDOW_SEC: f32 = 0.08;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageAObjectiveWeights {
    pub w_precision: f32,
    pub w_false_positive_rate: f32,
    pub w_count_penalty: f32,
    pub target_avg_predicted_pitches: f32,
    pub recall_floor: f32,
    pub w_recall_drop: f32,
}

impl Default for StageAObjectiveWeights {
    fn default() -> Self {
        // Objective explicitly rewards precision and penalizes noisy candidate sets.
        Self {
            w_precision: 0.45,
            w_false_positive_rate: 0.35,
            w_count_penalty: 0.15,
            target_avg_predicted_pitches: 2.0,
            recall_floor: 0.70,
            w_recall_drop: 0.30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageCObjectiveWeights {
    pub w_pitch_string_f1: f32,
    pub w_pitch_string_fret_f1: f32,
    pub w_exact_configuration: f32,
    pub w_decode_top1: f32,
    pub w_decode_top3: f32,
    pub w_decode_top5: f32,
    pub w_pitch_f1: f32,
    pub w_high_confidence_wrong_decode_penalty: f32,
}

impl Default for StageCObjectiveWeights {
    fn default() -> Self {
        // Prioritize decode quality and exactness while penalizing confident mistakes.
        Self {
            w_pitch_string_f1: 0.30,
            w_pitch_string_fret_f1: 0.25,
            w_exact_configuration: 0.15,
            w_decode_top1: 0.15,
            w_decode_top3: 0.08,
            w_decode_top5: 0.04,
            w_pitch_f1: 0.03,
            w_high_confidence_wrong_decode_penalty: 0.25,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrialMetrics {
    pub pitch_precision: Option<f32>,
    pub pitch_recall: Option<f32>,
    pub pitch_f1: Option<f32>,
    pub false_positives: Option<usize>,
    pub false_positive_rate: Option<f32>,
    pub avg_predicted_pitches_per_event: Option<f32>,
    pub octave_error_rate: Option<f32>,
    pub sub_octave_error_rate: Option<f32>,
    pub pitch_string_f1: Option<f32>,
    pub pitch_string_fret_f1: Option<f32>,
    pub exact_configuration_accuracy: Option<f32>,
    pub avg_candidate_positions_per_pitch: Option<f32>,
    pub avg_global_configurations_explored: Option<f32>,
    pub decode_top1_hit_rate: Option<f32>,
    pub decode_top3_hit_rate: Option<f32>,
    pub decode_top5_hit_rate: Option<f32>,
    pub failure_pitch_correct_string_wrong_count: Option<usize>,
    pub failure_string_correct_fret_wrong_count: Option<usize>,
    pub failure_full_wrong_despite_correct_pitch_count: Option<usize>,
    pub failure_full_decode_wrong_high_confidence_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningTrialResult {
    pub trial_id: usize,
    pub stage: String,
    pub sampled_parameters: BTreeMap<String, Value>,
    pub metrics: TrialMetrics,
    pub objective: f32,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StageSearchSummary {
    pub best_trial: TuningTrialResult,
    pub best_config: AppConfig,
    pub trials_json_path: PathBuf,
    pub trials_csv_path: PathBuf,
    pub best_config_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Phase1PolyMetrics {
    pub total_events: usize,
    pub evaluated_events: usize,
    pub skipped_events: usize,
    pub true_positive_count: usize,
    pub false_positive_count: usize,
    pub false_negative_count: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub false_positive_rate: f32,
    pub avg_predicted_pitches_per_event: f32,
    pub octave_error_count: usize,
    pub octave_error_rate: f32,
    pub sub_octave_error_count: usize,
    pub sub_octave_error_rate: f32,
}

#[derive(Debug, Clone)]
struct Phase1SearchSpace {
    win_lengths: Vec<usize>,
    hop_lengths: Vec<usize>,
    whitening_span_bins: (usize, usize),
    max_harmonics: (usize, usize),
    harmonic_alpha: (f32, f32),
    local_bandwidth_bins: (usize, usize),
    lambda_subharmonic: (f32, f32),
    nms_threshold_value: (f32, f32),
    nms_radius_midi_steps: (usize, usize),
    nms_max_polyphony: (usize, usize),
}

impl Default for Phase1SearchSpace {
    fn default() -> Self {
        Self {
            win_lengths: vec![2048, 4096, 8192],
            hop_lengths: vec![256, 512, 1024],
            whitening_span_bins: (15, 63),
            max_harmonics: (4, 10),
            harmonic_alpha: (0.5, 1.5),
            local_bandwidth_bins: (1, 4),
            lambda_subharmonic: (0.0, 0.8),
            nms_threshold_value: (0.2, 0.8),
            nms_radius_midi_steps: (1, 3),
            nms_max_polyphony: (2, 6),
        }
    }
}

#[derive(Debug, Clone)]
struct Phase1Sample {
    win_length: usize,
    hop_length: usize,
    whitening_span_bins: usize,
    max_harmonics: usize,
    harmonic_alpha: f32,
    local_bandwidth_bins: usize,
    lambda_subharmonic: f32,
    nms_threshold_value: f32,
    nms_radius_midi_steps: usize,
    nms_max_polyphony: usize,
}

impl Phase1SearchSpace {
    fn sample(&self, rng: &mut SeededRng) -> Phase1Sample {
        Phase1Sample {
            win_length: rng.choose(&self.win_lengths),
            hop_length: rng.choose(&self.hop_lengths),
            whitening_span_bins: rng
                .usize_inclusive(self.whitening_span_bins.0, self.whitening_span_bins.1),
            max_harmonics: rng.usize_inclusive(self.max_harmonics.0, self.max_harmonics.1),
            harmonic_alpha: rng.f32_inclusive(self.harmonic_alpha.0, self.harmonic_alpha.1),
            local_bandwidth_bins: rng
                .usize_inclusive(self.local_bandwidth_bins.0, self.local_bandwidth_bins.1),
            lambda_subharmonic: rng
                .f32_inclusive(self.lambda_subharmonic.0, self.lambda_subharmonic.1),
            nms_threshold_value: rng
                .f32_inclusive(self.nms_threshold_value.0, self.nms_threshold_value.1),
            nms_radius_midi_steps: rng
                .usize_inclusive(self.nms_radius_midi_steps.0, self.nms_radius_midi_steps.1),
            nms_max_polyphony: rng
                .usize_inclusive(self.nms_max_polyphony.0, self.nms_max_polyphony.1),
        }
    }
}

#[derive(Debug, Clone)]
struct Phase3SearchSpace {
    alpha: (f32, f32),
    beta: (f32, f32),
    lambda_conflict: (f32, f32),
    lambda_fret_spread: (f32, f32),
    lambda_position_prior: (f32, f32),
    max_candidates_per_pitch: (usize, usize),
}

impl Default for Phase3SearchSpace {
    fn default() -> Self {
        Self {
            alpha: (0.2, 3.0),
            beta: (0.2, 3.0),
            lambda_conflict: (100.0, 5000.0),
            lambda_fret_spread: (0.0, 2.0),
            lambda_position_prior: (0.0, 0.5),
            max_candidates_per_pitch: (1, 5),
        }
    }
}

#[derive(Debug, Clone)]
struct Phase3Sample {
    alpha: f32,
    beta: f32,
    lambda_conflict: f32,
    lambda_fret_spread: f32,
    lambda_position_prior: f32,
    max_candidates_per_pitch: usize,
}

impl Phase3SearchSpace {
    fn sample(&self, rng: &mut SeededRng) -> Phase3Sample {
        Phase3Sample {
            alpha: rng.f32_inclusive(self.alpha.0, self.alpha.1),
            beta: rng.f32_inclusive(self.beta.0, self.beta.1),
            lambda_conflict: rng.f32_inclusive(self.lambda_conflict.0, self.lambda_conflict.1),
            lambda_fret_spread: rng
                .f32_inclusive(self.lambda_fret_spread.0, self.lambda_fret_spread.1),
            lambda_position_prior: rng
                .f32_inclusive(self.lambda_position_prior.0, self.lambda_position_prior.1),
            max_candidates_per_pitch: rng.usize_inclusive(
                self.max_candidates_per_pitch.0,
                self.max_candidates_per_pitch.1,
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct Phase1EventCounts {
    tp: usize,
    fp: usize,
    fn_count: usize,
    predicted_count: usize,
    octave_error: bool,
    sub_octave_error: bool,
}

#[derive(Debug, Clone)]
struct PlannedPhase1Trial {
    trial_id: usize,
    cfg: AppConfig,
    sampled_parameters: BTreeMap<String, Value>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct PlannedPhase3Trial {
    trial_id: usize,
    cfg: AppConfig,
    sampled_parameters: BTreeMap<String, Value>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct SeededRng {
    state: u64,
}

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f32_unit(&mut self) -> f32 {
        let v = (self.next_u64() >> 40) as u32;
        v as f32 / (u32::MAX >> 8) as f32
    }

    fn usize_inclusive(&mut self, min: usize, max: usize) -> usize {
        if min >= max {
            return min;
        }
        let span = (max - min + 1) as u64;
        min + (self.next_u64() % span) as usize
    }

    fn f32_inclusive(&mut self, min: f32, max: f32) -> f32 {
        if !min.is_finite() || !max.is_finite() || min >= max {
            return min;
        }
        let t = self.next_f32_unit().clamp(0.0, 1.0);
        min + t * (max - min)
    }

    fn choose<T: Copy>(&mut self, values: &[T]) -> T {
        assert!(!values.is_empty(), "cannot sample from empty choices");
        let idx = self.usize_inclusive(0, values.len() - 1);
        values[idx]
    }
}

pub fn run_stage_a_random_search(
    base_cfg: &AppConfig,
    dataset: &[PolyEvalEvent],
    trials: usize,
    seed: u64,
    objective_weights: &StageAObjectiveWeights,
    tuning_dir: &Path,
) -> Result<StageSearchSummary> {
    if trials == 0 {
        bail!("stage-a random search requires trials > 0");
    }

    create_dir_all(tuning_dir).with_context(|| {
        format!(
            "failed to create tuning directory: {}",
            tuning_dir.display()
        )
    })?;

    let mut rng = SeededRng::new(seed);
    let search_space = Phase1SearchSpace::default();
    let mut planned = Vec::with_capacity(trials);
    for trial_id in 1..=trials {
        let sample = search_space.sample(&mut rng);
        let (cfg, sampled_parameters, warnings) = apply_phase1_sample(base_cfg, sample);
        planned.push(PlannedPhase1Trial {
            trial_id,
            cfg,
            sampled_parameters,
            warnings,
        });
    }

    let trial_cfg_by_id = planned
        .iter()
        .map(|x| (x.trial_id, x.cfg.clone()))
        .collect::<BTreeMap<_, _>>();

    let mut sorted = planned
        .into_par_iter()
        .map(|planned_trial| {
            let PlannedPhase1Trial {
                trial_id,
                cfg,
                sampled_parameters,
                mut warnings,
            } = planned_trial;

            let (metrics, objective) = match evaluate_phase1_poly_dataset(dataset, &cfg) {
                Ok(metrics) => {
                    let objective = objective_stage_a(&metrics, objective_weights);
                    warnings.extend(phase1_trial_warnings(&metrics));
                    (
                        TrialMetrics {
                            pitch_precision: Some(metrics.precision),
                            pitch_recall: Some(metrics.recall),
                            pitch_f1: Some(metrics.f1),
                            false_positives: Some(metrics.false_positive_count),
                            false_positive_rate: Some(metrics.false_positive_rate),
                            avg_predicted_pitches_per_event: Some(
                                metrics.avg_predicted_pitches_per_event,
                            ),
                            octave_error_rate: Some(metrics.octave_error_rate),
                            sub_octave_error_rate: Some(metrics.sub_octave_error_rate),
                            ..TrialMetrics::default()
                        },
                        objective,
                    )
                }
                Err(err) => {
                    warnings.push(format!("trial-evaluation-failed: {err:#}"));
                    (TrialMetrics::default(), f32::NEG_INFINITY)
                }
            };

            TuningTrialResult {
                trial_id,
                stage: "stage_a".to_string(),
                sampled_parameters,
                metrics,
                objective,
                warnings,
            }
        })
        .collect::<Vec<_>>();

    sorted.sort_by(|a, b| b.objective.total_cmp(&a.objective));

    let best_trial = sorted
        .iter()
        .find(|t| t.objective.is_finite())
        .cloned()
        .context("stage-a random search produced no valid trial")?;

    let best_config = trial_cfg_by_id
        .get(&best_trial.trial_id)
        .cloned()
        .context("best trial config not found")?;

    let trials_json_path = tuning_dir.join("stage_a_trials.json");
    let trials_csv_path = tuning_dir.join("stage_a_trials.csv");
    let best_config_path = tuning_dir.join("best_stage_a_pitch_config.toml");

    write_json_pretty(&trials_json_path, &sorted)?;
    write_phase1_trials_csv(&trials_csv_path, &sorted)?;
    write_config_toml(&best_config_path, &best_config)?;

    Ok(StageSearchSummary {
        best_trial,
        best_config,
        trials_json_path,
        trials_csv_path,
        best_config_path,
    })
}

pub fn run_stage_c_random_search(
    base_cfg: &AppConfig,
    dataset: &[PolyEvalEvent],
    templates: &GuitarTemplates,
    trials: usize,
    seed: u64,
    objective_weights: &StageCObjectiveWeights,
    tuning_dir: &Path,
) -> Result<StageSearchSummary> {
    if trials == 0 {
        bail!("stage-c random search requires trials > 0");
    }

    create_dir_all(tuning_dir).with_context(|| {
        format!(
            "failed to create tuning directory: {}",
            tuning_dir.display()
        )
    })?;

    let mut rng = SeededRng::new(seed);
    let search_space = Phase3SearchSpace::default();
    let mut planned = Vec::with_capacity(trials);
    for trial_id in 1..=trials {
        let sample = search_space.sample(&mut rng);
        let (cfg, sampled_parameters, warnings) = apply_phase3_sample(base_cfg, sample);
        planned.push(PlannedPhase3Trial {
            trial_id,
            cfg,
            sampled_parameters,
            warnings,
        });
    }

    let trial_cfg_by_id = planned
        .iter()
        .map(|x| (x.trial_id, x.cfg.clone()))
        .collect::<BTreeMap<_, _>>();

    let mut sorted = planned
        .into_par_iter()
        .map(|planned_trial| {
            let PlannedPhase3Trial {
                trial_id,
                cfg,
                sampled_parameters,
                mut warnings,
            } = planned_trial;

            let (metrics, objective) = match evaluate_full_dataset(dataset, templates, &cfg) {
                Ok(artifact) => {
                    let m = artifact.metrics;
                    let objective = objective_stage_c(&m, objective_weights);
                    warnings.extend(phase3_trial_warnings(&m));
                    (
                        TrialMetrics {
                            pitch_precision: Some(m.pitch.precision),
                            pitch_recall: Some(m.pitch.recall),
                            pitch_f1: Some(m.pitch.f1),
                            pitch_string_f1: Some(m.pitch_string.f1),
                            pitch_string_fret_f1: Some(m.pitch_string_fret.f1),
                            exact_configuration_accuracy: Some(m.exact_configuration_accuracy),
                            avg_candidate_positions_per_pitch: Some(
                                m.avg_candidate_positions_per_pitch,
                            ),
                            avg_global_configurations_explored: Some(
                                m.avg_global_configurations_explored,
                            ),
                            decode_top1_hit_rate: Some(m.decode_top1_hit_rate),
                            decode_top3_hit_rate: Some(m.decode_top3_hit_rate),
                            decode_top5_hit_rate: Some(m.decode_top5_hit_rate),
                            failure_pitch_correct_string_wrong_count: Some(
                                m.failure_pitch_correct_string_wrong_count,
                            ),
                            failure_string_correct_fret_wrong_count: Some(
                                m.failure_string_correct_fret_wrong_count,
                            ),
                            failure_full_wrong_despite_correct_pitch_count: Some(
                                m.failure_full_wrong_despite_correct_pitch_count,
                            ),
                            failure_full_decode_wrong_high_confidence_count: Some(
                                m.failure_full_decode_wrong_high_confidence_count,
                            ),
                            ..TrialMetrics::default()
                        },
                        objective,
                    )
                }
                Err(err) => {
                    warnings.push(format!("trial-evaluation-failed: {err:#}"));
                    (TrialMetrics::default(), f32::NEG_INFINITY)
                }
            };

            TuningTrialResult {
                trial_id,
                stage: "stage_c".to_string(),
                sampled_parameters,
                metrics,
                objective,
                warnings,
            }
        })
        .collect::<Vec<_>>();

    sorted.sort_by(|a, b| b.objective.total_cmp(&a.objective));

    let best_trial = sorted
        .iter()
        .find(|t| t.objective.is_finite())
        .cloned()
        .context("stage-c random search produced no valid trial")?;

    let best_config = trial_cfg_by_id
        .get(&best_trial.trial_id)
        .cloned()
        .context("best trial config not found")?;

    let trials_json_path = tuning_dir.join("stage_c_trials.json");
    let trials_csv_path = tuning_dir.join("stage_c_trials.csv");
    let best_config_path = tuning_dir.join("best_stage_c_decode_config.toml");

    write_json_pretty(&trials_json_path, &sorted)?;
    write_stage_c_trials_csv(&trials_csv_path, &sorted)?;
    write_config_toml(&best_config_path, &best_config)?;

    Ok(StageSearchSummary {
        best_trial,
        best_config,
        trials_json_path,
        trials_csv_path,
        best_config_path,
    })
}

pub fn build_tuning_report(tuning_dir: &Path) -> Result<PathBuf> {
    create_dir_all(tuning_dir).with_context(|| {
        format!(
            "failed to create tuning directory: {}",
            tuning_dir.display()
        )
    })?;

    let stage_a: Vec<TuningTrialResult> =
        read_json_optional(&tuning_dir.join("stage_a_trials.json"))?.unwrap_or_default();
    let stage_c: Vec<TuningTrialResult> =
        read_json_optional(&tuning_dir.join("stage_c_trials.json"))?.unwrap_or_default();

    let report_dir = tuning_dir.join("report");
    create_dir_all(&report_dir).with_context(|| {
        format!(
            "failed to create report directory: {}",
            report_dir.display()
        )
    })?;

    let html = render_tuning_html(&stage_a, &stage_c);
    let report_path = report_dir.join("index.html");
    std::fs::write(&report_path, html)
        .with_context(|| format!("failed to write report html: {}", report_path.display()))?;

    Ok(report_path)
}

fn apply_phase1_sample(
    base_cfg: &AppConfig,
    sample: Phase1Sample,
) -> (AppConfig, BTreeMap<String, Value>, Vec<String>) {
    let mut cfg = base_cfg.clone();
    let mut warnings = Vec::new();

    cfg.frontend.win_length = sample.win_length;
    cfg.frontend.hop_length = sample.hop_length;
    if cfg.frontend.nfft < cfg.frontend.win_length {
        cfg.frontend.nfft = cfg.frontend.win_length;
        warnings.push(
            "adjusted frontend.nfft to keep frontend.win_length <= frontend.nfft".to_string(),
        );
    }

    cfg.whitening.span_bins = sample.whitening_span_bins;
    cfg.pitch.max_harmonics = sample.max_harmonics;
    cfg.pitch.harmonic_alpha = sample.harmonic_alpha;
    cfg.pitch.local_bandwidth_bins = sample.local_bandwidth_bins;
    cfg.pitch.lambda_subharmonic = sample.lambda_subharmonic;
    cfg.nms.threshold_value = sample.nms_threshold_value;
    cfg.nms.radius_midi_steps = sample.nms_radius_midi_steps;
    cfg.nms.max_polyphony = sample.nms_max_polyphony;

    let mut sampled_parameters = BTreeMap::new();
    sampled_parameters.insert(
        "frontend.win_length".to_string(),
        json!(cfg.frontend.win_length),
    );
    sampled_parameters.insert(
        "frontend.hop_length".to_string(),
        json!(cfg.frontend.hop_length),
    );
    sampled_parameters.insert("frontend.nfft".to_string(), json!(cfg.frontend.nfft));
    sampled_parameters.insert(
        "whitening.span_bins".to_string(),
        json!(cfg.whitening.span_bins),
    );
    sampled_parameters.insert(
        "pitch.max_harmonics".to_string(),
        json!(cfg.pitch.max_harmonics),
    );
    sampled_parameters.insert(
        "pitch.harmonic_alpha".to_string(),
        json!(cfg.pitch.harmonic_alpha),
    );
    sampled_parameters.insert(
        "pitch.local_bandwidth_bins".to_string(),
        json!(cfg.pitch.local_bandwidth_bins),
    );
    sampled_parameters.insert(
        "pitch.lambda_subharmonic".to_string(),
        json!(cfg.pitch.lambda_subharmonic),
    );
    sampled_parameters.insert(
        "nms.threshold_value".to_string(),
        json!(cfg.nms.threshold_value),
    );
    sampled_parameters.insert(
        "nms.radius_midi_steps".to_string(),
        json!(cfg.nms.radius_midi_steps),
    );
    sampled_parameters.insert(
        "nms.max_polyphony".to_string(),
        json!(cfg.nms.max_polyphony),
    );

    (cfg, sampled_parameters, warnings)
}

fn apply_phase3_sample(
    base_cfg: &AppConfig,
    sample: Phase3Sample,
) -> (AppConfig, BTreeMap<String, Value>, Vec<String>) {
    let mut cfg = base_cfg.clone();

    cfg.decode.alpha = sample.alpha;
    cfg.decode.beta = sample.beta;
    cfg.decode.lambda_conflict = sample.lambda_conflict;
    cfg.decode.lambda_fret_spread = sample.lambda_fret_spread;
    cfg.decode.lambda_position_prior = sample.lambda_position_prior;
    cfg.decode.max_candidates_per_pitch = sample.max_candidates_per_pitch;

    let mut sampled_parameters = BTreeMap::new();
    sampled_parameters.insert("decode.alpha".to_string(), json!(cfg.decode.alpha));
    sampled_parameters.insert("decode.beta".to_string(), json!(cfg.decode.beta));
    sampled_parameters.insert(
        "decode.lambda_conflict".to_string(),
        json!(cfg.decode.lambda_conflict),
    );
    sampled_parameters.insert(
        "decode.lambda_fret_spread".to_string(),
        json!(cfg.decode.lambda_fret_spread),
    );
    sampled_parameters.insert(
        "decode.lambda_position_prior".to_string(),
        json!(cfg.decode.lambda_position_prior),
    );
    sampled_parameters.insert(
        "decode.max_candidates_per_pitch".to_string(),
        json!(cfg.decode.max_candidates_per_pitch),
    );

    (cfg, sampled_parameters, Vec::new())
}

fn objective_stage_a(metrics: &Phase1PolyMetrics, weights: &StageAObjectiveWeights) -> f32 {
    let count_penalty =
        (metrics.avg_predicted_pitches_per_event - weights.target_avg_predicted_pitches).max(0.0);
    let recall_drop = (weights.recall_floor - metrics.recall).max(0.0);

    metrics.f1 + weights.w_precision * metrics.precision
        - weights.w_false_positive_rate * metrics.false_positive_rate
        - weights.w_count_penalty * count_penalty
        - weights.w_recall_drop * recall_drop
}

fn objective_stage_c(metrics: &FullAggregateMetrics, weights: &StageCObjectiveWeights) -> f32 {
    let denom = metrics.evaluated_events.max(1) as f32;
    let high_confidence_wrong_decode_rate =
        metrics.failure_full_decode_wrong_high_confidence_count as f32 / denom;

    weights.w_pitch_string_f1 * metrics.pitch_string.f1
        + weights.w_pitch_string_fret_f1 * metrics.pitch_string_fret.f1
        + weights.w_exact_configuration * metrics.exact_configuration_accuracy
        + weights.w_decode_top1 * metrics.decode_top1_hit_rate
        + weights.w_decode_top3 * metrics.decode_top3_hit_rate
        + weights.w_decode_top5 * metrics.decode_top5_hit_rate
        + weights.w_pitch_f1 * metrics.pitch.f1
        - weights.w_high_confidence_wrong_decode_penalty * high_confidence_wrong_decode_rate
}

fn phase1_trial_warnings(metrics: &Phase1PolyMetrics) -> Vec<String> {
    let mut out = Vec::new();
    if metrics.precision < 0.25 && metrics.avg_predicted_pitches_per_event > 2.5 {
        out.push(
            "diagnostic:noisy pitch candidates (low precision + too many predictions/event)"
                .to_string(),
        );
    }
    if metrics.recall < 0.55 {
        out.push("diagnostic:recall collapse (pitch candidate set too strict)".to_string());
    }
    out
}

fn phase3_trial_warnings(metrics: &FullAggregateMetrics) -> Vec<String> {
    let mut out = Vec::new();
    if metrics.decode_top5_hit_rate > metrics.decode_top1_hit_rate + 0.15 {
        out.push(
            "diagnostic:poor score calibration / ranking (correct decode appears in top-k but not top-1)"
                .to_string(),
        );
    }
    if metrics.pitch.f1 > metrics.pitch_string.f1 + 0.15 {
        out.push(
            "diagnostic:weak local string ranking (pitch is right more often than string attribution)"
                .to_string(),
        );
    }
    let denom = metrics.evaluated_events.max(1) as f32;
    let wrong_despite_pitch = metrics.failure_full_wrong_despite_correct_pitch_count as f32 / denom;
    if wrong_despite_pitch > 0.20 {
        out.push(
            "diagnostic:weak global decode penalties (correct pitch set still collapses to wrong full decode)"
                .to_string(),
        );
    }
    if metrics.failure_full_decode_wrong_high_confidence_count as f32 / denom > 0.10 {
        out.push(
            "diagnostic:wrong decode with high confidence suggests calibration mismatch"
                .to_string(),
        );
    }
    out
}

fn evaluate_phase1_poly_dataset(
    dataset: &[PolyEvalEvent],
    cfg: &AppConfig,
) -> Result<Phase1PolyMetrics> {
    let stft_plan = build_stft_plan(&cfg.frontend);
    let maps = build_harmonic_maps(
        cfg.sample_rate_target,
        stft_plan.nfft,
        cfg.pitch.midi_min,
        cfg.pitch.midi_max,
        cfg.pitch.max_harmonics,
        cfg.pitch.local_bandwidth_bins,
    );
    let hweights = harmonic_weights(cfg.pitch.max_harmonics, cfg.pitch.harmonic_alpha);

    let mut grouped: BTreeMap<String, Vec<&PolyEvalEvent>> = BTreeMap::new();
    for event in dataset {
        grouped
            .entry(event.audio_path.clone())
            .or_default()
            .push(event);
    }

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_count = 0usize;
    let mut octave_error_count = 0usize;
    let mut sub_octave_error_count = 0usize;
    let mut predicted_pitch_count = 0usize;
    let mut evaluated_events = 0usize;
    let mut skipped_events = 0usize;

    for (audio_path, mut events) in grouped {
        events.sort_by(|a, b| {
            a.onset_sec
                .total_cmp(&b.onset_sec)
                .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
        });

        let audio = load_wav_mono(&audio_path)
            .with_context(|| format!("failed to load audio: {audio_path}"))?;

        if audio.sample_rate != cfg.sample_rate_target {
            skipped_events += events.len();
            continue;
        }

        for event in events {
            let Some(counts) = evaluate_phase1_event(
                &audio.samples,
                audio.sample_rate,
                event,
                cfg,
                &stft_plan,
                &maps,
                &hweights,
            ) else {
                skipped_events += 1;
                continue;
            };

            evaluated_events += 1;
            tp += counts.tp;
            fp += counts.fp;
            fn_count += counts.fn_count;
            predicted_pitch_count += counts.predicted_count;
            if counts.octave_error {
                octave_error_count += 1;
            }
            if counts.sub_octave_error {
                sub_octave_error_count += 1;
            }
        }
    }

    let precision = ratio(tp, tp + fp);
    let recall = ratio(tp, tp + fn_count);

    Ok(Phase1PolyMetrics {
        total_events: dataset.len(),
        evaluated_events,
        skipped_events,
        true_positive_count: tp,
        false_positive_count: fp,
        false_negative_count: fn_count,
        precision,
        recall,
        f1: f1(precision, recall),
        false_positive_rate: ratio(fp, tp + fp),
        avg_predicted_pitches_per_event: ratio(predicted_pitch_count, evaluated_events),
        octave_error_count,
        octave_error_rate: ratio(octave_error_count, evaluated_events),
        sub_octave_error_count,
        sub_octave_error_rate: ratio(sub_octave_error_count, evaluated_events),
    })
}

fn evaluate_phase1_event(
    audio_samples: &[f32],
    sample_rate: u32,
    event: &PolyEvalEvent,
    cfg: &AppConfig,
    stft_plan: &StftPlan,
    maps: &[MidiHarmonicMap],
    hweights: &[f32],
) -> Option<Phase1EventCounts> {
    if event.notes.is_empty() {
        return None;
    }

    let start_sec = (event.onset_sec - NOTE_CONTEXT_MARGIN_SEC).max(0.0);
    let end_sec = (event.offset_sec + NOTE_CONTEXT_MARGIN_SEC).max(start_sec);
    let mut segment = trim_audio_region(audio_samples, sample_rate, start_sec, end_sec);
    if segment.is_empty() {
        return None;
    }
    if segment.len() < stft_plan.win_length {
        segment.resize(stft_plan.win_length, 0.0);
    }

    let mut frames = compute_stft_frames(&segment, sample_rate, stft_plan);
    if frames.is_empty() {
        return None;
    }

    let bins = stft_plan.nfft / 2 + 1;
    let mut tmp_log = vec![0.0_f32; bins];
    let mut envelope = vec![0.0_f32; bins];
    let mut white = vec![0.0_f32; bins];

    for frame in &mut frames {
        whiten_log_spectrum(
            &frame.mag,
            &cfg.whitening,
            &mut tmp_log,
            &mut envelope,
            &mut white,
        );
        frame.white.copy_from_slice(&white);
    }

    let onset_time = (event.onset_sec - start_sec).max(0.0);
    let pooled_midi_scores =
        pooled_pitch_scores_onset_max(&frames, onset_time, maps, cfg, hweights)?;
    let predicted = select_pitch_candidates(&pooled_midi_scores, cfg.pitch.midi_min, &cfg.nms);

    let pred_set = predicted
        .iter()
        .map(|(midi, _)| *midi)
        .collect::<BTreeSet<_>>();
    let gt_set = event.notes.iter().map(|x| x.midi).collect::<BTreeSet<_>>();

    let (tp, fp, fn_count) = set_counts(&pred_set, &gt_set);

    let matched_any = tp > 0;
    let octave_error = !matched_any
        && gt_set.iter().any(|&gt| {
            pred_set.contains(&gt.saturating_add(12))
                || (gt >= 12 && pred_set.contains(&gt.saturating_sub(12)))
        });
    let sub_octave_error = !matched_any
        && gt_set
            .iter()
            .any(|&gt| gt >= 12 && pred_set.contains(&gt.saturating_sub(12)));

    Some(Phase1EventCounts {
        tp,
        fp,
        fn_count,
        predicted_count: pred_set.len(),
        octave_error,
        sub_octave_error,
    })
}

fn pooled_pitch_scores_onset_max(
    frames: &[SpectrumFrame],
    onset_time_sec: f32,
    maps: &[MidiHarmonicMap],
    cfg: &AppConfig,
    hweights: &[f32],
) -> Option<Vec<f32>> {
    let candidate_indices = onset_window_indices(frames, onset_time_sec, ONSET_POOL_WINDOW_SEC);
    let mut indices = if candidate_indices.is_empty() {
        (0..frames.len()).collect::<Vec<_>>()
    } else {
        candidate_indices
    };
    if indices.is_empty() {
        return None;
    }
    indices.sort_unstable();
    indices.dedup();

    let mut pooled: Option<Vec<f32>> = None;
    for idx in indices {
        let frame = frames.get(idx)?;
        let pitch_frame = score_pitch_frame(&frame.white, maps, &cfg.pitch, hweights);
        if let Some(acc) = pooled.as_mut() {
            if acc.len() != pitch_frame.midi_scores.len() {
                return None;
            }
            for (dst, &score) in acc.iter_mut().zip(&pitch_frame.midi_scores) {
                if score > *dst {
                    *dst = score;
                }
            }
        } else {
            pooled = Some(pitch_frame.midi_scores);
        }
    }
    pooled
}

fn onset_window_indices(
    frames: &[SpectrumFrame],
    onset_time_sec: f32,
    window_sec: f32,
) -> Vec<usize> {
    let start = onset_time_sec.max(0.0);
    let end = start + window_sec.max(0.0);
    frames
        .iter()
        .enumerate()
        .filter_map(|(idx, frame)| {
            if frame.time_sec >= start && frame.time_sec <= end {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

fn set_counts<T: Ord + Copy>(pred: &BTreeSet<T>, gt: &BTreeSet<T>) -> (usize, usize, usize) {
    let tp = pred.intersection(gt).count();
    let fp = pred.difference(gt).count();
    let fn_count = gt.difference(pred).count();
    (tp, fp, fn_count)
}

fn write_phase1_trials_csv(path: &Path, rows: &[TuningTrialResult]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create phase1 trial csv: {}", path.display()))?;

    writeln!(
        file,
        "trial_id,objective,frontend.win_length,frontend.hop_length,frontend.nfft,whitening.span_bins,pitch.max_harmonics,pitch.harmonic_alpha,pitch.local_bandwidth_bins,pitch.lambda_subharmonic,nms.threshold_value,nms.radius_midi_steps,nms.max_polyphony,pitch_precision,pitch_recall,pitch_f1,false_positives,false_positive_rate,avg_predicted_pitches_per_event,octave_error_rate,sub_octave_error_rate,warnings"
    )?;

    for row in rows {
        writeln!(
            file,
            "{},{:.6},{},{},{},{},{},{:.6},{},{:.6},{:.6},{},{},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{}",
            row.trial_id,
            row.objective,
            param_as_string(&row.sampled_parameters, "frontend.win_length"),
            param_as_string(&row.sampled_parameters, "frontend.hop_length"),
            param_as_string(&row.sampled_parameters, "frontend.nfft"),
            param_as_string(&row.sampled_parameters, "whitening.span_bins"),
            param_as_string(&row.sampled_parameters, "pitch.max_harmonics"),
            param_as_float_or_zero(&row.sampled_parameters, "pitch.harmonic_alpha"),
            param_as_string(&row.sampled_parameters, "pitch.local_bandwidth_bins"),
            param_as_float_or_zero(&row.sampled_parameters, "pitch.lambda_subharmonic"),
            param_as_float_or_zero(&row.sampled_parameters, "nms.threshold_value"),
            param_as_string(&row.sampled_parameters, "nms.radius_midi_steps"),
            param_as_string(&row.sampled_parameters, "nms.max_polyphony"),
            metric_or_zero(row.metrics.pitch_precision),
            metric_or_zero(row.metrics.pitch_recall),
            metric_or_zero(row.metrics.pitch_f1),
            row.metrics.false_positives.unwrap_or(0),
            metric_or_zero(row.metrics.false_positive_rate),
            metric_or_zero(row.metrics.avg_predicted_pitches_per_event),
            metric_or_zero(row.metrics.octave_error_rate),
            metric_or_zero(row.metrics.sub_octave_error_rate),
            csv_escape(&row.warnings.join(" | ")),
        )?;
    }

    Ok(())
}

fn write_stage_c_trials_csv(path: &Path, rows: &[TuningTrialResult]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create stage-c trial csv: {}", path.display()))?;

    writeln!(
        file,
        "trial_id,objective,decode.alpha,decode.beta,decode.lambda_conflict,decode.lambda_fret_spread,decode.lambda_position_prior,decode.max_candidates_per_pitch,pitch_f1,pitch_string_f1,pitch_string_fret_f1,exact_configuration,decode_top1,decode_top3,decode_top5,avg_candidate_positions_per_pitch,avg_global_configurations_explored,failure_pitch_correct_string_wrong_count,failure_string_correct_fret_wrong_count,failure_full_wrong_despite_correct_pitch_count,failure_full_decode_wrong_high_confidence_count,warnings"
    )?;

    for row in rows {
        writeln!(
            file,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{}",
            row.trial_id,
            row.objective,
            param_as_float_or_zero(&row.sampled_parameters, "decode.alpha"),
            param_as_float_or_zero(&row.sampled_parameters, "decode.beta"),
            param_as_float_or_zero(&row.sampled_parameters, "decode.lambda_conflict"),
            param_as_float_or_zero(&row.sampled_parameters, "decode.lambda_fret_spread"),
            param_as_float_or_zero(&row.sampled_parameters, "decode.lambda_position_prior"),
            param_as_string(&row.sampled_parameters, "decode.max_candidates_per_pitch"),
            metric_or_zero(row.metrics.pitch_f1),
            metric_or_zero(row.metrics.pitch_string_f1),
            metric_or_zero(row.metrics.pitch_string_fret_f1),
            metric_or_zero(row.metrics.exact_configuration_accuracy),
            metric_or_zero(row.metrics.decode_top1_hit_rate),
            metric_or_zero(row.metrics.decode_top3_hit_rate),
            metric_or_zero(row.metrics.decode_top5_hit_rate),
            metric_or_zero(row.metrics.avg_candidate_positions_per_pitch),
            metric_or_zero(row.metrics.avg_global_configurations_explored),
            row.metrics
                .failure_pitch_correct_string_wrong_count
                .unwrap_or(0),
            row.metrics
                .failure_string_correct_fret_wrong_count
                .unwrap_or(0),
            row.metrics
                .failure_full_wrong_despite_correct_pitch_count
                .unwrap_or(0),
            row.metrics
                .failure_full_decode_wrong_high_confidence_count
                .unwrap_or(0),
            csv_escape(&row.warnings.join(" | ")),
        )?;
    }

    Ok(())
}

fn render_tuning_html(stage_a: &[TuningTrialResult], stage_c: &[TuningTrialResult]) -> String {
    let mut html = String::new();
    let best_a = stage_a.first();
    let best_c = stage_c.first();

    html.push_str("<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Tuning Report</title>");
    html.push_str("<style>
    :root { --bg:#f8f6ef; --card:#fff; --ink:#1f2a25; --muted:#5c6b63; --accent:#126b4f; --line:#d8dfd9; }
    body { margin:0; font-family:'Source Sans 3','Segoe UI',sans-serif; background:linear-gradient(180deg,#efe7d5 0%,#f8f6ef 40%,#edf4ef 100%); color:var(--ink); }
    main { max-width: 1240px; margin: 0 auto; padding: 24px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(270px,1fr)); gap:14px; margin:14px 0 20px; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; box-shadow:0 4px 15px rgba(20,25,21,0.06); }
    h1,h2,h3 { margin:0 0 10px 0; font-family:'Space Grotesk','Segoe UI',sans-serif; }
    p { margin:6px 0; color:var(--muted); }
    .metric { display:flex; justify-content:space-between; border-bottom:1px dashed #e8ede8; padding:4px 0; }
    .metric:last-child { border-bottom:none; }
    table { width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }
    th,td { border:1px solid #dce4de; padding:6px; text-align:left; vertical-align:top; }
    th { background:#edf5ef; cursor:pointer; }
    tbody tr:nth-child(even) { background:#fafcfa; }
    .good { color:#126b4f; font-weight:700; }
    .bad { color:#8f2f2f; font-weight:700; }
    code { background:#edf1ed; border-radius:4px; padding:1px 4px; }
    .mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
    </style></head><body><main>");

    html.push_str("<h1>Tuning Diagnostics Report</h1>");
    html.push_str("<p>Role-separated workflow: Stage A solo pitch tuning, Stage B solo template+mono diagnostics, Stage C comp decode tuning, Stage D solo control after decoder tuning.</p>");

    html.push_str("<div class=\"grid\">");
    html.push_str("<div class=\"card\"><h3>Tuning Artifacts</h3><p><a href=\"../stage_a_trials.json\">stage_a_trials.json</a></p><p><a href=\"../stage_a_trials.csv\">stage_a_trials.csv</a></p><p><a href=\"../stage_c_trials.json\">stage_c_trials.json</a></p><p><a href=\"../stage_c_trials.csv\">stage_c_trials.csv</a></p><p><a href=\"../best_stage_a_pitch_config.toml\">best_stage_a_pitch_config.toml</a></p><p><a href=\"../best_stage_c_decode_config.toml\">best_stage_c_decode_config.toml</a></p><p><a href=\"../best_stage_b_templates.bin\">best_stage_b_templates.bin</a></p></div>");
    html.push_str("<div class=\"card\"><h3>Split Reports</h3><p><a href=\"../reports/1_solo_pitch_tuning/phase1_pitch_metrics.json\">1) Solo pitch tuning report</a></p><p><a href=\"../reports/2_solo_mono_template/phase2_string_metrics.json\">2) Solo mono string/template report</a></p><p><a href=\"../reports/3_comp_poly_decode/phase3_full_metrics.json\">3) Comp poly decode tuning report</a></p><p><a href=\"../reports/4_solo_control_after_decoder/phase3_full_metrics.json\">4) Solo control report after decoder tuning</a></p></div>");

    html.push_str("<div class=\"card\"><h3>Random Search</h3>");
    html.push_str(&format!(
        "<div class=\"metric\"><span>Stage A trials</span><strong>{}</strong></div>",
        stage_a.len()
    ));
    html.push_str(&format!(
        "<div class=\"metric\"><span>Stage C trials</span><strong>{}</strong></div>",
        stage_c.len()
    ));
    if let Some(best) = best_a {
        html.push_str(&format!(
            "<div class=\"metric\"><span>Best Stage A objective</span><strong>{:.4}</strong></div>",
            best.objective
        ));
    }
    if let Some(best) = best_c {
        html.push_str(&format!(
            "<div class=\"metric\"><span>Best Stage C objective</span><strong>{:.4}</strong></div>",
            best.objective
        ));
    }
    html.push_str("</div>");

    html.push_str("<div class=\"card\"><h3>Control Note</h3><p>Stage D solo report is a control/generalization check after decoder tuning on comp. It is not the optimization target for Stage C.</p></div>");
    html.push_str("</div>");

    render_leaderboard(&mut html, "Stage A Leaderboard", stage_a);
    render_leaderboard(&mut html, "Stage C Leaderboard", stage_c);
    render_metric_trend_table(&mut html, stage_a, stage_c);
    render_best_by_metric(&mut html, stage_a, stage_c);
    render_comparison_tables(&mut html, stage_a, stage_c);
    render_best_trial_details(&mut html, best_a, best_c);
    render_bad_informative_trials(&mut html, stage_a, stage_c);

    html.push_str("<script>
    function sortTable(tableId, colIndex) {
      const table = document.getElementById(tableId);
      if (!table) return;
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);
      const asc = table.getAttribute('data-sort-col') != colIndex || table.getAttribute('data-sort-dir') !== 'asc';
      rows.sort((a,b) => {
        const av = a.cells[colIndex].innerText.trim();
        const bv = b.cells[colIndex].innerText.trim();
        const an = Number(av); const bn = Number(bv);
        if (!Number.isNaN(an) && !Number.isNaN(bn)) return asc ? an - bn : bn - an;
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      });
      rows.forEach(r => tbody.appendChild(r));
      table.setAttribute('data-sort-col', colIndex);
      table.setAttribute('data-sort-dir', asc ? 'asc' : 'desc');
    }
    document.querySelectorAll('th[data-table][data-col]').forEach(th => {
      th.addEventListener('click', () => sortTable(th.dataset.table, Number(th.dataset.col)));
    });
    </script>");

    html.push_str("</main></body></html>");
    html
}

fn render_leaderboard(output: &mut String, title: &str, rows: &[TuningTrialResult]) {
    output.push_str(&format!(
        "<section class=\"card\"><h2>{}</h2>",
        escape_html(title)
    ));
    if rows.is_empty() {
        output.push_str("<p>No trials available.</p></section>");
        return;
    }

    let table_id = if title.contains("Stage A") {
        "tbl_stage_a"
    } else {
        "tbl_stage_c"
    };
    output.push_str(&format!("<table id=\"{}\"><thead><tr>", table_id));
    output.push_str(&format!(
        "<th data-table=\"{}\" data-col=\"0\">Trial</th><th data-table=\"{}\" data-col=\"1\">Objective</th><th data-table=\"{}\" data-col=\"2\">Pitch F1</th><th data-table=\"{}\" data-col=\"3\">Pitch+String F1</th><th data-table=\"{}\" data-col=\"4\">Exact</th><th data-table=\"{}\" data-col=\"5\">Warnings</th>",
        table_id, table_id, table_id, table_id, table_id, table_id
    ));
    output.push_str("</tr></thead><tbody>");

    for row in rows.iter().take(200) {
        output.push_str("<tr>");
        output.push_str(&format!("<td>{}</td>", row.trial_id));
        output.push_str(&format!("<td>{:.6}</td>", row.objective));
        output.push_str(&format!(
            "<td>{:.6}</td>",
            metric_or_zero(row.metrics.pitch_f1)
        ));
        output.push_str(&format!(
            "<td>{:.6}</td>",
            metric_or_zero(row.metrics.pitch_string_f1)
        ));
        output.push_str(&format!(
            "<td>{:.6}</td>",
            metric_or_zero(row.metrics.exact_configuration_accuracy)
        ));
        output.push_str(&format!(
            "<td>{}</td>",
            escape_html(&row.warnings.join(" | "))
        ));
        output.push_str("</tr>");
    }

    output.push_str("</tbody></table></section>");
}

fn render_metric_trend_table(
    output: &mut String,
    stage_a: &[TuningTrialResult],
    stage_c: &[TuningTrialResult],
) {
    output.push_str("<section class=\"card\"><h2>Metric Trends Across Trials</h2>");
    output
        .push_str("<p>Rows are sorted by trial id to show progression across random samples.</p>");

    let mut merged = stage_a
        .iter()
        .chain(stage_c.iter())
        .cloned()
        .collect::<Vec<_>>();
    merged.sort_by(|a, b| {
        a.stage
            .cmp(&b.stage)
            .then_with(|| a.trial_id.cmp(&b.trial_id))
    });

    if merged.is_empty() {
        output.push_str("<p>No trials available.</p></section>");
        return;
    }

    output.push_str("<table id=\"tbl_trends\"><thead><tr><th data-table=\"tbl_trends\" data-col=\"0\">Stage</th><th data-table=\"tbl_trends\" data-col=\"1\">Trial</th><th data-table=\"tbl_trends\" data-col=\"2\">Objective</th><th data-table=\"tbl_trends\" data-col=\"3\">Precision</th><th data-table=\"tbl_trends\" data-col=\"4\">Recall</th><th data-table=\"tbl_trends\" data-col=\"5\">Pitch F1</th><th data-table=\"tbl_trends\" data-col=\"6\">Pitch+String F1</th><th data-table=\"tbl_trends\" data-col=\"7\">Exact</th></tr></thead><tbody>");
    for row in merged.iter().take(400) {
        output.push_str("<tr>");
        output.push_str(&format!("<td>{}</td>", escape_html(&row.stage)));
        output.push_str(&format!("<td>{}</td>", row.trial_id));
        output.push_str(&format!("<td>{:.6}</td>", row.objective));
        output.push_str(&format!(
            "<td>{:.6}</td>",
            metric_or_zero(row.metrics.pitch_precision)
        ));
        output.push_str(&format!(
            "<td>{:.6}</td>",
            metric_or_zero(row.metrics.pitch_recall)
        ));
        output.push_str(&format!(
            "<td>{:.6}</td>",
            metric_or_zero(row.metrics.pitch_f1)
        ));
        output.push_str(&format!(
            "<td>{:.6}</td>",
            metric_or_zero(row.metrics.pitch_string_f1)
        ));
        output.push_str(&format!(
            "<td>{:.6}</td>",
            metric_or_zero(row.metrics.exact_configuration_accuracy)
        ));
        output.push_str("</tr>");
    }
    output.push_str("</tbody></table></section>");
}

fn render_best_by_metric(
    output: &mut String,
    stage_a: &[TuningTrialResult],
    stage_c: &[TuningTrialResult],
) {
    output.push_str("<section class=\"card\"><h2>Best By Metric</h2>");

    let best_precision = stage_a.iter().max_by(|a, b| {
        metric_or_zero(a.metrics.pitch_precision)
            .total_cmp(&metric_or_zero(b.metrics.pitch_precision))
    });
    let best_pitch_f1 = stage_a.iter().max_by(|a, b| {
        metric_or_zero(a.metrics.pitch_f1).total_cmp(&metric_or_zero(b.metrics.pitch_f1))
    });
    let best_exact = stage_c.iter().max_by(|a, b| {
        metric_or_zero(a.metrics.exact_configuration_accuracy)
            .total_cmp(&metric_or_zero(b.metrics.exact_configuration_accuracy))
    });

    if let Some(trial) = best_precision {
        output.push_str(&format!(
            "<p><strong>Best precision:</strong> trial {} ({:.4})</p>",
            trial.trial_id,
            metric_or_zero(trial.metrics.pitch_precision)
        ));
    }
    if let Some(trial) = best_pitch_f1 {
        output.push_str(&format!(
            "<p><strong>Best pitch F1:</strong> trial {} ({:.4})</p>",
            trial.trial_id,
            metric_or_zero(trial.metrics.pitch_f1)
        ));
    }
    if let Some(trial) = best_exact {
        output.push_str(&format!(
            "<p><strong>Best exact configuration:</strong> trial {} ({:.4})</p>",
            trial.trial_id,
            metric_or_zero(trial.metrics.exact_configuration_accuracy)
        ));
    }
    if best_precision.is_none() && best_pitch_f1.is_none() && best_exact.is_none() {
        output.push_str("<p>No trials available.</p>");
    }

    output.push_str("</section>");
}

fn render_comparison_tables(
    output: &mut String,
    stage_a: &[TuningTrialResult],
    stage_c: &[TuningTrialResult],
) {
    output.push_str("<section class=\"card\"><h2>Comparison Tables</h2>");

    output.push_str("<h3>Precision vs Recall (Stage A)</h3>");
    if stage_a.is_empty() {
        output.push_str("<p>No Stage A trials available.</p>");
    } else {
        output.push_str("<table><thead><tr><th>Trial</th><th>Precision</th><th>Recall</th><th>Pitch F1</th><th>Pred/Event</th></tr></thead><tbody>");
        for t in stage_a.iter().take(100) {
            output.push_str("<tr>");
            output.push_str(&format!("<td>{}</td>", t.trial_id));
            output.push_str(&format!(
                "<td>{:.6}</td>",
                metric_or_zero(t.metrics.pitch_precision)
            ));
            output.push_str(&format!(
                "<td>{:.6}</td>",
                metric_or_zero(t.metrics.pitch_recall)
            ));
            output.push_str(&format!(
                "<td>{:.6}</td>",
                metric_or_zero(t.metrics.pitch_f1)
            ));
            output.push_str(&format!(
                "<td>{:.6}</td>",
                metric_or_zero(t.metrics.avg_predicted_pitches_per_event)
            ));
            output.push_str("</tr>");
        }
        output.push_str("</tbody></table>");
    }

    output.push_str("<h3>Pitch F1 vs Pitch+String F1 (Stage C)</h3>");
    if stage_c.is_empty() {
        output.push_str("<p>No Stage C trials available.</p>");
    } else {
        output.push_str("<table><thead><tr><th>Trial</th><th>Pitch F1</th><th>Pitch+String F1</th><th>Pitch+String+Fret F1</th><th>Exact</th></tr></thead><tbody>");
        for t in stage_c.iter().take(100) {
            output.push_str("<tr>");
            output.push_str(&format!("<td>{}</td>", t.trial_id));
            output.push_str(&format!(
                "<td>{:.6}</td>",
                metric_or_zero(t.metrics.pitch_f1)
            ));
            output.push_str(&format!(
                "<td>{:.6}</td>",
                metric_or_zero(t.metrics.pitch_string_f1)
            ));
            output.push_str(&format!(
                "<td>{:.6}</td>",
                metric_or_zero(t.metrics.pitch_string_fret_f1)
            ));
            output.push_str(&format!(
                "<td>{:.6}</td>",
                metric_or_zero(t.metrics.exact_configuration_accuracy)
            ));
            output.push_str("</tr>");
        }
        output.push_str("</tbody></table>");
    }

    output.push_str("<h3>Top-1 vs Top-k Decode Success (Stage C)</h3>");
    if stage_c.is_empty() {
        output.push_str("<p>No Stage C trials available.</p>");
    } else {
        output.push_str("<table><thead><tr><th>Trial</th><th>Top-1</th><th>Top-3</th><th>Top-5</th><th>Top-5 minus Top-1</th></tr></thead><tbody>");
        for t in stage_c.iter().take(100) {
            let top1 = metric_or_zero(t.metrics.decode_top1_hit_rate);
            let top3 = metric_or_zero(t.metrics.decode_top3_hit_rate);
            let top5 = metric_or_zero(t.metrics.decode_top5_hit_rate);
            output.push_str("<tr>");
            output.push_str(&format!("<td>{}</td>", t.trial_id));
            output.push_str(&format!("<td>{:.6}</td>", top1));
            output.push_str(&format!("<td>{:.6}</td>", top3));
            output.push_str(&format!("<td>{:.6}</td>", top5));
            output.push_str(&format!("<td>{:.6}</td>", top5 - top1));
            output.push_str("</tr>");
        }
        output.push_str("</tbody></table>");
    }

    output.push_str("</section>");
}

fn render_best_trial_details(
    output: &mut String,
    best_a: Option<&TuningTrialResult>,
    best_c: Option<&TuningTrialResult>,
) {
    output.push_str("<section class=\"card\"><h2>Best Trial Details</h2>");
    if let Some(trial) = best_a {
        output.push_str("<h3>Best Stage A</h3>");
        output.push_str(&format!(
            "<p>trial={} objective={:.6}</p><pre class=\"mono\">{}</pre>",
            trial.trial_id,
            trial.objective,
            escape_html(&serde_json::to_string_pretty(trial).unwrap_or_else(|_| "{}".to_string()))
        ));
    }
    if let Some(trial) = best_c {
        output.push_str("<h3>Best Stage C</h3>");
        output.push_str(&format!(
            "<p>trial={} objective={:.6}</p><pre class=\"mono\">{}</pre>",
            trial.trial_id,
            trial.objective,
            escape_html(&serde_json::to_string_pretty(trial).unwrap_or_else(|_| "{}".to_string()))
        ));
    }
    if best_a.is_none() && best_c.is_none() {
        output.push_str("<p>No best-trial details available yet.</p>");
    }
    output.push_str("</section>");
}

fn render_bad_informative_trials(
    output: &mut String,
    stage_a: &[TuningTrialResult],
    stage_c: &[TuningTrialResult],
) {
    output.push_str("<section class=\"card\"><h2>Bad But Informative Trials</h2>");

    let mut picks = Vec::new();

    if let Some(t) = stage_a.iter().rev().find(|x| !x.warnings.is_empty()) {
        picks.push(("Stage A", t));
    } else if let Some(t) = stage_a.last() {
        picks.push(("Stage A", t));
    }

    if let Some(t) = stage_c.iter().find(|x| {
        metric_or_zero(x.metrics.decode_top5_hit_rate)
            - metric_or_zero(x.metrics.decode_top1_hit_rate)
            > 0.20
    }) {
        picks.push(("Stage C", t));
    } else if let Some(t) = stage_c.last() {
        picks.push(("Stage C", t));
    }

    if picks.is_empty() {
        output.push_str("<p>No informative bad trials available.</p></section>");
        return;
    }

    for (stage_name, trial) in picks {
        output.push_str(&format!(
            "<p><strong>{}</strong> trial {} objective {:.6}</p>",
            stage_name, trial.trial_id, trial.objective
        ));
        output.push_str(&format!(
            "<p>Warnings: {}</p>",
            escape_html(&trial.warnings.join(" | "))
        ));
    }

    output.push_str("</section>");
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(value)
        .with_context(|| format!("failed to serialize json: {}", path.display()))?;
    std::fs::write(path, json)
        .with_context(|| format!("failed to write json file: {}", path.display()))?;
    Ok(())
}

fn write_config_toml(path: &Path, cfg: &AppConfig) -> Result<()> {
    let toml = toml::to_string_pretty(cfg).context("failed to serialize best config to toml")?;
    std::fs::write(path, toml)
        .with_context(|| format!("failed to write config file: {}", path.display()))?;
    Ok(())
}

fn read_json_optional<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Option<T>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read json artifact: {}", path.display()))?;
    let parsed = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse json artifact: {}", path.display()))?;
    Ok(Some(parsed))
}

fn param_as_string(params: &BTreeMap<String, Value>, key: &str) -> String {
    match params.get(key) {
        Some(Value::String(v)) => v.clone(),
        Some(Value::Number(v)) => v.to_string(),
        Some(Value::Bool(v)) => v.to_string(),
        Some(Value::Null) | None => "NA".to_string(),
        Some(other) => other.to_string(),
    }
}

fn param_as_float_or_zero(params: &BTreeMap<String, Value>, key: &str) -> f32 {
    params
        .get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.0)
}

fn metric_or_zero(v: Option<f32>) -> f32 {
    v.unwrap_or(0.0)
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn f1(precision: f32, recall: f32) -> f32 {
    if precision + recall <= 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        objective_stage_a, objective_stage_c, Phase1PolyMetrics, SeededRng, StageAObjectiveWeights,
        StageCObjectiveWeights,
    };
    use crate::full_eval::{FullAggregateMetrics, SetMetrics};

    #[test]
    fn seeded_rng_is_reproducible() {
        let mut a = SeededRng::new(123);
        let mut b = SeededRng::new(123);
        let seq_a = (0..8).map(|_| a.next_u64()).collect::<Vec<_>>();
        let seq_b = (0..8).map(|_| b.next_u64()).collect::<Vec<_>>();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn stage_a_objective_penalizes_noisy_predictions() {
        let weights = StageAObjectiveWeights::default();
        let base = Phase1PolyMetrics {
            precision: 0.30,
            recall: 0.80,
            f1: 0.44,
            false_positive_rate: 0.70,
            avg_predicted_pitches_per_event: 2.0,
            ..Phase1PolyMetrics::default()
        };
        let noisy = Phase1PolyMetrics {
            avg_predicted_pitches_per_event: 4.5,
            false_positive_rate: 0.85,
            precision: 0.15,
            ..base.clone()
        };

        assert!(objective_stage_a(&base, &weights) > objective_stage_a(&noisy, &weights));
    }

    #[test]
    fn stage_c_objective_prefers_string_aware_gain() {
        let weights = StageCObjectiveWeights::default();
        let mk = |pitch_f1, ps_f1, psf_f1, exact, top1, top3, top5, high_conf_wrong| {
            FullAggregateMetrics {
                interpretation: String::new(),
                total_events: 1,
                evaluated_events: 1,
                skipped_events: 0,
                comp_filename_events: 1,
                solo_filename_events: 0,
                other_filename_events: 0,
                pitch: SetMetrics {
                    true_positive_count: 0,
                    false_positive_count: 0,
                    false_negative_count: 0,
                    precision: pitch_f1,
                    recall: pitch_f1,
                    f1: pitch_f1,
                },
                pitch_string: SetMetrics {
                    true_positive_count: 0,
                    false_positive_count: 0,
                    false_negative_count: 0,
                    precision: ps_f1,
                    recall: ps_f1,
                    f1: ps_f1,
                },
                pitch_string_fret: SetMetrics {
                    true_positive_count: 0,
                    false_positive_count: 0,
                    false_negative_count: 0,
                    precision: psf_f1,
                    recall: psf_f1,
                    f1: psf_f1,
                },
                exact_configuration_accuracy: exact,
                decode_top1_hit_count: 0,
                decode_top3_hit_count: 0,
                decode_top5_hit_count: 0,
                decode_top1_hit_rate: top1,
                decode_top3_hit_rate: top3,
                decode_top5_hit_rate: top5,
                avg_candidate_positions_per_pitch: 0.0,
                avg_global_configurations_explored: 0.0,
                failure_pitch_correct_string_wrong_count: 0,
                failure_string_correct_fret_wrong_count: 0,
                failure_full_wrong_despite_correct_pitch_count: 0,
                failure_full_decode_wrong_high_confidence_count: high_conf_wrong,
            }
        };

        let pitch_only_better = mk(0.70, 0.10, 0.05, 0.02, 0.10, 0.10, 0.10, 0);
        let string_better = mk(0.55, 0.35, 0.20, 0.08, 0.40, 0.55, 0.65, 0);
        let high_conf_wrong = mk(0.55, 0.35, 0.20, 0.08, 0.40, 0.55, 0.65, 1);

        assert!(
            objective_stage_c(&string_better, &weights)
                > objective_stage_c(&pitch_only_better, &weights)
        );
        assert!(
            objective_stage_c(&string_better, &weights)
                > objective_stage_c(&high_conf_wrong, &weights)
        );
    }
}
