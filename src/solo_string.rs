use std::collections::BTreeMap;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::AppConfig;
use crate::dataset::{filename_contains_solo, load_mono_annotations};
use crate::features::{extract_note_features_for_record, flatten_note_features};
use crate::fretboard::fret_region;
use crate::harmonic_map::build_harmonic_maps;
use crate::stft::build_stft_plan;
use crate::templates::{
    running_stats_finalize, running_stats_new, running_stats_update, save_templates,
    score_string_fret_candidates, GuitarTemplates,
};
use crate::types::{MonoAnnotation, NoteFeatures, OPEN_MIDI};

const STRING_COUNT: usize = 6;
const DEFAULT_TRIALS: usize = 500;
const TOPK_STORE: usize = 6;
const MAX_HIGHLIGHTS: usize = 12;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoloStringObjectiveWeights {
    pub w_top1: f32,
    pub w_top3: f32,
    pub w_margin: f32,
    pub w_high_confidence_wrong_rate: f32,
}

impl Default for SoloStringObjectiveWeights {
    fn default() -> Self {
        Self {
            w_top1: 0.70,
            w_top3: 0.10,
            w_margin: 0.20,
            w_high_confidence_wrong_rate: 0.25,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoloStringEvalThresholds {
    pub high_confidence_margin_threshold: f32,
    pub low_margin_threshold: f32,
}

impl Default for SoloStringEvalThresholds {
    fn default() -> Self {
        Self {
            high_confidence_margin_threshold: 0.40,
            low_margin_threshold: 0.08,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SoloStringRunSummary {
    pub phase_dir: PathBuf,
    pub trials_json_path: PathBuf,
    pub trials_csv_path: PathBuf,
    pub best_config_path: PathBuf,
    pub best_top1_config_path: PathBuf,
    pub best_balanced_config_path: PathBuf,
    pub best_robust_config_path: PathBuf,
    pub metrics_json_path: PathBuf,
    pub confusion_csv_path: PathBuf,
    pub examples_csv_path: PathBuf,
    pub report_path: PathBuf,
    pub best_trial: SoloStringTrialResult,
}

#[derive(Debug, Clone)]
pub struct SoloStringEvalSummary {
    pub phase_dir: PathBuf,
    pub metrics_json_path: PathBuf,
    pub confusion_csv_path: PathBuf,
    pub examples_csv_path: PathBuf,
    pub report_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoloStringTrialResult {
    pub trial_id: usize,
    pub sampled_parameters: BTreeMap<String, Value>,
    pub metrics: SoloStringTrialMetrics,
    pub objective: f32,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoloStringTrialMetrics {
    pub evaluated_examples: usize,
    pub skipped_examples: usize,
    pub top1_string_accuracy: f32,
    pub top2_string_accuracy: f32,
    pub top3_string_accuracy: f32,
    pub average_margin_top1_top2: f32,
    pub high_confidence_wrong_rate: f32,
    pub low_margin_ambiguous_rate: f32,
    pub adjacent_string_confusion_rate_overall: f32,
    pub adjacent_string_confusion_rate_of_errors: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassAccuracy {
    pub correct: usize,
    pub total: usize,
    pub accuracy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedCandidate {
    pub rank: usize,
    pub string: u8,
    pub fret: u8,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoloStringExampleRanking {
    pub example_id: String,
    pub file_id: String,
    pub true_midi: u8,
    pub true_string: u8,
    pub true_fret: u8,
    pub ranked_candidate_strings: Vec<u8>,
    pub ranked_candidate_positions: Vec<RankedCandidate>,
    pub top1_score: Option<f32>,
    pub top2_score: Option<f32>,
    pub margin_top1_top2: Option<f32>,
    pub truth_in_top1: bool,
    pub truth_in_top2: bool,
    pub truth_in_top3: bool,
    pub adjacent_string_confusion: bool,
    pub high_confidence_wrong: bool,
    pub low_margin_ambiguous: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoloStringConfusionMatrix {
    pub labels: Vec<String>,
    pub counts: Vec<Vec<u32>>,
    pub normalized: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoloStringMetrics {
    pub dataset_scope: String,
    pub records_total_before_filename_filter: usize,
    pub records_filtered_non_solo: usize,
    pub records_total_after_solo_filter: usize,
    pub records_evaluated: usize,
    pub records_skipped: usize,
    pub top1_string_accuracy: f32,
    pub top2_string_accuracy: f32,
    pub top3_string_accuracy: f32,
    pub exact_string_fret_meaningful: bool,
    pub exact_string_fret_accuracy: Option<f32>,
    pub accuracy_by_true_string: BTreeMap<u8, ClassAccuracy>,
    pub accuracy_by_midi_pitch: BTreeMap<u8, ClassAccuracy>,
    pub accuracy_by_fret_region: BTreeMap<String, ClassAccuracy>,
    pub average_margin_top1_top2: f32,
    pub median_margin_top1_top2: f32,
    pub mean_margin_correct: f32,
    pub median_margin_correct: f32,
    pub mean_margin_wrong: f32,
    pub median_margin_wrong: f32,
    pub high_confidence_margin_threshold: f32,
    pub high_confidence_wrong_count: usize,
    pub high_confidence_wrong_rate: f32,
    pub low_margin_threshold: f32,
    pub low_margin_ambiguous_count: usize,
    pub low_margin_ambiguous_rate: f32,
    pub adjacent_string_confusion_count: usize,
    pub adjacent_string_confusion_rate_overall: f32,
    pub adjacent_string_confusion_rate_of_errors: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureAblationRow {
    pub profile: String,
    pub top1_string_accuracy: f32,
    pub top3_string_accuracy: f32,
    pub average_margin_top1_top2: f32,
    pub high_confidence_wrong_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSeparabilityRow {
    pub feature: String,
    pub between_over_within: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoloStringRecommendations {
    pub objective_best_trial_id: usize,
    pub top1_best_trial_id: usize,
    pub balanced_best_trial_id: usize,
    pub robust_best_trial_id: usize,
    pub objective_formula: String,
    pub balanced_formula: String,
    pub robust_formula: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoloStringArtifact {
    pub metrics: SoloStringMetrics,
    pub confusion_matrix: SoloStringConfusionMatrix,
    pub examples: Vec<SoloStringExampleRanking>,
}

#[derive(Debug, Clone)]
struct SoloDatasetInput {
    records: Vec<MonoAnnotation>,
    total_before_filter: usize,
    filtered_non_solo: usize,
}

#[derive(Debug, Clone)]
struct PreparedSoloExample {
    note: MonoAnnotation,
    features: NoteFeatures,
    file_id: String,
}

#[derive(Debug, Clone)]
struct PreparedSoloDataset {
    examples: Vec<PreparedSoloExample>,
    total_after_solo_filter: usize,
    skipped_feature_extraction: usize,
    total_before_filter: usize,
    filtered_non_solo: usize,
}

#[derive(Debug, Clone)]
struct AccuracyCounter {
    correct: usize,
    total: usize,
}

#[derive(Debug, Clone)]
struct PlannedTrial {
    trial_id: usize,
    cfg: AppConfig,
    sampled_parameters: BTreeMap<String, Value>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct TrialEval {
    trial_metrics: SoloStringTrialMetrics,
    artifact: SoloStringArtifact,
    templates: GuitarTemplates,
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

    fn choose<T: Copy>(&mut self, values: &[T]) -> T {
        assert!(!values.is_empty(), "cannot sample from empty choices");
        let idx = self.usize_inclusive(0, values.len() - 1);
        values[idx]
    }

    fn usize_inclusive(&mut self, min: usize, max: usize) -> usize {
        if min >= max {
            return min;
        }
        let span = (max - min + 1) as u64;
        min + (self.next_u64() % span) as usize
    }

    fn log_f32(&mut self, min: f32, max: f32) -> f32 {
        if !min.is_finite() || !max.is_finite() || min <= 0.0 || max <= min {
            return min;
        }
        let t = self.next_f32_unit().clamp(0.0, 1.0);
        let lo = min.ln();
        let hi = max.ln();
        (lo + (hi - lo) * t).exp()
    }
}

#[derive(Debug, Clone, Copy)]
struct FeatureProfile {
    name: &'static str,
    include_harmonic_ratios: bool,
    include_centroid: bool,
    include_rolloff: bool,
    include_flux: bool,
    include_attack_ratio: bool,
    include_noise_ratio: bool,
    include_inharmonicity: bool,
}

const FEATURE_PROFILES: [FeatureProfile; 5] = [
    FeatureProfile {
        name: "harmonics_only",
        include_harmonic_ratios: false,
        include_centroid: false,
        include_rolloff: false,
        include_flux: false,
        include_attack_ratio: false,
        include_noise_ratio: false,
        include_inharmonicity: false,
    },
    FeatureProfile {
        name: "harmonics_plus_ratios",
        include_harmonic_ratios: true,
        include_centroid: false,
        include_rolloff: false,
        include_flux: false,
        include_attack_ratio: false,
        include_noise_ratio: false,
        include_inharmonicity: false,
    },
    FeatureProfile {
        name: "harmonics_ratios_centroid_rolloff",
        include_harmonic_ratios: true,
        include_centroid: true,
        include_rolloff: true,
        include_flux: false,
        include_attack_ratio: false,
        include_noise_ratio: false,
        include_inharmonicity: false,
    },
    FeatureProfile {
        name: "harmonics_ratios_flux_noise_inharm",
        include_harmonic_ratios: true,
        include_centroid: false,
        include_rolloff: false,
        include_flux: true,
        include_attack_ratio: false,
        include_noise_ratio: true,
        include_inharmonicity: true,
    },
    FeatureProfile {
        name: "full_default",
        include_harmonic_ratios: true,
        include_centroid: true,
        include_rolloff: true,
        include_flux: true,
        include_attack_ratio: false,
        include_noise_ratio: true,
        include_inharmonicity: true,
    },
];

const REGION_BOUND_SETS: [&[u8]; 6] = [
    &[0, 5, 10, 15, 21],
    &[0, 4, 8, 12, 21],
    &[0, 3, 7, 12, 21],
    &[0, 6, 12, 21],
    &[0, 7, 14, 21],
    &[0, 5, 9, 13, 17, 21],
];

pub fn optimize_string_solo(
    base_cfg: &AppConfig,
    mono_dataset_path: &str,
    dataset_root: &Path,
    out_dir: &Path,
    trials: Option<usize>,
    seed: Option<u64>,
    weights: &SoloStringObjectiveWeights,
    thresholds: &SoloStringEvalThresholds,
) -> Result<SoloStringRunSummary> {
    let trial_count = trials.unwrap_or(DEFAULT_TRIALS);
    if trial_count == 0 {
        bail!("optimize-string-solo requires trials > 0");
    }

    let seed_value = seed.unwrap_or(base_cfg.optimization.seed);
    let solo_input = load_solo_annotations_strict(mono_dataset_path, dataset_root)?;
    let prepared = prepare_solo_dataset(&solo_input, base_cfg)?;

    create_dir_all(out_dir)
        .with_context(|| format!("failed to create output directory: {}", out_dir.display()))?;

    let mut rng = SeededRng::new(seed_value);
    let mut planned = Vec::with_capacity(trial_count);
    for trial_id in 1..=trial_count {
        let (cfg, sampled_parameters, warnings) = sample_s1_trial(base_cfg, &mut rng);
        planned.push(PlannedTrial {
            trial_id,
            cfg,
            sampled_parameters,
            warnings,
        });
    }

    let trial_cfg_by_id = planned
        .iter()
        .map(|p| (p.trial_id, p.cfg.clone()))
        .collect::<BTreeMap<_, _>>();

    let mut trial_rows = planned
        .into_par_iter()
        .map(|planned_trial| {
            let mut warnings = planned_trial.warnings;
            let trial_metrics =
                match evaluate_trial(&prepared, &planned_trial.cfg, thresholds, false) {
                    Ok(eval) => eval.trial_metrics,
                    Err(err) => {
                        warnings.push(format!("trial-evaluation-failed: {err:#}"));
                        SoloStringTrialMetrics {
                            evaluated_examples: 0,
                            skipped_examples: prepared.total_after_solo_filter,
                            top1_string_accuracy: 0.0,
                            top2_string_accuracy: 0.0,
                            top3_string_accuracy: 0.0,
                            average_margin_top1_top2: 0.0,
                            high_confidence_wrong_rate: 1.0,
                            low_margin_ambiguous_rate: 0.0,
                            adjacent_string_confusion_rate_overall: 0.0,
                            adjacent_string_confusion_rate_of_errors: 0.0,
                        }
                    }
                };
            let objective = objective_s1(&trial_metrics, weights);

            SoloStringTrialResult {
                trial_id: planned_trial.trial_id,
                sampled_parameters: planned_trial.sampled_parameters,
                metrics: trial_metrics,
                objective,
                warnings,
            }
        })
        .collect::<Vec<_>>();

    trial_rows.sort_by(|a, b| b.objective.total_cmp(&a.objective));

    let best_trial = trial_rows
        .iter()
        .find(|t| t.metrics.evaluated_examples > 0)
        .cloned()
        .context("no valid Phase S1 trial was evaluated")?;

    let best_cfg = trial_cfg_by_id
        .get(&best_trial.trial_id)
        .cloned()
        .context("best trial config is missing")?;

    let best_top1_trial = trial_rows
        .iter()
        .filter(|t| t.metrics.evaluated_examples > 0)
        .max_by(|a, b| {
            a.metrics
                .top1_string_accuracy
                .total_cmp(&b.metrics.top1_string_accuracy)
                .then_with(|| a.objective.total_cmp(&b.objective))
        })
        .cloned()
        .context("failed to select top1-best trial")?;

    let best_balanced_trial = trial_rows
        .iter()
        .filter(|t| t.metrics.evaluated_examples > 0)
        .max_by(|a, b| {
            balanced_score(&a.metrics)
                .total_cmp(&balanced_score(&b.metrics))
                .then_with(|| a.objective.total_cmp(&b.objective))
        })
        .cloned()
        .context("failed to select balanced-best trial")?;

    let best_robust_trial = trial_rows
        .iter()
        .filter(|t| t.metrics.evaluated_examples > 0)
        .max_by(|a, b| {
            robust_score(&a.metrics)
                .total_cmp(&robust_score(&b.metrics))
                .then_with(|| a.objective.total_cmp(&b.objective))
        })
        .cloned()
        .context("failed to select robust-best trial")?;

    let best_top1_cfg = trial_cfg_by_id
        .get(&best_top1_trial.trial_id)
        .cloned()
        .context("best top1 trial config is missing")?;
    let best_balanced_cfg = trial_cfg_by_id
        .get(&best_balanced_trial.trial_id)
        .cloned()
        .context("best balanced trial config is missing")?;
    let best_robust_cfg = trial_cfg_by_id
        .get(&best_robust_trial.trial_id)
        .cloned()
        .context("best robust trial config is missing")?;

    let final_eval = evaluate_trial(&prepared, &best_cfg, thresholds, true)?;

    let feature_ablation = run_feature_ablation(&prepared, &best_cfg, thresholds)?;
    let feature_separability = compute_feature_separability(&prepared, &best_cfg);

    let recommendations = SoloStringRecommendations {
        objective_best_trial_id: best_trial.trial_id,
        top1_best_trial_id: best_top1_trial.trial_id,
        balanced_best_trial_id: best_balanced_trial.trial_id,
        robust_best_trial_id: best_robust_trial.trial_id,
        objective_formula:
            "w1*top1 + w2*top3 + w3*average_margin_top1_top2 - w4*high_confidence_wrong_rate"
                .to_string(),
        balanced_formula: "0.7*top1 + 0.3*top3".to_string(),
        robust_formula: "top1 + 0.5*average_margin_top1_top2 - high_confidence_wrong_rate"
            .to_string(),
    };

    let trials_json_path = out_dir.join("trials.json");
    let trials_csv_path = out_dir.join("trials.csv");
    let best_config_path = out_dir.join("best_config.toml");
    let best_top1_config_path = out_dir.join("best_config_top1.toml");
    let best_balanced_config_path = out_dir.join("best_config_balanced.toml");
    let best_robust_config_path = out_dir.join("best_config_robust.toml");

    write_json_pretty(&trials_json_path, &trial_rows)?;
    write_trials_csv(&trials_csv_path, &trial_rows)?;
    write_config_toml(&best_config_path, &best_cfg)?;
    write_config_toml(&best_top1_config_path, &best_top1_cfg)?;
    write_config_toml(&best_balanced_config_path, &best_balanced_cfg)?;
    write_config_toml(&best_robust_config_path, &best_robust_cfg)?;

    let metrics_json_path = out_dir.join("string_metrics.json");
    let confusion_json_path = out_dir.join("string_confusion_matrix.json");
    let confusion_csv_path = out_dir.join("string_confusion_matrix.csv");
    let examples_json_path = out_dir.join("example_rankings.json");
    let examples_csv_path = out_dir.join("example_rankings.csv");

    write_json_pretty(&metrics_json_path, &final_eval.artifact.metrics)?;
    write_json_pretty(&confusion_json_path, &final_eval.artifact.confusion_matrix)?;
    write_confusion_csv(&confusion_csv_path, &final_eval.artifact.confusion_matrix)?;
    write_json_pretty(&examples_json_path, &final_eval.artifact.examples)?;
    write_examples_csv(&examples_csv_path, &final_eval.artifact.examples)?;
    write_json_pretty(&out_dir.join("feature_ablation.json"), &feature_ablation)?;
    write_feature_ablation_csv(&out_dir.join("feature_ablation.csv"), &feature_ablation)?;
    write_json_pretty(
        &out_dir.join("feature_separability.json"),
        &feature_separability,
    )?;
    write_feature_separability_csv(
        &out_dir.join("feature_separability.csv"),
        &feature_separability,
    )?;
    write_json_pretty(
        &out_dir.join("best_config_recommendations.json"),
        &recommendations,
    )?;

    save_templates(
        &out_dir.join("best_templates.bin").to_string_lossy(),
        &final_eval.templates,
    )?;

    let report_path = build_string_solo_report(out_dir)?;

    Ok(SoloStringRunSummary {
        phase_dir: out_dir.to_path_buf(),
        trials_json_path,
        trials_csv_path,
        best_config_path,
        best_top1_config_path,
        best_balanced_config_path,
        best_robust_config_path,
        metrics_json_path,
        confusion_csv_path,
        examples_csv_path,
        report_path,
        best_trial,
    })
}

pub fn eval_string_solo(
    cfg: &AppConfig,
    mono_dataset_path: &str,
    dataset_root: &Path,
    out_dir: &Path,
    thresholds: &SoloStringEvalThresholds,
) -> Result<SoloStringEvalSummary> {
    let solo_input = load_solo_annotations_strict(mono_dataset_path, dataset_root)?;
    let prepared = prepare_solo_dataset(&solo_input, cfg)?;
    let eval = evaluate_trial(&prepared, cfg, thresholds, true)?;

    create_dir_all(out_dir)
        .with_context(|| format!("failed to create output directory: {}", out_dir.display()))?;

    let metrics_json_path = out_dir.join("string_metrics.json");
    let confusion_json_path = out_dir.join("string_confusion_matrix.json");
    let confusion_csv_path = out_dir.join("string_confusion_matrix.csv");
    let examples_json_path = out_dir.join("example_rankings.json");
    let examples_csv_path = out_dir.join("example_rankings.csv");

    write_json_pretty(&metrics_json_path, &eval.artifact.metrics)?;
    write_json_pretty(&confusion_json_path, &eval.artifact.confusion_matrix)?;
    write_confusion_csv(&confusion_csv_path, &eval.artifact.confusion_matrix)?;
    write_json_pretty(&examples_json_path, &eval.artifact.examples)?;
    write_examples_csv(&examples_csv_path, &eval.artifact.examples)?;
    write_config_toml(&out_dir.join("best_config.toml"), cfg)?;
    save_templates(
        &out_dir.join("best_templates.bin").to_string_lossy(),
        &eval.templates,
    )?;

    let feature_ablation = run_feature_ablation(&prepared, cfg, thresholds)?;
    let feature_separability = compute_feature_separability(&prepared, cfg);
    write_json_pretty(&out_dir.join("feature_ablation.json"), &feature_ablation)?;
    write_feature_ablation_csv(&out_dir.join("feature_ablation.csv"), &feature_ablation)?;
    write_json_pretty(
        &out_dir.join("feature_separability.json"),
        &feature_separability,
    )?;
    write_feature_separability_csv(
        &out_dir.join("feature_separability.csv"),
        &feature_separability,
    )?;

    let report_path = build_string_solo_report(out_dir)?;

    Ok(SoloStringEvalSummary {
        phase_dir: out_dir.to_path_buf(),
        metrics_json_path,
        confusion_csv_path,
        examples_csv_path,
        report_path,
    })
}

pub fn build_string_solo_report(out_dir: &Path) -> Result<PathBuf> {
    create_dir_all(out_dir)
        .with_context(|| format!("failed to create output directory: {}", out_dir.display()))?;
    let report_dir = out_dir.join("report");
    create_dir_all(&report_dir).with_context(|| {
        format!(
            "failed to create report directory: {}",
            report_dir.display()
        )
    })?;

    let trials: Vec<SoloStringTrialResult> =
        read_json_optional(&out_dir.join("trials.json"))?.unwrap_or_default();
    let metrics: Option<SoloStringMetrics> =
        read_json_optional(&out_dir.join("string_metrics.json"))?;
    let confusion: Option<SoloStringConfusionMatrix> =
        read_json_optional(&out_dir.join("string_confusion_matrix.json"))?;
    let examples: Vec<SoloStringExampleRanking> =
        read_json_optional(&out_dir.join("example_rankings.json"))?.unwrap_or_default();
    let ablation: Vec<FeatureAblationRow> =
        read_json_optional(&out_dir.join("feature_ablation.json"))?.unwrap_or_default();
    let separability: Vec<FeatureSeparabilityRow> =
        read_json_optional(&out_dir.join("feature_separability.json"))?.unwrap_or_default();
    let recommendations: Option<SoloStringRecommendations> =
        read_json_optional(&out_dir.join("best_config_recommendations.json"))?;

    let html = render_phase_s1_html(
        &trials,
        metrics.as_ref(),
        confusion.as_ref(),
        &examples,
        &ablation,
        &separability,
        recommendations.as_ref(),
    );

    let report_path = report_dir.join("index.html");
    std::fs::write(&report_path, html)
        .with_context(|| format!("failed to write report html: {}", report_path.display()))?;

    Ok(report_path)
}

fn objective_s1(metrics: &SoloStringTrialMetrics, weights: &SoloStringObjectiveWeights) -> f32 {
    weights.w_top1 * metrics.top1_string_accuracy
        + weights.w_top3 * metrics.top3_string_accuracy
        + weights.w_margin * metrics.average_margin_top1_top2
        - weights.w_high_confidence_wrong_rate * metrics.high_confidence_wrong_rate
}

fn balanced_score(metrics: &SoloStringTrialMetrics) -> f32 {
    0.7 * metrics.top1_string_accuracy + 0.3 * metrics.top3_string_accuracy
}

fn robust_score(metrics: &SoloStringTrialMetrics) -> f32 {
    metrics.top1_string_accuracy + 0.5 * metrics.average_margin_top1_top2
        - metrics.high_confidence_wrong_rate
}

fn load_solo_annotations_strict(path: &str, dataset_root: &Path) -> Result<SoloDatasetInput> {
    let all = load_mono_annotations(path, dataset_root, false)?;
    let total_before_filter = all.len();
    let records = all
        .into_iter()
        .filter(|r| filename_contains_solo(&r.audio_path))
        .collect::<Vec<_>>();
    if records.is_empty() {
        bail!(
            "Phase S1 requires SOLO records; no filenames containing 'solo' were found in {}",
            path
        );
    }
    Ok(SoloDatasetInput {
        filtered_non_solo: total_before_filter.saturating_sub(records.len()),
        total_before_filter,
        records,
    })
}

fn prepare_solo_dataset(input: &SoloDatasetInput, cfg: &AppConfig) -> Result<PreparedSoloDataset> {
    let mut sorted = input.records.clone();
    sorted.sort_by(|a, b| {
        a.audio_path
            .cmp(&b.audio_path)
            .then_with(|| a.onset_sec.total_cmp(&b.onset_sec))
            .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
            .then_with(|| a.midi.cmp(&b.midi))
            .then_with(|| a.string.cmp(&b.string))
            .then_with(|| a.fret.cmp(&b.fret))
    });

    let stft_plan = build_stft_plan(&cfg.frontend);
    let midi_min = sorted
        .iter()
        .map(|x| x.midi)
        .min()
        .unwrap_or(cfg.pitch.midi_min)
        .min(cfg.pitch.midi_min);
    let midi_max = sorted
        .iter()
        .map(|x| x.midi)
        .max()
        .unwrap_or(cfg.pitch.midi_max)
        .max(cfg.pitch.midi_max);

    let maps = build_harmonic_maps(
        cfg.sample_rate_target,
        stft_plan.nfft,
        midi_min,
        midi_max,
        cfg.templates.num_harmonics.max(cfg.pitch.max_harmonics),
        cfg.pitch.local_bandwidth_bins,
    );

    let mut grouped: BTreeMap<String, Vec<MonoAnnotation>> = BTreeMap::new();
    for rec in sorted {
        grouped.entry(rec.audio_path.clone()).or_default().push(rec);
    }

    let mut examples = Vec::new();
    let mut skipped_feature_extraction = 0usize;

    for (audio_path, notes) in grouped {
        let audio = crate::audio::load_wav_mono(&audio_path)
            .with_context(|| format!("failed to load audio for Phase S1: {audio_path}"))?;
        let file_id = file_id_for_path(&audio_path);
        if audio.sample_rate != cfg.sample_rate_target {
            skipped_feature_extraction += notes.len();
            continue;
        }

        for note in &notes {
            let Some(features) =
                extract_note_features_for_record(note, &audio, &stft_plan, cfg, &maps)?
            else {
                skipped_feature_extraction += 1;
                continue;
            };

            examples.push(PreparedSoloExample {
                note: note.clone(),
                features,
                file_id: file_id.clone(),
            });
        }
    }

    if examples.is_empty() {
        bail!("Phase S1 preparation produced zero evaluable SOLO examples");
    }

    Ok(PreparedSoloDataset {
        examples,
        total_after_solo_filter: input.records.len(),
        skipped_feature_extraction,
        total_before_filter: input.total_before_filter,
        filtered_non_solo: input.filtered_non_solo,
    })
}

fn sample_s1_trial(
    base_cfg: &AppConfig,
    rng: &mut SeededRng,
) -> (AppConfig, BTreeMap<String, Value>, Vec<String>) {
    let profile = FEATURE_PROFILES[rng.usize_inclusive(0, FEATURE_PROFILES.len() - 1)];
    let level = rng.choose(&["string", "string_region"]);
    let score_type = rng.choose(&["diag_mahalanobis", "cosine"]);
    let feature_normalization = rng.choose(&["none", "l2", "zscore"]);

    let mut cfg = base_cfg.clone();
    cfg.templates.level = level.to_string();
    cfg.templates.score_type = score_type.to_string();
    cfg.templates.feature_normalization = feature_normalization.to_string();
    cfg.templates.mahalanobis_std_floor = rng.log_f32(1e-5, 5e-2);
    cfg.templates.scorer_epsilon = rng.log_f32(1e-8, 1e-4);

    apply_feature_profile(&mut cfg, profile);

    let mut warnings = Vec::new();
    if cfg.templates.level == "string_region" {
        let chosen_bounds = REGION_BOUND_SETS[rng.usize_inclusive(0, REGION_BOUND_SETS.len() - 1)];
        cfg.templates.region_bounds = chosen_bounds.to_vec();
    } else if cfg.templates.region_bounds.len() < 2 {
        cfg.templates.region_bounds = vec![0, 21];
        warnings.push("templates.region_bounds repaired for non-region template level".to_string());
    }

    let mut sampled_parameters = BTreeMap::new();
    sampled_parameters.insert("templates.level".to_string(), json!(cfg.templates.level));
    sampled_parameters.insert(
        "templates.score_type".to_string(),
        json!(cfg.templates.score_type),
    );
    sampled_parameters.insert("feature_profile".to_string(), json!(profile.name));
    sampled_parameters.insert(
        "templates.feature_normalization".to_string(),
        json!(cfg.templates.feature_normalization),
    );
    sampled_parameters.insert(
        "templates.mahalanobis_std_floor".to_string(),
        json!(cfg.templates.mahalanobis_std_floor),
    );
    sampled_parameters.insert(
        "templates.scorer_epsilon".to_string(),
        json!(cfg.templates.scorer_epsilon),
    );
    sampled_parameters.insert(
        "templates.region_bounds".to_string(),
        json!(cfg.templates.region_bounds),
    );

    (cfg, sampled_parameters, warnings)
}

fn apply_feature_profile(cfg: &mut AppConfig, profile: FeatureProfile) {
    cfg.templates.include_harmonic_ratios = profile.include_harmonic_ratios;
    cfg.templates.include_centroid = profile.include_centroid;
    cfg.templates.include_rolloff = profile.include_rolloff;
    cfg.templates.include_flux = profile.include_flux;
    cfg.templates.include_attack_ratio = profile.include_attack_ratio;
    cfg.templates.include_noise_ratio = profile.include_noise_ratio;
    cfg.templates.include_inharmonicity = profile.include_inharmonicity;
}

fn evaluate_trial(
    prepared: &PreparedSoloDataset,
    cfg: &AppConfig,
    thresholds: &SoloStringEvalThresholds,
    include_examples: bool,
) -> Result<TrialEval> {
    let mut flattened = Vec::with_capacity(prepared.examples.len());
    let mut dim: Option<usize> = None;

    for (idx, example) in prepared.examples.iter().enumerate() {
        let z = flatten_note_features(&example.features, &cfg.templates);
        if z.is_empty() {
            continue;
        }
        match dim {
            Some(d) if d != z.len() => continue,
            None => dim = Some(z.len()),
            _ => {}
        }
        flattened.push((idx, z));
    }

    if flattened.is_empty() {
        bail!("no flattenable SOLO examples for current trial configuration");
    }

    let templates = train_templates_from_prepared(prepared, cfg, &flattened)?;

    let mut examples = Vec::new();
    let mut by_true_string: BTreeMap<u8, AccuracyCounter> = BTreeMap::new();
    let mut by_midi: BTreeMap<u8, AccuracyCounter> = BTreeMap::new();
    let mut by_region: BTreeMap<String, AccuracyCounter> = BTreeMap::new();
    let mut confusion_counts = vec![vec![0_u32; STRING_COUNT]; STRING_COUNT];

    let mut top1_hits = 0usize;
    let mut top2_hits = 0usize;
    let mut top3_hits = 0usize;
    let mut exact_hits = 0usize;
    let mut high_conf_wrong_count = 0usize;
    let mut low_margin_count = 0usize;
    let mut adjacent_confusion_count = 0usize;
    let mut wrong_count = 0usize;

    let mut margins_all = Vec::new();
    let mut margins_correct = Vec::new();
    let mut margins_wrong = Vec::new();

    for (flattened_idx, (prepared_idx, z)) in flattened.iter().enumerate() {
        let example = &prepared.examples[*prepared_idx];
        let candidates = score_string_fret_candidates(example.note.midi, z, &templates, cfg);
        if candidates.is_empty() {
            continue;
        }

        let top1 = candidates.first();
        let top2 = candidates.get(1);

        let truth_in_top1 = top1
            .map(|c| c.string == example.note.string)
            .unwrap_or(false);
        let truth_in_top2 = candidates
            .iter()
            .take(2)
            .any(|c| c.string == example.note.string);
        let truth_in_top3 = candidates
            .iter()
            .take(3)
            .any(|c| c.string == example.note.string);

        if truth_in_top1 {
            top1_hits += 1;
        } else {
            wrong_count += 1;
        }
        if truth_in_top2 {
            top2_hits += 1;
        }
        if truth_in_top3 {
            top3_hits += 1;
        }

        let exact_hit = top1
            .map(|c| c.string == example.note.string && c.fret == example.note.fret)
            .unwrap_or(false);
        if exact_hit {
            exact_hits += 1;
        }

        let margin = normalized_top1_top2_margin(&candidates, cfg.templates.scorer_epsilon);
        if let Some(m) = margin {
            margins_all.push(m);
            if truth_in_top1 {
                margins_correct.push(m);
            } else {
                margins_wrong.push(m);
            }
            if m <= thresholds.low_margin_threshold {
                low_margin_count += 1;
            }
        }

        let pred_top1 = top1.map(|c| c.string);
        let adjacent_confusion = pred_top1
            .map(|pred| pred != example.note.string && pred.abs_diff(example.note.string) == 1)
            .unwrap_or(false);

        if adjacent_confusion {
            adjacent_confusion_count += 1;
        }

        let high_conf_wrong = !truth_in_top1
            && margin
                .map(|m| m >= thresholds.high_confidence_margin_threshold)
                .unwrap_or(false);
        if high_conf_wrong {
            high_conf_wrong_count += 1;
        }

        let string_counter = by_true_string
            .entry(example.note.string)
            .or_insert(AccuracyCounter {
                correct: 0,
                total: 0,
            });
        string_counter.total += 1;
        if truth_in_top1 {
            string_counter.correct += 1;
        }

        let midi_counter = by_midi.entry(example.note.midi).or_insert(AccuracyCounter {
            correct: 0,
            total: 0,
        });
        midi_counter.total += 1;
        if truth_in_top1 {
            midi_counter.correct += 1;
        }

        let region = format_region_label(example.note.fret, &cfg.templates.region_bounds);
        let region_counter = by_region.entry(region).or_insert(AccuracyCounter {
            correct: 0,
            total: 0,
        });
        region_counter.total += 1;
        if truth_in_top1 {
            region_counter.correct += 1;
        }

        if let Some(pred) = top1 {
            let true_idx = example.note.string as usize;
            let pred_idx = pred.string as usize;
            if true_idx < STRING_COUNT && pred_idx < STRING_COUNT {
                confusion_counts[true_idx][pred_idx] += 1;
            }
        }

        if include_examples {
            let example_id = format!(
                "{}_{}",
                sanitize_for_id(&example.file_id),
                flattened_idx + 1
            );
            let ranked_candidate_positions = candidates
                .iter()
                .take(TOPK_STORE)
                .enumerate()
                .map(|(rank_idx, c)| RankedCandidate {
                    rank: rank_idx + 1,
                    string: c.string,
                    fret: c.fret,
                    score: c.combined_score,
                })
                .collect::<Vec<_>>();
            let ranked_candidate_strings = ranked_candidate_positions
                .iter()
                .map(|c| c.string)
                .collect::<Vec<_>>();

            examples.push(SoloStringExampleRanking {
                example_id,
                file_id: example.file_id.clone(),
                true_midi: example.note.midi,
                true_string: example.note.string,
                true_fret: example.note.fret,
                ranked_candidate_strings,
                ranked_candidate_positions,
                top1_score: top1.map(|c| c.combined_score),
                top2_score: top2.map(|c| c.combined_score),
                margin_top1_top2: margin,
                truth_in_top1,
                truth_in_top2,
                truth_in_top3,
                adjacent_string_confusion: adjacent_confusion,
                high_confidence_wrong: high_conf_wrong,
                low_margin_ambiguous: margin
                    .map(|m| m <= thresholds.low_margin_threshold)
                    .unwrap_or(false),
            });
        }
    }

    let evaluated = by_midi.values().map(|c| c.total).sum::<usize>();
    if evaluated == 0 {
        bail!("all SOLO examples were skipped during trial evaluation");
    }

    let confusion_matrix = build_confusion_matrix(confusion_counts);
    let exact_meaningful = cfg.templates.level == "string_fret";

    let metrics = SoloStringMetrics {
        dataset_scope: "Phase S1 SOLO-only (filename contains 'solo')".to_string(),
        records_total_before_filename_filter: prepared.total_before_filter,
        records_filtered_non_solo: prepared.filtered_non_solo,
        records_total_after_solo_filter: prepared.total_after_solo_filter,
        records_evaluated: evaluated,
        records_skipped: prepared
            .total_after_solo_filter
            .saturating_sub(evaluated)
            .saturating_add(prepared.skipped_feature_extraction),
        top1_string_accuracy: ratio(top1_hits, evaluated),
        top2_string_accuracy: ratio(top2_hits, evaluated),
        top3_string_accuracy: ratio(top3_hits, evaluated),
        exact_string_fret_meaningful: exact_meaningful,
        exact_string_fret_accuracy: if exact_meaningful {
            Some(ratio(exact_hits, evaluated))
        } else {
            None
        },
        accuracy_by_true_string: finalize_accuracy_map(by_true_string),
        accuracy_by_midi_pitch: finalize_accuracy_map(by_midi),
        accuracy_by_fret_region: finalize_accuracy_map(by_region),
        average_margin_top1_top2: mean(&margins_all),
        median_margin_top1_top2: median(&margins_all),
        mean_margin_correct: mean(&margins_correct),
        median_margin_correct: median(&margins_correct),
        mean_margin_wrong: mean(&margins_wrong),
        median_margin_wrong: median(&margins_wrong),
        high_confidence_margin_threshold: thresholds.high_confidence_margin_threshold,
        high_confidence_wrong_count: high_conf_wrong_count,
        high_confidence_wrong_rate: ratio(high_conf_wrong_count, evaluated),
        low_margin_threshold: thresholds.low_margin_threshold,
        low_margin_ambiguous_count: low_margin_count,
        low_margin_ambiguous_rate: ratio(low_margin_count, evaluated),
        adjacent_string_confusion_count: adjacent_confusion_count,
        adjacent_string_confusion_rate_overall: ratio(adjacent_confusion_count, evaluated),
        adjacent_string_confusion_rate_of_errors: ratio(adjacent_confusion_count, wrong_count),
    };

    let trial_metrics = SoloStringTrialMetrics {
        evaluated_examples: metrics.records_evaluated,
        skipped_examples: metrics.records_skipped,
        top1_string_accuracy: metrics.top1_string_accuracy,
        top2_string_accuracy: metrics.top2_string_accuracy,
        top3_string_accuracy: metrics.top3_string_accuracy,
        average_margin_top1_top2: metrics.average_margin_top1_top2,
        high_confidence_wrong_rate: metrics.high_confidence_wrong_rate,
        low_margin_ambiguous_rate: metrics.low_margin_ambiguous_rate,
        adjacent_string_confusion_rate_overall: metrics.adjacent_string_confusion_rate_overall,
        adjacent_string_confusion_rate_of_errors: metrics.adjacent_string_confusion_rate_of_errors,
    };

    Ok(TrialEval {
        trial_metrics,
        artifact: SoloStringArtifact {
            metrics,
            confusion_matrix,
            examples,
        },
        templates,
    })
}

fn train_templates_from_prepared(
    prepared: &PreparedSoloDataset,
    cfg: &AppConfig,
    flattened: &[(usize, Vec<f32>)],
) -> Result<GuitarTemplates> {
    let Some(dim) = flattened.first().map(|(_, z)| z.len()) else {
        bail!("cannot train templates from empty flattened feature set");
    };

    let region_count = cfg.templates.region_bounds.len().saturating_sub(1).max(1);
    let mut by_string_running = (0..STRING_COUNT)
        .map(|_| running_stats_new(dim))
        .collect::<Vec<_>>();
    let mut by_string_region_running: Vec<Vec<Option<_>>> = (0..STRING_COUNT)
        .map(|_| (0..region_count).map(|_| None).collect())
        .collect();

    for (prepared_idx, z) in flattened {
        let note = &prepared.examples[*prepared_idx].note;
        let s = note.string as usize;
        if s >= STRING_COUNT || z.len() != dim {
            continue;
        }
        running_stats_update(&mut by_string_running[s], z);

        let region = fret_region(note.fret, &cfg.templates.region_bounds).min(region_count - 1);
        let stats =
            by_string_region_running[s][region].get_or_insert_with(|| running_stats_new(dim));
        running_stats_update(stats, z);
    }

    let by_string = by_string_running
        .into_iter()
        .map(running_stats_finalize)
        .collect::<Vec<_>>();
    let by_string_region = by_string_region_running
        .into_iter()
        .map(|regions| {
            regions
                .into_iter()
                .map(|cell| cell.map(running_stats_finalize))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let fret_max = prepared
        .examples
        .iter()
        .map(|x| x.note.fret)
        .max()
        .unwrap_or(20)
        .max(20);

    let by_string_fret = (0..STRING_COUNT)
        .map(|_| {
            (0..=usize::from(fret_max))
                .map(|_| None)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Ok(GuitarTemplates {
        open_midi: OPEN_MIDI,
        fret_max,
        feature_names: crate::features::feature_names(&cfg.templates),
        by_string,
        by_string_region,
        by_string_fret,
    })
}

fn run_feature_ablation(
    prepared: &PreparedSoloDataset,
    base_cfg: &AppConfig,
    thresholds: &SoloStringEvalThresholds,
) -> Result<Vec<FeatureAblationRow>> {
    let mut rows = Vec::new();
    for profile in FEATURE_PROFILES {
        let mut cfg = base_cfg.clone();
        apply_feature_profile(&mut cfg, profile);
        let eval = evaluate_trial(prepared, &cfg, thresholds, false)?;
        rows.push(FeatureAblationRow {
            profile: profile.name.to_string(),
            top1_string_accuracy: eval.trial_metrics.top1_string_accuracy,
            top3_string_accuracy: eval.trial_metrics.top3_string_accuracy,
            average_margin_top1_top2: eval.trial_metrics.average_margin_top1_top2,
            high_confidence_wrong_rate: eval.trial_metrics.high_confidence_wrong_rate,
        });
    }

    rows.sort_by(|a, b| {
        b.top1_string_accuracy
            .total_cmp(&a.top1_string_accuracy)
            .then_with(|| b.top3_string_accuracy.total_cmp(&a.top3_string_accuracy))
    });
    Ok(rows)
}

fn compute_feature_separability(
    prepared: &PreparedSoloDataset,
    cfg: &AppConfig,
) -> Vec<FeatureSeparabilityRow> {
    let mut flattened = Vec::with_capacity(prepared.examples.len());
    let mut labels = Vec::with_capacity(prepared.examples.len());

    for example in &prepared.examples {
        let z = flatten_note_features(&example.features, &cfg.templates);
        if z.is_empty() {
            continue;
        }
        flattened.push(z);
        labels.push(example.note.string);
    }

    if flattened.is_empty() {
        return Vec::new();
    }

    let dim = flattened[0].len();
    let n = flattened.len();
    if n < 2 {
        return Vec::new();
    }

    let feature_names = crate::features::feature_names(&cfg.templates);
    let mut rows = Vec::new();

    for i in 0..dim {
        let mut global_sum = 0.0_f64;
        for v in &flattened {
            global_sum += v[i] as f64;
        }
        let global_mean = global_sum / n as f64;

        let mut by_class: BTreeMap<u8, Vec<f64>> = BTreeMap::new();
        for (idx, &label) in labels.iter().enumerate() {
            by_class
                .entry(label)
                .or_default()
                .push(flattened[idx][i] as f64);
        }

        let mut between = 0.0_f64;
        let mut within = 0.0_f64;
        for values in by_class.values() {
            if values.is_empty() {
                continue;
            }
            let count = values.len() as f64;
            let mean = values.iter().sum::<f64>() / count;
            between += count * (mean - global_mean) * (mean - global_mean);
            for value in values {
                let d = *value - mean;
                within += d * d;
            }
        }

        let score = (between / (within + 1e-12)).max(0.0) as f32;
        rows.push(FeatureSeparabilityRow {
            feature: feature_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("feature_{i}")),
            between_over_within: score,
        });
    }

    rows.sort_by(|a, b| b.between_over_within.total_cmp(&a.between_over_within));
    rows
}

fn normalized_top1_top2_margin(
    candidates: &[crate::types::CandidatePosition],
    eps: f32,
) -> Option<f32> {
    let top1 = candidates.first()?.combined_score;
    let top2 = candidates.get(1).map(|c| c.combined_score).unwrap_or(top1);

    let mut min_score = f32::INFINITY;
    let mut max_score = f32::NEG_INFINITY;
    for candidate in candidates {
        min_score = min_score.min(candidate.combined_score);
        max_score = max_score.max(candidate.combined_score);
    }

    if !min_score.is_finite() || !max_score.is_finite() {
        return None;
    }

    let denom = (max_score - min_score).abs().max(eps.max(1e-12));
    let n1 = (top1 - min_score) / denom;
    let n2 = (top2 - min_score) / denom;
    Some((n1 - n2).max(0.0))
}

fn build_confusion_matrix(counts: Vec<Vec<u32>>) -> SoloStringConfusionMatrix {
    let mut normalized = vec![vec![0.0_f32; STRING_COUNT]; STRING_COUNT];
    for (row_idx, row) in counts.iter().enumerate().take(STRING_COUNT) {
        let row_sum: u32 = row.iter().copied().sum();
        if row_sum == 0 {
            continue;
        }
        for col_idx in 0..STRING_COUNT {
            normalized[row_idx][col_idx] = row[col_idx] as f32 / row_sum as f32;
        }
    }

    SoloStringConfusionMatrix {
        labels: (0..STRING_COUNT).map(|x| x.to_string()).collect(),
        counts,
        normalized,
    }
}

fn finalize_accuracy_map<K: Ord>(
    counters: BTreeMap<K, AccuracyCounter>,
) -> BTreeMap<K, ClassAccuracy> {
    counters
        .into_iter()
        .map(|(key, counter)| {
            (
                key,
                ClassAccuracy {
                    correct: counter.correct,
                    total: counter.total,
                    accuracy: ratio(counter.correct, counter.total),
                },
            )
        })
        .collect()
}

fn format_region_label(fret: u8, bounds: &[u8]) -> String {
    if bounds.len() < 2 {
        return "all".to_string();
    }
    let idx = fret_region(fret, bounds).min(bounds.len() - 2);
    let lo = bounds[idx];
    let hi = bounds[idx + 1];
    format!("[{lo},{hi})")
}

fn file_id_for_path(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

fn sanitize_for_id(input: &str) -> String {
    input
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn render_phase_s1_html(
    trials: &[SoloStringTrialResult],
    metrics: Option<&SoloStringMetrics>,
    confusion: Option<&SoloStringConfusionMatrix>,
    examples: &[SoloStringExampleRanking],
    ablation: &[FeatureAblationRow],
    separability: &[FeatureSeparabilityRow],
    recommendations: Option<&SoloStringRecommendations>,
) -> String {
    let mut html = String::new();

    html.push_str(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Phase S1 SOLO String Detection Report</title>",
    );
    html.push_str(
        "<style>
        :root { --bg:#f7f4ec; --card:#ffffff; --ink:#19211c; --muted:#5b6a61; --line:#dbe2db; --accent:#136848; }
        body { margin:0; font-family:'Source Sans 3','Segoe UI',sans-serif; color:var(--ink); background:linear-gradient(180deg,#efe7d5 0%,#f7f4ec 45%,#ecf4ee 100%); }
        main { max-width:1240px; margin:0 auto; padding:24px; }
        h1,h2,h3 { margin:0 0 10px 0; font-family:'Space Grotesk','Segoe UI',sans-serif; }
        p { margin:6px 0; color:var(--muted); }
        .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; margin:12px 0 20px; }
        .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; box-shadow:0 5px 16px rgba(20,26,23,0.06); }
        .metric { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px dashed #e8ede8; }
        .metric:last-child { border-bottom:none; }
        table { width:100%; border-collapse:collapse; margin-top:8px; font-size:13px; }
        th,td { border:1px solid #dce4de; padding:6px; text-align:left; vertical-align:top; }
        th { background:#edf5ef; }
        tbody tr:nth-child(even) { background:#fafcfa; }
        .good { color:#136848; font-weight:700; }
        .bad { color:#8f2f2f; font-weight:700; }
        code { background:#eef2ee; border-radius:4px; padding:1px 4px; }
        </style></head><body><main>",
    );

    html.push_str("<h1>Phase S1 Report: Monophonic String Detection (SOLO only)</h1>");
    html.push_str("<p>Scope is strictly SOLO filename-filtered monophonic annotations with known MIDI pitch. Polyphonic/COMP evaluation is intentionally excluded.</p>");

    html.push_str("<div class=\"grid\">");
    html.push_str("<div class=\"card\"><h3>Artifacts</h3><p><a href=\"../trials.json\">trials.json</a></p><p><a href=\"../trials.csv\">trials.csv</a></p><p><a href=\"../best_config.toml\">best_config.toml</a></p><p><a href=\"../best_config_top1.toml\">best_config_top1.toml</a></p><p><a href=\"../best_config_balanced.toml\">best_config_balanced.toml</a></p><p><a href=\"../best_config_robust.toml\">best_config_robust.toml</a></p><p><a href=\"../string_metrics.json\">string_metrics.json</a></p><p><a href=\"../string_confusion_matrix.csv\">string_confusion_matrix.csv</a></p><p><a href=\"../example_rankings.csv\">example_rankings.csv</a></p></div>");

    if let Some(m) = metrics {
        html.push_str("<div class=\"card\"><h3>Summary Metrics</h3>");
        html.push_str(&format!(
            "<div class=\"metric\"><span>Records after SOLO filter</span><strong>{}</strong></div>",
            m.records_total_after_solo_filter
        ));
        html.push_str(&format!(
            "<div class=\"metric\"><span>Evaluated</span><strong>{}</strong></div>",
            m.records_evaluated
        ));
        html.push_str(&format!(
            "<div class=\"metric\"><span>Top-1 string accuracy</span><strong>{:.4}</strong></div>",
            m.top1_string_accuracy
        ));
        html.push_str(&format!(
            "<div class=\"metric\"><span>Top-2 string accuracy</span><strong>{:.4}</strong></div>",
            m.top2_string_accuracy
        ));
        html.push_str(&format!(
            "<div class=\"metric\"><span>Top-3 string accuracy</span><strong>{:.4}</strong></div>",
            m.top3_string_accuracy
        ));
        html.push_str(&format!(
            "<div class=\"metric\"><span>Avg margin top1-top2</span><strong>{:.4}</strong></div>",
            m.average_margin_top1_top2
        ));
        html.push_str(&format!("<div class=\"metric\"><span>High-confidence wrong rate</span><strong>{:.4}</strong></div>", m.high_confidence_wrong_rate));
        html.push_str(&format!("<div class=\"metric\"><span>Adjacent confusion rate (overall)</span><strong>{:.4}</strong></div>", m.adjacent_string_confusion_rate_overall));
        html.push_str("</div>");
    } else {
        html.push_str("<div class=\"card\"><h3>Summary Metrics</h3><p>Missing <code>string_metrics.json</code>.</p></div>");
    }

    html.push_str("<div class=\"card\"><h3>Optimization</h3>");
    html.push_str(&format!(
        "<div class=\"metric\"><span>Trials</span><strong>{}</strong></div>",
        trials.len()
    ));
    if let Some(best) = trials.first() {
        html.push_str(&format!(
            "<div class=\"metric\"><span>Best trial</span><strong>{}</strong></div>",
            best.trial_id
        ));
        html.push_str(&format!(
            "<div class=\"metric\"><span>Best objective</span><strong>{:.6}</strong></div>",
            best.objective
        ));
    }
    html.push_str("</div>");

    if let Some(rec) = recommendations {
        html.push_str("<div class=\"card\"><h3>Recommended Configs</h3>");
        html.push_str(&format!(
            "<div class=\"metric\"><span>Best for top-1</span><strong>trial {}</strong></div>",
            rec.top1_best_trial_id
        ));
        html.push_str(&format!("<div class=\"metric\"><span>Best balanced top1/top3</span><strong>trial {}</strong></div>", rec.balanced_best_trial_id));
        html.push_str(&format!(
            "<div class=\"metric\"><span>Best robust ranking</span><strong>trial {}</strong></div>",
            rec.robust_best_trial_id
        ));
        html.push_str("</div>");
    }

    html.push_str("</div>");

    html.push_str("<section class=\"card\"><h2>Top Trial Leaderboard</h2>");
    if trials.is_empty() {
        html.push_str("<p>No trials available.</p>");
    } else {
        html.push_str("<table><thead><tr><th>Trial</th><th>Objective</th><th>Top1</th><th>Top2</th><th>Top3</th><th>Avg margin</th><th>High-conf wrong</th><th>Feature profile</th><th>Level</th><th>Scorer</th></tr></thead><tbody>");
        for row in trials.iter().take(150) {
            html.push_str("<tr>");
            html.push_str(&format!("<td>{}</td>", row.trial_id));
            html.push_str(&format!("<td>{:.6}</td>", row.objective));
            html.push_str(&format!("<td>{:.4}</td>", row.metrics.top1_string_accuracy));
            html.push_str(&format!("<td>{:.4}</td>", row.metrics.top2_string_accuracy));
            html.push_str(&format!("<td>{:.4}</td>", row.metrics.top3_string_accuracy));
            html.push_str(&format!(
                "<td>{:.4}</td>",
                row.metrics.average_margin_top1_top2
            ));
            html.push_str(&format!(
                "<td>{:.4}</td>",
                row.metrics.high_confidence_wrong_rate
            ));
            html.push_str(&format!(
                "<td>{}</td>",
                escape_html(&param_as_string(&row.sampled_parameters, "feature_profile"))
            ));
            html.push_str(&format!(
                "<td>{}</td>",
                escape_html(&param_as_string(&row.sampled_parameters, "templates.level"))
            ));
            html.push_str(&format!(
                "<td>{}</td>",
                escape_html(&param_as_string(
                    &row.sampled_parameters,
                    "templates.score_type"
                ))
            ));
            html.push_str("</tr>");
        }
        html.push_str("</tbody></table>");
    }
    html.push_str("</section>");

    html.push_str("<section class=\"card\"><h2>Feature Diagnostics</h2>");
    if !ablation.is_empty() {
        html.push_str("<h3>Feature-group Ablation</h3>");
        html.push_str("<table><thead><tr><th>Profile</th><th>Top1</th><th>Top3</th><th>Avg margin</th><th>High-conf wrong</th></tr></thead><tbody>");
        for row in ablation {
            html.push_str("<tr>");
            html.push_str(&format!("<td>{}</td>", escape_html(&row.profile)));
            html.push_str(&format!("<td>{:.4}</td>", row.top1_string_accuracy));
            html.push_str(&format!("<td>{:.4}</td>", row.top3_string_accuracy));
            html.push_str(&format!("<td>{:.4}</td>", row.average_margin_top1_top2));
            html.push_str(&format!("<td>{:.4}</td>", row.high_confidence_wrong_rate));
            html.push_str("</tr>");
        }
        html.push_str("</tbody></table>");
    } else {
        html.push_str("<p>Missing feature ablation results.</p>");
    }

    if !separability.is_empty() {
        html.push_str("<h3>Per-feature Separability (between/within variance)</h3>");
        html.push_str("<table><thead><tr><th>Feature</th><th>Score</th></tr></thead><tbody>");
        for row in separability.iter().take(40) {
            html.push_str("<tr>");
            html.push_str(&format!("<td>{}</td>", escape_html(&row.feature)));
            html.push_str(&format!("<td>{:.6}</td>", row.between_over_within));
            html.push_str("</tr>");
        }
        html.push_str("</tbody></table>");
    } else {
        html.push_str("<p>Missing feature separability results.</p>");
    }
    html.push_str("</section>");

    html.push_str(
        "<section class=\"card\"><h2>Confusion Matrix (True string vs Predicted string)</h2>",
    );
    if let Some(matrix) = confusion {
        html.push_str("<table><thead><tr><th>True\\Pred</th>");
        for label in &matrix.labels {
            html.push_str(&format!("<th>{}</th>", escape_html(label)));
        }
        html.push_str("</tr></thead><tbody>");
        for (row_idx, row_counts) in matrix.counts.iter().enumerate() {
            html.push_str(&format!("<tr><th>{}</th>", row_idx));
            let norm_row = matrix.normalized.get(row_idx).cloned().unwrap_or_default();
            for (col_idx, &count) in row_counts.iter().enumerate() {
                let normalized = norm_row.get(col_idx).copied().unwrap_or(0.0);
                let intensity = (normalized.clamp(0.0, 1.0) * 0.75 + 0.08) as f64;
                html.push_str(&format!(
                    "<td style=\"background:rgba(19,104,72,{:.3});\"><span>{}</span><br><small>{:.3}</small></td>",
                    intensity, count, normalized
                ));
            }
            html.push_str("</tr>");
        }
        html.push_str("</tbody></table>");
    } else {
        html.push_str("<p>Missing <code>string_confusion_matrix.json</code>.</p>");
    }
    html.push_str("</section>");

    html.push_str("<section class=\"card\"><h2>Accuracy Breakdowns</h2>");
    if let Some(m) = metrics {
        render_accuracy_table_u8(
            &mut html,
            "Accuracy by true string",
            &m.accuracy_by_true_string,
        );
        render_accuracy_table_u8(
            &mut html,
            "Accuracy by MIDI pitch",
            &m.accuracy_by_midi_pitch,
        );
        render_accuracy_table_string(
            &mut html,
            "Accuracy by fret region",
            &m.accuracy_by_fret_region,
        );

        html.push_str("<h3>Margin and Confidence Summary</h3>");
        html.push_str(&format!(
            "<p>Average/median margin: {:.4} / {:.4}</p>",
            m.average_margin_top1_top2, m.median_margin_top1_top2
        ));
        html.push_str(&format!(
            "<p>Correct mean/median margin: {:.4} / {:.4}</p>",
            m.mean_margin_correct, m.median_margin_correct
        ));
        html.push_str(&format!(
            "<p>Wrong mean/median margin: {:.4} / {:.4}</p>",
            m.mean_margin_wrong, m.median_margin_wrong
        ));
        html.push_str(&format!(
            "<p>High-confidence wrongs: {} ({:.4}) at threshold {:.3}</p>",
            m.high_confidence_wrong_count,
            m.high_confidence_wrong_rate,
            m.high_confidence_margin_threshold
        ));
        html.push_str(&format!(
            "<p>Low-margin ambiguous: {} ({:.4}) at threshold {:.3}</p>",
            m.low_margin_ambiguous_count, m.low_margin_ambiguous_rate, m.low_margin_threshold
        ));
        html.push_str(&format!(
            "<p>Adjacent-string confusion count: {} (overall {:.4}, of errors {:.4})</p>",
            m.adjacent_string_confusion_count,
            m.adjacent_string_confusion_rate_overall,
            m.adjacent_string_confusion_rate_of_errors
        ));
    } else {
        html.push_str("<p>Missing metrics.</p>");
    }
    html.push_str("</section>");

    render_example_highlights(&mut html, examples);

    html.push_str("<section class=\"card\"><h2>Example Rankings</h2>");
    if examples.is_empty() {
        html.push_str("<p>No example rankings found.</p>");
    } else {
        html.push_str("<p>Showing up to 300 examples. Full export is available in <a href=\"../example_rankings.csv\">example_rankings.csv</a>.</p>");
        html.push_str("<table><thead><tr><th>Example</th><th>File</th><th>True (MIDI,S,F)</th><th>Top1 score</th><th>Top2 score</th><th>Margin</th><th>GT@1</th><th>GT@2</th><th>GT@3</th><th>Adjacent confusion</th></tr></thead><tbody>");
        for row in examples.iter().take(300) {
            html.push_str("<tr>");
            html.push_str(&format!("<td>{}</td>", escape_html(&row.example_id)));
            html.push_str(&format!("<td>{}</td>", escape_html(&row.file_id)));
            html.push_str(&format!(
                "<td>({}, {}, {})</td>",
                row.true_midi, row.true_string, row.true_fret
            ));
            html.push_str(&format!("<td>{}</td>", format_opt_float(row.top1_score)));
            html.push_str(&format!("<td>{}</td>", format_opt_float(row.top2_score)));
            html.push_str(&format!(
                "<td>{}</td>",
                format_opt_float(row.margin_top1_top2)
            ));
            html.push_str(&format!("<td>{}</td>", bool_badge(row.truth_in_top1)));
            html.push_str(&format!("<td>{}</td>", bool_badge(row.truth_in_top2)));
            html.push_str(&format!("<td>{}</td>", bool_badge(row.truth_in_top3)));
            html.push_str(&format!(
                "<td>{}</td>",
                bool_badge(row.adjacent_string_confusion)
            ));
            html.push_str("</tr>");
        }
        html.push_str("</tbody></table>");
    }
    html.push_str("</section>");

    html.push_str("</main></body></html>");
    html
}

fn render_accuracy_table_u8(html: &mut String, title: &str, map: &BTreeMap<u8, ClassAccuracy>) {
    html.push_str(&format!("<h3>{}</h3>", escape_html(title)));
    if map.is_empty() {
        html.push_str("<p>None.</p>");
        return;
    }
    html.push_str("<table><thead><tr><th>Class</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr></thead><tbody>");
    for (class, acc) in map {
        html.push_str("<tr>");
        html.push_str(&format!("<td>{}</td>", class));
        html.push_str(&format!("<td>{}</td>", acc.correct));
        html.push_str(&format!("<td>{}</td>", acc.total));
        html.push_str(&format!("<td>{:.4}</td>", acc.accuracy));
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table>");
}

fn render_accuracy_table_string(
    html: &mut String,
    title: &str,
    map: &BTreeMap<String, ClassAccuracy>,
) {
    html.push_str(&format!("<h3>{}</h3>", escape_html(title)));
    if map.is_empty() {
        html.push_str("<p>None.</p>");
        return;
    }
    html.push_str("<table><thead><tr><th>Class</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr></thead><tbody>");
    for (class, acc) in map {
        html.push_str("<tr>");
        html.push_str(&format!("<td>{}</td>", escape_html(class)));
        html.push_str(&format!("<td>{}</td>", acc.correct));
        html.push_str(&format!("<td>{}</td>", acc.total));
        html.push_str(&format!("<td>{:.4}</td>", acc.accuracy));
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table>");
}

fn render_example_highlights(html: &mut String, examples: &[SoloStringExampleRanking]) {
    html.push_str("<section class=\"card\"><h2>Representative Cases</h2>");
    if examples.is_empty() {
        html.push_str("<p>No examples available.</p></section>");
        return;
    }

    let mut highest_conf_correct = examples
        .iter()
        .filter(|x| x.truth_in_top1)
        .collect::<Vec<_>>();
    highest_conf_correct.sort_by(|a, b| {
        b.margin_top1_top2
            .unwrap_or(f32::NEG_INFINITY)
            .total_cmp(&a.margin_top1_top2.unwrap_or(f32::NEG_INFINITY))
    });

    let mut highest_conf_wrong = examples
        .iter()
        .filter(|x| !x.truth_in_top1)
        .collect::<Vec<_>>();
    highest_conf_wrong.sort_by(|a, b| {
        b.margin_top1_top2
            .unwrap_or(f32::NEG_INFINITY)
            .total_cmp(&a.margin_top1_top2.unwrap_or(f32::NEG_INFINITY))
    });

    let mut ambiguous = examples
        .iter()
        .filter(|x| x.low_margin_ambiguous)
        .collect::<Vec<_>>();
    ambiguous.sort_by(|a, b| {
        a.margin_top1_top2
            .unwrap_or(f32::INFINITY)
            .total_cmp(&b.margin_top1_top2.unwrap_or(f32::INFINITY))
    });

    let mut adjacent = examples
        .iter()
        .filter(|x| x.adjacent_string_confusion)
        .collect::<Vec<_>>();
    adjacent.sort_by(|a, b| {
        b.margin_top1_top2
            .unwrap_or(f32::NEG_INFINITY)
            .total_cmp(&a.margin_top1_top2.unwrap_or(f32::NEG_INFINITY))
    });

    render_highlight_table(html, "Highest-confidence correct", &highest_conf_correct);
    render_highlight_table(html, "Highest-confidence wrong", &highest_conf_wrong);
    render_highlight_table(html, "Ambiguous low-margin", &ambiguous);
    render_highlight_table(html, "Adjacent-string confusions", &adjacent);

    html.push_str("</section>");
}

fn render_highlight_table(html: &mut String, title: &str, rows: &[&SoloStringExampleRanking]) {
    html.push_str(&format!("<h3>{}</h3>", escape_html(title)));
    if rows.is_empty() {
        html.push_str("<p>None.</p>");
        return;
    }
    html.push_str("<table><thead><tr><th>Example</th><th>File</th><th>True S</th><th>Pred S</th><th>Margin</th><th>GT@1</th></tr></thead><tbody>");
    for row in rows.iter().take(MAX_HIGHLIGHTS) {
        let pred = row
            .ranked_candidate_strings
            .first()
            .map(|x| x.to_string())
            .unwrap_or_else(|| "NA".to_string());
        html.push_str("<tr>");
        html.push_str(&format!("<td>{}</td>", escape_html(&row.example_id)));
        html.push_str(&format!("<td>{}</td>", escape_html(&row.file_id)));
        html.push_str(&format!("<td>{}</td>", row.true_string));
        html.push_str(&format!("<td>{}</td>", pred));
        html.push_str(&format!(
            "<td>{}</td>",
            format_opt_float(row.margin_top1_top2)
        ));
        html.push_str(&format!("<td>{}</td>", bool_badge(row.truth_in_top1)));
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table>");
}

fn format_opt_float(v: Option<f32>) -> String {
    v.map(|x| format!("{x:.4}"))
        .unwrap_or_else(|| "NA".to_string())
}

fn bool_badge(v: bool) -> String {
    if v {
        "<span class=\"good\">true</span>".to_string()
    } else {
        "<span class=\"bad\">false</span>".to_string()
    }
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

fn write_trials_csv(path: &Path, rows: &[SoloStringTrialResult]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create trials csv: {}", path.display()))?;

    writeln!(
        file,
        "trial_id,objective,templates.level,templates.score_type,feature_profile,templates.feature_normalization,templates.mahalanobis_std_floor,templates.scorer_epsilon,top1,top2,top3,average_margin_top1_top2,high_confidence_wrong_rate,low_margin_ambiguous_rate,adjacent_string_confusion_rate_overall,adjacent_string_confusion_rate_of_errors,evaluated_examples,skipped_examples,warnings"
    )?;

    for row in rows {
        writeln!(
            file,
            "{},{:.6},{},{},{},{},{:.8},{:.8},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{}",
            row.trial_id,
            row.objective,
            csv_escape(&param_as_string(&row.sampled_parameters, "templates.level")),
            csv_escape(&param_as_string(&row.sampled_parameters, "templates.score_type")),
            csv_escape(&param_as_string(&row.sampled_parameters, "feature_profile")),
            csv_escape(&param_as_string(&row.sampled_parameters, "templates.feature_normalization")),
            param_as_f32(&row.sampled_parameters, "templates.mahalanobis_std_floor"),
            param_as_f32(&row.sampled_parameters, "templates.scorer_epsilon"),
            row.metrics.top1_string_accuracy,
            row.metrics.top2_string_accuracy,
            row.metrics.top3_string_accuracy,
            row.metrics.average_margin_top1_top2,
            row.metrics.high_confidence_wrong_rate,
            row.metrics.low_margin_ambiguous_rate,
            row.metrics.adjacent_string_confusion_rate_overall,
            row.metrics.adjacent_string_confusion_rate_of_errors,
            row.metrics.evaluated_examples,
            row.metrics.skipped_examples,
            csv_escape(&row.warnings.join(" | ")),
        )?;
    }

    Ok(())
}

fn param_as_f32(params: &BTreeMap<String, Value>, key: &str) -> f32 {
    params
        .get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.0)
}

fn write_confusion_csv(path: &Path, matrix: &SoloStringConfusionMatrix) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create confusion csv: {}", path.display()))?;
    writeln!(file, "true_string,pred_string,count,normalized")?;

    for (true_idx, counts_row) in matrix.counts.iter().enumerate() {
        let norm_row = matrix.normalized.get(true_idx).cloned().unwrap_or_default();
        for (pred_idx, &count) in counts_row.iter().enumerate() {
            let normalized = norm_row.get(pred_idx).copied().unwrap_or(0.0);
            writeln!(
                file,
                "{},{},{},{:.6}",
                true_idx, pred_idx, count, normalized
            )?;
        }
    }

    Ok(())
}

fn write_examples_csv(path: &Path, rows: &[SoloStringExampleRanking]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create examples csv: {}", path.display()))?;
    writeln!(
        file,
        "example_id,file_id,true_midi,true_string,true_fret,ranked_candidate_strings,ranked_candidate_positions,top1_score,top2_score,margin_top1_top2,truth_in_top1,truth_in_top2,truth_in_top3,adjacent_string_confusion,high_confidence_wrong,low_margin_ambiguous"
    )?;

    for row in rows {
        let strings = row
            .ranked_candidate_strings
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let positions = row
            .ranked_candidate_positions
            .iter()
            .map(|p| format!("({},{},{:.4})", p.string, p.fret, p.score))
            .collect::<Vec<_>>()
            .join(" ");

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            csv_escape(&row.example_id),
            csv_escape(&row.file_id),
            row.true_midi,
            row.true_string,
            row.true_fret,
            csv_escape(&strings),
            csv_escape(&positions),
            format_opt_float(row.top1_score),
            format_opt_float(row.top2_score),
            format_opt_float(row.margin_top1_top2),
            row.truth_in_top1,
            row.truth_in_top2,
            row.truth_in_top3,
            row.adjacent_string_confusion,
            row.high_confidence_wrong,
            row.low_margin_ambiguous,
        )?;
    }

    Ok(())
}

fn write_feature_ablation_csv(path: &Path, rows: &[FeatureAblationRow]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create feature ablation csv: {}", path.display()))?;
    writeln!(
        file,
        "profile,top1_string_accuracy,top3_string_accuracy,average_margin_top1_top2,high_confidence_wrong_rate"
    )?;

    for row in rows {
        writeln!(
            file,
            "{},{:.6},{:.6},{:.6},{:.6}",
            csv_escape(&row.profile),
            row.top1_string_accuracy,
            row.top3_string_accuracy,
            row.average_margin_top1_top2,
            row.high_confidence_wrong_rate,
        )?;
    }

    Ok(())
}

fn write_feature_separability_csv(path: &Path, rows: &[FeatureSeparabilityRow]) -> Result<()> {
    let mut file = File::create(path).with_context(|| {
        format!(
            "failed to create feature separability csv: {}",
            path.display()
        )
    })?;
    writeln!(file, "feature,between_over_within")?;

    for row in rows {
        writeln!(
            file,
            "{},{:.6}",
            csv_escape(&row.feature),
            row.between_over_within
        )?;
    }

    Ok(())
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(value)
        .with_context(|| format!("failed to serialize json: {}", path.display()))?;
    std::fs::write(path, json)
        .with_context(|| format!("failed to write json file: {}", path.display()))?;
    Ok(())
}

fn write_config_toml(path: &Path, cfg: &AppConfig) -> Result<()> {
    let toml = toml::to_string_pretty(cfg).context("failed to serialize config to toml")?;
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

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut tmp = values.to_vec();
    tmp.sort_by(|a, b| a.total_cmp(b));
    let mid = tmp.len() / 2;
    if tmp.len() % 2 == 0 {
        (tmp[mid - 1] + tmp[mid]) * 0.5
    } else {
        tmp[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::{
        balanced_score, median, normalized_top1_top2_margin, objective_s1, robust_score,
        SoloStringObjectiveWeights, SoloStringTrialMetrics,
    };
    use crate::types::CandidatePosition;

    #[test]
    fn objective_rewards_top1_and_penalizes_wrong_confidence() {
        let w = SoloStringObjectiveWeights::default();
        let strong = SoloStringTrialMetrics {
            evaluated_examples: 100,
            skipped_examples: 0,
            top1_string_accuracy: 0.75,
            top2_string_accuracy: 0.90,
            top3_string_accuracy: 0.98,
            average_margin_top1_top2: 0.30,
            high_confidence_wrong_rate: 0.04,
            low_margin_ambiguous_rate: 0.10,
            adjacent_string_confusion_rate_overall: 0.05,
            adjacent_string_confusion_rate_of_errors: 0.20,
        };
        let weak = SoloStringTrialMetrics {
            top1_string_accuracy: 0.62,
            high_confidence_wrong_rate: 0.16,
            ..strong.clone()
        };

        assert!(objective_s1(&strong, &w) > objective_s1(&weak, &w));
        assert!(balanced_score(&strong) > balanced_score(&weak));
        assert!(robust_score(&strong) > robust_score(&weak));
    }

    #[test]
    fn normalized_margin_is_zero_when_two_best_equal() {
        let cands = vec![
            CandidatePosition {
                midi: 64,
                string: 5,
                fret: 0,
                pitch_score: 0.0,
                template_score: 0.0,
                pitch_score_norm: 0.0,
                template_score_norm: 0.0,
                combined_score: 1.0,
            },
            CandidatePosition {
                midi: 64,
                string: 4,
                fret: 5,
                pitch_score: 0.0,
                template_score: 0.0,
                pitch_score_norm: 0.0,
                template_score_norm: 0.0,
                combined_score: 1.0,
            },
        ];
        let margin = normalized_top1_top2_margin(&cands, 1e-6).unwrap_or(1.0);
        assert!(margin.abs() < 1e-6);
    }

    #[test]
    fn median_handles_even_and_odd() {
        assert!((median(&[1.0, 3.0, 2.0]) - 2.0).abs() < 1e-6);
        assert!((median(&[1.0, 2.0, 4.0, 10.0]) - 3.0).abs() < 1e-6);
    }
}
