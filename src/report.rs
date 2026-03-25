use std::collections::{BTreeMap, BTreeSet};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::audio::{load_wav_mono, trim_audio_region};
use crate::config::AppConfig;
use crate::features::{extract_note_features_for_record, flatten_note_features};
use crate::fretboard::fret_region;
use crate::full_eval::{FullAggregateMetrics, FullExampleDiagnostic, FullFailureCases};
use crate::harmonic_map::build_harmonic_maps;
use crate::infer::run_pitch_only_inference;
use crate::masp::MaspValidationSummary;
use crate::metrics::evaluate_string_predictions;
use crate::stft::build_stft_plan;
use crate::templates::{score_string_fret_candidates, GuitarTemplates};
use crate::types::{AudioBuffer, MonoAnnotation};

pub const DEFAULT_DEBUG_DIR: &str = "output/debug";

const NOTE_CONTEXT_MARGIN_SEC: f32 = 0.005;
const TOPK_LIMIT: usize = 3;
const STRING_COUNT: usize = 6;
const MAX_HIGHLIGHTS: usize = 10;
const MAX_EXAMPLE_ROWS_IN_HTML: usize = 300;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchAggregateMetrics {
    pub interpretation: String,
    pub total_records: usize,
    pub evaluated_records: usize,
    pub skipped_records: usize,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchPerFileDiagnostic {
    pub file_id: String,
    pub evaluated_events: usize,
    pub skipped_events: usize,
    pub ground_truth_pitches: String,
    pub predicted_pitches: String,
    pub matched_notes: usize,
    pub missed_notes: usize,
    pub false_positive_notes: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchPrediction {
    pub midi: u8,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchExampleDiagnostic {
    pub example_id: String,
    pub file_id: String,
    pub onset_sec: f32,
    pub offset_sec: f32,
    pub representative_frame_index: usize,
    pub representative_frame_time_sec: f32,
    pub ground_truth_pitches: Vec<u8>,
    pub predicted_pitches: Vec<PitchPrediction>,
    pub matched_pitches: Vec<u8>,
    pub missed_pitches: Vec<u8>,
    pub false_positive_pitches: Vec<u8>,
    pub octave_error: bool,
    pub sub_octave_error: bool,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchDiagnosticsArtifact {
    pub metrics: PitchAggregateMetrics,
    pub per_file: Vec<PitchPerFileDiagnostic>,
    pub examples: Vec<PitchExampleDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase1AbComparisonArtifact {
    pub baseline_frontend: String,
    pub variant_frontend: String,
    pub baseline_metrics: PitchAggregateMetrics,
    pub variant_metrics: PitchAggregateMetrics,
    pub delta_precision: f32,
    pub delta_recall: f32,
    pub delta_f1: f32,
    pub delta_false_positive_rate: f32,
    pub delta_avg_predicted_pitches_per_event: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassAccuracy {
    pub correct: usize,
    pub total: usize,
    pub accuracy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonoAggregateMetrics {
    pub total_records: usize,
    pub evaluated_records: usize,
    pub skipped_records: usize,
    pub top1_string_accuracy: f32,
    pub top3_string_accuracy: f32,
    pub exact_string_fret_meaningful: bool,
    pub exact_string_fret_accuracy: Option<f32>,
    pub accuracy_by_true_string: BTreeMap<u8, ClassAccuracy>,
    pub accuracy_by_midi_pitch: BTreeMap<u8, ClassAccuracy>,
    pub accuracy_by_fret_region: BTreeMap<String, ClassAccuracy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedCandidate {
    pub rank: usize,
    pub string: u8,
    pub fret: u8,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonoExampleDiagnostic {
    pub example_id: String,
    pub file_id: String,
    pub onset_sec: f32,
    pub offset_sec: f32,
    pub true_midi: u8,
    pub true_string: u8,
    pub true_fret: u8,
    pub predicted_top1_string: Option<u8>,
    pub predicted_top1_fret: Option<u8>,
    pub top1_score: Option<f32>,
    pub top2_candidate: Option<RankedCandidate>,
    pub top3_candidate: Option<RankedCandidate>,
    pub candidates_topk: Vec<RankedCandidate>,
    pub ground_truth_in_top1: bool,
    pub ground_truth_in_top3: bool,
    pub margin_top1_top2: Option<f32>,
    pub exact_string_fret_hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonoPerFileDiagnostic {
    pub file_id: String,
    pub evaluated_examples: usize,
    pub skipped_examples: usize,
    pub top1_accuracy: f32,
    pub top3_accuracy: f32,
    pub exact_string_fret_accuracy: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub labels: Vec<String>,
    pub counts: Vec<Vec<u32>>,
    pub normalized: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub feature_names: Vec<String>,
    pub count: usize,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleHighlight {
    pub example_id: String,
    pub file_id: String,
    pub true_midi: u8,
    pub true_string: u8,
    pub predicted_top1_string: Option<u8>,
    pub top1_score: Option<f32>,
    pub margin_top1_top2: Option<f32>,
    pub ground_truth_in_top1: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHighlights {
    pub highest_confidence_correct: Vec<ExampleHighlight>,
    pub highest_confidence_wrong: Vec<ExampleHighlight>,
    pub smallest_margin_examples: Vec<ExampleHighlight>,
    pub largest_margin_wrong_examples: Vec<ExampleHighlight>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonoDiagnosticsArtifact {
    pub metrics: MonoAggregateMetrics,
    pub confusion_matrix: ConfusionMatrix,
    pub per_file: Vec<MonoPerFileDiagnostic>,
    pub examples: Vec<MonoExampleDiagnostic>,
    pub feature_stats: FeatureStats,
    pub highlights: ErrorHighlights,
}

#[derive(Debug, Clone)]
struct PitchFileAccumulator {
    evaluated_events: usize,
    skipped_events: usize,
    gt_hist: BTreeMap<u8, usize>,
    pred_hist: BTreeMap<u8, usize>,
    matched_notes: usize,
    missed_notes: usize,
    false_positive_notes: usize,
}

#[derive(Debug, Clone)]
struct MonoFileAccumulator {
    evaluated_examples: usize,
    skipped_examples: usize,
    top1_hits: usize,
    top3_hits: usize,
    exact_hits: usize,
}

#[derive(Debug, Clone)]
struct AccuracyCounter {
    correct: usize,
    total: usize,
}

#[derive(Debug, Clone)]
struct FeatureStatsAccumulator {
    count: usize,
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    min: Vec<f32>,
    max: Vec<f32>,
}

impl FeatureStatsAccumulator {
    fn new(dim: usize) -> Self {
        Self {
            count: 0,
            sum: vec![0.0; dim],
            sum_sq: vec![0.0; dim],
            min: vec![f32::INFINITY; dim],
            max: vec![f32::NEG_INFINITY; dim],
        }
    }

    fn update(&mut self, x: &[f32]) {
        if x.len() != self.sum.len() {
            return;
        }
        self.count += 1;
        for (i, &value) in x.iter().enumerate() {
            let value_f64 = value as f64;
            self.sum[i] += value_f64;
            self.sum_sq[i] += value_f64 * value_f64;
            self.min[i] = self.min[i].min(value);
            self.max[i] = self.max[i].max(value);
        }
    }

    fn finalize(self, feature_names: Vec<String>) -> FeatureStats {
        let dim = self.sum.len();
        if self.count == 0 {
            return FeatureStats {
                feature_names,
                count: 0,
                mean: vec![0.0; dim],
                std: vec![0.0; dim],
                min: vec![0.0; dim],
                max: vec![0.0; dim],
            };
        }

        let mut mean = vec![0.0_f32; dim];
        let mut std = vec![0.0_f32; dim];
        let n = self.count as f64;

        for i in 0..dim {
            let m = self.sum[i] / n;
            let var = ((self.sum_sq[i] / n) - (m * m)).max(0.0);
            mean[i] = m as f32;
            std[i] = var.sqrt() as f32;
        }

        FeatureStats {
            feature_names,
            count: self.count,
            mean,
            std,
            min: self.min,
            max: self.max,
        }
    }
}

pub fn evaluate_pitch_dataset(
    dataset: &[MonoAnnotation],
    cfg: &AppConfig,
) -> Result<PitchDiagnosticsArtifact> {
    let mut sorted = dataset.to_vec();
    sorted.sort_by(|a, b| {
        a.audio_path
            .cmp(&b.audio_path)
            .then_with(|| a.onset_sec.total_cmp(&b.onset_sec))
            .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
            .then_with(|| a.midi.cmp(&b.midi))
    });

    let mut grouped: BTreeMap<String, Vec<MonoAnnotation>> = BTreeMap::new();
    for record in sorted {
        grouped
            .entry(record.audio_path.clone())
            .or_default()
            .push(record);
    }

    let mut examples = Vec::new();
    let mut per_file_acc: BTreeMap<String, PitchFileAccumulator> = BTreeMap::new();

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_count = 0usize;
    let mut octave_error_count = 0usize;
    let mut sub_octave_error_count = 0usize;
    let mut predicted_pitch_count = 0usize;

    for (audio_path, notes) in grouped {
        let file_id = file_id_for_path(&audio_path);
        let mut file_acc = PitchFileAccumulator {
            evaluated_events: 0,
            skipped_events: 0,
            gt_hist: BTreeMap::new(),
            pred_hist: BTreeMap::new(),
            matched_notes: 0,
            missed_notes: 0,
            false_positive_notes: 0,
        };

        let audio = load_wav_mono(&audio_path).with_context(|| {
            format!("failed to load audio during pitch diagnostics: {audio_path}")
        })?;

        if audio.sample_rate != cfg.sample_rate_target {
            file_acc.skipped_events += notes.len();
            per_file_acc.insert(file_id, file_acc);
            continue;
        }

        for (idx, note) in notes.iter().enumerate() {
            let Some(mut diag) = evaluate_pitch_event(&audio, note, cfg, &file_id, idx + 1) else {
                file_acc.skipped_events += 1;
                continue;
            };

            file_acc.evaluated_events += 1;
            *file_acc.gt_hist.entry(note.midi).or_insert(0) += 1;
            for prediction in &diag.predicted_pitches {
                *file_acc.pred_hist.entry(prediction.midi).or_insert(0) += 1;
            }

            tp += diag.matched_pitches.len();
            fp += diag.false_positive_pitches.len();
            fn_count += diag.missed_pitches.len();
            predicted_pitch_count += diag.predicted_pitches.len();

            if diag.octave_error {
                octave_error_count += 1;
            }
            if diag.sub_octave_error {
                sub_octave_error_count += 1;
            }

            file_acc.matched_notes += diag.matched_pitches.len();
            file_acc.missed_notes += diag.missed_pitches.len();
            file_acc.false_positive_notes += diag.false_positive_pitches.len();

            diag.example_id = format!("{}_{}", sanitize_for_id(&file_id), idx + 1);
            examples.push(diag);
        }

        per_file_acc.insert(file_id, file_acc);
    }

    let precision = ratio(tp, tp + fp);
    let recall = ratio(tp, tp + fn_count);
    let f1_score = f1(precision, recall);

    let evaluated_records = examples.len();
    let skipped_records = dataset.len().saturating_sub(evaluated_records);
    let false_positive_rate = ratio(fp, tp + fp);

    let metrics = PitchAggregateMetrics {
        interpretation: "Event-based: one representative frame (nearest note midpoint) is evaluated per annotation event.".to_string(),
        total_records: dataset.len(),
        evaluated_records,
        skipped_records,
        true_positive_count: tp,
        false_positive_count: fp,
        false_negative_count: fn_count,
        precision,
        recall,
        f1: f1_score,
        false_positive_rate,
        avg_predicted_pitches_per_event: ratio(predicted_pitch_count, evaluated_records),
        octave_error_count,
        octave_error_rate: ratio(octave_error_count, evaluated_records),
        sub_octave_error_count,
        sub_octave_error_rate: ratio(sub_octave_error_count, evaluated_records),
    };

    let mut per_file = Vec::with_capacity(per_file_acc.len());
    for (file_id, acc) in per_file_acc {
        let p = ratio(
            acc.matched_notes,
            acc.matched_notes + acc.false_positive_notes,
        );
        let r = ratio(acc.matched_notes, acc.matched_notes + acc.missed_notes);
        per_file.push(PitchPerFileDiagnostic {
            file_id,
            evaluated_events: acc.evaluated_events,
            skipped_events: acc.skipped_events,
            ground_truth_pitches: midi_histogram_to_string(&acc.gt_hist),
            predicted_pitches: midi_histogram_to_string(&acc.pred_hist),
            matched_notes: acc.matched_notes,
            missed_notes: acc.missed_notes,
            false_positive_notes: acc.false_positive_notes,
            precision: p,
            recall: r,
            f1: f1(p, r),
        });
    }

    Ok(PitchDiagnosticsArtifact {
        metrics,
        per_file,
        examples,
    })
}

pub fn evaluate_mono_dataset(
    dataset: &[MonoAnnotation],
    templates: &GuitarTemplates,
    cfg: &AppConfig,
) -> Result<MonoDiagnosticsArtifact> {
    let stft_plan = build_stft_plan(&cfg.frontend);
    let midi_min = dataset
        .iter()
        .map(|x| x.midi)
        .min()
        .unwrap_or(cfg.pitch.midi_min)
        .min(cfg.pitch.midi_min);
    let midi_max = dataset
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

    let mut sorted = dataset.to_vec();
    sorted.sort_by(|a, b| {
        a.audio_path
            .cmp(&b.audio_path)
            .then_with(|| a.onset_sec.total_cmp(&b.onset_sec))
            .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
            .then_with(|| a.midi.cmp(&b.midi))
    });

    let mut grouped: BTreeMap<String, Vec<MonoAnnotation>> = BTreeMap::new();
    for record in sorted {
        grouped
            .entry(record.audio_path.clone())
            .or_default()
            .push(record);
    }

    let mut per_file_acc: BTreeMap<String, MonoFileAccumulator> = BTreeMap::new();
    let mut examples = Vec::new();
    let mut ranked_candidates = Vec::new();
    let mut ground_truth = Vec::new();
    let mut by_true_string: BTreeMap<u8, AccuracyCounter> = BTreeMap::new();
    let mut by_midi_pitch: BTreeMap<u8, AccuracyCounter> = BTreeMap::new();
    let mut by_fret_region: BTreeMap<String, AccuracyCounter> = BTreeMap::new();

    let mut confusion_counts = vec![vec![0_u32; STRING_COUNT]; STRING_COUNT];

    let feature_dim = templates.feature_names.len();
    let mut feature_stats_acc = FeatureStatsAccumulator::new(feature_dim);

    for (audio_path, notes) in grouped {
        let file_id = file_id_for_path(&audio_path);
        let mut file_acc = MonoFileAccumulator {
            evaluated_examples: 0,
            skipped_examples: 0,
            top1_hits: 0,
            top3_hits: 0,
            exact_hits: 0,
        };

        let audio = load_wav_mono(&audio_path).with_context(|| {
            format!("failed to load audio during mono diagnostics: {audio_path}")
        })?;

        if audio.sample_rate != cfg.sample_rate_target {
            file_acc.skipped_examples += notes.len();
            per_file_acc.insert(file_id, file_acc);
            continue;
        }

        for (idx, note) in notes.iter().enumerate() {
            let Some(feat) =
                extract_note_features_for_record(note, &audio, &stft_plan, cfg, &maps)?
            else {
                file_acc.skipped_examples += 1;
                continue;
            };

            let z = flatten_note_features(&feat, &cfg.templates);
            if z.len() != feature_dim {
                file_acc.skipped_examples += 1;
                continue;
            }

            feature_stats_acc.update(&z);

            let candidates = score_string_fret_candidates(note.midi, &z, templates, cfg);
            if candidates.is_empty() {
                file_acc.skipped_examples += 1;
                continue;
            }

            let top1 = candidates.first();
            let top2 = candidates.get(1);
            let top3 = candidates.get(2);

            let gt_in_top1 = top1.map(|c| c.string == note.string).unwrap_or(false);
            let gt_in_top3 = candidates.iter().take(3).any(|c| c.string == note.string);
            let exact_hit = top1
                .map(|c| c.string == note.string && c.fret == note.fret)
                .unwrap_or(false);

            file_acc.evaluated_examples += 1;
            if gt_in_top1 {
                file_acc.top1_hits += 1;
            }
            if gt_in_top3 {
                file_acc.top3_hits += 1;
            }
            if exact_hit {
                file_acc.exact_hits += 1;
            }

            let string_counter = by_true_string
                .entry(note.string)
                .or_insert(AccuracyCounter {
                    correct: 0,
                    total: 0,
                });
            string_counter.total += 1;
            if gt_in_top1 {
                string_counter.correct += 1;
            }

            let midi_counter = by_midi_pitch.entry(note.midi).or_insert(AccuracyCounter {
                correct: 0,
                total: 0,
            });
            midi_counter.total += 1;
            if gt_in_top1 {
                midi_counter.correct += 1;
            }

            let region_label = format_region_label(note.fret, &cfg.templates.region_bounds);
            let region_counter = by_fret_region
                .entry(region_label)
                .or_insert(AccuracyCounter {
                    correct: 0,
                    total: 0,
                });
            region_counter.total += 1;
            if gt_in_top1 {
                region_counter.correct += 1;
            }

            if let Some(pred) = top1 {
                let true_idx = note.string as usize;
                let pred_idx = pred.string as usize;
                if true_idx < STRING_COUNT && pred_idx < STRING_COUNT {
                    confusion_counts[true_idx][pred_idx] += 1;
                }
            }

            let example_id = format!("{}_{}", sanitize_for_id(&file_id), idx + 1);
            let margin = top1
                .zip(top2)
                .map(|(a, b)| a.combined_score - b.combined_score);
            let diag = MonoExampleDiagnostic {
                example_id,
                file_id: file_id.clone(),
                onset_sec: note.onset_sec,
                offset_sec: note.offset_sec,
                true_midi: note.midi,
                true_string: note.string,
                true_fret: note.fret,
                predicted_top1_string: top1.map(|c| c.string),
                predicted_top1_fret: top1.map(|c| c.fret),
                top1_score: top1.map(|c| c.combined_score),
                top2_candidate: top2.map(|c| RankedCandidate {
                    rank: 2,
                    string: c.string,
                    fret: c.fret,
                    score: c.combined_score,
                }),
                top3_candidate: top3.map(|c| RankedCandidate {
                    rank: 3,
                    string: c.string,
                    fret: c.fret,
                    score: c.combined_score,
                }),
                candidates_topk: candidates
                    .iter()
                    .take(TOPK_LIMIT)
                    .enumerate()
                    .map(|(rank_idx, c)| RankedCandidate {
                        rank: rank_idx + 1,
                        string: c.string,
                        fret: c.fret,
                        score: c.combined_score,
                    })
                    .collect(),
                ground_truth_in_top1: gt_in_top1,
                ground_truth_in_top3: gt_in_top3,
                margin_top1_top2: margin,
                exact_string_fret_hit: exact_hit,
            };

            ranked_candidates.push(candidates);
            ground_truth.push(note.clone());
            examples.push(diag);
        }

        per_file_acc.insert(file_id, file_acc);
    }

    let exact_meaningful = cfg.templates.level == "string_fret";
    let metric_report =
        evaluate_string_predictions(&ranked_candidates, &ground_truth, exact_meaningful);

    let mut per_file = Vec::with_capacity(per_file_acc.len());
    for (file_id, acc) in per_file_acc {
        per_file.push(MonoPerFileDiagnostic {
            file_id,
            evaluated_examples: acc.evaluated_examples,
            skipped_examples: acc.skipped_examples,
            top1_accuracy: ratio(acc.top1_hits, acc.evaluated_examples),
            top3_accuracy: ratio(acc.top3_hits, acc.evaluated_examples),
            exact_string_fret_accuracy: if exact_meaningful {
                Some(ratio(acc.exact_hits, acc.evaluated_examples))
            } else {
                None
            },
        });
    }

    let confusion_matrix = build_confusion_matrix(confusion_counts);

    let metrics = MonoAggregateMetrics {
        total_records: dataset.len(),
        evaluated_records: examples.len(),
        skipped_records: dataset.len().saturating_sub(examples.len()),
        top1_string_accuracy: metric_report.top1,
        top3_string_accuracy: metric_report.top3,
        exact_string_fret_meaningful: exact_meaningful,
        exact_string_fret_accuracy: if exact_meaningful {
            Some(metric_report.exact_string_fret)
        } else {
            None
        },
        accuracy_by_true_string: finalize_accuracy_map(by_true_string),
        accuracy_by_midi_pitch: finalize_accuracy_map(by_midi_pitch),
        accuracy_by_fret_region: finalize_accuracy_map(by_fret_region),
    };

    let feature_stats = feature_stats_acc.finalize(templates.feature_names.clone());
    let highlights = build_error_highlights(&examples);

    Ok(MonoDiagnosticsArtifact {
        metrics,
        confusion_matrix,
        per_file,
        examples,
        feature_stats,
        highlights,
    })
}

pub fn write_pitch_artifacts(debug_dir: &Path, artifact: &PitchDiagnosticsArtifact) -> Result<()> {
    create_dir_all(debug_dir)
        .with_context(|| format!("failed to create debug directory: {}", debug_dir.display()))?;

    write_json_pretty(
        &debug_dir.join("phase1_pitch_metrics.json"),
        &artifact.metrics,
    )?;
    write_phase1_per_file_csv(&debug_dir.join("phase1_per_file.csv"), &artifact.per_file)?;

    let examples_dir = debug_dir.join("phase1_pitch_scores");
    create_dir_all(&examples_dir).with_context(|| {
        format!(
            "failed to create phase1 example directory: {}",
            examples_dir.display()
        )
    })?;
    for example in &artifact.examples {
        let path = examples_dir.join(format!("{}.json", example.example_id));
        write_json_pretty(&path, example)?;
    }

    Ok(())
}

pub fn write_pitch_artifacts_variant(
    debug_dir: &Path,
    artifact: &PitchDiagnosticsArtifact,
    suffix: &str,
) -> Result<()> {
    create_dir_all(debug_dir)
        .with_context(|| format!("failed to create debug directory: {}", debug_dir.display()))?;

    let suffix = sanitize_artifact_suffix(suffix);
    write_json_pretty(
        &debug_dir.join(format!("phase1_pitch_metrics_{suffix}.json")),
        &artifact.metrics,
    )?;
    write_phase1_per_file_csv(
        &debug_dir.join(format!("phase1_per_file_{suffix}.csv")),
        &artifact.per_file,
    )?;

    let examples_dir = debug_dir.join(format!("phase1_pitch_scores_{suffix}"));
    create_dir_all(&examples_dir).with_context(|| {
        format!(
            "failed to create phase1 variant example directory: {}",
            examples_dir.display()
        )
    })?;
    for example in &artifact.examples {
        let path = examples_dir.join(format!("{}.json", example.example_id));
        write_json_pretty(&path, example)?;
    }

    Ok(())
}

pub fn build_phase1_ab_comparison(
    baseline_frontend: &str,
    baseline: &PitchDiagnosticsArtifact,
    variant_frontend: &str,
    variant: &PitchDiagnosticsArtifact,
) -> Phase1AbComparisonArtifact {
    Phase1AbComparisonArtifact {
        baseline_frontend: baseline_frontend.to_string(),
        variant_frontend: variant_frontend.to_string(),
        baseline_metrics: baseline.metrics.clone(),
        variant_metrics: variant.metrics.clone(),
        delta_precision: variant.metrics.precision - baseline.metrics.precision,
        delta_recall: variant.metrics.recall - baseline.metrics.recall,
        delta_f1: variant.metrics.f1 - baseline.metrics.f1,
        delta_false_positive_rate: variant.metrics.false_positive_rate
            - baseline.metrics.false_positive_rate,
        delta_avg_predicted_pitches_per_event: variant.metrics.avg_predicted_pitches_per_event
            - baseline.metrics.avg_predicted_pitches_per_event,
    }
}

pub fn write_phase1_ab_comparison(
    debug_dir: &Path,
    comparison: &Phase1AbComparisonArtifact,
) -> Result<()> {
    create_dir_all(debug_dir)
        .with_context(|| format!("failed to create debug directory: {}", debug_dir.display()))?;
    write_json_pretty(&debug_dir.join("phase1_ab_comparison.json"), comparison)
}

pub fn write_mono_artifacts(debug_dir: &Path, artifact: &MonoDiagnosticsArtifact) -> Result<()> {
    create_dir_all(debug_dir)
        .with_context(|| format!("failed to create debug directory: {}", debug_dir.display()))?;

    write_json_pretty(
        &debug_dir.join("phase2_string_metrics.json"),
        &artifact.metrics,
    )?;
    write_json_pretty(
        &debug_dir.join("phase2_confusion_matrix.json"),
        &artifact.confusion_matrix,
    )?;
    write_phase2_confusion_csv(
        &debug_dir.join("phase2_confusion_matrix.csv"),
        &artifact.confusion_matrix,
    )?;
    write_json_pretty(
        &debug_dir.join("phase2_feature_stats.json"),
        &artifact.feature_stats,
    )?;
    write_json_pretty(
        &debug_dir.join("phase2_examples_topk.json"),
        &artifact.examples,
    )?;
    write_phase2_examples_csv(
        &debug_dir.join("phase2_examples_topk.csv"),
        &artifact.examples,
    )?;
    write_phase2_per_file_csv(&debug_dir.join("phase2_per_file.csv"), &artifact.per_file)?;
    write_json_pretty(
        &debug_dir.join("phase2_error_highlights.json"),
        &artifact.highlights,
    )?;

    let examples_dir = debug_dir.join("phase2_mono_scores");
    create_dir_all(&examples_dir).with_context(|| {
        format!(
            "failed to create phase2 example directory: {}",
            examples_dir.display()
        )
    })?;

    for example in &artifact.examples {
        let path = examples_dir.join(format!("{}.json", example.example_id));
        write_json_pretty(&path, example)?;
    }

    Ok(())
}

pub fn build_html_report(debug_dir: &Path) -> Result<PathBuf> {
    let report_dir = debug_dir.join("report");
    create_dir_all(&report_dir).with_context(|| {
        format!(
            "failed to create report directory: {}",
            report_dir.display()
        )
    })?;

    let pitch_metrics: Option<PitchAggregateMetrics> =
        read_json_optional(&debug_dir.join("phase1_pitch_metrics.json"))?;
    let phase1_ab: Option<Phase1AbComparisonArtifact> =
        read_json_optional(&debug_dir.join("phase1_ab_comparison.json"))?;
    let mono_metrics: Option<MonoAggregateMetrics> =
        read_json_optional(&debug_dir.join("phase2_string_metrics.json"))?;
    let confusion_matrix: Option<ConfusionMatrix> =
        read_json_optional(&debug_dir.join("phase2_confusion_matrix.json"))?;
    let topk_examples: Option<Vec<MonoExampleDiagnostic>> =
        read_json_optional(&debug_dir.join("phase2_examples_topk.json"))?;
    let highlights: Option<ErrorHighlights> =
        read_json_optional(&debug_dir.join("phase2_error_highlights.json"))?;
    let full_metrics: Option<FullAggregateMetrics> =
        read_json_optional(&debug_dir.join("phase3_full_metrics.json"))?;
    let full_examples: Option<Vec<FullExampleDiagnostic>> =
        read_json_optional(&debug_dir.join("phase3_full_examples.json"))?;
    let full_failures: Option<FullFailureCases> =
        read_json_optional(&debug_dir.join("phase3_failure_cases.json"))?;
    let masp_summary: Option<MaspValidationSummary> =
        read_json_optional(&debug_dir.join("masp_validation_summary.json"))?;

    let html = render_report_html(
        pitch_metrics.as_ref(),
        phase1_ab.as_ref(),
        mono_metrics.as_ref(),
        confusion_matrix.as_ref(),
        topk_examples.as_deref().unwrap_or(&[]),
        highlights.as_ref(),
        full_metrics.as_ref(),
        full_examples.as_deref().unwrap_or(&[]),
        full_failures.as_ref(),
        masp_summary.as_ref(),
    );

    let index_path = report_dir.join("index.html");
    std::fs::write(&index_path, html)
        .with_context(|| format!("failed to write report html: {}", index_path.display()))?;

    Ok(index_path)
}

pub fn try_open_in_browser(path: &Path) {
    let target = path.to_string_lossy().to_string();

    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", &target])
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(&target).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(&target).spawn();
    }
}

fn evaluate_pitch_event(
    audio: &AudioBuffer,
    note: &MonoAnnotation,
    cfg: &AppConfig,
    file_id: &str,
    index: usize,
) -> Option<PitchExampleDiagnostic> {
    let start_sec = (note.onset_sec - NOTE_CONTEXT_MARGIN_SEC).max(0.0);
    let end_sec = (note.offset_sec + NOTE_CONTEXT_MARGIN_SEC).max(start_sec);

    let mut segment = trim_audio_region(&audio.samples, audio.sample_rate, start_sec, end_sec);
    if segment.is_empty() {
        return None;
    }

    let win_length = cfg.frontend.win_length;
    if segment.len() < win_length {
        segment.resize(win_length, 0.0);
    }

    let segment_audio = AudioBuffer {
        sample_rate: audio.sample_rate,
        samples: segment,
    };

    let frames = run_pitch_only_inference(&segment_audio, cfg);
    if frames.is_empty() {
        return None;
    }

    let target_time = ((note.onset_sec + note.offset_sec) * 0.5 - start_sec).max(0.0);
    let (frame_index, frame) = frames.iter().enumerate().min_by(|(_, a), (_, b)| {
        (a.time_sec - target_time)
            .abs()
            .total_cmp(&(b.time_sec - target_time).abs())
    })?;

    let mut gt_set = BTreeSet::new();
    gt_set.insert(note.midi);

    let pred_notes = frame
        .notes
        .iter()
        .map(|candidate| PitchPrediction {
            midi: candidate.midi,
            score: candidate.score,
        })
        .collect::<Vec<_>>();

    let pred_set = pred_notes.iter().map(|x| x.midi).collect::<BTreeSet<_>>();

    let matched = set_intersection(&gt_set, &pred_set);
    let missed = set_difference(&gt_set, &pred_set);
    let false_positive = set_difference(&pred_set, &gt_set);

    let precision = ratio(matched.len(), matched.len() + false_positive.len());
    let recall = ratio(matched.len(), matched.len() + missed.len());

    let octave_above = note.midi.saturating_add(12);
    let octave_below = note.midi.saturating_sub(12);
    let octave_error = matched.is_empty()
        && (pred_set.contains(&octave_above)
            || (note.midi >= 12 && pred_set.contains(&octave_below)));
    let sub_octave_error =
        matched.is_empty() && note.midi >= 12 && pred_set.contains(&octave_below);

    Some(PitchExampleDiagnostic {
        example_id: format!("{}_{}", sanitize_for_id(file_id), index),
        file_id: file_id.to_string(),
        onset_sec: note.onset_sec,
        offset_sec: note.offset_sec,
        representative_frame_index: frame_index,
        representative_frame_time_sec: frame.time_sec,
        ground_truth_pitches: gt_set.into_iter().collect(),
        predicted_pitches: pred_notes,
        matched_pitches: matched,
        missed_pitches: missed,
        false_positive_pitches: false_positive,
        octave_error,
        sub_octave_error,
        precision,
        recall,
        f1: f1(precision, recall),
    })
}

fn write_phase1_per_file_csv(path: &Path, rows: &[PitchPerFileDiagnostic]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create phase1 per-file csv: {}", path.display()))?;
    writeln!(
        file,
        "file_id,evaluated_events,skipped_events,ground_truth_pitches,predicted_pitches,matched_notes,missed_notes,false_positive_notes,precision,recall,f1"
    )?;

    for row in rows {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6}",
            csv_escape(&row.file_id),
            row.evaluated_events,
            row.skipped_events,
            csv_escape(&row.ground_truth_pitches),
            csv_escape(&row.predicted_pitches),
            row.matched_notes,
            row.missed_notes,
            row.false_positive_notes,
            row.precision,
            row.recall,
            row.f1
        )?;
    }

    Ok(())
}

fn write_phase2_per_file_csv(path: &Path, rows: &[MonoPerFileDiagnostic]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create phase2 per-file csv: {}", path.display()))?;
    writeln!(
        file,
        "file_id,evaluated_examples,skipped_examples,top1_accuracy,top3_accuracy,exact_string_fret_accuracy"
    )?;

    for row in rows {
        let exact = row
            .exact_string_fret_accuracy
            .map(|x| format!("{x:.6}"))
            .unwrap_or_else(|| "NA".to_string());
        writeln!(
            file,
            "{},{},{},{:.6},{:.6},{}",
            csv_escape(&row.file_id),
            row.evaluated_examples,
            row.skipped_examples,
            row.top1_accuracy,
            row.top3_accuracy,
            exact
        )?;
    }

    Ok(())
}

fn write_phase2_examples_csv(path: &Path, rows: &[MonoExampleDiagnostic]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create phase2 examples csv: {}", path.display()))?;

    writeln!(
        file,
        "example_id,file_id,true_midi,true_string,true_fret,pred_top1_string,pred_top1_fret,top1_score,pred_top2_string,pred_top2_fret,top2_score,pred_top3_string,pred_top3_fret,top3_score,gt_in_top1,gt_in_top3,margin_top1_top2,example_json"
    )?;

    for row in rows {
        let top2 = row.top2_candidate.as_ref();
        let top3 = row.top3_candidate.as_ref();

        let pred_top1_string = row
            .predicted_top1_string
            .map(|v| v.to_string())
            .unwrap_or_else(|| "NA".to_string());
        let pred_top1_fret = row
            .predicted_top1_fret
            .map(|v| v.to_string())
            .unwrap_or_else(|| "NA".to_string());
        let top1_score = row
            .top1_score
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "NA".to_string());

        let pred_top2_string = top2
            .map(|c| c.string.to_string())
            .unwrap_or_else(|| "NA".to_string());
        let pred_top2_fret = top2
            .map(|c| c.fret.to_string())
            .unwrap_or_else(|| "NA".to_string());
        let top2_score = top2
            .map(|c| format!("{:.6}", c.score))
            .unwrap_or_else(|| "NA".to_string());

        let pred_top3_string = top3
            .map(|c| c.string.to_string())
            .unwrap_or_else(|| "NA".to_string());
        let pred_top3_fret = top3
            .map(|c| c.fret.to_string())
            .unwrap_or_else(|| "NA".to_string());
        let top3_score = top3
            .map(|c| format!("{:.6}", c.score))
            .unwrap_or_else(|| "NA".to_string());

        let margin = row
            .margin_top1_top2
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "NA".to_string());

        let json_path = format!("phase2_mono_scores/{}.json", row.example_id);

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            csv_escape(&row.example_id),
            csv_escape(&row.file_id),
            row.true_midi,
            row.true_string,
            row.true_fret,
            pred_top1_string,
            pred_top1_fret,
            top1_score,
            pred_top2_string,
            pred_top2_fret,
            top2_score,
            pred_top3_string,
            pred_top3_fret,
            top3_score,
            row.ground_truth_in_top1,
            row.ground_truth_in_top3,
            margin,
            csv_escape(&json_path)
        )?;
    }

    Ok(())
}

fn write_phase2_confusion_csv(path: &Path, matrix: &ConfusionMatrix) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create phase2 confusion csv: {}", path.display()))?;

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

fn build_confusion_matrix(counts: Vec<Vec<u32>>) -> ConfusionMatrix {
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

    ConfusionMatrix {
        labels: (0..STRING_COUNT).map(|x| x.to_string()).collect(),
        counts,
        normalized,
    }
}

fn build_error_highlights(examples: &[MonoExampleDiagnostic]) -> ErrorHighlights {
    let mut correct = examples
        .iter()
        .filter(|x| x.ground_truth_in_top1)
        .cloned()
        .collect::<Vec<_>>();
    correct.sort_by(|a, b| {
        b.top1_score
            .unwrap_or(f32::NEG_INFINITY)
            .total_cmp(&a.top1_score.unwrap_or(f32::NEG_INFINITY))
    });

    let mut wrong = examples
        .iter()
        .filter(|x| !x.ground_truth_in_top1)
        .cloned()
        .collect::<Vec<_>>();
    wrong.sort_by(|a, b| {
        b.top1_score
            .unwrap_or(f32::NEG_INFINITY)
            .total_cmp(&a.top1_score.unwrap_or(f32::NEG_INFINITY))
    });

    let mut smallest_margin = examples
        .iter()
        .filter(|x| x.margin_top1_top2.is_some())
        .cloned()
        .collect::<Vec<_>>();
    smallest_margin.sort_by(|a, b| {
        a.margin_top1_top2
            .unwrap_or(f32::INFINITY)
            .total_cmp(&b.margin_top1_top2.unwrap_or(f32::INFINITY))
    });

    let mut largest_margin_wrong = examples
        .iter()
        .filter(|x| !x.ground_truth_in_top1 && x.margin_top1_top2.is_some())
        .cloned()
        .collect::<Vec<_>>();
    largest_margin_wrong.sort_by(|a, b| {
        b.margin_top1_top2
            .unwrap_or(f32::NEG_INFINITY)
            .total_cmp(&a.margin_top1_top2.unwrap_or(f32::NEG_INFINITY))
    });

    ErrorHighlights {
        highest_confidence_correct: correct
            .into_iter()
            .take(MAX_HIGHLIGHTS)
            .map(to_highlight)
            .collect(),
        highest_confidence_wrong: wrong
            .into_iter()
            .take(MAX_HIGHLIGHTS)
            .map(to_highlight)
            .collect(),
        smallest_margin_examples: smallest_margin
            .into_iter()
            .take(MAX_HIGHLIGHTS)
            .map(to_highlight)
            .collect(),
        largest_margin_wrong_examples: largest_margin_wrong
            .into_iter()
            .take(MAX_HIGHLIGHTS)
            .map(to_highlight)
            .collect(),
    }
}

fn to_highlight(example: MonoExampleDiagnostic) -> ExampleHighlight {
    ExampleHighlight {
        example_id: example.example_id,
        file_id: example.file_id,
        true_midi: example.true_midi,
        true_string: example.true_string,
        predicted_top1_string: example.predicted_top1_string,
        top1_score: example.top1_score,
        margin_top1_top2: example.margin_top1_top2,
        ground_truth_in_top1: example.ground_truth_in_top1,
    }
}

fn render_report_html(
    pitch_metrics: Option<&PitchAggregateMetrics>,
    phase1_ab: Option<&Phase1AbComparisonArtifact>,
    mono_metrics: Option<&MonoAggregateMetrics>,
    confusion: Option<&ConfusionMatrix>,
    examples: &[MonoExampleDiagnostic],
    highlights: Option<&ErrorHighlights>,
    full_metrics: Option<&FullAggregateMetrics>,
    full_examples: &[FullExampleDiagnostic],
    full_failures: Option<&FullFailureCases>,
    masp_summary: Option<&MaspValidationSummary>,
) -> String {
    let mut html = String::new();

    html.push_str(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Diagnostics Report (Phase 1-3)</title>",
    );
    html.push_str(
        "<style>
        :root { --bg:#f6f5f0; --card:#ffffff; --ink:#1c221f; --muted:#5f6b64; --accent:#1f6f49; --border:#d8ddd9; }
        body { margin:0; font-family: 'Source Sans 3','Segoe UI',sans-serif; background: radial-gradient(circle at top left, #efe9da 0%, #f6f5f0 38%, #eef3ed 100%); color:var(--ink); }
        main { max-width: 1200px; margin: 0 auto; padding: 24px; }
        h1,h2,h3 { margin: 0 0 10px 0; font-family: 'Space Grotesk','Segoe UI',sans-serif; }
        p { margin: 6px 0; color: var(--muted); }
        .grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); gap:14px; margin:12px 0 20px; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px; box-shadow: 0 5px 16px rgba(19,27,22,0.06); }
        .metric { display:flex; justify-content:space-between; border-bottom:1px dashed #e8ece8; padding:4px 0; }
        .metric:last-child { border-bottom:none; }
        table { width:100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
        th, td { border: 1px solid #dbe1dc; padding: 6px; text-align: left; vertical-align: top; }
        th { background:#edf3ee; }
        tbody tr:nth-child(even) { background:#f9fbf9; }
        .good { color:#1f6f49; font-weight:600; }
        .bad { color:#9b2c2c; font-weight:600; }
        code { background: #eef2ee; border-radius: 4px; padding: 1px 4px; }
        details { margin-top: 10px; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
        </style></head><body><main>",
    );

    html.push_str("<h1>Evaluation Report (Phase 1-3)</h1>");
    html.push_str(
        "<p>Artifacts are generated by the Rust CLI only. No external plotting tools are required.</p>",
    );

    html.push_str("<div class=\"grid\">");
    html.push_str(
        "<div class=\"card\"><h3>Artifacts</h3><p><a href=\"../phase1_pitch_metrics.json\">phase1_pitch_metrics.json</a></p><p><a href=\"../phase1_ab_comparison.json\">phase1_ab_comparison.json</a></p><p><a href=\"../phase1_pitch_metrics_qdft.json\">phase1_pitch_metrics_qdft.json</a></p><p><a href=\"../phase2_string_metrics.json\">phase2_string_metrics.json</a></p><p><a href=\"../phase2_confusion_matrix.csv\">phase2_confusion_matrix.csv</a></p><p><a href=\"../phase2_examples_topk.csv\">phase2_examples_topk.csv</a></p><p><a href=\"../phase3_full_metrics.json\">phase3_full_metrics.json</a></p><p><a href=\"../phase3_examples_topk.csv\">phase3_examples_topk.csv</a></p><p><a href=\"../masp_validation_summary.json\">masp_validation_summary.json</a></p><p><a href=\"../masp_validation_results.jsonl\">masp_validation_results.jsonl</a></p><p><a href=\"../tuning/report/index.html\">tuning/report/index.html</a></p></div>",
    );

    match pitch_metrics {
        Some(metrics) => {
            html.push_str("<div class=\"card\"><h3>Pitch Summary (Phase 1)</h3>");
            html.push_str(&format!(
                "<div class=\"metric\"><span>Total records</span><strong>{}</strong></div>",
                metrics.total_records
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Evaluated events</span><strong>{}</strong></div>",
                metrics.evaluated_records
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Precision</span><strong>{:.4}</strong></div>",
                metrics.precision
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Recall</span><strong>{:.4}</strong></div>",
                metrics.recall
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>F1</span><strong>{:.4}</strong></div>",
                metrics.f1
            ));
            html.push_str("</div>");
        }
        None => {
            html.push_str(
                "<div class=\"card\"><h3>Pitch Summary (Phase 1)</h3><p>Missing artifact <code>phase1_pitch_metrics.json</code>.</p></div>",
            );
        }
    }

    if let Some(ab) = phase1_ab {
        html.push_str("<div class=\"card\"><h3>Phase 1 A/B (STFT vs QDFT)</h3>");
        html.push_str(&format!(
            "<div class=\"metric\"><span>{} F1</span><strong>{:.4}</strong></div>",
            escape_html(&ab.baseline_frontend),
            ab.baseline_metrics.f1
        ));
        html.push_str(&format!(
            "<div class=\"metric\"><span>{} F1</span><strong>{:.4}</strong></div>",
            escape_html(&ab.variant_frontend),
            ab.variant_metrics.f1
        ));
        html.push_str(&format!(
            "<div class=\"metric\"><span>Delta F1 ({})</span><strong>{:+.4}</strong></div>",
            escape_html(&ab.variant_frontend),
            ab.delta_f1
        ));
        html.push_str(&format!(
            "<div class=\"metric\"><span>Delta precision</span><strong>{:+.4}</strong></div>",
            ab.delta_precision
        ));
        html.push_str(&format!(
            "<div class=\"metric\"><span>Delta recall</span><strong>{:+.4}</strong></div>",
            ab.delta_recall
        ));
        html.push_str("</div>");
    }

    match mono_metrics {
        Some(metrics) => {
            html.push_str("<div class=\"card\"><h3>Mono String Summary (Phase 2)</h3>");
            html.push_str(&format!(
                "<div class=\"metric\"><span>Total records</span><strong>{}</strong></div>",
                metrics.total_records
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Evaluated examples</span><strong>{}</strong></div>",
                metrics.evaluated_records
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Top-1 accuracy</span><strong>{:.4}</strong></div>",
                metrics.top1_string_accuracy
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Top-3 accuracy</span><strong>{:.4}</strong></div>",
                metrics.top3_string_accuracy
            ));
            if let Some(exact) = metrics.exact_string_fret_accuracy {
                html.push_str(&format!(
                    "<div class=\"metric\"><span>Exact string+fret</span><strong>{:.4}</strong></div>",
                    exact
                ));
            } else {
                html.push_str("<div class=\"metric\"><span>Exact string+fret</span><strong>not meaningful at current template level</strong></div>");
            }
            html.push_str("</div>");
        }
        None => {
            html.push_str(
                "<div class=\"card\"><h3>Mono String Summary (Phase 2)</h3><p>Missing artifact <code>phase2_string_metrics.json</code>.</p></div>",
            );
        }
    }

    match full_metrics {
        Some(metrics) => {
            html.push_str("<div class=\"card\"><h3>Poly Full Summary (Phase 3)</h3>");
            html.push_str(&format!(
                "<div class=\"metric\"><span>Total events</span><strong>{}</strong></div>",
                metrics.total_events
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Evaluated events</span><strong>{}</strong></div>",
                metrics.evaluated_events
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Pitch F1</span><strong>{:.4}</strong></div>",
                metrics.pitch.f1
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Pitch+String F1</span><strong>{:.4}</strong></div>",
                metrics.pitch_string.f1
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Pitch+String+Fret F1</span><strong>{:.4}</strong></div>",
                metrics.pitch_string_fret.f1
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Exact configuration</span><strong>{:.4}</strong></div>",
                metrics.exact_configuration_accuracy
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Decode hit@1 / @3 / @5</span><strong>{:.4} / {:.4} / {:.4}</strong></div>",
                metrics.decode_top1_hit_rate, metrics.decode_top3_hit_rate, metrics.decode_top5_hit_rate
            ));
            html.push_str("</div>");
        }
        None => {
            html.push_str(
                "<div class=\"card\"><h3>Poly Full Summary (Phase 3)</h3><p>Missing artifact <code>phase3_full_metrics.json</code>.</p></div>",
            );
        }
    }

    match masp_summary {
        Some(summary) => {
            html.push_str("<div class=\"card\"><h3>MASP Validation</h3>");
            html.push_str(&format!(
                "<div class=\"metric\"><span>Total windows</span><strong>{}</strong></div>",
                summary.total
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Passed</span><strong>{}</strong></div>",
                summary.passed
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Failed</span><strong>{}</strong></div>",
                summary.failed
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>Pass rate</span><strong>{:.4}</strong></div>",
                summary.pass_rate
            ));
            html.push_str(&format!(
                "<div class=\"metric\"><span>False accept / reject</span><strong>{} / {}</strong></div>",
                summary.false_accept_count, summary.false_reject_count
            ));
            if let Some(acc) = summary.accuracy {
                html.push_str(&format!(
                    "<div class=\"metric\"><span>Labeled accuracy</span><strong>{:.4}</strong></div>",
                    acc
                ));
            }
            html.push_str("</div>");
        }
        None => {
            html.push_str(
                "<div class=\"card\"><h3>MASP Validation</h3><p>Missing artifact <code>masp_validation_summary.json</code>.</p></div>",
            );
        }
    }
    html.push_str("</div>");

    html.push_str("<section class=\"card\"><h2>Metric Notes</h2>");
    html.push_str("<p><strong>Pitch precision/recall/F1:</strong> event-level comparison between ground-truth pitch and midpoint-frame predictions.</p>");
    html.push_str("<p><strong>Top-1 / Top-3:</strong> string classification hit rate for the best 1 or 3 candidates.</p>");
    html.push_str("<p><strong>Margin (top1-top2):</strong> confidence separation; small margins often indicate ambiguity.</p>");
    html.push_str("</section>");

    html.push_str(
        "<section class=\"card\"><h2>Confusion Matrix (True string vs Predicted string)</h2>",
    );
    match confusion {
        Some(matrix) => {
            html.push_str("<table><thead><tr><th>True \\ Pred</th>");
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
                        "<td style=\"background:rgba(31,111,73,{:.3});\"><span class=\"mono\">{}</span><br><small>{:.3}</small></td>",
                        intensity,
                        count,
                        normalized
                    ));
                }
                html.push_str("</tr>");
            }
            html.push_str("</tbody></table>");
        }
        None => {
            html.push_str("<p>Missing artifact <code>phase2_confusion_matrix.json</code>.</p>");
        }
    }
    html.push_str("</section>");

    html.push_str("<section class=\"card\"><h2>Representative Success/Failure Cases</h2>");
    if let Some(highlights) = highlights {
        render_highlight_block(
            &mut html,
            "Highest-confidence correct",
            &highlights.highest_confidence_correct,
        );
        render_highlight_block(
            &mut html,
            "Highest-confidence wrong",
            &highlights.highest_confidence_wrong,
        );
        render_highlight_block(
            &mut html,
            "Smallest top1-top2 margins",
            &highlights.smallest_margin_examples,
        );
        render_highlight_block(
            &mut html,
            "Largest-margin wrong predictions",
            &highlights.largest_margin_wrong_examples,
        );
    } else {
        html.push_str("<p>Missing artifact <code>phase2_error_highlights.json</code>.</p>");
    }
    html.push_str("</section>");

    html.push_str("<section class=\"card\"><h2>Top-k Examples</h2>");
    if examples.is_empty() {
        html.push_str("<p>No mono top-k examples were found.</p>");
    } else {
        html.push_str("<p>Showing up to 300 examples. Full table is available at <a href=\"../phase2_examples_topk.csv\">phase2_examples_topk.csv</a>.</p>");
        html.push_str("<table><thead><tr><th>Example</th><th>File</th><th>True (MIDI,S,F)</th><th>Top1 (S,F,score)</th><th>Top2</th><th>Top3</th><th>GT@1</th><th>GT@3</th><th>Margin</th><th>JSON</th></tr></thead><tbody>");

        for example in examples.iter().take(MAX_EXAMPLE_ROWS_IN_HTML) {
            let top1 = match (
                example.predicted_top1_string,
                example.predicted_top1_fret,
                example.top1_score,
            ) {
                (Some(s), Some(f), Some(score)) => format!("({}, {}, {:.4})", s, f, score),
                _ => "NA".to_string(),
            };
            let top2 = example
                .top2_candidate
                .as_ref()
                .map(|c| format!("({}, {}, {:.4})", c.string, c.fret, c.score))
                .unwrap_or_else(|| "NA".to_string());
            let top3 = example
                .top3_candidate
                .as_ref()
                .map(|c| format!("({}, {}, {:.4})", c.string, c.fret, c.score))
                .unwrap_or_else(|| "NA".to_string());

            let gt1_class = if example.ground_truth_in_top1 {
                "good"
            } else {
                "bad"
            };
            let gt3_class = if example.ground_truth_in_top3 {
                "good"
            } else {
                "bad"
            };

            let margin = example
                .margin_top1_top2
                .map(|x| format!("{x:.4}"))
                .unwrap_or_else(|| "NA".to_string());

            html.push_str("<tr>");
            html.push_str(&format!("<td>{}</td>", escape_html(&example.example_id)));
            html.push_str(&format!("<td>{}</td>", escape_html(&example.file_id)));
            html.push_str(&format!(
                "<td>({}, {}, {})</td>",
                example.true_midi, example.true_string, example.true_fret
            ));
            html.push_str(&format!("<td>{}</td>", escape_html(&top1)));
            html.push_str(&format!("<td>{}</td>", escape_html(&top2)));
            html.push_str(&format!("<td>{}</td>", escape_html(&top3)));
            html.push_str(&format!(
                "<td><span class=\"{}\">{}</span></td>",
                gt1_class, example.ground_truth_in_top1
            ));
            html.push_str(&format!(
                "<td><span class=\"{}\">{}</span></td>",
                gt3_class, example.ground_truth_in_top3
            ));
            html.push_str(&format!("<td>{}</td>", escape_html(&margin)));
            html.push_str(&format!(
                "<td><a href=\"../phase2_mono_scores/{}.json\">open</a></td>",
                escape_html(&example.example_id)
            ));
            html.push_str("</tr>");
        }

        html.push_str("</tbody></table>");
    }
    html.push_str("</section>");

    html.push_str("<section class=\"card\"><h2>Polyphonic Diagnostics (Phase 3)</h2>");
    match full_metrics {
        Some(metrics) => {
            html.push_str("<div class=\"metric\"><span>Pitch precision / recall / F1</span>");
            html.push_str(&format!(
                "<strong>{:.4} / {:.4} / {:.4}</strong></div>",
                metrics.pitch.precision, metrics.pitch.recall, metrics.pitch.f1
            ));
            html.push_str(
                "<div class=\"metric\"><span>Pitch+String precision / recall / F1</span>",
            );
            html.push_str(&format!(
                "<strong>{:.4} / {:.4} / {:.4}</strong></div>",
                metrics.pitch_string.precision,
                metrics.pitch_string.recall,
                metrics.pitch_string.f1
            ));
            html.push_str(
                "<div class=\"metric\"><span>Pitch+String+Fret precision / recall / F1</span>",
            );
            html.push_str(&format!(
                "<strong>{:.4} / {:.4} / {:.4}</strong></div>",
                metrics.pitch_string_fret.precision,
                metrics.pitch_string_fret.recall,
                metrics.pitch_string_fret.f1
            ));
            html.push_str(&format!(
                "<p>Filename split counts: comp={}, solo={}, other={}</p>",
                metrics.comp_filename_events,
                metrics.solo_filename_events,
                metrics.other_filename_events
            ));
            html.push_str(&format!(
                "<p>Failure counts: pitch-correct/string-wrong={}, string-correct/fret-wrong={}, full-decode-wrong-high-confidence={}</p>",
                metrics.failure_pitch_correct_string_wrong_count,
                metrics.failure_string_correct_fret_wrong_count,
                metrics.failure_full_decode_wrong_high_confidence_count
            ));
            html.push_str(&format!(
                "<p>Decode top-k success: top1={:.4}, top3={:.4}, top5={:.4}</p>",
                metrics.decode_top1_hit_rate,
                metrics.decode_top3_hit_rate,
                metrics.decode_top5_hit_rate
            ));
            html.push_str(&format!(
                "<p>Search breadth: avg candidates per pitch={:.3}, avg global configurations explored={:.3}</p>",
                metrics.avg_candidate_positions_per_pitch, metrics.avg_global_configurations_explored
            ));
            html.push_str(&format!(
                "<p>Full decode wrong despite correct pitch set: {}</p>",
                metrics.failure_full_wrong_despite_correct_pitch_count
            ));
        }
        None => {
            html.push_str("<p>Missing artifact <code>phase3_full_metrics.json</code>.</p>");
        }
    }
    html.push_str("</section>");

    html.push_str("<section class=\"card\"><h2>Phase 3 Failure Buckets</h2>");
    if let Some(failures) = full_failures {
        html.push_str(&format!(
            "<p>pitch correct/string wrong: {} examples</p>",
            failures.pitch_correct_string_wrong.len()
        ));
        html.push_str(&format!(
            "<p>string correct/fret wrong: {} examples</p>",
            failures.string_correct_fret_wrong.len()
        ));
        html.push_str(&format!(
            "<p>full decode wrong with high confidence: {} examples</p>",
            failures.full_decode_wrong_high_confidence.len()
        ));
    } else {
        html.push_str("<p>Missing artifact <code>phase3_failure_cases.json</code>.</p>");
    }
    html.push_str("</section>");

    html.push_str("<section class=\"card\"><h2>Phase 3 Examples</h2>");
    if full_examples.is_empty() {
        html.push_str("<p>No polyphonic examples were found.</p>");
    } else {
        html.push_str("<p>Showing up to 300 examples. Full table is available at <a href=\"../phase3_examples_topk.csv\">phase3_examples_topk.csv</a>.</p>");
        html.push_str("<table><thead><tr><th>Example</th><th>File</th><th>Split</th><th>GT pitches</th><th>Pred pitches</th><th>Exact</th><th>Decode top-k</th><th>Top1-Top2 margin</th><th>Best score</th><th>Failures</th><th>JSON</th></tr></thead><tbody>");
        for example in full_examples.iter().take(MAX_EXAMPLE_ROWS_IN_HTML) {
            let gt = example
                .ground_truth_pitches
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            let pred = example
                .predicted_pitches
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            let margin = example
                .confidence_margin_top1_top2
                .map(|x| format!("{x:.4}"))
                .unwrap_or_else(|| "NA".to_string());
            let best_score = example
                .best_total_score
                .map(|x| format!("{x:.4}"))
                .unwrap_or_else(|| "NA".to_string());
            let failures = format!(
                "p/s={} s/f={} full-wrong-correct-pitch={} high-conf={}",
                example.failure_pitch_correct_string_wrong,
                example.failure_string_correct_fret_wrong,
                example.failure_full_wrong_despite_correct_pitch,
                example.failure_full_decode_wrong_high_confidence
            );

            html.push_str("<tr>");
            html.push_str(&format!("<td>{}</td>", escape_html(&example.example_id)));
            html.push_str(&format!("<td>{}</td>", escape_html(&example.file_id)));
            html.push_str(&format!(
                "<td>{}</td>",
                escape_html(&example.filename_split)
            ));
            html.push_str(&format!("<td>{}</td>", escape_html(&gt)));
            html.push_str(&format!("<td>{}</td>", escape_html(&pred)));
            html.push_str(&format!("<td>{}</td>", example.exact_configuration_hit));
            html.push_str(&format!(
                "<td>top1={} top3={} top5={}</td>",
                example.decode_hit_top1, example.decode_hit_top3, example.decode_hit_top5
            ));
            html.push_str(&format!("<td>{}</td>", escape_html(&margin)));
            html.push_str(&format!("<td>{}</td>", escape_html(&best_score)));
            html.push_str(&format!("<td>{}</td>", escape_html(&failures)));
            html.push_str(&format!(
                "<td><a href=\"../phase3_poly_scores/{}.json\">open</a></td>",
                escape_html(&example.example_id)
            ));
            html.push_str("</tr>");
        }
        html.push_str("</tbody></table>");
    }
    html.push_str("</section>");

    html.push_str("</main></body></html>");
    html
}

fn render_highlight_block(output: &mut String, title: &str, rows: &[ExampleHighlight]) {
    output.push_str(&format!(
        "<details><summary><strong>{}</strong></summary>",
        escape_html(title)
    ));
    if rows.is_empty() {
        output.push_str("<p>None.</p></details>");
        return;
    }
    output.push_str("<table><thead><tr><th>Example</th><th>File</th><th>True MIDI</th><th>True S</th><th>Pred S</th><th>Top1 score</th><th>Margin</th><th>GT@1</th><th>JSON</th></tr></thead><tbody>");
    for row in rows {
        output.push_str("<tr>");
        output.push_str(&format!("<td>{}</td>", escape_html(&row.example_id)));
        output.push_str(&format!("<td>{}</td>", escape_html(&row.file_id)));
        output.push_str(&format!("<td>{}</td>", row.true_midi));
        output.push_str(&format!("<td>{}</td>", row.true_string));
        let pred = row
            .predicted_top1_string
            .map(|v| v.to_string())
            .unwrap_or_else(|| "NA".to_string());
        let score = row
            .top1_score
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "NA".to_string());
        let margin = row
            .margin_top1_top2
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "NA".to_string());
        output.push_str(&format!("<td>{}</td>", escape_html(&pred)));
        output.push_str(&format!("<td>{}</td>", escape_html(&score)));
        output.push_str(&format!("<td>{}</td>", escape_html(&margin)));
        output.push_str(&format!("<td>{}</td>", row.ground_truth_in_top1));
        output.push_str(&format!(
            "<td><a href=\"../phase2_mono_scores/{}.json\">open</a></td>",
            escape_html(&row.example_id)
        ));
        output.push_str("</tr>");
    }
    output.push_str("</tbody></table></details>");
}

fn file_id_for_path(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

fn midi_histogram_to_string(hist: &BTreeMap<u8, usize>) -> String {
    hist.iter()
        .map(|(midi, count)| format!("{midi}x{count}"))
        .collect::<Vec<_>>()
        .join(";")
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

fn set_intersection(left: &BTreeSet<u8>, right: &BTreeSet<u8>) -> Vec<u8> {
    left.intersection(right).copied().collect()
}

fn set_difference(left: &BTreeSet<u8>, right: &BTreeSet<u8>) -> Vec<u8> {
    left.difference(right).copied().collect()
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

fn sanitize_for_id(input: &str) -> String {
    input
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn sanitize_artifact_suffix(input: &str) -> String {
    let trimmed = input.trim().to_ascii_lowercase();
    let cleaned: String = trimmed
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if cleaned.is_empty() {
        "variant".to_string()
    } else {
        cleaned
    }
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(value)
        .with_context(|| format!("failed to serialize json: {}", path.display()))?;
    std::fs::write(path, json)
        .with_context(|| format!("failed to write json file: {}", path.display()))?;
    Ok(())
}

fn read_json_optional<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Option<T>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read json file: {}", path.display()))?;
    let value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse json file: {}", path.display()))?;
    Ok(Some(value))
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

#[cfg(test)]
mod tests {
    use super::{build_confusion_matrix, f1, sanitize_for_id};

    #[test]
    fn sanitize_for_id_replaces_non_alnum() {
        assert_eq!(sanitize_for_id("a/b:c"), "a_b_c");
    }

    #[test]
    fn f1_handles_zero_values() {
        assert_eq!(f1(0.0, 0.0), 0.0);
        assert!((f1(0.5, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn confusion_matrix_normalizes_rows() {
        let matrix = build_confusion_matrix(vec![
            vec![3, 1, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 2, 0, 0, 0],
            vec![0, 0, 0, 1, 0, 0],
            vec![0, 0, 0, 0, 1, 0],
            vec![0, 0, 0, 0, 0, 1],
        ]);
        assert!((matrix.normalized[0][0] - 0.75).abs() < 1e-6);
        assert!((matrix.normalized[0][1] - 0.25).abs() < 1e-6);
        assert_eq!(matrix.normalized[1][1], 0.0);
    }
}
