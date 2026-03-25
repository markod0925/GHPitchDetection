use std::collections::{BTreeMap, BTreeSet};
use std::fs::{create_dir_all, File};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use num::complex::Complex64;
use qdft::QDFT;
use serde::{Deserialize, Serialize};

use crate::audio::{
    load_wav_mono, normalize_audio_rms_in_place, resample_linear, rms, trim_audio_region,
};
use crate::config::{AppConfig, MaspWeightsConfig};
use crate::dataset::PolyEvalEvent;
use crate::fretboard::pitch_to_hz;
use crate::harmonic_map::build_harmonic_maps;
use crate::stft::{build_stft_plan, compute_stft_frames, StftPlan};
use crate::types::{AudioBuffer, MidiHarmonicMap, MonoAnnotation, PitchScoreFrame, SpectrumFrame};

const EPS: f32 = 1e-12;
const NOTE_CONTEXT_MARGIN_SEC: f32 = 0.005;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaspManifest {
    pub version: u32,
    pub config_hash: u64,
    #[serde(default)]
    pub model_params: MaspModelParams,
    #[serde(default)]
    pub validation_rule: MaspValidationRule,
    pub mode: String,
    pub strict_sample_rate: u32,
    pub bins_per_octave: usize,
    pub max_harmonics: usize,
    pub b_exponent: f32,
    pub cent_tolerance: f32,
    pub rms_window_ms: u32,
    pub rms_h_relax: f32,
    pub validation_score_threshold: f32,
    pub pretrain_trials: usize,
    pub target_mono_recall: f32,
    pub target_comp_recall: f32,
    pub calibrated_precision: Option<f32>,
    pub calibrated_mono_recall: Option<f32>,
    pub calibrated_comp_recall: Option<f32>,
    pub weights: MaspWeightsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaspModelParams {
    pub mode: String,
    pub strict_sample_rate: u32,
    pub bins_per_octave: usize,
    pub max_harmonics: usize,
    pub b_exponent: f32,
    pub cent_tolerance: f32,
    pub rms_window_ms: u32,
    pub rms_h_relax: f32,
    pub pretrain_trials: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaspValidationRule {
    pub score_threshold: f32,
    pub score_weights: MaspWeightsConfig,
    pub target_mono_recall: f32,
    pub target_comp_recall: f32,
    pub calibrated_precision: Option<f32>,
    pub calibrated_mono_recall: Option<f32>,
    pub calibrated_comp_recall: Option<f32>,
}

impl MaspManifest {
    fn sync_structured_fields_from_legacy(&mut self) {
        if self.model_params.mode.is_empty() {
            self.model_params = MaspModelParams {
                mode: self.mode.clone(),
                strict_sample_rate: self.strict_sample_rate,
                bins_per_octave: self.bins_per_octave,
                max_harmonics: self.max_harmonics,
                b_exponent: self.b_exponent,
                cent_tolerance: self.cent_tolerance,
                rms_window_ms: self.rms_window_ms,
                rms_h_relax: self.rms_h_relax,
                pretrain_trials: self.pretrain_trials,
            };
        }

        let weights_sum = self.validation_rule.score_weights.har
            + self.validation_rule.score_weights.mbw
            + self.validation_rule.score_weights.cent
            + self.validation_rule.score_weights.rms;
        if weights_sum <= 0.0 {
            self.validation_rule = MaspValidationRule {
                score_threshold: self.validation_score_threshold,
                score_weights: self.weights.clone(),
                target_mono_recall: self.target_mono_recall,
                target_comp_recall: self.target_comp_recall,
                calibrated_precision: self.calibrated_precision,
                calibrated_mono_recall: self.calibrated_mono_recall,
                calibrated_comp_recall: self.calibrated_comp_recall,
            };
        }
    }

    fn sync_legacy_fields_from_structured(&mut self) {
        self.mode = self.model_params.mode.clone();
        self.strict_sample_rate = self.model_params.strict_sample_rate;
        self.bins_per_octave = self.model_params.bins_per_octave;
        self.max_harmonics = self.model_params.max_harmonics;
        self.b_exponent = self.model_params.b_exponent;
        self.cent_tolerance = self.model_params.cent_tolerance;
        self.rms_window_ms = self.model_params.rms_window_ms;
        self.rms_h_relax = self.model_params.rms_h_relax;
        self.pretrain_trials = self.model_params.pretrain_trials;
        self.validation_score_threshold = self.validation_rule.score_threshold;
        self.weights = self.validation_rule.score_weights.clone();
        self.target_mono_recall = self.validation_rule.target_mono_recall;
        self.target_comp_recall = self.validation_rule.target_comp_recall;
        self.calibrated_precision = self.validation_rule.calibrated_precision;
        self.calibrated_mono_recall = self.validation_rule.calibrated_mono_recall;
        self.calibrated_comp_recall = self.validation_rule.calibrated_comp_recall;
    }

    fn finalize_compat(&mut self) {
        self.sync_structured_fields_from_legacy();
        self.sync_legacy_fields_from_structured();
    }

    fn validation_weights(&self) -> &MaspWeightsConfig {
        &self.validation_rule.score_weights
    }

    fn validation_threshold(&self) -> f32 {
        self.validation_rule.score_threshold
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaspNoteSignature {
    pub key: String,
    pub midi: u8,
    pub string: u8,
    pub fret: u8,
    pub count: usize,
    pub mean_pitch_spectrum: Vec<f32>,
    pub h_target: f32,
    pub har_target: f32,
    pub mbw_target: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaspJointSignature {
    pub key: String,
    pub expected_midis: Vec<u8>,
    pub count: usize,
    pub mean_pitch_spectrum: Vec<f32>,
    pub h_target: f32,
    pub har_target: f32,
    pub mbw_target: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaspPretrainArtifacts {
    pub manifest: MaspManifest,
    pub note_signatures: Vec<MaspNoteSignature>,
    pub joint_signatures: Vec<MaspJointSignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaspValidationRequest {
    pub id: Option<String>,
    pub audio_path: String,
    pub start_sec: f32,
    pub end_sec: f32,
    pub expected_midis: Vec<u8>,
    pub expected_valid: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaspValidationResult {
    pub id: Option<String>,
    pub audio_path: String,
    pub expected_midis: Vec<u8>,
    pub pass: bool,
    pub reason: String,
    pub h_real: f32,
    pub h_target: f32,
    pub h_dynamic_threshold: f32,
    pub har: f32,
    pub mbw: f32,
    pub cent_error: f32,
    pub cent_tolerance: f32,
    pub noise_rms: f32,
    pub weighted_score: f32,
    pub score_threshold: f32,
    pub execution_ms: Option<f64>,
    pub expected_valid: Option<bool>,
    pub false_accept: bool,
    pub false_reject: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaspFileSummary {
    pub file_id: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaspValidationSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub pass_rate: f32,
    pub labeled_total: usize,
    pub false_accept_count: usize,
    pub false_reject_count: usize,
    pub accuracy: Option<f32>,
    pub timing_ms_mean: Option<f32>,
    pub timing_ms_p95: Option<f32>,
    pub timing_ms_max: Option<f32>,
    pub per_file: Vec<MaspFileSummary>,
}

#[derive(Debug, Clone)]
struct EventMetrics {
    pitch_spectrum: Vec<f32>,
    h: f32,
    har: f32,
    mbw: f32,
    cent_error: f32,
    noise_rms: f32,
}

#[derive(Debug, Clone)]
struct SignatureAccumulator {
    sum_pitch_spectrum: Vec<f64>,
    sum_h: f64,
    sum_har: f64,
    sum_mbw: f64,
    count: usize,
}

#[derive(Debug, Clone)]
struct CalibrationCorpus {
    mono_positive: Vec<EventMetrics>,
    mono_negative: Vec<EventMetrics>,
    comp_positive: Vec<EventMetrics>,
    comp_negative: Vec<EventMetrics>,
}

struct MaspValidationTuning {
    b_exponent: f32,
    weights: MaspWeightsConfig,
    threshold: f32,
    selection: ThresholdSelection,
}

#[derive(Debug, Clone, Copy)]
struct ThresholdSelection {
    threshold: f32,
    precision: f32,
    mono_recall: f32,
    comp_recall: f32,
    objective: f32,
}

impl SignatureAccumulator {
    fn new(dim: usize) -> Self {
        Self {
            sum_pitch_spectrum: vec![0.0; dim],
            sum_h: 0.0,
            sum_har: 0.0,
            sum_mbw: 0.0,
            count: 0,
        }
    }

    fn update(&mut self, metrics: &EventMetrics) {
        if self.sum_pitch_spectrum.len() != metrics.pitch_spectrum.len() {
            return;
        }
        for (dst, src) in self
            .sum_pitch_spectrum
            .iter_mut()
            .zip(&metrics.pitch_spectrum)
        {
            *dst += *src as f64;
        }
        self.sum_h += metrics.h as f64;
        self.sum_har += metrics.har as f64;
        self.sum_mbw += metrics.mbw as f64;
        self.count += 1;
    }
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

    fn next_f32(&mut self) -> f32 {
        let v = (self.next_u64() >> 40) as u32;
        v as f32 / (u32::MAX >> 8) as f32
    }

    fn f32_inclusive(&mut self, min: f32, max: f32) -> f32 {
        if !min.is_finite() || !max.is_finite() || min >= max {
            return min;
        }
        let t = self.next_f32().clamp(0.0, 1.0);
        min + t * (max - min)
    }
}

pub fn score_pitch_frames_masp(audio: &AudioBuffer, cfg: &AppConfig) -> Vec<PitchScoreFrame> {
    let (samples, effective_sr, _) =
        preprocess_samples_for_masp(&audio.samples, audio.sample_rate, cfg);
    let plan = build_stft_plan(&cfg.frontend);
    let frames = compute_phase1_frontend_frames_masp(&samples, effective_sr, &plan, cfg);
    if frames.is_empty() {
        return Vec::new();
    }

    let maps = build_harmonic_maps(
        effective_sr,
        plan.nfft,
        cfg.pitch.midi_min,
        cfg.pitch.midi_max,
        cfg.masp.max_harmonics,
        cfg.pitch.local_bandwidth_bins,
    );

    let mut out = Vec::with_capacity(frames.len());
    for frame in frames {
        let midi_scores = score_masp_midi_frame(&frame.mag, &maps, cfg);
        out.push(PitchScoreFrame {
            time_sec: frame.time_sec,
            midi_scores,
        });
    }
    out
}

pub fn pretrain_from_guitarset(
    cfg: &AppConfig,
    mono_dataset: &[MonoAnnotation],
    comp_dataset: &[PolyEvalEvent],
    target_mono_recall: f32,
    target_comp_recall: f32,
) -> Result<MaspPretrainArtifacts> {
    let mut audio_cache: BTreeMap<String, AudioBuffer> = BTreeMap::new();
    let tuned = tune_masp_validation(
        cfg,
        mono_dataset,
        comp_dataset,
        &mut audio_cache,
        target_mono_recall,
        target_comp_recall,
    )?;

    let mut tuned_cfg = cfg.clone();
    tuned_cfg.masp.b_exponent = tuned.b_exponent;

    let (note_signatures, joint_signatures) = build_signatures_from_guitarset(
        &tuned_cfg,
        mono_dataset,
        comp_dataset,
        &mut audio_cache,
    )?;

    let mut manifest = build_manifest(&tuned_cfg);
    manifest.validation_rule.score_weights = tuned.weights;
    manifest.validation_rule.score_threshold = tuned.threshold;
    manifest.validation_rule.target_mono_recall = target_mono_recall;
    manifest.validation_rule.target_comp_recall = target_comp_recall;
    manifest.validation_rule.calibrated_precision = Some(tuned.selection.precision);
    manifest.validation_rule.calibrated_mono_recall = Some(tuned.selection.mono_recall);
    manifest.validation_rule.calibrated_comp_recall = Some(tuned.selection.comp_recall);
    manifest.finalize_compat();

    Ok(MaspPretrainArtifacts {
        manifest,
        note_signatures,
        joint_signatures,
    })
}

fn build_signatures_from_guitarset(
    cfg: &AppConfig,
    mono_dataset: &[MonoAnnotation],
    comp_dataset: &[PolyEvalEvent],
    audio_cache: &mut BTreeMap<String, AudioBuffer>,
) -> Result<(Vec<MaspNoteSignature>, Vec<MaspJointSignature>)> {
    let mut note_acc: BTreeMap<String, (u8, u8, u8, SignatureAccumulator)> = BTreeMap::new();
    let mut joint_acc: BTreeMap<String, (Vec<u8>, SignatureAccumulator)> = BTreeMap::new();
    let dim = usize::from(cfg.pitch.midi_max.saturating_sub(cfg.pitch.midi_min)) + 1;

    for note in mono_dataset {
        let audio = load_or_cached_audio(audio_cache, &note.audio_path)?;
        let metrics =
            segment_metrics_for_audio(audio, note.onset_sec, note.offset_sec, &[note.midi], cfg)?;
        let key = format!("s{}_f{}_m{}", note.string, note.fret, note.midi);
        let entry = note_acc.entry(key).or_insert_with(|| {
            (
                note.midi,
                note.string,
                note.fret,
                SignatureAccumulator::new(dim),
            )
        });
        entry.3.update(&metrics);
    }

    for event in comp_dataset {
        let mut expected = event.notes.iter().map(|x| x.midi).collect::<Vec<_>>();
        expected.sort_unstable();
        expected.dedup();
        if expected.is_empty() {
            continue;
        }
        let audio = load_or_cached_audio(audio_cache, &event.audio_path)?;
        let metrics =
            segment_metrics_for_audio(audio, event.onset_sec, event.offset_sec, &expected, cfg)?;
        let key = midi_key(&expected);
        let entry = joint_acc
            .entry(key)
            .or_insert_with(|| (expected.clone(), SignatureAccumulator::new(dim)));
        entry.1.update(&metrics);
    }

    Ok((
        finalize_note_signatures(note_acc),
        finalize_joint_signatures(joint_acc),
    ))
}

pub fn write_pretrain_artifacts(out_dir: &Path, artifacts: &MaspPretrainArtifacts) -> Result<()> {
    create_dir_all(out_dir)
        .with_context(|| format!("failed to create output directory: {}", out_dir.display()))?;

    write_json_pretty(&out_dir.join("masp_manifest.json"), &artifacts.manifest)?;
    write_jsonl(
        &out_dir.join("masp_note_signatures.jsonl"),
        &artifacts.note_signatures,
    )?;
    write_jsonl(
        &out_dir.join("masp_joint_signatures.jsonl"),
        &artifacts.joint_signatures,
    )?;

    Ok(())
}

pub fn load_pretrain_artifacts(in_dir: &Path) -> Result<MaspPretrainArtifacts> {
    let mut manifest: MaspManifest = read_json(&in_dir.join("masp_manifest.json"))?;
    manifest.finalize_compat();
    let note_signatures =
        read_jsonl::<MaspNoteSignature>(&in_dir.join("masp_note_signatures.jsonl"))?;
    let joint_signatures =
        read_jsonl::<MaspJointSignature>(&in_dir.join("masp_joint_signatures.jsonl"))?;

    Ok(MaspPretrainArtifacts {
        manifest,
        note_signatures,
        joint_signatures,
    })
}

pub fn validate_expected_segment(
    audio: &AudioBuffer,
    start_sec: f32,
    end_sec: f32,
    expected_midis: &[u8],
    cfg: &AppConfig,
    artifacts: &MaspPretrainArtifacts,
) -> Result<MaspValidationResult> {
    if expected_midis.is_empty() {
        bail!("expected_midis must not be empty");
    }

    let mut expected = expected_midis.to_vec();
    expected.sort_unstable();
    expected.dedup();

    let metrics = segment_metrics_for_audio(audio, start_sec, end_sec, &expected, cfg)?;
    let targets = resolve_targets(&expected, artifacts);

    let cent_gate = metrics.cent_error.abs() <= cfg.masp.cent_tolerance;
    let h_dynamic_threshold =
        (targets.0 * (1.0 - cfg.masp.rms_h_relax * metrics.noise_rms.clamp(0.0, 1.0))).max(0.0);
    let h_gate = metrics.h >= h_dynamic_threshold;

    let cent_score = cent_score(metrics.cent_error, cfg.masp.cent_tolerance);
    let rms_score = (1.0 - metrics.noise_rms).clamp(0.0, 1.0);

    let weights = artifacts.manifest.validation_weights();
    let weighted_score = weights.har * metrics.har
        + weights.mbw * metrics.mbw
        + weights.cent * cent_score
        + weights.rms * rms_score;

    let score_gate = weighted_score >= artifacts.manifest.validation_threshold();
    let pass = score_gate;

    let reason = if pass {
        if cent_gate && h_gate {
            "validated".to_string()
        } else if !cent_gate && !h_gate {
            "validated_soft_score_without_cent_or_h_gate".to_string()
        } else if !cent_gate {
            "validated_soft_score_without_cent_gate".to_string()
        } else {
            "validated_soft_score_without_h_gate".to_string()
        }
    } else if !cent_gate {
        "cent_gate_failed".to_string()
    } else if !h_gate {
        "harmonicity_threshold_failed".to_string()
    } else {
        "weighted_score_failed".to_string()
    };

    Ok(MaspValidationResult {
        id: None,
        audio_path: String::new(),
        expected_midis: expected,
        pass,
        reason,
        h_real: metrics.h,
        h_target: targets.0,
        h_dynamic_threshold,
        har: metrics.har,
        mbw: metrics.mbw,
        cent_error: metrics.cent_error,
        cent_tolerance: cfg.masp.cent_tolerance,
        noise_rms: metrics.noise_rms,
        weighted_score,
        score_threshold: artifacts.manifest.validation_threshold(),
        execution_ms: None,
        expected_valid: None,
        false_accept: false,
        false_reject: false,
    })
}

pub fn write_validation_artifacts(
    out_dir: &Path,
    results: &[MaspValidationResult],
) -> Result<MaspValidationSummary> {
    create_dir_all(out_dir)
        .with_context(|| format!("failed to create output directory: {}", out_dir.display()))?;

    write_jsonl(&out_dir.join("masp_validation_results.jsonl"), results)?;

    let mut per_file_acc: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut passed = 0usize;
    let mut false_accept_count = 0usize;
    let mut false_reject_count = 0usize;
    let mut labeled_total = 0usize;
    let mut labeled_correct = 0usize;
    let mut timings_ms = Vec::<f64>::new();

    for result in results {
        if result.pass {
            passed += 1;
        }

        let file_id = file_id_for_path(&result.audio_path);
        let entry = per_file_acc.entry(file_id).or_insert((0, 0));
        entry.0 += 1;
        if result.pass {
            entry.1 += 1;
        }

        if result.false_accept {
            false_accept_count += 1;
        }
        if result.false_reject {
            false_reject_count += 1;
        }

        if let Some(expected_valid) = result.expected_valid {
            labeled_total += 1;
            if expected_valid == result.pass {
                labeled_correct += 1;
            }
        }
        if let Some(ms) = result.execution_ms {
            if ms.is_finite() && ms >= 0.0 {
                timings_ms.push(ms);
            }
        }
    }

    let total = results.len();
    let failed = total.saturating_sub(passed);

    let mut per_file = Vec::with_capacity(per_file_acc.len());
    for (file_id, (count, pass_count)) in per_file_acc {
        per_file.push(MaspFileSummary {
            file_id,
            total: count,
            passed: pass_count,
            failed: count.saturating_sub(pass_count),
        });
    }

    timings_ms.sort_by(|a, b| a.total_cmp(b));
    let timing_ms_mean = if timings_ms.is_empty() {
        None
    } else {
        Some((timings_ms.iter().sum::<f64>() / timings_ms.len() as f64) as f32)
    };
    let timing_ms_p95 = if timings_ms.is_empty() {
        None
    } else {
        let idx = ((timings_ms.len() - 1) as f64 * 0.95).round() as usize;
        Some(timings_ms[idx] as f32)
    };
    let timing_ms_max = timings_ms.last().copied().map(|x| x as f32);

    let summary = MaspValidationSummary {
        total,
        passed,
        failed,
        pass_rate: ratio(passed, total),
        labeled_total,
        false_accept_count,
        false_reject_count,
        accuracy: if labeled_total > 0 {
            Some(ratio(labeled_correct, labeled_total))
        } else {
            None
        },
        timing_ms_mean,
        timing_ms_p95,
        timing_ms_max,
        per_file,
    };

    write_json_pretty(&out_dir.join("masp_validation_summary.json"), &summary)?;
    Ok(summary)
}

pub fn read_validation_requests(path: &Path) -> Result<Vec<MaspValidationRequest>> {
    read_jsonl(path)
}

pub fn write_validation_request_template(path: &Path) -> Result<()> {
    let sample = MaspValidationRequest {
        id: Some("example_1".to_string()),
        audio_path: "input/guitarset/audio/00_BN1-129-Eb_comp_mic.wav".to_string(),
        start_sec: 1.0,
        end_sec: 1.3,
        expected_midis: vec![51, 58, 63],
        expected_valid: None,
    };
    write_jsonl(path, &[sample])
}

pub fn apply_log_shift(
    midi_scores: &[f32],
    semitone_shift: f32,
    bins_per_octave: usize,
) -> Vec<f32> {
    if midi_scores.is_empty() {
        return Vec::new();
    }
    let max_bins = midi_scores.len();
    let mut out = vec![0.0_f32; max_bins];
    let shift_log2 = semitone_shift / 12.0;
    for (idx, value) in midi_scores.iter().copied().enumerate() {
        let dst = log_shifted_bin_index(idx, shift_log2, bins_per_octave, max_bins);
        out[dst] += value;
    }
    out
}

pub fn log_shifted_bin_index(
    bin: usize,
    shift_log2: f32,
    bins_per_octave: usize,
    max_bins: usize,
) -> usize {
    if max_bins == 0 {
        return 0;
    }
    let shift_bins = shift_log2 * bins_per_octave as f32;
    let idx = bin as f32 + shift_bins;
    idx.round().clamp(0.0, (max_bins - 1) as f32) as usize
}

fn tune_masp_validation(
    cfg: &AppConfig,
    mono_dataset: &[MonoAnnotation],
    comp_dataset: &[PolyEvalEvent],
    audio_cache: &mut BTreeMap<String, AudioBuffer>,
    target_mono_recall: f32,
    target_comp_recall: f32,
) -> Result<MaspValidationTuning> {
    if mono_dataset.is_empty() && comp_dataset.is_empty() {
        return Ok(fallback_validation_tuning(cfg));
    }

    let mut best = fallback_validation_tuning(cfg);
    best.selection.objective = f32::NEG_INFINITY;
    let mut rng = SeededRng::new(cfg.optimization.seed ^ 0x4D_41_53_50);

    let b_trials = sampled_b_exponents(cfg, &mut rng);
    let mut weight_trials = Vec::new();
    weight_trials.push(cfg.masp.weights.clone());
    for _ in 0..cfg.masp.pretrain_trials {
        let mut sampled = MaspWeightsConfig {
            har: rng.f32_inclusive(0.05, 1.0),
            mbw: rng.f32_inclusive(0.05, 1.0),
            cent: rng.f32_inclusive(0.05, 1.0),
            rms: rng.f32_inclusive(0.05, 1.0),
        };
        normalize_weights(&mut sampled);
        weight_trials.push(sampled);
    }

    for b_exponent in b_trials {
        let mut trial_cfg = cfg.clone();
        trial_cfg.masp.b_exponent = b_exponent;
        let corpus = build_calibration_corpus(&trial_cfg, mono_dataset, comp_dataset, audio_cache)?;
        if corpus.mono_positive.is_empty() && corpus.comp_positive.is_empty() {
            continue;
        }

        for weights in &weight_trials {
            let selection = select_threshold_for_targets(
                &corpus,
                weights,
                cfg.masp.cent_tolerance,
                target_mono_recall,
                target_comp_recall,
            );
            if selection.objective > best.selection.objective {
                best = MaspValidationTuning {
                    b_exponent,
                    weights: weights.clone(),
                    threshold: selection.threshold,
                    selection,
                };
            }
        }
    }

    Ok(best)
}

fn fallback_validation_tuning(cfg: &AppConfig) -> MaspValidationTuning {
    MaspValidationTuning {
        b_exponent: cfg.masp.b_exponent,
        weights: cfg.masp.weights.clone(),
        threshold: cfg.masp.validation_score_threshold,
        selection: ThresholdSelection {
            threshold: cfg.masp.validation_score_threshold,
            precision: 0.0,
            mono_recall: 0.0,
            comp_recall: 0.0,
            objective: 0.0,
        },
    }
}

fn sampled_b_exponents(cfg: &AppConfig, rng: &mut SeededRng) -> Vec<f32> {
    const B_MIN: f32 = 0.0;
    const B_MAX: f32 = 1.0;

    let target_count = ((cfg.masp.pretrain_trials as f64).sqrt().ceil() as usize).max(3);
    let mut values = Vec::with_capacity(target_count + 3);
    values.push(cfg.masp.b_exponent.clamp(B_MIN, B_MAX));
    values.push(0.25);
    values.push(0.5);
    while values.len() < target_count + 3 {
        values.push(rng.f32_inclusive(B_MIN, B_MAX));
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    values
}

fn build_decoy_sets(expected: &[u8], cfg: &AppConfig) -> Vec<Vec<u8>> {
    let mut out = Vec::new();

    let mut sub = Vec::new();
    for &midi in expected {
        if midi >= cfg.pitch.midi_min + 12 {
            sub.push(midi - 12);
        }
    }
    if !sub.is_empty() {
        sub.sort_unstable();
        sub.dedup();
        if sub != expected {
            out.push(sub);
        }
    }

    let mut up = Vec::new();
    for &midi in expected {
        if midi <= cfg.pitch.midi_max.saturating_sub(12) {
            up.push(midi + 12);
        }
    }
    if !up.is_empty() {
        up.sort_unstable();
        up.dedup();
        if up != expected {
            out.push(up);
        }
    }

    let root = expected.iter().copied().min().unwrap_or(cfg.pitch.midi_min);
    if root >= cfg.pitch.midi_min + 12 {
        out.push(vec![root - 12]);
    }

    out.sort();
    out.dedup();
    out
}

fn segment_metrics_for_audio(
    audio: &AudioBuffer,
    start_sec: f32,
    end_sec: f32,
    expected_midis: &[u8],
    cfg: &AppConfig,
) -> Result<EventMetrics> {
    let start = (start_sec - NOTE_CONTEXT_MARGIN_SEC).max(0.0);
    let end = (end_sec + NOTE_CONTEXT_MARGIN_SEC).max(start);

    let segment_raw = trim_audio_region(&audio.samples, audio.sample_rate, start, end);
    if segment_raw.is_empty() {
        bail!("empty segment in [{start_sec:.3}, {end_sec:.3}]");
    }

    let raw_noise_rms = rms(&segment_raw).clamp(0.0, 1.0);

    let (mut segment, effective_sr, _) =
        preprocess_samples_for_masp(&segment_raw, audio.sample_rate, cfg);
    if segment.is_empty() {
        bail!("segment became empty after preprocessing");
    }

    normalize_audio_rms_in_place(&mut segment, 0.1);

    let plan = build_stft_plan(&cfg.frontend);
    if segment.len() < plan.win_length {
        segment.resize(plan.win_length, 0.0);
    }
    let frames = compute_phase1_frontend_frames_masp(&segment, effective_sr, &plan, cfg);
    if frames.is_empty() {
        bail!("no frontend frames for segment");
    }

    let maps = build_harmonic_maps(
        effective_sr,
        plan.nfft,
        cfg.pitch.midi_min,
        cfg.pitch.midi_max,
        cfg.masp.max_harmonics,
        cfg.pitch.local_bandwidth_bins,
    );

    let mut best: Option<EventMetrics> = None;
    for frame in &frames {
        let pitch_spectrum = score_masp_midi_frame(&frame.mag, &maps, cfg);
        let candidate = EventMetrics {
            h: compute_harmonicity_h(&frame.mag, &pitch_spectrum, effective_sr, plan.nfft, cfg),
            har: compute_har(&pitch_spectrum, expected_midis, cfg.pitch.midi_min),
            mbw: compute_mbw(&frame.mag, expected_midis, &maps),
            cent_error: compute_cent_error(&pitch_spectrum, expected_midis, cfg.pitch.midi_min),
            pitch_spectrum,
            noise_rms: raw_noise_rms,
        };

        let should_replace = match &best {
            None => true,
            Some(previous) => is_better_interval_match(&candidate, previous),
        };
        if should_replace {
            best = Some(candidate);
        }
    }

    best.with_context(|| format!("no MASP frame metrics for [{start_sec:.3}, {end_sec:.3}]"))
}

fn resolve_targets(expected_midis: &[u8], artifacts: &MaspPretrainArtifacts) -> (f32, f32, f32) {
    if expected_midis.len() > 1 {
        let key = midi_key(expected_midis);
        if let Some(sig) = artifacts.joint_signatures.iter().find(|x| x.key == key) {
            return (sig.h_target, sig.har_target, sig.mbw_target);
        }

        let expected_set = expected_midis.iter().copied().collect::<BTreeSet<_>>();
        let mut best = None::<(f32, &MaspJointSignature)>;
        for sig in &artifacts.joint_signatures {
            let sig_set = sig.expected_midis.iter().copied().collect::<BTreeSet<_>>();
            let inter = sig_set.intersection(&expected_set).count();
            let union = sig_set.union(&expected_set).count().max(1);
            let j = inter as f32 / union as f32;
            match best {
                Some((bj, _)) if bj >= j => {}
                _ => best = Some((j, sig)),
            }
        }
        if let Some((_, sig)) = best {
            return (sig.h_target, sig.har_target, sig.mbw_target);
        }
    }

    let mut count = 0usize;
    let mut h_sum = 0.0_f32;
    let mut har_sum = 0.0_f32;
    let mut mbw_sum = 0.0_f32;

    for &midi in expected_midis {
        for sig in &artifacts.note_signatures {
            if sig.midi == midi {
                count += 1;
                h_sum += sig.h_target;
                har_sum += sig.har_target;
                mbw_sum += sig.mbw_target;
            }
        }
    }

    if count > 0 {
        let inv = 1.0 / count as f32;
        (h_sum * inv, har_sum * inv, mbw_sum * inv)
    } else {
        (1.0, 0.5, 0.5)
    }
}

fn finalize_note_signatures(
    acc: BTreeMap<String, (u8, u8, u8, SignatureAccumulator)>,
) -> Vec<MaspNoteSignature> {
    let mut out = Vec::with_capacity(acc.len());

    for (key, (midi, string, fret, s)) in acc {
        if s.count == 0 {
            continue;
        }
        let inv = 1.0 / s.count as f32;
        let mean_pitch_spectrum = s
            .sum_pitch_spectrum
            .iter()
            .map(|x| (*x as f32) * inv)
            .collect::<Vec<_>>();

        out.push(MaspNoteSignature {
            key,
            midi,
            string,
            fret,
            count: s.count,
            mean_pitch_spectrum,
            h_target: (s.sum_h as f32) * inv,
            har_target: (s.sum_har as f32) * inv,
            mbw_target: (s.sum_mbw as f32) * inv,
        });
    }

    out.sort_by(|a, b| a.key.cmp(&b.key));
    out
}

fn finalize_joint_signatures(
    acc: BTreeMap<String, (Vec<u8>, SignatureAccumulator)>,
) -> Vec<MaspJointSignature> {
    let mut out = Vec::with_capacity(acc.len());

    for (key, (expected_midis, s)) in acc {
        if s.count == 0 {
            continue;
        }
        let inv = 1.0 / s.count as f32;
        let mean_pitch_spectrum = s
            .sum_pitch_spectrum
            .iter()
            .map(|x| (*x as f32) * inv)
            .collect::<Vec<_>>();

        out.push(MaspJointSignature {
            key,
            expected_midis,
            count: s.count,
            mean_pitch_spectrum,
            h_target: (s.sum_h as f32) * inv,
            har_target: (s.sum_har as f32) * inv,
            mbw_target: (s.sum_mbw as f32) * inv,
        });
    }

    out.sort_by(|a, b| a.key.cmp(&b.key));
    out
}

fn score_masp_midi_frame(mag: &[f32], maps: &[MidiHarmonicMap], cfg: &AppConfig) -> Vec<f32> {
    if mag.is_empty() {
        return vec![0.0; maps.len()];
    }

    let mean_x = mag.iter().copied().sum::<f32>() / mag.len() as f32;
    let sum_x = mag.iter().copied().sum::<f32>();

    let mut out = vec![0.0_f32; maps.len()];
    for (idx, map) in maps.iter().enumerate() {
        let mut product = 1.0_f32;
        let mut smooth_sum = 0.0_f32;
        let mut used = 0usize;

        for (h_idx, range) in map.ranges.iter().enumerate().take(cfg.masp.max_harmonics) {
            let k = h_idx + 1;
            let local =
                aggregate_local_masp(mag, range.start, range.end, &cfg.pitch.local_aggregation);
            let a = smoothing_factor(k, cfg.masp.b_exponent);
            let xk = (a * local + (1.0 - a) * mean_x).max(EPS);
            product *= xk;
            smooth_sum += xk;
            used += 1;
        }

        if used == 0 {
            out[idx] = 0.0;
            continue;
        }

        let s = (1.0 + sum_x).ln() / (smooth_sum + EPS);
        out[idx] = (s * product).max(0.0);
    }

    out
}

fn compute_harmonicity_h(
    mag: &[f32],
    pitch_spectrum: &[f32],
    sample_rate: u32,
    nfft: usize,
    cfg: &AppConfig,
) -> f32 {
    if mag.is_empty() || pitch_spectrum.is_empty() || nfft == 0 || sample_rate == 0 {
        return 1.0;
    }

    let mut mag_sum = 0.0_f64;
    let mut mag_expect = 0.0_f64;
    let hz_per_bin = sample_rate as f64 / nfft as f64;

    for (bin_idx, &x) in mag.iter().enumerate().skip(1) {
        if x <= 0.0 {
            continue;
        }
        let freq = (bin_idx as f64 * hz_per_bin).max(1.0);
        let w = x as f64;
        mag_sum += w;
        mag_expect += w * freq.log2();
    }
    if mag_sum <= 0.0 {
        return 1.0;
    }
    let m_f = mag_expect / mag_sum;

    let mut pitch_sum = 0.0_f64;
    let mut pitch_expect = 0.0_f64;
    for (idx, &x) in pitch_spectrum.iter().enumerate() {
        if x <= 0.0 {
            continue;
        }
        let midi = cfg.pitch.midi_min as usize + idx;
        let freq = pitch_to_hz(midi as u8).max(1.0) as f64;
        let w = x as f64;
        pitch_sum += w;
        pitch_expect += w * freq.log2();
    }
    if pitch_sum <= 0.0 {
        return 1.0;
    }
    let m_p = pitch_expect / pitch_sum;

    2.0_f64.powf(m_f - m_p) as f32
}

fn compute_har(pitch_spectrum: &[f32], expected_midis: &[u8], midi_min: u8) -> f32 {
    if pitch_spectrum.is_empty() {
        return 0.0;
    }
    let total = pitch_spectrum.iter().copied().sum::<f32>();
    if total <= 0.0 {
        return 0.0;
    }

    let mut explained = 0.0_f32;
    for &midi in expected_midis {
        if midi < midi_min {
            continue;
        }
        let idx = usize::from(midi - midi_min);
        if let Some(v) = pitch_spectrum.get(idx) {
            explained += *v;
        }
    }
    (explained / total).clamp(0.0, 1.0)
}

fn compute_mbw(mag: &[f32], expected_midis: &[u8], maps: &[MidiHarmonicMap]) -> f32 {
    if mag.is_empty() || expected_midis.is_empty() {
        return 0.0;
    }

    let mut selected_bins = BTreeSet::new();
    for &midi in expected_midis {
        if let Some(map) = maps.iter().find(|m| m.midi == midi) {
            for range in &map.ranges {
                let center = (range.start + range.end) / 2;
                selected_bins.insert(center.min(mag.len().saturating_sub(1)));
            }
        }
    }

    if selected_bins.len() < 2 {
        return 0.0;
    }

    let mut prev = None::<f32>;
    let mut abs_diff_sum = 0.0_f32;
    let mut count = 0usize;

    for bin in selected_bins {
        let x = (mag[bin] + EPS).ln();
        if let Some(p) = prev {
            abs_diff_sum += (x - p).abs();
            count += 1;
        }
        prev = Some(x);
    }

    if count == 0 {
        return 0.0;
    }

    let mad = abs_diff_sum / count as f32;
    (1.0 / (1.0 + mad)).clamp(0.0, 1.0)
}

fn compute_cent_error(pitch_spectrum: &[f32], expected_midis: &[u8], midi_min: u8) -> f32 {
    if pitch_spectrum.is_empty() || expected_midis.is_empty() {
        return 0.0;
    }

    let (max_idx, _) = pitch_spectrum
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, 0.0));

    let predicted_midi = midi_min as f32 + max_idx as f32;
    let nearest_expected = expected_midis
        .iter()
        .copied()
        .map(|m| m as f32)
        .min_by(|a, b| {
            (predicted_midi - *a)
                .abs()
                .total_cmp(&(predicted_midi - *b).abs())
        })
        .unwrap_or(predicted_midi);

    100.0 * (predicted_midi - nearest_expected)
}

fn weighted_metrics_score(
    metrics: &EventMetrics,
    weights: &MaspWeightsConfig,
    cent_tol: f32,
) -> f32 {
    let cent = cent_score(metrics.cent_error, cent_tol);
    let rms_score = (1.0 - metrics.noise_rms).clamp(0.0, 1.0);
    weights.har * metrics.har
        + weights.mbw * metrics.mbw
        + weights.cent * cent
        + weights.rms * rms_score
}

fn build_calibration_corpus(
    cfg: &AppConfig,
    mono_dataset: &[MonoAnnotation],
    comp_dataset: &[PolyEvalEvent],
    audio_cache: &mut BTreeMap<String, AudioBuffer>,
) -> Result<CalibrationCorpus> {
    let mut corpus = CalibrationCorpus {
        mono_positive: Vec::new(),
        mono_negative: Vec::new(),
        comp_positive: Vec::new(),
        comp_negative: Vec::new(),
    };

    for note in mono_dataset {
        let audio = load_or_cached_audio(audio_cache, &note.audio_path)?;
        corpus.mono_positive.push(segment_metrics_for_audio(
            audio,
            note.onset_sec,
            note.offset_sec,
            &[note.midi],
            cfg,
        )?);

        for decoy in build_decoy_sets(&[note.midi], cfg) {
            corpus.mono_negative.push(segment_metrics_for_audio(
                audio,
                note.onset_sec,
                note.offset_sec,
                &decoy,
                cfg,
            )?);
        }
    }

    for event in comp_dataset {
        let mut expected = event.notes.iter().map(|x| x.midi).collect::<Vec<_>>();
        expected.sort_unstable();
        expected.dedup();
        if expected.is_empty() {
            continue;
        }

        let audio = load_or_cached_audio(audio_cache, &event.audio_path)?;
        corpus.comp_positive.push(segment_metrics_for_audio(
            audio,
            event.onset_sec,
            event.offset_sec,
            &expected,
            cfg,
        )?);

        for decoy in build_decoy_sets(&expected, cfg) {
            corpus.comp_negative.push(segment_metrics_for_audio(
                audio,
                event.onset_sec,
                event.offset_sec,
                &decoy,
                cfg,
            )?);
        }
    }

    Ok(corpus)
}

fn select_threshold_for_targets(
    corpus: &CalibrationCorpus,
    weights: &MaspWeightsConfig,
    cent_tol: f32,
    target_mono_recall: f32,
    target_comp_recall: f32,
) -> ThresholdSelection {
    #[derive(Clone, Copy)]
    enum Label {
        MonoPos,
        CompPos,
        Neg,
    }

    let mono_total = corpus.mono_positive.len();
    let comp_total = corpus.comp_positive.len();
    let mut scored = Vec::<(f32, Label)>::with_capacity(
        corpus.mono_positive.len()
            + corpus.comp_positive.len()
            + corpus.mono_negative.len()
            + corpus.comp_negative.len(),
    );

    for metrics in &corpus.mono_positive {
        scored.push((
            weighted_metrics_score(metrics, weights, cent_tol),
            Label::MonoPos,
        ));
    }
    for metrics in &corpus.comp_positive {
        scored.push((
            weighted_metrics_score(metrics, weights, cent_tol),
            Label::CompPos,
        ));
    }
    for metrics in &corpus.mono_negative {
        scored.push((
            weighted_metrics_score(metrics, weights, cent_tol),
            Label::Neg,
        ));
    }
    for metrics in &corpus.comp_negative {
        scored.push((
            weighted_metrics_score(metrics, weights, cent_tol),
            Label::Neg,
        ));
    }

    if scored.is_empty() {
        return ThresholdSelection {
            threshold: 0.0,
            precision: 0.0,
            mono_recall: 0.0,
            comp_recall: 0.0,
            objective: f32::NEG_INFINITY,
        };
    }

    scored.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut mono_tp = 0usize;
    let mut comp_tp = 0usize;
    let mut fp = 0usize;
    let mut best = ThresholdSelection {
        threshold: scored[0].0 + 1e-6,
        precision: 0.0,
        mono_recall: 0.0,
        comp_recall: 0.0,
        objective: f32::NEG_INFINITY,
    };

    let mut idx = 0usize;
    while idx < scored.len() {
        let score = scored[idx].0;
        while idx < scored.len() && scored[idx].0 == score {
            match scored[idx].1 {
                Label::MonoPos => mono_tp += 1,
                Label::CompPos => comp_tp += 1,
                Label::Neg => fp += 1,
            }
            idx += 1;
        }

        let mono_recall = ratio(mono_tp, mono_total);
        let comp_recall = ratio(comp_tp, comp_total);
        let precision = ratio(mono_tp + comp_tp, mono_tp + comp_tp + fp);
        let mono_deficit = (target_mono_recall - mono_recall).max(0.0);
        let comp_deficit = (target_comp_recall - comp_recall).max(0.0);
        let objective = if mono_deficit <= 0.0 && comp_deficit <= 0.0 {
            precision + 0.02 * (mono_recall + comp_recall)
        } else {
            precision - 5.0 * (mono_deficit + comp_deficit)
        };

        if objective > best.objective {
            best = ThresholdSelection {
                threshold: score,
                precision,
                mono_recall,
                comp_recall,
                objective,
            };
        }
    }

    best
}

fn is_better_interval_match(candidate: &EventMetrics, previous: &EventMetrics) -> bool {
    candidate.har > previous.har
        || (candidate.har == previous.har && candidate.cent_error.abs() < previous.cent_error.abs())
        || (candidate.har == previous.har
            && candidate.cent_error.abs() == previous.cent_error.abs()
            && candidate.h > previous.h)
        || (candidate.har == previous.har
            && candidate.cent_error.abs() == previous.cent_error.abs()
            && candidate.h == previous.h
            && candidate.mbw > previous.mbw)
}

fn cent_score(cent_error: f32, tolerance: f32) -> f32 {
    if tolerance <= 0.0 {
        return 0.0;
    }
    (1.0 - cent_error.abs() / tolerance).clamp(0.0, 1.0)
}

fn normalize_weights(weights: &mut MaspWeightsConfig) {
    let sum = (weights.har + weights.mbw + weights.cent + weights.rms).max(EPS);
    weights.har /= sum;
    weights.mbw /= sum;
    weights.cent /= sum;
    weights.rms /= sum;
}

fn smoothing_factor(k: usize, b: f32) -> f32 {
    1.0 / (1.0 + (k as f32).powf(b.max(0.0)))
}

fn aggregate_local_masp(mag: &[f32], start: usize, end: usize, mode: &str) -> f32 {
    if mag.is_empty() {
        return 0.0;
    }
    let s = start.min(mag.len() - 1);
    let e = end.min(mag.len() - 1);
    if s > e {
        return 0.0;
    }

    if mode == "max" {
        mag[s..=e].iter().copied().fold(0.0_f32, f32::max)
    } else {
        let mut sum = 0.0_f32;
        let mut n = 0usize;
        for &x in &mag[s..=e] {
            sum += x;
            n += 1;
        }
        if n == 0 {
            0.0
        } else {
            sum / n as f32
        }
    }
}

fn preprocess_samples_for_masp(
    samples: &[f32],
    sample_rate: u32,
    cfg: &AppConfig,
) -> (Vec<f32>, u32, bool) {
    if cfg.masp.mode == "strict" && sample_rate > cfg.masp.strict_sample_rate {
        (
            resample_linear(samples, sample_rate, cfg.masp.strict_sample_rate),
            cfg.masp.strict_sample_rate,
            true,
        )
    } else {
        (samples.to_vec(), sample_rate, false)
    }
}

fn compute_phase1_frontend_frames_masp(
    samples: &[f32],
    sample_rate: u32,
    plan: &StftPlan,
    cfg: &AppConfig,
) -> Vec<SpectrumFrame> {
    match cfg.frontend.kind.trim().to_ascii_lowercase().as_str() {
        "stft" => compute_stft_frames(samples, sample_rate, plan),
        "qdft" => compute_qdft_frames(samples, sample_rate, plan, cfg),
        _ => compute_stft_frames(samples, sample_rate, plan),
    }
}

fn compute_qdft_frames(
    samples: &[f32],
    sample_rate: u32,
    plan: &StftPlan,
    cfg: &AppConfig,
) -> Vec<SpectrumFrame> {
    if plan.win_length == 0
        || plan.hop_length == 0
        || plan.nfft == 0
        || samples.len() < plan.win_length
    {
        return Vec::new();
    }

    let frame_count = 1 + (samples.len() - plan.win_length) / plan.hop_length;
    let bins = plan.nfft / 2 + 1;
    let hz_per_bin = sample_rate as f64 / plan.nfft as f64;
    let nyquist = (sample_rate as f64 * 0.5).max(1.0);
    let mut min_hz = pitch_to_hz(cfg.pitch.midi_min).max(20.0) as f64;
    if min_hz >= nyquist {
        min_hz = (nyquist * 0.5).max(1.0);
    }
    let max_hz = nyquist.max(min_hz + hz_per_bin);
    let resolution = qdft_resolution_from_target_window(sample_rate, plan.win_length, min_hz);

    let mut qdft = QDFT::<f32, f64>::new(
        sample_rate as f64,
        (min_hz, max_hz),
        resolution,
        0.0,
        Some((0.5, -0.5)),
    );
    let src_bins = qdft.size();
    if src_bins == 0 {
        return Vec::new();
    }

    let mut map_bin0 = vec![0usize; src_bins];
    let mut map_bin1 = vec![0usize; src_bins];
    let mut map_frac = vec![0.0_f32; src_bins];
    for (i, &freq_hz) in qdft.frequencies().iter().enumerate() {
        let bin_f = (freq_hz / hz_per_bin).clamp(0.0, (bins - 1) as f64);
        let bin0 = bin_f.floor() as usize;
        let bin1 = (bin0 + 1).min(bins - 1);
        map_bin0[i] = bin0;
        map_bin1[i] = bin1;
        map_frac[i] = (bin_f - bin0 as f64) as f32;
    }

    let mut ring = vec![0.0_f32; plan.win_length * src_bins];
    let mut src_acc = vec![0.0_f32; src_bins];
    let mut spectrum = vec![Complex64::new(0.0, 0.0); src_bins];
    let mut out = Vec::with_capacity(frame_count);

    for (sample_idx, &sample) in samples.iter().enumerate() {
        qdft.qdft_scalar(&sample, &mut spectrum);

        let ring_base = (sample_idx % plan.win_length) * src_bins;
        for src_idx in 0..src_bins {
            let mag = spectrum[src_idx].norm() as f32;
            let ring_idx = ring_base + src_idx;
            let old = ring[ring_idx];
            ring[ring_idx] = mag;
            src_acc[src_idx] += mag - old;
        }

        if sample_idx + 1 < plan.win_length {
            continue;
        }
        let start = sample_idx + 1 - plan.win_length;
        if start % plan.hop_length != 0 {
            continue;
        }

        let mut mag = vec![0.0_f32; bins];
        let inv_win = 1.0 / plan.win_length as f32;
        for src_idx in 0..src_bins {
            let energy = src_acc[src_idx] * inv_win;
            let bin0 = map_bin0[src_idx];
            let bin1 = map_bin1[src_idx];
            let frac = map_frac[src_idx];
            let left = energy * (1.0 - frac);
            let right = energy * frac;

            mag[bin0] += left;
            if bin1 != bin0 {
                mag[bin1] += right;
            } else {
                mag[bin0] += right;
            }
        }

        out.push(SpectrumFrame {
            frame_index: out.len(),
            time_sec: start as f32 / sample_rate as f32,
            mag,
            white: vec![0.0_f32; bins],
        });

        if out.len() == frame_count {
            break;
        }
    }

    out
}

fn qdft_resolution_from_target_window(sample_rate: u32, win_length: usize, min_hz: f64) -> f64 {
    if sample_rate == 0 || win_length == 0 || !min_hz.is_finite() || min_hz <= 0.0 {
        return 12.0;
    }
    let quality = (min_hz * win_length as f64 / sample_rate as f64).max(1e-6);
    let resolution = 1.0 / (1.0 + 1.0 / quality).log2();
    resolution.clamp(4.0, 96.0)
}

fn build_manifest(cfg: &AppConfig) -> MaspManifest {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let as_toml = toml::to_string(cfg).unwrap_or_else(|_| String::new());
    as_toml.hash(&mut hasher);

    let mut manifest = MaspManifest {
        version: 1,
        config_hash: hasher.finish(),
        model_params: MaspModelParams {
            mode: cfg.masp.mode.clone(),
            strict_sample_rate: cfg.masp.strict_sample_rate,
            bins_per_octave: cfg.masp.bins_per_octave,
            max_harmonics: cfg.masp.max_harmonics,
            b_exponent: cfg.masp.b_exponent,
            cent_tolerance: cfg.masp.cent_tolerance,
            rms_window_ms: cfg.masp.rms_window_ms,
            rms_h_relax: cfg.masp.rms_h_relax,
            pretrain_trials: cfg.masp.pretrain_trials,
        },
        validation_rule: MaspValidationRule {
            score_threshold: cfg.masp.validation_score_threshold,
            score_weights: cfg.masp.weights.clone(),
            target_mono_recall: 0.0,
            target_comp_recall: 0.0,
            calibrated_precision: None,
            calibrated_mono_recall: None,
            calibrated_comp_recall: None,
        },
        mode: cfg.masp.mode.clone(),
        strict_sample_rate: cfg.masp.strict_sample_rate,
        bins_per_octave: cfg.masp.bins_per_octave,
        max_harmonics: cfg.masp.max_harmonics,
        b_exponent: cfg.masp.b_exponent,
        cent_tolerance: cfg.masp.cent_tolerance,
        rms_window_ms: cfg.masp.rms_window_ms,
        rms_h_relax: cfg.masp.rms_h_relax,
        validation_score_threshold: cfg.masp.validation_score_threshold,
        pretrain_trials: cfg.masp.pretrain_trials,
        target_mono_recall: 0.0,
        target_comp_recall: 0.0,
        calibrated_precision: None,
        calibrated_mono_recall: None,
        calibrated_comp_recall: None,
        weights: cfg.masp.weights.clone(),
    };
    manifest.finalize_compat();
    manifest
}

fn load_or_cached_audio<'a>(
    cache: &'a mut BTreeMap<String, AudioBuffer>,
    path: &str,
) -> Result<&'a AudioBuffer> {
    if !cache.contains_key(path) {
        let audio = load_wav_mono(path)
            .with_context(|| format!("failed to load audio during MASP processing: {path}"))?;
        cache.insert(path.to_string(), audio);
    }
    cache
        .get(path)
        .with_context(|| format!("audio cache unexpectedly missing path: {path}"))
}

fn midi_key(midis: &[u8]) -> String {
    midis
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join("+")
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(value)
        .with_context(|| format!("failed to serialize json: {}", path.display()))?;
    std::fs::write(path, json)
        .with_context(|| format!("failed to write json file: {}", path.display()))?;
    Ok(())
}

fn write_jsonl<T: Serialize>(path: &Path, values: &[T]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create jsonl file: {}", path.display()))?;
    for value in values {
        let line = serde_json::to_string(value)
            .with_context(|| format!("failed to serialize jsonl row: {}", path.display()))?;
        writeln!(file, "{line}")
            .with_context(|| format!("failed to write jsonl row: {}", path.display()))?;
    }
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read json file: {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse json file: {}", path.display()))
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open jsonl file: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read jsonl line {} in {}",
                line_idx + 1,
                path.display()
            )
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "failed to parse jsonl line {} in {}",
                line_idx + 1,
                path.display()
            )
        })?;
        out.push(row);
    }

    Ok(out)
}

fn file_id_for_path(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or(path)
        .to_string()
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

#[cfg(test)]
mod tests {
    use super::{
        cent_score, compute_cent_error, compute_har, compute_harmonicity_h, compute_mbw,
        log_shifted_bin_index, preprocess_samples_for_masp, score_masp_midi_frame,
        smoothing_factor, weighted_metrics_score, EventMetrics,
    };
    use crate::config::{AppConfig, MaspWeightsConfig};
    use crate::fretboard::pitch_to_hz;
    use crate::harmonic_map::build_harmonic_maps;
    use crate::types::{HarmonicBinRange, MidiHarmonicMap};
    use serde::Deserialize;

    const CONSISTENCY_FIXTURE_JSON: &str = include_str!("../ts/masp/consistencyFixtures.json");

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TsMaspFixture {
        params: TsMaspParams,
        smoothing_factor: Vec<TsSmoothingCase>,
        cent_score: Vec<TsCentScoreCase>,
        frame_scoring: TsFrameScoringCase,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TsMaspParams {
        mode: String,
        strict_sample_rate: u32,
        bins_per_octave: usize,
        max_harmonics: usize,
        b_exponent: f32,
        cent_tolerance: f32,
        rms_window_ms: u32,
        rms_h_relax: f32,
        score_threshold: f32,
        score_weights: TsWeights,
        local_aggregation: String,
        midi_min: u8,
    }

    #[derive(Debug, Deserialize)]
    struct TsWeights {
        har: f32,
        mbw: f32,
        cent: f32,
        rms: f32,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TsSmoothingCase {
        k: usize,
        b_exponent: f32,
        expected: f32,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TsCentScoreCase {
        cent_error: f32,
        tolerance: f32,
        expected: f32,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TsFrameScoringCase {
        mag: Vec<f32>,
        maps: Vec<TsMidiMap>,
        midi_min: u8,
        sample_rate: u32,
        nfft: usize,
        expected_midis: Vec<u8>,
        pitch_spectrum: Vec<f32>,
        metrics: TsFrameMetrics,
        weighted_metrics_score: f32,
        validation_decision: TsValidationDecision,
    }

    #[derive(Debug, Deserialize)]
    struct TsMidiMap {
        midi: u8,
        ranges: Vec<TsRange>,
    }

    #[derive(Debug, Deserialize)]
    struct TsRange {
        start: usize,
        end: usize,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TsFrameMetrics {
        h: f32,
        har: f32,
        mbw: f32,
        cent_error: f32,
        noise_rms: f32,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TsValidationDecision {
        cent_gate: bool,
        h_gate: bool,
        h_dynamic_threshold: f32,
        weighted_score: f32,
        pass: bool,
        reason: String,
    }

    fn load_ts_fixture() -> TsMaspFixture {
        serde_json::from_str(CONSISTENCY_FIXTURE_JSON).expect("valid TS MASP consistency fixture")
    }

    fn build_cfg_from_ts_params(params: &TsMaspParams) -> AppConfig {
        let mut cfg = AppConfig::default();
        cfg.masp.mode = params.mode.clone();
        cfg.masp.strict_sample_rate = params.strict_sample_rate;
        cfg.masp.bins_per_octave = params.bins_per_octave;
        cfg.masp.max_harmonics = params.max_harmonics;
        cfg.masp.b_exponent = params.b_exponent;
        cfg.masp.cent_tolerance = params.cent_tolerance;
        cfg.masp.rms_window_ms = params.rms_window_ms;
        cfg.masp.rms_h_relax = params.rms_h_relax;
        cfg.masp.validation_score_threshold = params.score_threshold;
        cfg.masp.weights = MaspWeightsConfig {
            har: params.score_weights.har,
            mbw: params.score_weights.mbw,
            cent: params.score_weights.cent,
            rms: params.score_weights.rms,
        };
        cfg.pitch.midi_min = params.midi_min;
        cfg.pitch.local_aggregation = params.local_aggregation.clone();
        cfg
    }

    fn build_maps_from_fixture(maps: &[TsMidiMap]) -> Vec<MidiHarmonicMap> {
        maps.iter()
            .map(|map| MidiHarmonicMap {
                midi: map.midi,
                f0_hz: pitch_to_hz(map.midi),
                ranges: map
                    .ranges
                    .iter()
                    .map(|range| HarmonicBinRange {
                        start: range.start,
                        end: range.end,
                    })
                    .collect(),
            })
            .collect()
    }

    fn assert_close(actual: f32, expected: f32, eps: f32, label: &str) {
        assert!(
            (actual - expected).abs() <= eps,
            "{label} mismatch: got {actual}, expected {expected}"
        );
    }

    #[test]
    fn smoothing_factor_decreases_with_harmonic_order() {
        let a1 = smoothing_factor(1, 0.25);
        let a4 = smoothing_factor(4, 0.25);
        assert!(a1 > a4);
    }

    #[test]
    fn log_shift_index_moves_up_for_positive_shift() {
        let idx = log_shifted_bin_index(10, 0.25, 36, 100);
        assert!(idx > 10);
    }

    #[test]
    fn strict_mode_downsamples_when_needed() {
        let mut cfg = AppConfig::default();
        cfg.masp.mode = "strict".to_string();
        cfg.masp.strict_sample_rate = 22_050;
        let x = vec![0.0_f32; 44_100];
        let (_y, sr, downsampled) = preprocess_samples_for_masp(&x, 44_100, &cfg);
        assert!(downsampled);
        assert_eq!(sr, 22_050);
    }

    #[test]
    fn cent_gate_boundary_scores() {
        assert!((cent_score(49.0, 50.0) - 0.02).abs() < 1e-6);
        assert_eq!(cent_score(50.0, 50.0), 0.0);
        assert_eq!(cent_score(51.0, 50.0), 0.0);
    }

    #[test]
    fn masp_scoring_produces_positive_energy_when_harmonics_present() {
        let cfg = AppConfig::default();
        let maps = build_harmonic_maps(8_000, 1024, 69, 69, cfg.masp.max_harmonics, 1);
        let mut mag = vec![0.0_f32; 1024 / 2 + 1];
        for range in &maps[0].ranges {
            mag[(range.start + range.end) / 2] = 1.0;
        }
        let scores = score_masp_midi_frame(&mag, &maps, &cfg);
        assert!(scores[0] > 0.0);
    }

    #[test]
    fn ts_consistency_fixture_matches_rust_masp_math() {
        let fixture = load_ts_fixture();
        let cfg = build_cfg_from_ts_params(&fixture.params);
        let frame = &fixture.frame_scoring;
        let maps = build_maps_from_fixture(&frame.maps);

        for case in &fixture.smoothing_factor {
            assert_close(
                smoothing_factor(case.k, case.b_exponent),
                case.expected,
                1e-6,
                "smoothing_factor",
            );
        }

        for case in &fixture.cent_score {
            assert_close(
                cent_score(case.cent_error, case.tolerance),
                case.expected,
                1e-6,
                "cent_score",
            );
        }

        let pitch_spectrum = score_masp_midi_frame(&frame.mag, &maps, &cfg);
        assert_eq!(pitch_spectrum.len(), frame.pitch_spectrum.len());
        for (idx, (actual, expected)) in pitch_spectrum
            .iter()
            .zip(frame.pitch_spectrum.iter())
            .enumerate()
        {
            assert_close(*actual, *expected, 1e-6, &format!("pitch_spectrum[{idx}]"));
        }

        let h = compute_harmonicity_h(
            &frame.mag,
            &pitch_spectrum,
            frame.sample_rate,
            frame.nfft,
            &cfg,
        );
        assert_close(h, frame.metrics.h, 1e-6, "compute_harmonicity_h");

        let har = compute_har(&pitch_spectrum, &frame.expected_midis, frame.midi_min);
        assert_close(har, frame.metrics.har, 1e-6, "compute_har");

        let mbw = compute_mbw(&frame.mag, &frame.expected_midis, &maps);
        assert_close(mbw, frame.metrics.mbw, 1e-6, "compute_mbw");

        let cent_error = compute_cent_error(&pitch_spectrum, &frame.expected_midis, frame.midi_min);
        assert_close(
            cent_error,
            frame.metrics.cent_error,
            1e-6,
            "compute_cent_error",
        );

        let metrics = EventMetrics {
            h,
            har,
            mbw,
            cent_error,
            pitch_spectrum: pitch_spectrum.clone(),
            noise_rms: frame.metrics.noise_rms,
        };
        let weighted_score =
            weighted_metrics_score(&metrics, &cfg.masp.weights, cfg.masp.cent_tolerance);
        assert_close(
            weighted_score,
            frame.weighted_metrics_score,
            1e-6,
            "weighted_metrics_score",
        );

        let h_target = 1.75_f32;
        let cent_gate = metrics.cent_error.abs() <= cfg.masp.cent_tolerance;
        let h_dynamic_threshold =
            (h_target * (1.0 - cfg.masp.rms_h_relax * metrics.noise_rms.clamp(0.0, 1.0))).max(0.0);
        let h_gate = metrics.h >= h_dynamic_threshold;
        let pass = weighted_score >= cfg.masp.validation_score_threshold;
        let reason = if pass {
            if cent_gate && h_gate {
                "validated".to_string()
            } else if !cent_gate && !h_gate {
                "validated_soft_score_without_cent_or_h_gate".to_string()
            } else if !cent_gate {
                "validated_soft_score_without_cent_gate".to_string()
            } else {
                "validated_soft_score_without_h_gate".to_string()
            }
        } else if !cent_gate {
            "cent_gate_failed".to_string()
        } else if !h_gate {
            "harmonicity_threshold_failed".to_string()
        } else {
            "weighted_score_failed".to_string()
        };

        assert_eq!(cent_gate, frame.validation_decision.cent_gate);
        assert_eq!(h_gate, frame.validation_decision.h_gate);
        assert_close(
            h_dynamic_threshold,
            frame.validation_decision.h_dynamic_threshold,
            1e-6,
            "h_dynamic_threshold",
        );
        assert_close(
            weighted_score,
            frame.validation_decision.weighted_score,
            1e-6,
            "validation_weighted_score",
        );
        assert_eq!(pass, frame.validation_decision.pass);
        assert_eq!(reason, frame.validation_decision.reason);
    }
}
