use std::collections::{BTreeMap, BTreeSet};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::audio::{load_wav_mono, trim_audio_region};
use crate::config::AppConfig;
use crate::dataset::{filename_contains_comp, filename_contains_solo, PolyEvalEvent};
use crate::decode::{
    decode_global_solutions, normalize_and_combine_candidates, prune_top_k_candidates,
    DecodedSolution,
};
use crate::features::{extract_note_features, flatten_note_features, harmonic_map_for_midi};
use crate::harmonic_map::build_harmonic_maps;
use crate::nms::select_pitch_candidates;
use crate::pitch::{harmonic_weights, score_pitch_frame};
use crate::stft::{build_stft_plan, compute_stft_frames};
use crate::templates::{score_string_fret_candidates, GuitarTemplates};
use crate::types::{AudioBuffer, CandidatePosition, PolyEvent, SpectrumFrame};
use crate::whitening::whiten_log_spectrum;

const NOTE_CONTEXT_MARGIN_SEC: f32 = 0.005;
const HIGH_CONFIDENCE_MARGIN: f32 = 0.25;
const MAX_ALT_SOLUTIONS_IN_EVENT: usize = 5;
const ONSET_POOL_WINDOW_SEC: f32 = 0.08;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetMetrics {
    pub true_positive_count: usize,
    pub false_positive_count: usize,
    pub false_negative_count: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullAggregateMetrics {
    pub interpretation: String,
    pub total_events: usize,
    pub evaluated_events: usize,
    pub skipped_events: usize,
    pub comp_filename_events: usize,
    pub solo_filename_events: usize,
    pub other_filename_events: usize,
    pub pitch: SetMetrics,
    pub pitch_string: SetMetrics,
    pub pitch_string_fret: SetMetrics,
    pub exact_configuration_accuracy: f32,
    #[serde(default)]
    pub decode_top1_hit_count: usize,
    #[serde(default)]
    pub decode_top3_hit_count: usize,
    #[serde(default)]
    pub decode_top5_hit_count: usize,
    #[serde(default)]
    pub decode_top1_hit_rate: f32,
    #[serde(default)]
    pub decode_top3_hit_rate: f32,
    #[serde(default)]
    pub decode_top5_hit_rate: f32,
    #[serde(default)]
    pub avg_candidate_positions_per_pitch: f32,
    #[serde(default)]
    pub avg_global_configurations_explored: f32,
    pub failure_pitch_correct_string_wrong_count: usize,
    pub failure_string_correct_fret_wrong_count: usize,
    #[serde(default)]
    pub failure_full_wrong_despite_correct_pitch_count: usize,
    pub failure_full_decode_wrong_high_confidence_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullPerFileDiagnostic {
    pub file_id: String,
    pub evaluated_events: usize,
    pub skipped_events: usize,
    pub pitch_f1: f32,
    pub pitch_string_f1: f32,
    pub pitch_string_fret_f1: f32,
    pub exact_configuration_accuracy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionTriplet {
    pub midi: u8,
    pub string: u8,
    pub fret: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchCandidateDiagnostics {
    pub rank: usize,
    pub midi: u8,
    pub string: u8,
    pub fret: u8,
    pub pitch_score: f32,
    pub template_score: f32,
    pub pitch_score_norm: f32,
    pub template_score_norm: f32,
    pub combined_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerPitchCandidatesDiagnostics {
    pub midi: u8,
    pub pitch_score: f32,
    pub pitch_score_norm: f32,
    pub top_candidates: Vec<PitchCandidateDiagnostics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedSolutionDiagnostics {
    pub rank: usize,
    pub total_score: f32,
    pub sum_combined_score: f32,
    pub conflict_count: usize,
    pub fret_spread: f32,
    pub position_prior: f32,
    pub conflict_penalty: f32,
    pub fret_spread_penalty: f32,
    pub position_prior_penalty: f32,
    pub positions: Vec<PositionTriplet>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullExampleDiagnostic {
    pub example_id: String,
    pub file_id: String,
    pub filename_split: String,
    pub onset_sec: f32,
    pub offset_sec: f32,
    pub representative_frame_index: usize,
    pub representative_frame_time_sec: f32,
    pub ground_truth_pitches: Vec<u8>,
    pub predicted_pitches: Vec<u8>,
    pub ground_truth_positions: Vec<PositionTriplet>,
    pub predicted_positions: Vec<PositionTriplet>,
    pub top_k_candidates_per_pitch: Vec<PerPitchCandidatesDiagnostics>,
    pub alternative_solutions: Vec<DecodedSolutionDiagnostics>,
    pub best_total_score: Option<f32>,
    pub confidence_margin_top1_top2: Option<f32>,
    #[serde(default)]
    pub decode_hit_top1: bool,
    #[serde(default)]
    pub decode_hit_top3: bool,
    #[serde(default)]
    pub decode_hit_top5: bool,
    #[serde(default)]
    pub avg_candidate_positions_per_pitch: f32,
    #[serde(default)]
    pub global_configurations_explored: usize,
    #[serde(default)]
    pub global_configurations_returned: usize,
    pub pitch_tp: usize,
    pub pitch_fp: usize,
    pub pitch_fn: usize,
    pub pitch_string_tp: usize,
    pub pitch_string_fp: usize,
    pub pitch_string_fn: usize,
    pub pitch_string_fret_tp: usize,
    pub pitch_string_fret_fp: usize,
    pub pitch_string_fret_fn: usize,
    pub exact_configuration_hit: bool,
    pub failure_pitch_correct_string_wrong: bool,
    pub failure_string_correct_fret_wrong: bool,
    #[serde(default)]
    pub failure_full_wrong_despite_correct_pitch: bool,
    pub failure_full_decode_wrong_high_confidence: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullFailureCases {
    pub pitch_correct_string_wrong: Vec<FullExampleDiagnostic>,
    pub string_correct_fret_wrong: Vec<FullExampleDiagnostic>,
    pub full_decode_wrong_high_confidence: Vec<FullExampleDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullDiagnosticsArtifact {
    pub metrics: FullAggregateMetrics,
    pub per_file: Vec<FullPerFileDiagnostic>,
    pub examples: Vec<FullExampleDiagnostic>,
    pub failures: FullFailureCases,
}

#[derive(Debug, Clone, Default)]
struct EventCounts {
    pitch_tp: usize,
    pitch_fp: usize,
    pitch_fn: usize,
    pitch_string_tp: usize,
    pitch_string_fp: usize,
    pitch_string_fn: usize,
    pitch_string_fret_tp: usize,
    pitch_string_fret_fp: usize,
    pitch_string_fret_fn: usize,
    exact_hits: usize,
}

#[derive(Debug, Clone, Default)]
struct FileAccumulator {
    evaluated_events: usize,
    skipped_events: usize,
    counts: EventCounts,
}

#[derive(Debug, Clone)]
struct FullEventResult {
    diagnostic: FullExampleDiagnostic,
    counts: EventCounts,
}

#[derive(Debug, Clone)]
struct PooledPitchSummary {
    midi_scores: Vec<f32>,
    best_frame_by_midi: Vec<usize>,
    representative_frame_index: usize,
}

pub fn evaluate_full_dataset(
    dataset: &[PolyEvalEvent],
    templates: &GuitarTemplates,
    cfg: &AppConfig,
) -> Result<FullDiagnosticsArtifact> {
    let stft_plan = build_stft_plan(&cfg.frontend);
    let maps = build_harmonic_maps(
        cfg.sample_rate_target,
        stft_plan.nfft,
        cfg.pitch.midi_min,
        cfg.pitch.midi_max,
        cfg.templates.num_harmonics.max(cfg.pitch.max_harmonics),
        cfg.pitch.local_bandwidth_bins,
    );
    let hweights = harmonic_weights(cfg.pitch.max_harmonics, cfg.pitch.harmonic_alpha);

    let mut grouped: BTreeMap<String, Vec<PolyEvalEvent>> = BTreeMap::new();
    for event in dataset {
        grouped
            .entry(event.audio_path.clone())
            .or_default()
            .push(event.clone());
    }
    for events in grouped.values_mut() {
        events.sort_by(|a, b| {
            a.onset_sec
                .total_cmp(&b.onset_sec)
                .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
        });
    }

    let mut examples = Vec::new();
    let mut per_file = Vec::new();
    let mut global_counts = EventCounts::default();

    for (audio_path, events) in grouped {
        let file_id = file_id_for_path(&audio_path);
        let mut file_acc = FileAccumulator::default();

        let audio = load_wav_mono(&audio_path).with_context(|| {
            format!("failed to load audio during full evaluation: {audio_path}")
        })?;
        if audio.sample_rate != cfg.sample_rate_target {
            file_acc.skipped_events += events.len();
            per_file.push(FullPerFileDiagnostic {
                file_id,
                evaluated_events: 0,
                skipped_events: events.len(),
                pitch_f1: 0.0,
                pitch_string_f1: 0.0,
                pitch_string_fret_f1: 0.0,
                exact_configuration_accuracy: 0.0,
            });
            continue;
        }

        for (idx, event) in events.iter().enumerate() {
            let Some(result) = evaluate_full_event(
                &audio,
                event,
                templates,
                cfg,
                &stft_plan,
                &maps,
                &hweights,
                &file_id,
                idx + 1,
            )?
            else {
                file_acc.skipped_events += 1;
                continue;
            };

            file_acc.evaluated_events += 1;
            accumulate_counts(&mut file_acc.counts, &result.counts);
            accumulate_counts(&mut global_counts, &result.counts);
            examples.push(result.diagnostic);
        }

        per_file.push(FullPerFileDiagnostic {
            file_id,
            evaluated_events: file_acc.evaluated_events,
            skipped_events: file_acc.skipped_events,
            pitch_f1: counts_to_metrics(
                file_acc.counts.pitch_tp,
                file_acc.counts.pitch_fp,
                file_acc.counts.pitch_fn,
            )
            .f1,
            pitch_string_f1: counts_to_metrics(
                file_acc.counts.pitch_string_tp,
                file_acc.counts.pitch_string_fp,
                file_acc.counts.pitch_string_fn,
            )
            .f1,
            pitch_string_fret_f1: counts_to_metrics(
                file_acc.counts.pitch_string_fret_tp,
                file_acc.counts.pitch_string_fret_fp,
                file_acc.counts.pitch_string_fret_fn,
            )
            .f1,
            exact_configuration_accuracy: ratio(
                file_acc.counts.exact_hits,
                file_acc.evaluated_events,
            ),
        });
    }

    let comp_filename_events = dataset
        .iter()
        .filter(|e| filename_contains_comp(&e.audio_path))
        .count();
    let solo_filename_events = dataset
        .iter()
        .filter(|e| filename_contains_solo(&e.audio_path))
        .count();
    let other_filename_events = dataset
        .len()
        .saturating_sub(comp_filename_events + solo_filename_events);

    let failure_pitch_correct_string_wrong = examples
        .iter()
        .filter(|x| x.failure_pitch_correct_string_wrong)
        .count();
    let failure_string_correct_fret_wrong = examples
        .iter()
        .filter(|x| x.failure_string_correct_fret_wrong)
        .count();
    let failure_full_wrong_despite_correct_pitch = examples
        .iter()
        .filter(|x| x.failure_full_wrong_despite_correct_pitch)
        .count();
    let failure_full_decode_wrong_high_confidence = examples
        .iter()
        .filter(|x| x.failure_full_decode_wrong_high_confidence)
        .count();
    let decode_top1_hit_count = examples.iter().filter(|x| x.decode_hit_top1).count();
    let decode_top3_hit_count = examples.iter().filter(|x| x.decode_hit_top3).count();
    let decode_top5_hit_count = examples.iter().filter(|x| x.decode_hit_top5).count();
    let avg_candidate_positions_per_pitch = if examples.is_empty() {
        0.0
    } else {
        examples
            .iter()
            .map(|x| x.avg_candidate_positions_per_pitch)
            .sum::<f32>()
            / examples.len() as f32
    };
    let avg_global_configurations_explored = if examples.is_empty() {
        0.0
    } else {
        examples
            .iter()
            .map(|x| x.global_configurations_explored as f32)
            .sum::<f32>()
            / examples.len() as f32
    };

    let metrics = FullAggregateMetrics {
        interpretation: "Event-based polyphonic evaluation with set-based matching. Candidate ranking uses normalized pitch+template combined score; global decoding maximizes sum(combined_score)-penalties.".to_string(),
        total_events: dataset.len(),
        evaluated_events: examples.len(),
        skipped_events: dataset.len().saturating_sub(examples.len()),
        comp_filename_events,
        solo_filename_events,
        other_filename_events,
        pitch: counts_to_metrics(
            global_counts.pitch_tp,
            global_counts.pitch_fp,
            global_counts.pitch_fn,
        ),
        pitch_string: counts_to_metrics(
            global_counts.pitch_string_tp,
            global_counts.pitch_string_fp,
            global_counts.pitch_string_fn,
        ),
        pitch_string_fret: counts_to_metrics(
            global_counts.pitch_string_fret_tp,
            global_counts.pitch_string_fret_fp,
            global_counts.pitch_string_fret_fn,
        ),
        exact_configuration_accuracy: ratio(global_counts.exact_hits, examples.len()),
        decode_top1_hit_count,
        decode_top3_hit_count,
        decode_top5_hit_count,
        decode_top1_hit_rate: ratio(decode_top1_hit_count, examples.len()),
        decode_top3_hit_rate: ratio(decode_top3_hit_count, examples.len()),
        decode_top5_hit_rate: ratio(decode_top5_hit_count, examples.len()),
        avg_candidate_positions_per_pitch,
        avg_global_configurations_explored,
        failure_pitch_correct_string_wrong_count: failure_pitch_correct_string_wrong,
        failure_string_correct_fret_wrong_count: failure_string_correct_fret_wrong,
        failure_full_wrong_despite_correct_pitch_count: failure_full_wrong_despite_correct_pitch,
        failure_full_decode_wrong_high_confidence_count: failure_full_decode_wrong_high_confidence,
    };

    let failures = FullFailureCases {
        pitch_correct_string_wrong: examples
            .iter()
            .filter(|x| x.failure_pitch_correct_string_wrong)
            .take(200)
            .cloned()
            .collect(),
        string_correct_fret_wrong: examples
            .iter()
            .filter(|x| x.failure_string_correct_fret_wrong)
            .take(200)
            .cloned()
            .collect(),
        full_decode_wrong_high_confidence: examples
            .iter()
            .filter(|x| x.failure_full_decode_wrong_high_confidence)
            .take(200)
            .cloned()
            .collect(),
    };

    Ok(FullDiagnosticsArtifact {
        metrics,
        per_file,
        examples,
        failures,
    })
}

pub fn write_full_artifacts(debug_dir: &Path, artifact: &FullDiagnosticsArtifact) -> Result<()> {
    create_dir_all(debug_dir)
        .with_context(|| format!("failed to create debug directory: {}", debug_dir.display()))?;

    write_json_pretty(
        &debug_dir.join("phase3_full_metrics.json"),
        &artifact.metrics,
    )?;
    write_json_pretty(
        &debug_dir.join("phase3_full_examples.json"),
        &artifact.examples,
    )?;
    write_json_pretty(
        &debug_dir.join("phase3_failure_cases.json"),
        &artifact.failures,
    )?;
    write_phase3_per_file_csv(&debug_dir.join("phase3_per_file.csv"), &artifact.per_file)?;
    write_phase3_examples_csv(
        &debug_dir.join("phase3_examples_topk.csv"),
        &artifact.examples,
    )?;

    let examples_dir = debug_dir.join("phase3_poly_scores");
    create_dir_all(&examples_dir).with_context(|| {
        format!(
            "failed to create phase3 example directory: {}",
            examples_dir.display()
        )
    })?;

    for example in &artifact.examples {
        write_json_pretty(
            &examples_dir.join(format!("{}.json", example.example_id)),
            example,
        )?;
    }

    Ok(())
}

fn evaluate_full_event(
    audio: &AudioBuffer,
    event: &PolyEvalEvent,
    templates: &GuitarTemplates,
    cfg: &AppConfig,
    stft_plan: &crate::stft::StftPlan,
    maps: &[crate::types::MidiHarmonicMap],
    hweights: &[f32],
    file_id: &str,
    index: usize,
) -> Result<Option<FullEventResult>> {
    if event.notes.is_empty() {
        return Ok(None);
    }

    let start_sec = (event.onset_sec - NOTE_CONTEXT_MARGIN_SEC).max(0.0);
    let end_sec = (event.offset_sec + NOTE_CONTEXT_MARGIN_SEC).max(start_sec);
    let mut segment = trim_audio_region(&audio.samples, audio.sample_rate, start_sec, end_sec);
    if segment.is_empty() {
        return Ok(None);
    }

    if segment.len() < stft_plan.win_length {
        segment.resize(stft_plan.win_length, 0.0);
    }

    let mut frames = compute_stft_frames(&segment, audio.sample_rate, stft_plan);
    if frames.is_empty() {
        return Ok(None);
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
    let Some(pooled) = pooled_pitch_scores_onset_max(&frames, onset_time, maps, cfg, hweights)
    else {
        return Ok(None);
    };
    let rep_idx = pooled.representative_frame_index;
    let Some(rep_frame) = frames.get(rep_idx) else {
        return Ok(None);
    };
    let detected_pitches =
        select_pitch_candidates(&pooled.midi_scores, cfg.pitch.midi_min, &cfg.nms);

    let mut per_pitch_candidates = Vec::<Vec<CandidatePosition>>::new();
    for (midi, pitch_score) in detected_pitches {
        let midi_idx = usize::from(midi.saturating_sub(cfg.pitch.midi_min));
        let midi_frame_idx = pooled
            .best_frame_by_midi
            .get(midi_idx)
            .copied()
            .unwrap_or(rep_idx);
        let feature_frame = frames.get(midi_frame_idx).unwrap_or(rep_frame);
        let previous_white = if midi_frame_idx > 0 {
            frames.get(midi_frame_idx - 1).map(|f| f.white.as_slice())
        } else {
            None
        };
        let Some(map) = harmonic_map_for_midi(maps, midi) else {
            continue;
        };
        let feat = extract_note_features(
            &feature_frame.white,
            previous_white,
            midi,
            map,
            audio.sample_rate,
            stft_plan.nfft,
            &cfg.templates,
        );
        let z = flatten_note_features(&feat, &cfg.templates);
        let mut candidates = score_string_fret_candidates(midi, &z, templates, cfg);
        for candidate in candidates.iter_mut() {
            candidate.pitch_score = pitch_score;
        }
        per_pitch_candidates.push(candidates);
    }

    normalize_and_combine_candidates(&mut per_pitch_candidates, &cfg.decode);
    prune_top_k_candidates(
        &mut per_pitch_candidates,
        cfg.decode.max_candidates_per_pitch,
    );

    let all_solutions = if per_pitch_candidates.is_empty() {
        Vec::new()
    } else {
        decode_global_solutions(
            &per_pitch_candidates,
            &cfg.decode,
            cfg.decode.max_global_solutions,
        )
    };
    let global_configurations_explored = count_global_configurations(&per_pitch_candidates);
    let global_configurations_returned = all_solutions.len();

    let best_solution = all_solutions.first();
    let top_margin = if all_solutions.len() >= 2 {
        Some(all_solutions[0].total_score - all_solutions[1].total_score)
    } else {
        None
    };

    let predicted_positions = best_solution
        .map(|x| positions_to_triplets(&x.positions))
        .unwrap_or_default();
    let predicted_pitch_set = predicted_positions
        .iter()
        .map(|x| x.midi)
        .collect::<BTreeSet<_>>();
    let predicted_string_set = predicted_positions
        .iter()
        .map(|x| (x.midi, x.string))
        .collect::<BTreeSet<_>>();
    let predicted_fret_set = predicted_positions
        .iter()
        .map(|x| (x.midi, x.string, x.fret))
        .collect::<BTreeSet<_>>();

    let gt_positions = ground_truth_positions(&event.notes);
    let gt_pitch_set = gt_positions.iter().map(|x| x.midi).collect::<BTreeSet<_>>();
    let gt_string_set = gt_positions
        .iter()
        .map(|x| (x.midi, x.string))
        .collect::<BTreeSet<_>>();
    let gt_fret_set = gt_positions
        .iter()
        .map(|x| (x.midi, x.string, x.fret))
        .collect::<BTreeSet<_>>();

    let decode_hit_top1 = decode_hit_top_k(&all_solutions, &gt_fret_set, 1);
    let decode_hit_top3 = decode_hit_top_k(&all_solutions, &gt_fret_set, 3);
    let decode_hit_top5 = decode_hit_top_k(&all_solutions, &gt_fret_set, 5);

    let (pitch_tp, pitch_fp, pitch_fn) = set_counts(&predicted_pitch_set, &gt_pitch_set);
    let (pitch_string_tp, pitch_string_fp, pitch_string_fn) =
        set_counts(&predicted_string_set, &gt_string_set);
    let (pitch_string_fret_tp, pitch_string_fret_fp, pitch_string_fret_fn) =
        set_counts(&predicted_fret_set, &gt_fret_set);

    let exact_configuration_hit = predicted_fret_set == gt_fret_set;

    let failure_pitch_correct_string_wrong =
        predicted_pitch_set == gt_pitch_set && predicted_string_set != gt_string_set;
    let failure_string_correct_fret_wrong =
        predicted_string_set == gt_string_set && predicted_fret_set != gt_fret_set;
    let failure_full_wrong_despite_correct_pitch =
        predicted_pitch_set == gt_pitch_set && !exact_configuration_hit;
    let failure_full_decode_wrong_high_confidence =
        !exact_configuration_hit && top_margin.unwrap_or(0.0) >= HIGH_CONFIDENCE_MARGIN;
    let avg_candidate_positions_per_pitch = if per_pitch_candidates.is_empty() {
        0.0
    } else {
        per_pitch_candidates
            .iter()
            .map(|c| c.len() as f32)
            .sum::<f32>()
            / per_pitch_candidates.len() as f32
    };

    let top_k_candidates_per_pitch = per_pitch_candidates
        .iter()
        .filter_map(|candidates| {
            let top = candidates.first()?;
            Some(PerPitchCandidatesDiagnostics {
                midi: top.midi,
                pitch_score: top.pitch_score,
                pitch_score_norm: top.pitch_score_norm,
                top_candidates: candidates
                    .iter()
                    .enumerate()
                    .map(|(rank_idx, candidate)| PitchCandidateDiagnostics {
                        rank: rank_idx + 1,
                        midi: candidate.midi,
                        string: candidate.string,
                        fret: candidate.fret,
                        pitch_score: candidate.pitch_score,
                        template_score: candidate.template_score,
                        pitch_score_norm: candidate.pitch_score_norm,
                        template_score_norm: candidate.template_score_norm,
                        combined_score: candidate.combined_score,
                    })
                    .collect(),
            })
        })
        .collect::<Vec<_>>();

    let alternative_solutions = all_solutions
        .iter()
        .take(MAX_ALT_SOLUTIONS_IN_EVENT)
        .enumerate()
        .map(|(rank_idx, s)| to_solution_diag(rank_idx + 1, s))
        .collect::<Vec<_>>();

    let diagnostic = FullExampleDiagnostic {
        example_id: format!("{}_{}", sanitize_for_id(file_id), index),
        file_id: file_id.to_string(),
        filename_split: filename_split(&event.audio_path),
        onset_sec: event.onset_sec,
        offset_sec: event.offset_sec,
        representative_frame_index: rep_idx,
        representative_frame_time_sec: rep_frame.time_sec,
        ground_truth_pitches: gt_pitch_set.iter().copied().collect(),
        predicted_pitches: predicted_pitch_set.iter().copied().collect(),
        ground_truth_positions: gt_positions,
        predicted_positions,
        top_k_candidates_per_pitch,
        alternative_solutions,
        best_total_score: best_solution.map(|x| x.total_score),
        confidence_margin_top1_top2: top_margin,
        decode_hit_top1,
        decode_hit_top3,
        decode_hit_top5,
        avg_candidate_positions_per_pitch,
        global_configurations_explored,
        global_configurations_returned,
        pitch_tp,
        pitch_fp,
        pitch_fn,
        pitch_string_tp,
        pitch_string_fp,
        pitch_string_fn,
        pitch_string_fret_tp,
        pitch_string_fret_fp,
        pitch_string_fret_fn,
        exact_configuration_hit,
        failure_pitch_correct_string_wrong,
        failure_string_correct_fret_wrong,
        failure_full_wrong_despite_correct_pitch,
        failure_full_decode_wrong_high_confidence,
    };

    let counts = EventCounts {
        pitch_tp,
        pitch_fp,
        pitch_fn,
        pitch_string_tp,
        pitch_string_fp,
        pitch_string_fn,
        pitch_string_fret_tp,
        pitch_string_fret_fp,
        pitch_string_fret_fn,
        exact_hits: usize::from(exact_configuration_hit),
    };

    Ok(Some(FullEventResult { diagnostic, counts }))
}

fn filename_split(path: &str) -> String {
    if filename_contains_comp(path) {
        "comp".to_string()
    } else if filename_contains_solo(path) {
        "solo".to_string()
    } else {
        "other".to_string()
    }
}

fn pooled_pitch_scores_onset_max(
    frames: &[SpectrumFrame],
    onset_time_sec: f32,
    maps: &[crate::types::MidiHarmonicMap],
    cfg: &AppConfig,
    hweights: &[f32],
) -> Option<PooledPitchSummary> {
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

    let mut pooled_scores: Option<Vec<f32>> = None;
    let mut best_frame_by_midi: Option<Vec<usize>> = None;
    let mut representative_frame_index = indices[0];
    let mut representative_frame_strength = f32::NEG_INFINITY;

    for idx in indices {
        let frame = frames.get(idx)?;
        let pitch_frame = score_pitch_frame(&frame.white, maps, &cfg.pitch, hweights);
        let frame_strength = pitch_frame.midi_scores.iter().sum::<f32>();
        if frame_strength > representative_frame_strength {
            representative_frame_strength = frame_strength;
            representative_frame_index = idx;
        }

        if let Some(acc) = pooled_scores.as_mut() {
            if acc.len() != pitch_frame.midi_scores.len() {
                return None;
            }
            let best = best_frame_by_midi.as_mut()?;
            for midi_idx in 0..acc.len() {
                let score = pitch_frame.midi_scores[midi_idx];
                if score > acc[midi_idx] {
                    acc[midi_idx] = score;
                    best[midi_idx] = idx;
                }
            }
        } else {
            pooled_scores = Some(pitch_frame.midi_scores.clone());
            best_frame_by_midi = Some(vec![idx; pitch_frame.midi_scores.len()]);
        }
    }

    Some(PooledPitchSummary {
        midi_scores: pooled_scores?,
        best_frame_by_midi: best_frame_by_midi?,
        representative_frame_index,
    })
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

fn ground_truth_positions(notes: &[PolyEvent]) -> Vec<PositionTriplet> {
    notes
        .iter()
        .filter_map(|note| {
            let string = note.string?;
            let fret = note.fret?;
            Some(PositionTriplet {
                midi: note.midi,
                string,
                fret,
            })
        })
        .collect::<Vec<_>>()
}

fn positions_to_triplets(positions: &[CandidatePosition]) -> Vec<PositionTriplet> {
    positions
        .iter()
        .map(|x| PositionTriplet {
            midi: x.midi,
            string: x.string,
            fret: x.fret,
        })
        .collect()
}

fn to_solution_diag(rank: usize, solution: &DecodedSolution) -> DecodedSolutionDiagnostics {
    DecodedSolutionDiagnostics {
        rank,
        total_score: solution.total_score,
        sum_combined_score: solution.sum_combined_score,
        conflict_count: solution.conflict_count,
        fret_spread: solution.fret_spread,
        position_prior: solution.position_prior,
        conflict_penalty: solution.conflict_penalty,
        fret_spread_penalty: solution.fret_spread_penalty,
        position_prior_penalty: solution.position_prior_penalty,
        positions: positions_to_triplets(&solution.positions),
    }
}

fn decode_hit_top_k(
    solutions: &[DecodedSolution],
    gt_fret_set: &BTreeSet<(u8, u8, u8)>,
    k: usize,
) -> bool {
    solutions
        .iter()
        .take(k)
        .any(|solution| solution_matches_gt(solution, gt_fret_set))
}

fn solution_matches_gt(solution: &DecodedSolution, gt_fret_set: &BTreeSet<(u8, u8, u8)>) -> bool {
    let pred_set = solution
        .positions
        .iter()
        .map(|x| (x.midi, x.string, x.fret))
        .collect::<BTreeSet<_>>();
    pred_set == *gt_fret_set
}

fn count_global_configurations(per_pitch_candidates: &[Vec<CandidatePosition>]) -> usize {
    if per_pitch_candidates.is_empty() || per_pitch_candidates.iter().any(|x| x.is_empty()) {
        return 0;
    }
    let product = per_pitch_candidates
        .iter()
        .fold(1_u128, |acc, x| acc.saturating_mul(x.len() as u128));
    product.min(usize::MAX as u128) as usize
}

fn set_counts<T: Ord + Copy>(pred: &BTreeSet<T>, gt: &BTreeSet<T>) -> (usize, usize, usize) {
    let tp = pred.intersection(gt).count();
    let fp = pred.difference(gt).count();
    let fn_count = gt.difference(pred).count();
    (tp, fp, fn_count)
}

fn counts_to_metrics(tp: usize, fp: usize, fn_count: usize) -> SetMetrics {
    let precision = ratio(tp, tp + fp);
    let recall = ratio(tp, tp + fn_count);
    SetMetrics {
        true_positive_count: tp,
        false_positive_count: fp,
        false_negative_count: fn_count,
        precision,
        recall,
        f1: f1(precision, recall),
    }
}

fn accumulate_counts(target: &mut EventCounts, delta: &EventCounts) {
    target.pitch_tp += delta.pitch_tp;
    target.pitch_fp += delta.pitch_fp;
    target.pitch_fn += delta.pitch_fn;
    target.pitch_string_tp += delta.pitch_string_tp;
    target.pitch_string_fp += delta.pitch_string_fp;
    target.pitch_string_fn += delta.pitch_string_fn;
    target.pitch_string_fret_tp += delta.pitch_string_fret_tp;
    target.pitch_string_fret_fp += delta.pitch_string_fret_fp;
    target.pitch_string_fret_fn += delta.pitch_string_fret_fn;
    target.exact_hits += delta.exact_hits;
}

fn write_phase3_per_file_csv(path: &Path, rows: &[FullPerFileDiagnostic]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create phase3 per-file csv: {}", path.display()))?;
    writeln!(
        file,
        "file_id,evaluated_events,skipped_events,pitch_f1,pitch_string_f1,pitch_string_fret_f1,exact_configuration_accuracy"
    )?;

    for row in rows {
        writeln!(
            file,
            "{},{},{},{:.6},{:.6},{:.6},{:.6}",
            csv_escape(&row.file_id),
            row.evaluated_events,
            row.skipped_events,
            row.pitch_f1,
            row.pitch_string_f1,
            row.pitch_string_fret_f1,
            row.exact_configuration_accuracy
        )?;
    }

    Ok(())
}

fn write_phase3_examples_csv(path: &Path, rows: &[FullExampleDiagnostic]) -> Result<()> {
    let mut file = File::create(path)
        .with_context(|| format!("failed to create phase3 examples csv: {}", path.display()))?;

    writeln!(
        file,
        "example_id,file_id,split,onset_sec,offset_sec,pitch_tp,pitch_fp,pitch_fn,pitch_string_tp,pitch_string_fp,pitch_string_fn,pitch_string_fret_tp,pitch_string_fret_fp,pitch_string_fret_fn,exact_configuration_hit,decode_hit_top1,decode_hit_top3,decode_hit_top5,avg_candidate_positions_per_pitch,global_configurations_explored,global_configurations_returned,margin_top1_top2,best_total_score,failure_pitch_correct_string_wrong,failure_string_correct_fret_wrong,failure_full_wrong_despite_correct_pitch,failure_full_decode_wrong_high_confidence,example_json"
    )?;

    for row in rows {
        let margin = row
            .confidence_margin_top1_top2
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "NA".to_string());
        let best_score = row
            .best_total_score
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "NA".to_string());
        let json_path = format!("phase3_poly_scores/{}.json", row.example_id);

        writeln!(
            file,
            "{},{},{},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{}",
            csv_escape(&row.example_id),
            csv_escape(&row.file_id),
            csv_escape(&row.filename_split),
            row.onset_sec,
            row.offset_sec,
            row.pitch_tp,
            row.pitch_fp,
            row.pitch_fn,
            row.pitch_string_tp,
            row.pitch_string_fp,
            row.pitch_string_fn,
            row.pitch_string_fret_tp,
            row.pitch_string_fret_fp,
            row.pitch_string_fret_fn,
            row.exact_configuration_hit,
            row.decode_hit_top1,
            row.decode_hit_top3,
            row.decode_hit_top5,
            row.avg_candidate_positions_per_pitch,
            row.global_configurations_explored,
            row.global_configurations_returned,
            margin,
            best_score,
            row.failure_pitch_correct_string_wrong,
            row.failure_string_correct_fret_wrong,
            row.failure_full_wrong_despite_correct_pitch,
            row.failure_full_decode_wrong_high_confidence,
            csv_escape(&json_path),
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

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
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
    use std::collections::BTreeSet;

    use super::set_counts;

    #[test]
    fn set_matching_is_order_independent() {
        let pred = BTreeSet::from([60_u8, 64_u8]);
        let gt = BTreeSet::from([64_u8, 60_u8]);
        let (tp, fp, fn_count) = set_counts(&pred, &gt);
        assert_eq!(tp, 2);
        assert_eq!(fp, 0);
        assert_eq!(fn_count, 0);
    }
}
