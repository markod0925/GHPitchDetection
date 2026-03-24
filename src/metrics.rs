use std::collections::BTreeMap;

use anyhow::{Context, Result};

use crate::audio::load_wav_mono;
use crate::config::AppConfig;
use crate::features::{extract_note_features_for_record, flatten_note_features};
use crate::harmonic_map::build_harmonic_maps;
use crate::stft::build_stft_plan;
use crate::templates::{score_string_fret_candidates, GuitarTemplates};
use crate::types::{CandidatePosition, MonoAnnotation};

#[derive(Debug, Clone)]
pub struct PitchMetrics {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

#[derive(Debug, Clone)]
pub struct StringMetrics {
    pub top1: f32,
    pub top3: f32,
    pub exact_string_fret: f32,
}

#[derive(Debug, Clone)]
pub struct MonoEvalReport {
    pub metrics: StringMetrics,
    pub total_records: usize,
    pub evaluated_records: usize,
    pub skipped_records: usize,
    pub exact_string_fret_supported: bool,
}

pub fn evaluate_string_predictions(
    ranked_candidates: &[Vec<CandidatePosition>],
    ground_truth: &[MonoAnnotation],
    exact_string_fret_supported: bool,
) -> StringMetrics {
    if ranked_candidates.is_empty() || ranked_candidates.len() != ground_truth.len() {
        return StringMetrics {
            top1: 0.0,
            top3: 0.0,
            exact_string_fret: 0.0,
        };
    }

    let mut top1_hits = 0usize;
    let mut top3_hits = 0usize;
    let mut exact_hits = 0usize;

    for (ranked, gt) in ranked_candidates.iter().zip(ground_truth) {
        if ranked.first().map(|c| c.string) == Some(gt.string) {
            top1_hits += 1;
        }
        if ranked.iter().take(3).any(|c| c.string == gt.string) {
            top3_hits += 1;
        }
        if exact_string_fret_supported
            && ranked
                .first()
                .map(|c| c.string == gt.string && c.fret == gt.fret)
                .unwrap_or(false)
        {
            exact_hits += 1;
        }
    }

    let n = ranked_candidates.len() as f32;
    StringMetrics {
        top1: top1_hits as f32 / n,
        top3: top3_hits as f32 / n,
        exact_string_fret: if exact_string_fret_supported {
            exact_hits as f32 / n
        } else {
            0.0
        },
    }
}

pub fn evaluate_mono_dataset(
    dataset: &[MonoAnnotation],
    templates: &GuitarTemplates,
    cfg: &AppConfig,
) -> Result<MonoEvalReport> {
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

    let mut grouped: BTreeMap<&str, Vec<&MonoAnnotation>> = BTreeMap::new();
    for ann in dataset {
        grouped.entry(&ann.audio_path).or_default().push(ann);
    }

    let mut ranked = Vec::new();
    let mut ground_truth = Vec::new();
    let mut skipped = 0usize;

    for (audio_path, notes) in grouped {
        let audio = load_wav_mono(audio_path).with_context(|| {
            format!("failed to load audio during mono evaluation: {audio_path}")
        })?;
        if audio.sample_rate != cfg.sample_rate_target {
            skipped += notes.len();
            continue;
        }

        for note in notes {
            let Some(feat) =
                extract_note_features_for_record(note, &audio, &stft_plan, cfg, &maps)?
            else {
                skipped += 1;
                continue;
            };
            let z = flatten_note_features(&feat, &cfg.templates);
            let candidates = score_string_fret_candidates(note.midi, &z, templates, cfg);
            if candidates.is_empty() {
                skipped += 1;
                continue;
            }
            ranked.push(candidates);
            ground_truth.push(note.clone());
        }
    }

    let exact_supported = cfg.templates.level == "string_fret";
    let metrics = evaluate_string_predictions(&ranked, &ground_truth, exact_supported);

    Ok(MonoEvalReport {
        metrics,
        total_records: dataset.len(),
        evaluated_records: ranked.len(),
        skipped_records: skipped,
        exact_string_fret_supported: exact_supported,
    })
}

#[cfg(test)]
mod tests {
    use crate::types::CandidatePosition;

    use super::{evaluate_string_predictions, MonoEvalReport};

    #[test]
    fn string_metrics_compute_topk_and_exact() {
        let ranked = vec![
            vec![
                CandidatePosition {
                    midi: 64,
                    string: 5,
                    fret: 0,
                    pitch_score: 0.0,
                    template_score: 1.0,
                    pitch_score_norm: 0.0,
                    template_score_norm: 0.0,
                    combined_score: 1.0,
                },
                CandidatePosition {
                    midi: 64,
                    string: 4,
                    fret: 5,
                    pitch_score: 0.0,
                    template_score: 0.8,
                    pitch_score_norm: 0.0,
                    template_score_norm: 0.0,
                    combined_score: 0.8,
                },
            ],
            vec![
                CandidatePosition {
                    midi: 62,
                    string: 3,
                    fret: 7,
                    pitch_score: 0.0,
                    template_score: 0.9,
                    pitch_score_norm: 0.0,
                    template_score_norm: 0.0,
                    combined_score: 0.9,
                },
                CandidatePosition {
                    midi: 62,
                    string: 2,
                    fret: 12,
                    pitch_score: 0.0,
                    template_score: 0.7,
                    pitch_score_norm: 0.0,
                    template_score_norm: 0.0,
                    combined_score: 0.7,
                },
            ],
        ];
        let gt = vec![
            crate::types::MonoAnnotation {
                audio_path: "a.wav".to_string(),
                sample_rate: 44_100,
                midi: 64,
                string: 5,
                fret: 0,
                onset_sec: 0.0,
                offset_sec: 0.1,
                player_id: None,
                guitar_id: None,
                style: None,
                pickup_mode: None,
            },
            crate::types::MonoAnnotation {
                audio_path: "b.wav".to_string(),
                sample_rate: 44_100,
                midi: 62,
                string: 2,
                fret: 12,
                onset_sec: 0.0,
                offset_sec: 0.1,
                player_id: None,
                guitar_id: None,
                style: None,
                pickup_mode: None,
            },
        ];

        let m = evaluate_string_predictions(&ranked, &gt, true);
        assert!((m.top1 - 0.5).abs() < 1e-6);
        assert!((m.top3 - 1.0).abs() < 1e-6);
        assert!((m.exact_string_fret - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mono_eval_report_is_constructible() {
        let report = MonoEvalReport {
            metrics: super::StringMetrics {
                top1: 0.0,
                top3: 0.0,
                exact_string_fret: 0.0,
            },
            total_records: 0,
            evaluated_records: 0,
            skipped_records: 0,
            exact_string_fret_supported: false,
        };
        assert!(!report.exact_string_fret_supported);
    }
}
