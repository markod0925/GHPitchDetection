use std::cmp::Ordering;
use std::collections::BTreeMap;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::audio::load_wav_mono;
use crate::config::AppConfig;
use crate::features::{
    extract_note_features_for_record, feature_names, flatten_note_features, validate_feature_dim,
};
use crate::fretboard::{fret_region, midi_to_fret_positions};
use crate::harmonic_map::build_harmonic_maps;
use crate::stft::build_stft_plan;
use crate::types::{CandidatePosition, MonoAnnotation, OPEN_MIDI};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateStats {
    pub count: u32,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuitarTemplates {
    pub open_midi: [u8; 6],
    pub fret_max: u8,
    pub feature_names: Vec<String>,
    pub by_string: Vec<TemplateStats>,
    pub by_string_region: Vec<Vec<Option<TemplateStats>>>,
    pub by_string_fret: Vec<Vec<Option<TemplateStats>>>,
}

#[derive(Debug, Clone)]
pub struct RunningStats {
    pub count: u64,
    pub mean: Vec<f64>,
    pub m2: Vec<f64>,
}

pub fn running_stats_new(dim: usize) -> RunningStats {
    RunningStats {
        count: 0,
        mean: vec![0.0; dim],
        m2: vec![0.0; dim],
    }
}

pub fn running_stats_update(stats: &mut RunningStats, x: &[f32]) {
    debug_assert_eq!(stats.mean.len(), x.len());
    if stats.mean.len() != x.len() {
        return;
    }

    stats.count += 1;
    let n = stats.count as f64;
    for (i, &value) in x.iter().enumerate() {
        let value = value as f64;
        let delta = value - stats.mean[i];
        stats.mean[i] += delta / n;
        let delta2 = value - stats.mean[i];
        stats.m2[i] += delta * delta2;
    }
}

pub fn running_stats_finalize(stats: RunningStats) -> TemplateStats {
    let dim = stats.mean.len();
    let mut mean = vec![0.0_f32; dim];
    let mut std = vec![0.0_f32; dim];
    for i in 0..dim {
        mean[i] = stats.mean[i] as f32;
        let var = if stats.count > 1 {
            stats.m2[i] / (stats.count as f64 - 1.0)
        } else {
            0.0
        };
        std[i] = var.max(0.0).sqrt() as f32;
    }
    TemplateStats {
        count: stats.count.min(u32::MAX as u64) as u32,
        mean,
        std,
    }
}

pub fn train_templates_from_mono_dataset(
    dataset: &[MonoAnnotation],
    cfg: &AppConfig,
) -> anyhow::Result<GuitarTemplates> {
    let dim = validate_feature_dim(&cfg.templates)?;
    let feature_names = feature_names(&cfg.templates);
    let region_count = cfg.templates.region_bounds.len().saturating_sub(1).max(1);
    let fret_max = dataset.iter().map(|x| x.fret).max().unwrap_or(20).max(20);

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

    let mut by_string_running: Vec<RunningStats> = (0..6).map(|_| running_stats_new(dim)).collect();
    let mut by_string_region_running: Vec<Vec<Option<RunningStats>>> = (0..6)
        .map(|_| (0..region_count).map(|_| None).collect())
        .collect();

    let mut grouped: BTreeMap<&str, Vec<&MonoAnnotation>> = BTreeMap::new();
    for ann in dataset {
        grouped.entry(&ann.audio_path).or_default().push(ann);
    }

    for (audio_path, notes) in grouped {
        let audio = load_wav_mono(audio_path)
            .with_context(|| format!("failed to load audio for template training: {audio_path}"))?;
        if audio.sample_rate != cfg.sample_rate_target {
            // TODO(phase-3): support resampling when training data sample rates differ.
            continue;
        }

        for note in notes {
            let string_idx = note.string as usize;
            if string_idx >= 6 {
                continue;
            }

            let Some(feat) =
                extract_note_features_for_record(note, &audio, &stft_plan, cfg, &maps)?
            else {
                continue;
            };
            let z = flatten_note_features(&feat, &cfg.templates);
            if z.len() != dim {
                continue;
            }

            running_stats_update(&mut by_string_running[string_idx], &z);

            let region = fret_region(note.fret, &cfg.templates.region_bounds).min(region_count - 1);
            let region_stats = by_string_region_running[string_idx][region]
                .get_or_insert_with(|| running_stats_new(dim));
            running_stats_update(region_stats, &z);
        }
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
                .map(|stats| stats.map(running_stats_finalize))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // TODO(phase-3): train string_fret templates once phase-3 decoding needs fine-grained classes.
    let by_string_fret = (0..6)
        .map(|_| {
            (0..=usize::from(fret_max))
                .map(|_| None)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Ok(GuitarTemplates {
        open_midi: OPEN_MIDI,
        fret_max,
        feature_names,
        by_string,
        by_string_region,
        by_string_fret,
    })
}

pub fn score_diag_mahalanobis(z: &[f32], mean: &[f32], std: &[f32], eps: f32) -> f32 {
    if z.len() != mean.len() || z.len() != std.len() || z.is_empty() {
        return f32::NEG_INFINITY;
    }
    let eps = eps.max(1e-12);
    let mut score = 0.0_f32;
    for i in 0..z.len() {
        let d = z[i] - mean[i];
        let denom = std[i] * std[i] + eps;
        score -= d * d / denom;
    }
    score
}

pub fn score_cosine(z: &[f32], mean: &[f32], eps: f32) -> f32 {
    if z.len() != mean.len() || z.is_empty() {
        return f32::NEG_INFINITY;
    }
    let eps = eps.max(1e-12);
    let mut dot = 0.0_f32;
    let mut zn = 0.0_f32;
    let mut mn = 0.0_f32;
    for i in 0..z.len() {
        dot += z[i] * mean[i];
        zn += z[i] * z[i];
        mn += mean[i] * mean[i];
    }
    dot / ((zn.sqrt() * mn.sqrt()) + eps)
}

pub fn score_string_fret_candidates(
    midi: u8,
    z: &[f32],
    templates: &GuitarTemplates,
    cfg: &AppConfig,
) -> Vec<CandidatePosition> {
    let mut scored = Vec::new();
    for (string, fret) in midi_to_fret_positions(midi, &templates.open_midi, templates.fret_max) {
        if let Some(stats) = select_template_stats(templates, cfg, string, fret) {
            if stats.count == 0 || stats.mean.len() != z.len() || stats.std.len() != z.len() {
                continue;
            }
            let local_score = if cfg.templates.score_type == "cosine" {
                score_cosine(z, &stats.mean, 1e-6)
            } else {
                score_diag_mahalanobis(z, &stats.mean, &stats.std, 1e-6)
            };
            scored.push(CandidatePosition {
                midi,
                string,
                fret,
                local_score,
            });
        }
    }

    scored.sort_by(|a, b| {
        b.local_score
            .partial_cmp(&a.local_score)
            .unwrap_or(Ordering::Equal)
    });
    scored
}

pub fn save_templates(path: &str, templates: &GuitarTemplates) -> anyhow::Result<()> {
    let bytes =
        bincode::serialize(templates).context("failed to serialize templates with bincode")?;
    std::fs::write(path, bytes)
        .with_context(|| format!("failed to write template file: {path}"))?;
    Ok(())
}

pub fn load_templates(path: &str) -> anyhow::Result<GuitarTemplates> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read template file: {path}"))?;
    let templates: GuitarTemplates =
        bincode::deserialize(&bytes).context("failed to deserialize templates from bincode")?;
    Ok(templates)
}

fn select_template_stats<'a>(
    templates: &'a GuitarTemplates,
    cfg: &AppConfig,
    string: u8,
    fret: u8,
) -> Option<&'a TemplateStats> {
    let s = string as usize;
    if s >= templates.by_string.len() {
        return None;
    }
    let string_stats = templates.by_string.get(s).filter(|t| t.count > 0);
    match cfg.templates.level.as_str() {
        "string" => string_stats,
        "string_region" => {
            let region = fret_region(fret, &cfg.templates.region_bounds);
            let region_stats = templates
                .by_string_region
                .get(s)
                .and_then(|regions| regions.get(region))
                .and_then(|x| x.as_ref())
                .filter(|t| t.count > 0);
            region_stats.or(string_stats)
        }
        "string_fret" => {
            let fret_stats = templates
                .by_string_fret
                .get(s)
                .and_then(|frets| frets.get(fret as usize))
                .and_then(|x| x.as_ref())
                .filter(|t| t.count > 0);
            fret_stats
                .or_else(|| {
                    let region = fret_region(fret, &cfg.templates.region_bounds);
                    templates
                        .by_string_region
                        .get(s)
                        .and_then(|regions| regions.get(region))
                        .and_then(|x| x.as_ref())
                        .filter(|t| t.count > 0)
                })
                .or(string_stats)
        }
        _ => string_stats,
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{AppConfig, TemplateConfig};

    use super::{
        running_stats_finalize, running_stats_new, running_stats_update, score_cosine,
        score_diag_mahalanobis, score_string_fret_candidates, GuitarTemplates, TemplateStats,
    };

    #[test]
    fn running_stats_matches_expected_mean_and_std() {
        let mut stats = running_stats_new(2);
        running_stats_update(&mut stats, &[1.0, 2.0]);
        running_stats_update(&mut stats, &[3.0, 4.0]);
        let finalized = running_stats_finalize(stats);
        assert_eq!(finalized.count, 2);
        assert!((finalized.mean[0] - 2.0).abs() < 1e-6);
        assert!((finalized.mean[1] - 3.0).abs() < 1e-6);
        assert!((finalized.std[0] - (2.0_f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn scorers_prefer_matching_vectors() {
        let z = [1.0, 2.0, 3.0];
        let mean_same = [1.0, 2.0, 3.0];
        let mean_diff = [3.0, 2.0, 1.0];
        let std = [0.5, 0.5, 0.5];
        assert!(
            score_diag_mahalanobis(&z, &mean_same, &std, 1e-6)
                > score_diag_mahalanobis(&z, &mean_diff, &std, 1e-6)
        );
        assert!(score_cosine(&z, &mean_same, 1e-6) > score_cosine(&z, &mean_diff, 1e-6));
    }

    #[test]
    fn candidate_scoring_uses_string_level_templates() {
        let cfg = AppConfig {
            templates: TemplateConfig {
                level: "string".to_string(),
                score_type: "cosine".to_string(),
                num_harmonics: 1,
                include_harmonic_ratios: false,
                include_centroid: false,
                include_rolloff: false,
                include_flux: false,
                include_attack_ratio: false,
                include_noise_ratio: false,
                include_inharmonicity: false,
                region_bounds: vec![0, 5, 10],
            },
            ..AppConfig::default()
        };
        let by_string = (0..6)
            .map(|i| TemplateStats {
                count: 1,
                mean: vec![if i == 5 { 1.0 } else { 0.0 }],
                std: vec![1.0],
            })
            .collect::<Vec<_>>();
        let templates = GuitarTemplates {
            open_midi: crate::types::OPEN_MIDI,
            fret_max: 20,
            feature_names: vec!["harmonic_amps[0]".to_string()],
            by_string,
            by_string_region: vec![vec![None; 2]; 6],
            by_string_fret: vec![vec![None; 21]; 6],
        };

        let ranked = score_string_fret_candidates(64, &[1.0], &templates, &cfg);
        assert!(!ranked.is_empty());
        assert_eq!(ranked[0].string, 5);
        assert_eq!(ranked[0].fret, 0);
    }
}
