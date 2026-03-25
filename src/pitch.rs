use crate::config::PitchConfig;
use crate::types::{MidiHarmonicMap, PitchScoreFrame};

pub fn harmonic_weights(max_harmonics: usize, alpha: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(max_harmonics);
    for h in 1..=max_harmonics {
        out.push(1.0 / (h as f32).powf(alpha));
    }
    out
}

pub fn score_pitch_frame(
    white: &[f32],
    maps: &[MidiHarmonicMap],
    cfg: &PitchConfig,
    harmonic_weights: &[f32],
) -> PitchScoreFrame {
    let mut midi_scores = vec![0.0_f32; maps.len()];

    for (midi_idx, map) in maps.iter().enumerate() {
        let mut score = 0.0_f32;
        let mut missing_count = 0usize;

        for (h, range) in map.ranges.iter().enumerate() {
            let local = aggregate_local(white, range.start, range.end, &cfg.local_aggregation);
            let w = harmonic_weights.get(h).copied().unwrap_or(0.0);
            score += w * local;
            if local <= 0.0 {
                missing_count += 1;
            }
        }

        if cfg.use_subharmonic_penalty {
            if let Some(first_range) = map.ranges.first() {
                let center = (first_range.start + first_range.end) / 2;
                if center > 0 {
                    let width = (first_range.end - first_range.start) / 2;
                    let sub_center = center / 2;
                    let sub_start = sub_center.saturating_sub(width);
                    let sub_end = (sub_center + width).min(white.len().saturating_sub(1));
                    let sub = aggregate_local(white, sub_start, sub_end, &cfg.local_aggregation);
                    score -= cfg.lambda_subharmonic * sub;
                }
            }
        }

        if cfg.use_missing_penalty && !map.ranges.is_empty() {
            let ratio = missing_count as f32 / map.ranges.len() as f32;
            score -= cfg.lambda_missing * ratio;
        }

        midi_scores[midi_idx] = score;
    }

    PitchScoreFrame {
        // TODO(phase-1.5): thread frame timestamp through this API.
        time_sec: 0.0,
        midi_scores,
    }
}

fn aggregate_local(white: &[f32], start: usize, end: usize, mode: &str) -> f32 {
    if white.is_empty() {
        return 0.0;
    }
    let s = start.min(white.len() - 1);
    let e = end.min(white.len() - 1);
    if s > e {
        return 0.0;
    }

    if mode == "mean" {
        let mut sum = 0.0_f32;
        let mut n = 0usize;
        for &x in &white[s..=e] {
            sum += x;
            n += 1;
        }
        if n == 0 {
            0.0
        } else {
            sum / n as f32
        }
    } else {
        white[s..=e].iter().copied().fold(0.0_f32, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::PitchConfig;
    use crate::harmonic_map::build_harmonic_maps;

    use super::{harmonic_weights, score_pitch_frame};

    #[test]
    fn weighted_score_prefers_fundamental_with_energy() {
        let cfg = PitchConfig {
            backend: "legacy".to_string(),
            midi_min: 69,
            midi_max: 69,
            max_harmonics: 3,
            harmonic_alpha: 0.0,
            local_bandwidth_bins: 1,
            local_aggregation: "max".to_string(),
            use_subharmonic_penalty: false,
            lambda_subharmonic: 0.0,
            use_missing_penalty: false,
            lambda_missing: 0.0,
        };

        let maps = build_harmonic_maps(
            8_000,
            1024,
            69,
            69,
            cfg.max_harmonics,
            cfg.local_bandwidth_bins,
        );
        let mut white = vec![0.0_f32; 1024 / 2 + 1];
        for r in &maps[0].ranges {
            white[(r.start + r.end) / 2] = 1.0;
        }

        let w = harmonic_weights(cfg.max_harmonics, cfg.harmonic_alpha);
        let scored = score_pitch_frame(&white, &maps, &cfg, &w);
        assert!(scored.midi_scores[0] > 0.0);
    }
}
