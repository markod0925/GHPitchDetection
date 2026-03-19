use crate::config::NmsConfig;

pub fn select_pitch_candidates(
    midi_scores: &[f32],
    midi_min: u8,
    cfg: &NmsConfig,
) -> Vec<(u8, f32)> {
    if midi_scores.is_empty() || cfg.max_polyphony == 0 {
        return Vec::new();
    }

    let max_score = midi_scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let threshold = if cfg.threshold_mode == "absolute" {
        cfg.threshold_value
    } else if max_score.is_finite() {
        cfg.threshold_value * max_score
    } else {
        f32::INFINITY
    };

    let mut candidates: Vec<(u8, f32)> = Vec::new();
    for (i, &score) in midi_scores.iter().enumerate() {
        if score < threshold {
            continue;
        }
        if !is_local_maximum(midi_scores, i) {
            continue;
        }
        let midi = midi_min.saturating_add(i as u8);
        candidates.push((midi, score));
    }

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let radius = cfg.radius_midi_steps as u8;
    let mut selected: Vec<(u8, f32)> = Vec::with_capacity(cfg.max_polyphony.min(candidates.len()));
    'outer: for candidate in candidates {
        if selected.len() >= cfg.max_polyphony {
            break;
        }
        for &(midi, _) in &selected {
            if midi.abs_diff(candidate.0) <= radius {
                continue 'outer;
            }
        }
        selected.push(candidate);
    }

    selected
}

fn is_local_maximum(scores: &[f32], idx: usize) -> bool {
    let s = scores[idx];
    let left_ok = idx == 0 || s >= scores[idx - 1];
    let right_ok = idx + 1 >= scores.len() || s >= scores[idx + 1];
    left_ok && right_ok
}

#[cfg(test)]
mod tests {
    use crate::config::NmsConfig;

    use super::select_pitch_candidates;

    #[test]
    fn nms_keeps_best_non_overlapping_pitches() {
        let cfg = NmsConfig {
            threshold_mode: "absolute".to_string(),
            threshold_value: 0.1,
            radius_midi_steps: 1,
            max_polyphony: 3,
        };
        let scores = vec![0.2, 0.9, 0.8, 0.1, 0.7];
        let selected = select_pitch_candidates(&scores, 60, &cfg);

        assert_eq!(selected[0].0, 61);
        assert_eq!(selected[1].0, 64);
        assert_eq!(selected.len(), 2);
    }
}
