use serde::{Deserialize, Serialize};

use crate::config::DecodeConfig;
use crate::types::{CandidatePosition, DecodedFrame};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedSolution {
    pub positions: Vec<CandidatePosition>,
    pub sum_combined_score: f32,
    pub conflict_count: usize,
    pub fret_spread: f32,
    pub position_prior: f32,
    pub conflict_penalty: f32,
    pub fret_spread_penalty: f32,
    pub position_prior_penalty: f32,
    pub total_score: f32,
}

pub fn min_max_normalize(value: f32, min_value: f32, max_value: f32, eps: f32) -> f32 {
    (value - min_value) / ((max_value - min_value) + eps.max(1e-12))
}

pub fn normalize_and_combine_candidates(
    per_pitch_candidates: &mut [Vec<CandidatePosition>],
    cfg: &DecodeConfig,
) {
    let mut pitch_min = f32::INFINITY;
    let mut pitch_max = f32::NEG_INFINITY;

    for candidates in per_pitch_candidates.iter() {
        for candidate in candidates {
            pitch_min = pitch_min.min(candidate.pitch_score);
            pitch_max = pitch_max.max(candidate.pitch_score);
        }
    }

    if !pitch_min.is_finite() || !pitch_max.is_finite() {
        return;
    }

    for candidates in per_pitch_candidates.iter_mut() {
        if candidates.is_empty() {
            continue;
        }

        let mut template_min = f32::INFINITY;
        let mut template_max = f32::NEG_INFINITY;
        for candidate in candidates.iter() {
            template_min = template_min.min(candidate.template_score);
            template_max = template_max.max(candidate.template_score);
        }

        for candidate in candidates.iter_mut() {
            candidate.pitch_score_norm = min_max_normalize(
                candidate.pitch_score,
                pitch_min,
                pitch_max,
                cfg.normalization_epsilon,
            );
            candidate.template_score_norm = min_max_normalize(
                candidate.template_score,
                template_min,
                template_max,
                cfg.normalization_epsilon,
            );
            candidate.combined_score =
                cfg.alpha * candidate.pitch_score_norm + cfg.beta * candidate.template_score_norm;
        }

        candidates.sort_by(|a, b| {
            b.combined_score
                .total_cmp(&a.combined_score)
                .then_with(|| a.string.cmp(&b.string))
                .then_with(|| a.fret.cmp(&b.fret))
        });
    }
}

pub fn prune_top_k_candidates(per_pitch_candidates: &mut [Vec<CandidatePosition>], top_k: usize) {
    let keep = top_k.max(1);
    for candidates in per_pitch_candidates.iter_mut() {
        if candidates.len() > keep {
            candidates.truncate(keep);
        }
    }
}

pub fn decode_global_configuration(
    per_pitch_candidates: &[Vec<CandidatePosition>],
    cfg: &DecodeConfig,
) -> Option<DecodedFrame> {
    let best = decode_global_solutions(per_pitch_candidates, cfg, 1)
        .into_iter()
        .next()?;

    Some(DecodedFrame {
        time_sec: 0.0,
        positions: best.positions,
        total_score: best.total_score,
    })
}

pub fn decode_global_solutions(
    per_pitch_candidates: &[Vec<CandidatePosition>],
    cfg: &DecodeConfig,
    limit: usize,
) -> Vec<DecodedSolution> {
    if per_pitch_candidates.is_empty() || per_pitch_candidates.iter().any(|x| x.is_empty()) {
        return Vec::new();
    }

    let max_per_pitch = cfg.max_candidates_per_pitch.max(1);
    let keep = limit.max(1);

    let mut solutions = Vec::new();
    let mut current = Vec::with_capacity(per_pitch_candidates.len());
    recurse_solutions(
        0,
        per_pitch_candidates,
        max_per_pitch,
        cfg,
        &mut current,
        &mut solutions,
    );

    solutions.sort_by(|a, b| {
        b.total_score
            .total_cmp(&a.total_score)
            .then_with(|| b.sum_combined_score.total_cmp(&a.sum_combined_score))
    });
    if solutions.len() > keep {
        solutions.truncate(keep);
    }
    solutions
}

fn recurse_solutions(
    pitch_idx: usize,
    per_pitch_candidates: &[Vec<CandidatePosition>],
    max_per_pitch: usize,
    cfg: &DecodeConfig,
    current: &mut Vec<CandidatePosition>,
    out: &mut Vec<DecodedSolution>,
) {
    if pitch_idx >= per_pitch_candidates.len() {
        out.push(score_solution(current, cfg));
        return;
    }

    let candidates = &per_pitch_candidates[pitch_idx];
    let bound = candidates.len().min(max_per_pitch);
    for candidate in candidates.iter().take(bound) {
        current.push(candidate.clone());
        recurse_solutions(
            pitch_idx + 1,
            per_pitch_candidates,
            max_per_pitch,
            cfg,
            current,
            out,
        );
        current.pop();
    }
}

fn score_solution(positions: &[CandidatePosition], cfg: &DecodeConfig) -> DecodedSolution {
    let sum_combined_score = positions.iter().map(|x| x.combined_score).sum::<f32>();
    let conflict_count = count_string_conflicts(positions);
    let fret_spread = compute_fret_spread(positions);
    let position_prior = compute_position_prior(positions);

    let conflict_penalty = cfg.lambda_conflict * conflict_count as f32;
    let fret_spread_penalty = cfg.lambda_fret_spread * fret_spread;
    let position_prior_penalty = cfg.lambda_position_prior * position_prior;
    let total_score =
        sum_combined_score - conflict_penalty - fret_spread_penalty - position_prior_penalty;

    DecodedSolution {
        positions: positions.to_vec(),
        sum_combined_score,
        conflict_count,
        fret_spread,
        position_prior,
        conflict_penalty,
        fret_spread_penalty,
        position_prior_penalty,
        total_score,
    }
}

fn count_string_conflicts(positions: &[CandidatePosition]) -> usize {
    let mut counts = [0usize; 6];
    for pos in positions {
        let idx = pos.string as usize;
        if idx < counts.len() {
            counts[idx] += 1;
        }
    }
    counts.into_iter().map(|n| n.saturating_sub(1)).sum()
}

fn compute_fret_spread(positions: &[CandidatePosition]) -> f32 {
    if positions.len() <= 1 {
        return 0.0;
    }

    let min_fret = positions.iter().map(|x| x.fret).min().unwrap_or(0) as f32;
    let max_fret = positions.iter().map(|x| x.fret).max().unwrap_or(0) as f32;
    max_fret - min_fret
}

fn compute_position_prior(positions: &[CandidatePosition]) -> f32 {
    if positions.is_empty() {
        return 0.0;
    }
    let sum = positions.iter().map(|x| x.fret as f32).sum::<f32>();
    sum / positions.len() as f32
}

#[cfg(test)]
mod tests {
    use crate::config::DecodeConfig;
    use crate::types::CandidatePosition;

    use super::{
        decode_global_solutions, min_max_normalize, normalize_and_combine_candidates,
        prune_top_k_candidates,
    };

    fn candidate(midi: u8, string: u8, fret: u8, pitch: f32, template: f32) -> CandidatePosition {
        CandidatePosition {
            midi,
            string,
            fret,
            pitch_score: pitch,
            template_score: template,
            pitch_score_norm: 0.0,
            template_score_norm: 0.0,
            combined_score: 0.0,
        }
    }

    #[test]
    fn min_max_normalization_uses_eps() {
        let n = min_max_normalize(2.0, 2.0, 2.0, 1e-6);
        assert_eq!(n, 0.0);
    }

    #[test]
    fn normalize_and_combine_populates_scores() {
        let mut per_pitch = vec![
            vec![candidate(60, 0, 1, 0.2, 1.0), candidate(60, 1, 2, 0.2, 2.0)],
            vec![candidate(64, 2, 3, 0.8, 3.0), candidate(64, 3, 4, 0.8, 4.0)],
        ];
        let cfg = DecodeConfig::default();

        normalize_and_combine_candidates(&mut per_pitch, &cfg);

        assert!((per_pitch[0][0].pitch_score_norm - 0.0).abs() < 1e-6);
        assert!((per_pitch[1][0].pitch_score_norm - 1.0).abs() < 1e-4);
        assert!(per_pitch[0][0].combined_score >= per_pitch[0][1].combined_score);
        assert!(per_pitch[1][0].combined_score >= per_pitch[1][1].combined_score);
    }

    #[test]
    fn prune_keeps_only_top_k() {
        let mut per_pitch = vec![vec![
            candidate(60, 0, 1, 0.1, 0.1),
            candidate(60, 1, 2, 0.1, 0.2),
            candidate(60, 2, 3, 0.1, 0.3),
        ]];
        prune_top_k_candidates(&mut per_pitch, 2);
        assert_eq!(per_pitch[0].len(), 2);
    }

    #[test]
    fn global_decode_prefers_non_conflicting_strings_when_penalized() {
        let mut cfg = DecodeConfig::default();
        cfg.lambda_conflict = 100.0;

        let mut per_pitch = vec![
            vec![candidate(60, 0, 5, 1.0, 2.0), candidate(60, 1, 5, 1.0, 1.0)],
            vec![candidate(64, 0, 9, 1.0, 2.0), candidate(64, 2, 9, 1.0, 1.8)],
        ];
        normalize_and_combine_candidates(&mut per_pitch, &cfg);

        let best = decode_global_solutions(&per_pitch, &cfg, 1)
            .into_iter()
            .next()
            .expect("best solution");

        assert_eq!(best.conflict_count, 0);
        assert_ne!(best.positions[0].string, best.positions[1].string);
    }
}
