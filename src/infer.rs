use num::complex::Complex64;
use qdft::QDFT;
use serde::Serialize;

use crate::config::AppConfig;
use crate::fretboard::pitch_to_hz;
use crate::harmonic_map::build_harmonic_maps;
use crate::nms::select_pitch_candidates;
use crate::pitch::{harmonic_weights, score_pitch_frame};
use crate::stft::{build_stft_plan, compute_stft_frames, StftPlan};
use crate::types::{AudioBuffer, SpectrumFrame};
use crate::whitening::whiten_log_spectrum;

#[derive(Debug, Clone, Serialize)]
pub struct PitchCandidate {
    pub midi: u8,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct PitchFrameInference {
    pub time_sec: f32,
    pub notes: Vec<PitchCandidate>,
}

pub fn run_pitch_only_inference(audio: &AudioBuffer, cfg: &AppConfig) -> Vec<PitchFrameInference> {
    let stft_plan = build_stft_plan(&cfg.frontend);
    let mut frames =
        compute_phase1_frontend_frames(&audio.samples, audio.sample_rate, &stft_plan, cfg);
    let maps = build_harmonic_maps(
        audio.sample_rate,
        stft_plan.nfft,
        cfg.pitch.midi_min,
        cfg.pitch.midi_max,
        cfg.pitch.max_harmonics,
        cfg.pitch.local_bandwidth_bins,
    );
    let hweights = harmonic_weights(cfg.pitch.max_harmonics, cfg.pitch.harmonic_alpha);

    let bins = stft_plan.nfft / 2 + 1;
    let mut tmp_log = vec![0.0_f32; bins];
    let mut envelope = vec![0.0_f32; bins];
    let mut white = vec![0.0_f32; bins];

    let mut out = Vec::with_capacity(frames.len());
    for frame in frames.iter_mut() {
        whiten_log_spectrum(
            &frame.mag,
            &cfg.whitening,
            &mut tmp_log,
            &mut envelope,
            &mut white,
        );
        frame.white.copy_from_slice(&white);

        let mut scored = score_pitch_frame(&frame.white, &maps, &cfg.pitch, &hweights);
        scored.time_sec = frame.time_sec;

        let notes = select_pitch_candidates(&scored.midi_scores, cfg.pitch.midi_min, &cfg.nms)
            .into_iter()
            .map(|(midi, score)| PitchCandidate { midi, score })
            .collect();
        out.push(PitchFrameInference {
            time_sec: frame.time_sec,
            notes,
        });
    }

    // TODO(phase-3): route pitch candidates into string/fret scoring and global decode.
    out
}

fn compute_phase1_frontend_frames(
    samples: &[f32],
    sample_rate: u32,
    plan: &StftPlan,
    cfg: &AppConfig,
) -> Vec<SpectrumFrame> {
    match cfg.frontend.kind.trim().to_ascii_lowercase().as_str() {
        "stft" => compute_stft_frames(samples, sample_rate, plan),
        "qdft" => compute_qdft_frames(samples, sample_rate, plan, cfg),
        other => {
            eprintln!(
                "warning: unsupported frontend.kind '{}' for phase1; falling back to stft",
                other
            );
            compute_stft_frames(samples, sample_rate, plan)
        }
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
