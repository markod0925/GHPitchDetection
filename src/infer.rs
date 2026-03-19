use serde::Serialize;

use crate::config::AppConfig;
use crate::harmonic_map::build_harmonic_maps;
use crate::nms::select_pitch_candidates;
use crate::pitch::{harmonic_weights, score_pitch_frame};
use crate::stft::{build_stft_plan, compute_stft_frames};
use crate::types::AudioBuffer;
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
    let mut frames = compute_stft_frames(&audio.samples, audio.sample_rate, &stft_plan);
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
