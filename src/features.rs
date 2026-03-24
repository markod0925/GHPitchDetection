use anyhow::{Context, Result};

use crate::audio::trim_audio_region;
use crate::config::{AppConfig, TemplateConfig};
use crate::stft::{compute_stft_frames, StftPlan};
use crate::types::{AudioBuffer, MidiHarmonicMap, MonoAnnotation, NoteFeatures, MAX_HARMONICS};
use crate::whitening::whiten_log_spectrum;

const NOTE_CONTEXT_MARGIN_SEC: f32 = 0.005;
const EPS: f32 = 1e-8;
const ROLLOFF_RATIO: f32 = 0.85;

pub fn extract_note_features(
    current_white: &[f32],
    previous_white: Option<&[f32]>,
    midi: u8,
    map: &MidiHarmonicMap,
    sample_rate: u32,
    nfft: usize,
    cfg: &TemplateConfig,
) -> NoteFeatures {
    let harmonic_count = selected_harmonic_count(cfg, map);

    let mut harmonic_amps = [0.0_f32; MAX_HARMONICS];
    for (h, dst) in harmonic_amps.iter_mut().enumerate().take(harmonic_count) {
        let range = &map.ranges[h];
        *dst = local_max(current_white, range.start, range.end);
    }

    let mut harmonic_ratios = [0.0_f32; MAX_HARMONICS];
    let denom = harmonic_amps[0].max(EPS);
    for (h, dst) in harmonic_ratios.iter_mut().enumerate().take(harmonic_count) {
        *dst = harmonic_amps[h] / denom;
    }

    let (local_start, local_end) = local_note_region(map, harmonic_count, current_white.len());
    let hz_per_bin = sample_rate as f32 / nfft.max(1) as f32;
    let (centroid, rolloff) =
        centroid_and_rolloff(current_white, local_start, local_end, hz_per_bin);
    let flux = spectral_flux(current_white, previous_white, local_start, local_end);
    let noise_ratio = local_noise_ratio(current_white, map, harmonic_count, local_start, local_end);
    let inharmonicity = estimate_inharmonicity(
        current_white,
        map,
        harmonic_count,
        hz_per_bin,
        local_start,
        local_end,
    );

    NoteFeatures {
        midi,
        f0_hz: map.f0_hz,
        harmonic_amps,
        harmonic_ratios,
        centroid,
        rolloff,
        flux,
        // TODO(phase-3): compute attack_ratio using onset-proximal vs sustain frames.
        attack_ratio: 0.0,
        noise_ratio,
        inharmonicity,
    }
}

pub fn extract_note_features_from_annotation(
    annotation: &MonoAnnotation,
    audio: &AudioBuffer,
    stft_plan: &StftPlan,
    cfg: &AppConfig,
    map: &MidiHarmonicMap,
) -> Result<Option<NoteFeatures>> {
    if audio.sample_rate != cfg.sample_rate_target {
        return Ok(None);
    }

    let start_sec = (annotation.onset_sec - NOTE_CONTEXT_MARGIN_SEC).max(0.0);
    let end_sec = (annotation.offset_sec + NOTE_CONTEXT_MARGIN_SEC).max(start_sec);
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

    let mut acc = NoteFeatureAccumulator::new(annotation.midi, map.f0_hz);
    for (idx, frame) in frames.iter().enumerate() {
        let previous = if idx > 0 {
            Some(frames[idx - 1].white.as_slice())
        } else {
            None
        };
        let feat = extract_note_features(
            &frame.white,
            previous,
            annotation.midi,
            map,
            audio.sample_rate,
            stft_plan.nfft,
            &cfg.templates,
        );
        acc.update(&feat);
    }
    Ok(acc.finalize())
}

pub fn flatten_note_features(features: &NoteFeatures, cfg: &TemplateConfig) -> Vec<f32> {
    let names = feature_names(cfg);
    let mut out = Vec::with_capacity(names.len());
    let harmonic_count = cfg.num_harmonics.min(MAX_HARMONICS);

    out.extend_from_slice(&features.harmonic_amps[..harmonic_count]);
    if cfg.include_harmonic_ratios {
        out.extend_from_slice(&features.harmonic_ratios[..harmonic_count]);
    }
    if cfg.include_centroid {
        out.push(features.centroid);
    }
    if cfg.include_rolloff {
        out.push(features.rolloff);
    }
    if cfg.include_flux {
        out.push(features.flux);
    }
    if cfg.include_attack_ratio {
        out.push(features.attack_ratio);
    }
    if cfg.include_noise_ratio {
        out.push(features.noise_ratio);
    }
    if cfg.include_inharmonicity {
        out.push(features.inharmonicity);
    }

    normalize_feature_vector(out, &cfg.feature_normalization, cfg.normalization_epsilon)
}

fn normalize_feature_vector(mut x: Vec<f32>, mode: &str, eps: f32) -> Vec<f32> {
    match mode {
        "l2" => {
            let denom = x
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .max(eps.max(1e-12));
            for value in &mut x {
                *value /= denom;
            }
            x
        }
        "zscore" => {
            if x.is_empty() {
                return x;
            }
            let mean = x.iter().sum::<f32>() / x.len() as f32;
            let var = x
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f32>()
                / x.len() as f32;
            let std = var.sqrt().max(eps.max(1e-12));
            for value in &mut x {
                *value = (*value - mean) / std;
            }
            x
        }
        _ => x,
    }
}

pub fn feature_names(cfg: &TemplateConfig) -> Vec<String> {
    let mut names = Vec::new();
    let harmonic_count = cfg.num_harmonics.min(MAX_HARMONICS);

    for h in 0..harmonic_count {
        names.push(format!("harmonic_amps[{h}]"));
    }
    if cfg.include_harmonic_ratios {
        for h in 0..harmonic_count {
            names.push(format!("harmonic_ratios[{h}]"));
        }
    }
    if cfg.include_centroid {
        names.push("centroid".to_string());
    }
    if cfg.include_rolloff {
        names.push("rolloff".to_string());
    }
    if cfg.include_flux {
        names.push("flux".to_string());
    }
    if cfg.include_attack_ratio {
        names.push("attack_ratio".to_string());
    }
    if cfg.include_noise_ratio {
        names.push("noise_ratio".to_string());
    }
    if cfg.include_inharmonicity {
        names.push("inharmonicity".to_string());
    }

    names
}

pub fn harmonic_map_for_midi<'a>(
    maps: &'a [MidiHarmonicMap],
    midi: u8,
) -> Option<&'a MidiHarmonicMap> {
    if maps.is_empty() {
        return None;
    }
    let first = maps[0].midi;
    let idx = midi.checked_sub(first)? as usize;
    maps.get(idx).filter(|map| map.midi == midi)
}

pub fn validate_feature_dim(cfg: &TemplateConfig) -> Result<usize> {
    let dim = feature_names(cfg).len();
    if dim == 0 {
        anyhow::bail!("template feature set is empty");
    }
    Ok(dim)
}

fn selected_harmonic_count(cfg: &TemplateConfig, map: &MidiHarmonicMap) -> usize {
    cfg.num_harmonics.min(MAX_HARMONICS).min(map.ranges.len())
}

fn local_max(x: &[f32], start: usize, end: usize) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let s = start.min(x.len() - 1);
    let e = end.min(x.len() - 1);
    if s > e {
        return 0.0;
    }
    x[s..=e].iter().copied().fold(0.0_f32, f32::max)
}

fn local_note_region(map: &MidiHarmonicMap, harmonic_count: usize, len: usize) -> (usize, usize) {
    if harmonic_count == 0 || len == 0 {
        return (0, 0);
    }
    let mut start = usize::MAX;
    let mut end = 0usize;
    for range in map.ranges.iter().take(harmonic_count) {
        start = start.min(range.start.min(len - 1));
        end = end.max(range.end.min(len - 1));
    }
    if start == usize::MAX || start > end {
        (0, len.saturating_sub(1))
    } else {
        (start, end)
    }
}

fn centroid_and_rolloff(white: &[f32], start: usize, end: usize, hz_per_bin: f32) -> (f32, f32) {
    if white.is_empty() || start > end || start >= white.len() {
        return (0.0, 0.0);
    }
    let end = end.min(white.len() - 1);
    let mut energy_sum = 0.0_f32;
    let mut centroid_num = 0.0_f32;
    for (bin, &energy) in white.iter().enumerate().take(end + 1).skip(start) {
        let e = energy.max(0.0);
        let freq = bin as f32 * hz_per_bin;
        energy_sum += e;
        centroid_num += e * freq;
    }
    let centroid = if energy_sum > EPS {
        centroid_num / energy_sum
    } else {
        0.0
    };

    let target = energy_sum * ROLLOFF_RATIO;
    let mut accum = 0.0_f32;
    let mut rolloff_bin = start;
    for (bin, &energy) in white.iter().enumerate().take(end + 1).skip(start) {
        accum += energy.max(0.0);
        if accum >= target {
            rolloff_bin = bin;
            break;
        }
        rolloff_bin = bin;
    }

    (centroid, rolloff_bin as f32 * hz_per_bin)
}

fn spectral_flux(current: &[f32], previous: Option<&[f32]>, start: usize, end: usize) -> f32 {
    let Some(previous) = previous else {
        return 0.0;
    };
    if current.is_empty() || previous.is_empty() || start > end {
        return 0.0;
    }
    let max_len = current.len().min(previous.len());
    if start >= max_len {
        return 0.0;
    }
    let end = end.min(max_len - 1);
    let mut sum = 0.0_f32;
    let mut n = 0usize;
    for idx in start..=end {
        sum += (current[idx] - previous[idx]).abs();
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}

fn local_noise_ratio(
    white: &[f32],
    map: &MidiHarmonicMap,
    harmonic_count: usize,
    start: usize,
    end: usize,
) -> f32 {
    if white.is_empty() || start > end || start >= white.len() {
        return 0.0;
    }
    let end = end.min(white.len() - 1);
    let len = end - start + 1;
    let mut harmonic_mask = vec![false; len];
    for range in map.ranges.iter().take(harmonic_count) {
        let hs = range.start.max(start).min(end);
        let he = range.end.max(start).min(end);
        if hs > he {
            continue;
        }
        for idx in hs..=he {
            harmonic_mask[idx - start] = true;
        }
    }

    let mut harmonic_energy = 0.0_f32;
    let mut noise_energy = 0.0_f32;
    for idx in start..=end {
        let e = white[idx].max(0.0);
        if harmonic_mask[idx - start] {
            harmonic_energy += e;
        } else {
            noise_energy += e;
        }
    }
    noise_energy / (harmonic_energy + EPS)
}

fn estimate_inharmonicity(
    white: &[f32],
    map: &MidiHarmonicMap,
    harmonic_count: usize,
    hz_per_bin: f32,
    start: usize,
    end: usize,
) -> f32 {
    if white.is_empty() || harmonic_count == 0 || start > end {
        return 0.0;
    }
    let mut weighted_dev = 0.0_f32;
    let mut weight_sum = 0.0_f32;

    for (h_idx, range) in map.ranges.iter().take(harmonic_count).enumerate() {
        let hs = range.start.max(start).min(end);
        let he = range.end.max(start).min(end);
        if hs > he || hs >= white.len() {
            continue;
        }
        let he = he.min(white.len() - 1);
        let mut peak_bin = hs;
        let mut peak_val = 0.0_f32;
        for (bin, &value) in white.iter().enumerate().take(he + 1).skip(hs) {
            let v = value.max(0.0);
            if v > peak_val {
                peak_val = v;
                peak_bin = bin;
            }
        }
        if peak_val <= 0.0 {
            continue;
        }

        let harmonic_number = (h_idx + 1) as f32;
        let ideal_hz = map.f0_hz * harmonic_number;
        let observed_hz = peak_bin as f32 * hz_per_bin;
        let rel_dev = (observed_hz - ideal_hz).abs() / (ideal_hz + EPS);
        weighted_dev += rel_dev * peak_val;
        weight_sum += peak_val;
    }

    if weight_sum > 0.0 {
        weighted_dev / weight_sum
    } else {
        0.0
    }
}

struct NoteFeatureAccumulator {
    midi: u8,
    f0_hz: f32,
    count: usize,
    harmonic_amps: [f64; MAX_HARMONICS],
    harmonic_ratios: [f64; MAX_HARMONICS],
    centroid: f64,
    rolloff: f64,
    flux: f64,
    attack_ratio: f64,
    noise_ratio: f64,
    inharmonicity: f64,
}

impl NoteFeatureAccumulator {
    fn new(midi: u8, f0_hz: f32) -> Self {
        Self {
            midi,
            f0_hz,
            count: 0,
            harmonic_amps: [0.0; MAX_HARMONICS],
            harmonic_ratios: [0.0; MAX_HARMONICS],
            centroid: 0.0,
            rolloff: 0.0,
            flux: 0.0,
            attack_ratio: 0.0,
            noise_ratio: 0.0,
            inharmonicity: 0.0,
        }
    }

    fn update(&mut self, feat: &NoteFeatures) {
        self.count += 1;
        for i in 0..MAX_HARMONICS {
            self.harmonic_amps[i] += feat.harmonic_amps[i] as f64;
            self.harmonic_ratios[i] += feat.harmonic_ratios[i] as f64;
        }
        self.centroid += feat.centroid as f64;
        self.rolloff += feat.rolloff as f64;
        self.flux += feat.flux as f64;
        self.attack_ratio += feat.attack_ratio as f64;
        self.noise_ratio += feat.noise_ratio as f64;
        self.inharmonicity += feat.inharmonicity as f64;
    }

    fn finalize(self) -> Option<NoteFeatures> {
        if self.count == 0 {
            return None;
        }
        let inv = 1.0 / self.count as f64;
        let mut harmonic_amps = [0.0_f32; MAX_HARMONICS];
        let mut harmonic_ratios = [0.0_f32; MAX_HARMONICS];
        for i in 0..MAX_HARMONICS {
            harmonic_amps[i] = (self.harmonic_amps[i] * inv) as f32;
            harmonic_ratios[i] = (self.harmonic_ratios[i] * inv) as f32;
        }

        Some(NoteFeatures {
            midi: self.midi,
            f0_hz: self.f0_hz,
            harmonic_amps,
            harmonic_ratios,
            centroid: (self.centroid * inv) as f32,
            rolloff: (self.rolloff * inv) as f32,
            flux: (self.flux * inv) as f32,
            attack_ratio: (self.attack_ratio * inv) as f32,
            noise_ratio: (self.noise_ratio * inv) as f32,
            inharmonicity: (self.inharmonicity * inv) as f32,
        })
    }
}

pub fn extract_note_features_for_record(
    annotation: &MonoAnnotation,
    audio: &AudioBuffer,
    stft_plan: &StftPlan,
    cfg: &AppConfig,
    maps: &[MidiHarmonicMap],
) -> Result<Option<NoteFeatures>> {
    let map = harmonic_map_for_midi(maps, annotation.midi).with_context(|| {
        format!(
            "no harmonic map for midi={} in configured range [{}, {}]",
            annotation.midi,
            maps.first().map(|m| m.midi).unwrap_or(0),
            maps.last().map(|m| m.midi).unwrap_or(0)
        )
    })?;
    extract_note_features_from_annotation(annotation, audio, stft_plan, cfg, map)
}

#[cfg(test)]
mod tests {
    use crate::config::TemplateConfig;
    use crate::types::{HarmonicBinRange, MidiHarmonicMap, NoteFeatures, MAX_HARMONICS};

    use super::{extract_note_features, feature_names, flatten_note_features};

    #[test]
    fn flatten_order_matches_feature_names() {
        let cfg = TemplateConfig {
            num_harmonics: 3,
            include_harmonic_ratios: true,
            include_centroid: true,
            include_rolloff: false,
            include_flux: true,
            include_attack_ratio: true,
            include_noise_ratio: false,
            include_inharmonicity: true,
            ..TemplateConfig::default()
        };

        let mut harmonic_amps = [0.0_f32; MAX_HARMONICS];
        let mut harmonic_ratios = [0.0_f32; MAX_HARMONICS];
        harmonic_amps[..3].copy_from_slice(&[1.0, 2.0, 3.0]);
        harmonic_ratios[..3].copy_from_slice(&[0.5, 0.25, 0.125]);
        let f = NoteFeatures {
            midi: 60,
            f0_hz: 261.63,
            harmonic_amps,
            harmonic_ratios,
            centroid: 100.0,
            rolloff: 200.0,
            flux: 0.5,
            attack_ratio: 0.2,
            noise_ratio: 0.1,
            inharmonicity: 0.05,
        };

        let z = flatten_note_features(&f, &cfg);
        let names = feature_names(&cfg);
        assert_eq!(z.len(), names.len());
        assert_eq!(
            z,
            vec![1.0, 2.0, 3.0, 0.5, 0.25, 0.125, 100.0, 0.5, 0.2, 0.05]
        );
    }

    #[test]
    fn extract_features_produces_expected_basic_harmonics() {
        let cfg = TemplateConfig {
            num_harmonics: 2,
            include_harmonic_ratios: true,
            ..TemplateConfig::default()
        };
        let map = MidiHarmonicMap {
            midi: 69,
            f0_hz: 440.0,
            ranges: vec![
                HarmonicBinRange { start: 1, end: 2 },
                HarmonicBinRange { start: 3, end: 4 },
            ],
        };

        let current = vec![0.0, 1.0, 2.0, 4.0, 3.0, 0.0];
        let prev = vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let f = extract_note_features(&current, Some(&prev), 69, &map, 8_000, 1024, &cfg);
        assert_eq!(f.harmonic_amps[0], 2.0);
        assert_eq!(f.harmonic_amps[1], 4.0);
        assert!((f.harmonic_ratios[0] - 1.0).abs() < 1e-6);
        assert!((f.harmonic_ratios[1] - 2.0).abs() < 1e-6);
        assert!(f.flux > 0.0);
    }

    #[test]
    fn flatten_applies_l2_normalization_when_enabled() {
        let cfg = TemplateConfig {
            num_harmonics: 2,
            include_harmonic_ratios: false,
            include_centroid: false,
            include_rolloff: false,
            include_flux: false,
            include_attack_ratio: false,
            include_noise_ratio: false,
            include_inharmonicity: false,
            feature_normalization: "l2".to_string(),
            ..TemplateConfig::default()
        };
        let mut harmonic_amps = [0.0_f32; MAX_HARMONICS];
        harmonic_amps[..2].copy_from_slice(&[3.0, 4.0]);
        let f = NoteFeatures {
            midi: 60,
            f0_hz: 261.63,
            harmonic_amps,
            harmonic_ratios: [0.0_f32; MAX_HARMONICS],
            centroid: 0.0,
            rolloff: 0.0,
            flux: 0.0,
            attack_ratio: 0.0,
            noise_ratio: 0.0,
            inharmonicity: 0.0,
        };

        let z = flatten_note_features(&f, &cfg);
        assert_eq!(z.len(), 2);
        assert!((z[0] - 0.6).abs() < 1e-6);
        assert!((z[1] - 0.8).abs() < 1e-6);
    }
}
