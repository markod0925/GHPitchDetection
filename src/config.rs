use std::fs;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub sample_rate_target: u32,
    pub frontend: FrontendConfig,
    pub whitening: WhiteningConfig,
    pub pitch: PitchConfig,
    pub nms: NmsConfig,
    pub templates: TemplateConfig,
    pub decode: DecodeConfig,
    pub tracking: TrackingConfig,
    pub optimization: OptimizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FrontendConfig {
    pub kind: String,
    pub win_length: usize,
    pub hop_length: usize,
    pub nfft: usize,
    pub window: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WhiteningConfig {
    pub method: String,
    pub domain: String,
    pub span_bins: usize,
    pub rectify_positive: bool,
    pub floor_eps: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PitchConfig {
    pub midi_min: u8,
    pub midi_max: u8,
    pub max_harmonics: usize,
    pub harmonic_alpha: f32,
    pub local_bandwidth_bins: usize,
    pub local_aggregation: String,
    pub use_subharmonic_penalty: bool,
    pub lambda_subharmonic: f32,
    pub use_missing_penalty: bool,
    pub lambda_missing: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NmsConfig {
    pub threshold_mode: String,
    pub threshold_value: f32,
    pub radius_midi_steps: usize,
    pub max_polyphony: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TemplateConfig {
    pub level: String,
    pub num_harmonics: usize,
    pub include_harmonic_ratios: bool,
    pub include_centroid: bool,
    pub include_rolloff: bool,
    pub include_flux: bool,
    pub include_attack_ratio: bool,
    pub include_noise_ratio: bool,
    pub include_inharmonicity: bool,
    pub score_type: String,
    pub feature_normalization: String,
    pub normalization_epsilon: f32,
    pub mahalanobis_std_floor: f32,
    pub scorer_epsilon: f32,
    pub region_bounds: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DecodeConfig {
    pub alpha: f32,
    pub beta: f32,
    pub normalization_epsilon: f32,
    pub lambda_conflict: f32,
    pub lambda_fret_spread: f32,
    pub lambda_position_prior: f32,
    pub max_candidates_per_pitch: usize,
    pub max_global_solutions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackingConfig {
    pub mode: String,
    pub smooth_frames: usize,
    pub lambda_switch_string: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizationConfig {
    pub search_strategy: String,
    pub trials: usize,
    pub seed: u64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            sample_rate_target: 44_100,
            frontend: FrontendConfig::default(),
            whitening: WhiteningConfig::default(),
            pitch: PitchConfig::default(),
            nms: NmsConfig::default(),
            templates: TemplateConfig::default(),
            decode: DecodeConfig::default(),
            tracking: TrackingConfig::default(),
            optimization: OptimizationConfig::default(),
        }
    }
}

impl Default for FrontendConfig {
    fn default() -> Self {
        Self {
            kind: "stft".to_string(),
            win_length: 4096,
            hop_length: 512,
            nfft: 4096,
            window: "hann".to_string(),
        }
    }
}

impl Default for WhiteningConfig {
    fn default() -> Self {
        Self {
            method: "moving_average".to_string(),
            domain: "logmag".to_string(),
            span_bins: 31,
            rectify_positive: true,
            floor_eps: 1e-8,
        }
    }
}

impl Default for PitchConfig {
    fn default() -> Self {
        Self {
            midi_min: 40,
            midi_max: 88,
            max_harmonics: 8,
            harmonic_alpha: 0.8,
            local_bandwidth_bins: 2,
            local_aggregation: "max".to_string(),
            use_subharmonic_penalty: true,
            lambda_subharmonic: 0.15,
            use_missing_penalty: false,
            lambda_missing: 0.0,
        }
    }
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self {
            threshold_mode: "relative".to_string(),
            threshold_value: 0.30,
            radius_midi_steps: 1,
            max_polyphony: 6,
        }
    }
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            level: "string_region".to_string(),
            num_harmonics: 8,
            include_harmonic_ratios: true,
            include_centroid: true,
            include_rolloff: true,
            include_flux: true,
            include_attack_ratio: false,
            include_noise_ratio: true,
            include_inharmonicity: true,
            score_type: "diag_mahalanobis".to_string(),
            feature_normalization: "none".to_string(),
            normalization_epsilon: 1e-6,
            mahalanobis_std_floor: 1e-3,
            scorer_epsilon: 1e-6,
            region_bounds: vec![0, 5, 10, 15, 21],
        }
    }
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            normalization_epsilon: 1e-6,
            lambda_conflict: 1000.0,
            lambda_fret_spread: 0.1,
            lambda_position_prior: 0.02,
            max_candidates_per_pitch: 3,
            max_global_solutions: 256,
        }
    }
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            mode: "event".to_string(),
            smooth_frames: 3,
            lambda_switch_string: 0.2,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            search_strategy: "random".to_string(),
            trials: 100,
            seed: 12_345,
        }
    }
}

pub fn load_app_config(path: &str) -> Result<AppConfig> {
    let raw = fs::read_to_string(path).with_context(|| format!("failed to read config: {path}"))?;
    let cfg: AppConfig =
        toml::from_str(&raw).with_context(|| format!("failed to parse TOML: {path}"))?;
    validate_config(&cfg)?;
    Ok(cfg)
}

fn validate_config(cfg: &AppConfig) -> Result<()> {
    if cfg.frontend.kind != "stft" && cfg.frontend.kind != "qdft" {
        bail!("unsupported frontend.kind: {}", cfg.frontend.kind);
    }
    if cfg.frontend.window != "hann" {
        bail!("unsupported frontend.window: {}", cfg.frontend.window);
    }
    if cfg.frontend.win_length == 0 || cfg.frontend.hop_length == 0 || cfg.frontend.nfft == 0 {
        bail!("frontend lengths must be positive");
    }
    if cfg.frontend.win_length > cfg.frontend.nfft {
        bail!("frontend.win_length must be <= frontend.nfft");
    }
    if cfg.whitening.method != "moving_average" {
        bail!("unsupported whitening.method: {}", cfg.whitening.method);
    }
    if cfg.whitening.domain != "logmag" {
        bail!("unsupported whitening.domain: {}", cfg.whitening.domain);
    }
    if cfg.pitch.midi_min > cfg.pitch.midi_max {
        bail!("pitch.midi_min must be <= pitch.midi_max");
    }
    if cfg.pitch.max_harmonics == 0 {
        bail!("pitch.max_harmonics must be > 0");
    }
    if cfg.pitch.local_aggregation != "max" && cfg.pitch.local_aggregation != "mean" {
        bail!(
            "unsupported pitch.local_aggregation: {}",
            cfg.pitch.local_aggregation
        );
    }
    if cfg.nms.threshold_mode != "relative" && cfg.nms.threshold_mode != "absolute" {
        bail!("unsupported nms.threshold_mode: {}", cfg.nms.threshold_mode);
    }
    if cfg.templates.level != "string"
        && cfg.templates.level != "string_region"
        && cfg.templates.level != "string_fret"
    {
        bail!("unsupported templates.level: {}", cfg.templates.level);
    }
    if cfg.templates.score_type != "diag_mahalanobis" && cfg.templates.score_type != "cosine" {
        bail!(
            "unsupported templates.score_type: {}",
            cfg.templates.score_type
        );
    }
    if cfg.templates.feature_normalization != "none"
        && cfg.templates.feature_normalization != "l2"
        && cfg.templates.feature_normalization != "zscore"
    {
        bail!(
            "unsupported templates.feature_normalization: {}",
            cfg.templates.feature_normalization
        );
    }
    if cfg.templates.num_harmonics == 0 {
        bail!("templates.num_harmonics must be > 0");
    }
    if !cfg.templates.normalization_epsilon.is_finite()
        || cfg.templates.normalization_epsilon <= 0.0
    {
        bail!("templates.normalization_epsilon must be > 0");
    }
    if !cfg.templates.mahalanobis_std_floor.is_finite() || cfg.templates.mahalanobis_std_floor < 0.0
    {
        bail!("templates.mahalanobis_std_floor must be >= 0");
    }
    if !cfg.templates.scorer_epsilon.is_finite() || cfg.templates.scorer_epsilon <= 0.0 {
        bail!("templates.scorer_epsilon must be > 0");
    }
    if cfg.templates.level == "string_region" && cfg.templates.region_bounds.len() < 2 {
        bail!("templates.region_bounds must contain at least 2 bounds for string_region");
    }
    if !cfg.decode.alpha.is_finite() || !cfg.decode.beta.is_finite() {
        bail!("decode.alpha and decode.beta must be finite");
    }
    if !cfg.decode.normalization_epsilon.is_finite() || cfg.decode.normalization_epsilon <= 0.0 {
        bail!("decode.normalization_epsilon must be > 0");
    }
    if cfg.decode.max_candidates_per_pitch == 0 {
        bail!("decode.max_candidates_per_pitch must be > 0");
    }
    if cfg.decode.max_global_solutions == 0 {
        bail!("decode.max_global_solutions must be > 0");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::AppConfig;

    #[test]
    fn defaults_match_spec_core_values() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.sample_rate_target, 44_100);
        assert_eq!(cfg.frontend.nfft, 4096);
        assert_eq!(cfg.pitch.midi_min, 40);
        assert_eq!(cfg.pitch.midi_max, 88);
        assert_eq!(cfg.nms.max_polyphony, 6);
    }
}
