use serde::{Deserialize, Serialize};

pub const OPEN_MIDI: [u8; 6] = [40, 45, 50, 55, 59, 64];
pub const MAX_HARMONICS: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonoAnnotation {
    pub audio_path: String,
    pub sample_rate: u32,
    pub midi: u8,
    pub string: u8,
    pub fret: u8,
    pub onset_sec: f32,
    pub offset_sec: f32,
    pub player_id: Option<String>,
    pub guitar_id: Option<String>,
    pub style: Option<String>,
    pub pickup_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyEvent {
    pub onset_sec: f32,
    pub offset_sec: f32,
    pub midi: u8,
    pub string: Option<u8>,
    pub fret: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyAnnotation {
    pub audio_path: String,
    pub sample_rate: u32,
    pub events: Vec<PolyEvent>,
    pub player_id: Option<String>,
    pub guitar_id: Option<String>,
    pub style: Option<String>,
    pub pickup_mode: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SpectrumFrame {
    pub frame_index: usize,
    pub time_sec: f32,
    pub mag: Vec<f32>,
    pub white: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct HarmonicBinRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct MidiHarmonicMap {
    pub midi: u8,
    pub f0_hz: f32,
    pub ranges: Vec<HarmonicBinRange>,
}

#[derive(Debug, Clone)]
pub struct PitchScoreFrame {
    pub time_sec: f32,
    pub midi_scores: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct NoteFeatures {
    pub midi: u8,
    pub f0_hz: f32,
    pub harmonic_amps: [f32; MAX_HARMONICS],
    pub harmonic_ratios: [f32; MAX_HARMONICS],
    pub centroid: f32,
    pub rolloff: f32,
    pub flux: f32,
    pub attack_ratio: f32,
    pub noise_ratio: f32,
    pub inharmonicity: f32,
}

#[derive(Debug, Clone)]
pub struct CandidatePosition {
    pub midi: u8,
    pub string: u8,
    pub fret: u8,
    pub local_score: f32,
}

#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub time_sec: f32,
    pub positions: Vec<CandidatePosition>,
    pub total_score: f32,
}
