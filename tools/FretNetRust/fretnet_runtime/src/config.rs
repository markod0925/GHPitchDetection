use std::path::PathBuf;

pub const DEFAULT_INPUT_NAME: &str = "feats";
pub const DEFAULT_OUTPUT_NAMES: [&str; 3] = ["tablature", "tablature_rel", "onsets"];
pub const FRONTEND_SAMPLE_RATE: u32 = 22_050;
pub const FRONTEND_HOP_LENGTH: usize = 512;
pub const FRONTEND_FMIN_HZ: f32 = 82.406_89;
pub const FRONTEND_HARMONICS: [f32; 6] = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
pub const FRONTEND_N_BINS: usize = 144;
pub const FRONTEND_BINS_PER_OCTAVE: usize = 36;
pub const FRONTEND_FRAME_WIDTH: usize = 9;
pub const FRONTEND_TOP_DB: f32 = 80.0;
pub const FRONTEND_SPARSITY: f32 = 0.01;

pub fn default_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../model/model.onnx")
}

pub fn default_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/test_features/00_BN1-129-Eb_comp_mic_f130.npz")
}

pub fn default_fixture_metadata_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/test_features/00_BN1-129-Eb_comp_mic_f130.json")
}

pub fn default_fixture_paths() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/test_features");
    vec![
        root.join("00_BN1-129-Eb_comp_mic_f130.npz"),
        root.join("00_Jazz1-130-D_comp_mic_f130.npz"),
    ]
}

pub fn default_feature_fixture_paths() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/test_features_python");
    vec![
        root.join("00_BN1-129-Eb_comp_mic_features_f130.npz"),
        root.join("00_Jazz1-130-D_comp_mic_features_f130.npz"),
    ]
}

pub fn default_stage_fixture_paths() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/test_features_python_stages");
    vec![
        root.join("00_BN1-129-Eb_comp_mic"),
        root.join("00_Jazz1-130-D_comp_mic"),
    ]
}
