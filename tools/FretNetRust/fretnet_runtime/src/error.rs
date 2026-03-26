use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FrontendError {
    #[error("audio path does not exist: {0}")]
    MissingAudioFile(PathBuf),

    #[error("unsupported sample rate: expected {expected}, got {actual}")]
    UnsupportedSampleRate { expected: u32, actual: u32 },

    #[error("invalid audio input: {0}")]
    InvalidAudio(String),

    #[error("audio I/O failed for '{path}': {message}")]
    AudioIo { path: PathBuf, message: String },

    #[error("feature fixture file does not exist: {0}")]
    MissingFeatureFixtureFile(PathBuf),

    #[error("failed to read feature fixture '{path}': {message}")]
    FeatureFixtureIo { path: PathBuf, message: String },

    #[error("invalid feature fixture '{path}': {message}")]
    FeatureFixtureFormat { path: PathBuf, message: String },

    #[error("stage fixture path does not exist: {0}")]
    MissingStageFixture(PathBuf),

    #[error("failed to read stage fixture '{path}': {message}")]
    StageFixtureIo { path: PathBuf, message: String },

    #[error("invalid stage fixture '{path}': {message}")]
    StageFixtureFormat { path: PathBuf, message: String },

    #[error("frontend configuration mismatch: {0}")]
    ConfigMismatch(String),

    #[error("frontend shape mismatch: {0}")]
    ShapeMismatch(String),
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("model file does not exist: {0}")]
    MissingModelFile(PathBuf),

    #[error("fixture file does not exist: {0}")]
    MissingFixtureFile(PathBuf),

    #[error("failed to create ONNX Runtime session: {0}")]
    SessionCreation(String),

    #[error("invalid feature shape: expected rank {expected_rank}, got {actual_rank}")]
    InvalidFeatureRank {
        expected_rank: usize,
        actual_rank: usize,
    },

    #[error("feature dimension mismatch on axis {axis}: expected {expected}, got {actual}")]
    FeatureDimensionMismatch {
        axis: usize,
        expected: usize,
        actual: usize,
    },

    #[error("unsupported element type: {0}")]
    UnsupportedElementType(String),

    #[error("unexpected number of model inputs: expected 1, got {0}")]
    UnexpectedInputCount(usize),

    #[error("unexpected number of model outputs: expected at least 1, got {0}")]
    UnexpectedOutputCount(usize),

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("failed to extract output tensor '{name}': {message}")]
    OutputExtraction { name: String, message: String },

    #[error("failed to read fixture '{path}': {message}")]
    FixtureIo { path: PathBuf, message: String },

    #[error("invalid fixture '{path}': {message}")]
    FixtureFormat { path: PathBuf, message: String },

    #[error("decoding failed: {0}")]
    Decode(String),
}
