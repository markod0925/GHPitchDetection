pub mod audio;
pub mod config;
pub mod fretboard;
pub mod harmonic_map;
pub mod infer;
pub mod nms;
pub mod pitch;
pub mod stft;
pub mod types;
pub mod whitening;

// TODO(phase-2): implement dataset parsing and feature extraction modules.
pub mod dataset;
pub mod features;
pub mod templates;

// TODO(phase-3): implement candidate scoring and global decoding modules.
pub mod decode;

// TODO(phase-4): implement tracking and stability modules.
pub mod tracking;

// TODO(later): add evaluation and hyperparameter optimization modules.
pub mod cli_eval;
pub mod cli_infer;
pub mod cli_optimize;
pub mod cli_train;
pub mod metrics;
pub mod optimize;
