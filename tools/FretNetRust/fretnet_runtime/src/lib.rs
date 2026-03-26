pub mod audio;
pub mod benchmark;
pub mod config;
pub mod error;
pub mod fixture;
pub mod frontend;
pub mod hcqt;
pub mod inference;
pub mod model;
pub mod postprocess;
pub mod profile;
pub mod types;

pub use crate::{
    audio::{load_audio_for_frontend, load_wav_mono, AudioBuffer},
    benchmark::{
        benchmark_end_to_end_audio, benchmark_frontend_profile_audio, benchmark_runtime,
        BenchmarkSummary, EndToEndBenchmarkSummary, FrontendHarmonicProfileSummary,
        FrontendOctaveProfileSummary, FrontendOctavePyramidSummary,
        FrontendProfileBenchmarkSummary, OutputTensorSummary, ScalarProfileSummary,
        StageLatencySummary,
    },
    config::{
        default_feature_fixture_paths, default_fixture_metadata_path, default_fixture_path,
        default_fixture_paths, default_model_path, default_stage_fixture_paths,
    },
    error::{FrontendError, RuntimeError},
    fixture::{
        default_existing_feature_fixture_paths, default_existing_stage_audio_inputs,
        default_existing_stage_fixture_paths, load_feature_reference_fixture,
        load_regression_fixture, load_stage_reference_fixture, stage_fixture_to_outputs,
        try_default_regression_fixture, FeatureReferenceFixture, RegressionFixture,
        StageAudioInput, StageFixtureMetadata, StageFrontendMetadata, StageHarmonicMetadata,
        StageHarmonicReference, StageReferenceFixture,
    },
    frontend::FeatureExtractor,
    hcqt::{FrontendConfig, HcqtExtractor},
    model::FretNetRuntime,
    profile::{
        FrontendInitProfile, FrontendRunProfile, HarmonicInitProfile, HarmonicRunProfile,
        OctaveInitProfile, OctavePyramidProfile, OctaveRunProfile,
    },
    types::{
        DecodedOutput, FeatureBatch, FrontendStageOutputs, HarmonicStageOutput, HcqtFeatures,
        ModelMetadata, ModelOutput, NamedTensorOutput, TensorMetadata,
    },
};
