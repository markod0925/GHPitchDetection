use std::{
    fs,
    fs::File,
    path::{Path, PathBuf},
};

use ndarray::{Array1, Array2, Array3, Array5, ArrayD};
use ndarray_npy::NpzReader;
use serde::Deserialize;

use crate::{
    config::{
        default_feature_fixture_paths, default_fixture_path, default_stage_fixture_paths,
        DEFAULT_OUTPUT_NAMES,
    },
    error::{FrontendError, RuntimeError},
    types::{
        FeatureBatch, FrontendStageOutputs, HarmonicStageOutput, HcqtFeatures, ModelOutput,
        NamedTensorOutput,
    },
};

#[derive(Debug, Clone)]
pub struct RegressionFixture {
    pub features: FeatureBatch,
    pub reference_output: ModelOutput,
}

pub fn load_regression_fixture(path: impl AsRef<Path>) -> Result<RegressionFixture, RuntimeError> {
    let path = path.as_ref().to_path_buf();
    if !path.exists() {
        return Err(RuntimeError::MissingFixtureFile(path));
    }

    let file = File::open(&path).map_err(|err| RuntimeError::FixtureIo {
        path: path.clone(),
        message: err.to_string(),
    })?;
    let mut archive = NpzReader::new(file).map_err(|err| RuntimeError::FixtureFormat {
        path: path.clone(),
        message: err.to_string(),
    })?;

    let features: Array5<f32> =
        archive
            .by_name("features.npy")
            .map_err(|err| RuntimeError::FixtureFormat {
                path: path.clone(),
                message: format!("failed to load features.npy: {err}"),
            })?;

    let mut tensors = Vec::new();
    for name in DEFAULT_OUTPUT_NAMES {
        let array: ArrayD<f32> =
            archive
                .by_name(&format!("{name}.npy"))
                .map_err(|err| RuntimeError::FixtureFormat {
                    path: path.clone(),
                    message: format!("failed to load {name}.npy: {err}"),
                })?;

        tensors.push(NamedTensorOutput {
            name: name.to_owned(),
            data: array,
        });
    }

    Ok(RegressionFixture {
        features: FeatureBatch::new(features),
        reference_output: ModelOutput { tensors },
    })
}

pub fn try_default_regression_fixture() -> Option<PathBuf> {
    let path = default_fixture_path();
    path.exists().then_some(path)
}

#[derive(Debug, Clone)]
pub struct FeatureReferenceFixture {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    pub hcqt: HcqtFeatures,
    pub times: Vec<f32>,
}

pub fn load_feature_reference_fixture(
    path: impl AsRef<Path>,
) -> Result<FeatureReferenceFixture, FrontendError> {
    let path = path.as_ref().to_path_buf();
    if !path.exists() {
        return Err(FrontendError::MissingFeatureFixtureFile(path));
    }

    let file = File::open(&path).map_err(|err| FrontendError::FeatureFixtureIo {
        path: path.clone(),
        message: err.to_string(),
    })?;
    let mut archive = NpzReader::new(file).map_err(|err| FrontendError::FeatureFixtureFormat {
        path: path.clone(),
        message: err.to_string(),
    })?;

    let audio: Array1<f32> =
        archive
            .by_name("audio.npy")
            .map_err(|err| FrontendError::FeatureFixtureFormat {
                path: path.clone(),
                message: format!("failed to load audio.npy: {err}"),
            })?;
    let hcqt: Array3<f32> =
        archive
            .by_name("hcqt.npy")
            .map_err(|err| FrontendError::FeatureFixtureFormat {
                path: path.clone(),
                message: format!("failed to load hcqt.npy: {err}"),
            })?;
    let times: Array1<f32> =
        archive
            .by_name("times.npy")
            .map_err(|err| FrontendError::FeatureFixtureFormat {
                path: path.clone(),
                message: format!("failed to load times.npy: {err}"),
            })?;

    let sample_rate = 22_050;
    Ok(FeatureReferenceFixture {
        audio: audio.to_vec(),
        sample_rate,
        hcqt: HcqtFeatures::new(hcqt, sample_rate, 512),
        times: times.to_vec(),
    })
}

pub fn default_existing_feature_fixture_paths() -> Vec<PathBuf> {
    default_feature_fixture_paths()
        .into_iter()
        .filter(|path| path.exists())
        .collect()
}

#[derive(Debug, Clone, Deserialize)]
pub struct StageAudioMetadata {
    pub sample_rate: u32,
    pub num_samples: usize,
    pub duration_seconds: f32,
    pub channels: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StageFileMetadata {
    pub path: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub axis_semantics: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StageHarmonicMetadata {
    pub harmonic: f32,
    pub label: String,
    pub magnitude: StageFileMetadata,
    pub post_processed: StageFileMetadata,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StageFrontendMetadata {
    pub sample_rate: u32,
    pub hop_length: usize,
    pub fmin_hz: f32,
    pub harmonics: Vec<f32>,
    pub n_bins: usize,
    pub bins_per_octave: usize,
    pub top_db: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StageFixtureMetadata {
    pub fixture_name: String,
    pub source_audio_path: String,
    pub original_audio: StageAudioMetadata,
    pub resampled_audio: StageFileMetadata,
    pub times: StageFileMetadata,
    pub frontend: StageFrontendMetadata,
    pub harmonics: Vec<StageHarmonicMetadata>,
    pub hcqt: StageFileMetadata,
    pub max_frames: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct StageHarmonicReference {
    pub harmonic: f32,
    pub label: String,
    pub magnitude: Array2<f32>,
    pub post_processed: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct StageReferenceFixture {
    pub root: PathBuf,
    pub metadata: StageFixtureMetadata,
    pub resampled_audio: Vec<f32>,
    pub times: Vec<f32>,
    pub harmonics: Vec<StageHarmonicReference>,
    pub hcqt: HcqtFeatures,
}

#[derive(Debug, Clone)]
pub struct StageAudioInput {
    pub fixture_name: String,
    pub audio_path: PathBuf,
    pub sample_rate: u32,
    pub num_samples: usize,
    pub duration_seconds: f32,
}

pub fn load_stage_reference_fixture(
    path: impl AsRef<Path>,
) -> Result<StageReferenceFixture, FrontendError> {
    let root = path.as_ref().to_path_buf();
    if !root.exists() {
        return Err(FrontendError::MissingStageFixture(root));
    }
    if !root.is_dir() {
        return Err(FrontendError::StageFixtureFormat {
            path: root.clone(),
            message: "stage fixture path must be a directory".to_owned(),
        });
    }

    let metadata_path = root.join("metadata.json");
    let metadata_text =
        fs::read_to_string(&metadata_path).map_err(|err| FrontendError::StageFixtureIo {
            path: metadata_path.clone(),
            message: err.to_string(),
        })?;
    let metadata: StageFixtureMetadata =
        serde_json::from_str(&metadata_text).map_err(|err| FrontendError::StageFixtureFormat {
            path: metadata_path.clone(),
            message: err.to_string(),
        })?;

    let resampled_audio = load_stage_vector(&root, &metadata.resampled_audio.path, "audio.npy")?;
    let times = load_stage_vector(&root, &metadata.times.path, "times.npy")?;
    let hcqt_array = load_stage_array3(&root, &metadata.hcqt.path, "hcqt.npy")?;
    let hcqt = HcqtFeatures::new(
        hcqt_array,
        metadata.frontend.sample_rate,
        metadata.frontend.hop_length,
    );

    let mut harmonics = Vec::with_capacity(metadata.harmonics.len());
    for harmonic in &metadata.harmonics {
        let magnitude = load_stage_array2(&root, &harmonic.magnitude.path, "magnitude.npy")?;
        let post_processed =
            load_stage_array2(&root, &harmonic.post_processed.path, "post_processed.npy")?;
        harmonics.push(StageHarmonicReference {
            harmonic: harmonic.harmonic,
            label: harmonic.label.clone(),
            magnitude,
            post_processed,
        });
    }

    Ok(StageReferenceFixture {
        root,
        metadata,
        resampled_audio,
        times,
        harmonics,
        hcqt,
    })
}

pub fn default_existing_stage_fixture_paths() -> Vec<PathBuf> {
    default_stage_fixture_paths()
        .into_iter()
        .filter(|path| path.exists())
        .collect()
}

pub fn default_existing_stage_audio_inputs() -> Result<Vec<StageAudioInput>, FrontendError> {
    let mut inputs = Vec::new();
    for path in default_existing_stage_fixture_paths() {
        let fixture = load_stage_reference_fixture(&path)?;
        let audio_path = PathBuf::from(&fixture.metadata.source_audio_path);
        if !audio_path.exists() {
            return Err(FrontendError::MissingAudioFile(audio_path));
        }

        inputs.push(StageAudioInput {
            fixture_name: fixture.metadata.fixture_name,
            audio_path,
            sample_rate: fixture.metadata.original_audio.sample_rate,
            num_samples: fixture.metadata.original_audio.num_samples,
            duration_seconds: fixture.metadata.original_audio.duration_seconds,
        });
    }

    Ok(inputs)
}

pub fn stage_fixture_to_outputs(fixture: &StageReferenceFixture) -> FrontendStageOutputs {
    let harmonics = fixture
        .harmonics
        .iter()
        .map(|harmonic| HarmonicStageOutput {
            harmonic: harmonic.harmonic,
            magnitude: harmonic.magnitude.clone(),
            post_processed: harmonic.post_processed.clone(),
        })
        .collect();

    FrontendStageOutputs {
        hcqt: fixture.hcqt.clone(),
        harmonics,
    }
}

fn load_stage_vector(
    root: &Path,
    relative_path: &str,
    array_name: &str,
) -> Result<Vec<f32>, FrontendError> {
    let array = load_stage_array1(root, relative_path, array_name)?;
    Ok(array.to_vec())
}

fn load_stage_array2(
    root: &Path,
    relative_path: &str,
    array_name: &str,
) -> Result<Array2<f32>, FrontendError> {
    let path = root.join(relative_path);
    let file = File::open(&path).map_err(|err| FrontendError::StageFixtureIo {
        path: path.clone(),
        message: err.to_string(),
    })?;
    let mut archive = NpzReader::new(file).map_err(|err| FrontendError::StageFixtureFormat {
        path: path.clone(),
        message: err.to_string(),
    })?;

    archive
        .by_name(array_name)
        .map_err(|err| FrontendError::StageFixtureFormat {
            path,
            message: format!("failed to load {array_name}: {err}"),
        })
}

fn load_stage_array3(
    root: &Path,
    relative_path: &str,
    array_name: &str,
) -> Result<Array3<f32>, FrontendError> {
    let path = root.join(relative_path);
    let file = File::open(&path).map_err(|err| FrontendError::StageFixtureIo {
        path: path.clone(),
        message: err.to_string(),
    })?;
    let mut archive = NpzReader::new(file).map_err(|err| FrontendError::StageFixtureFormat {
        path: path.clone(),
        message: err.to_string(),
    })?;

    archive
        .by_name(array_name)
        .map_err(|err| FrontendError::StageFixtureFormat {
            path,
            message: format!("failed to load {array_name}: {err}"),
        })
}

fn load_stage_array1(
    root: &Path,
    relative_path: &str,
    array_name: &str,
) -> Result<Array1<f32>, FrontendError> {
    let path = root.join(relative_path);
    let file = File::open(&path).map_err(|err| FrontendError::StageFixtureIo {
        path: path.clone(),
        message: err.to_string(),
    })?;
    let mut archive = NpzReader::new(file).map_err(|err| FrontendError::StageFixtureFormat {
        path: path.clone(),
        message: err.to_string(),
    })?;

    archive
        .by_name(array_name)
        .map_err(|err| FrontendError::StageFixtureFormat {
            path,
            message: format!("failed to load {array_name}: {err}"),
        })
}
