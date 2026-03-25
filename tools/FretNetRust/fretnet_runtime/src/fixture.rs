use std::{
    fs::File,
    path::{Path, PathBuf},
};

use ndarray::{Array1, Array3, Array5, ArrayD};
use ndarray_npy::NpzReader;

use crate::{
    config::{DEFAULT_OUTPUT_NAMES, default_feature_fixture_paths, default_fixture_path},
    error::{FrontendError, RuntimeError},
    types::{FeatureBatch, HcqtFeatures, ModelOutput, NamedTensorOutput},
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

    let features: Array5<f32> = archive.by_name("features.npy").map_err(|err| RuntimeError::FixtureFormat {
        path: path.clone(),
        message: format!("failed to load features.npy: {err}"),
    })?;

    let mut tensors = Vec::new();
    for name in DEFAULT_OUTPUT_NAMES {
        let array: ArrayD<f32> = archive.by_name(&format!("{name}.npy")).map_err(|err| RuntimeError::FixtureFormat {
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

pub fn load_feature_reference_fixture(path: impl AsRef<Path>) -> Result<FeatureReferenceFixture, FrontendError> {
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

    let audio: Array1<f32> = archive.by_name("audio.npy").map_err(|err| FrontendError::FeatureFixtureFormat {
        path: path.clone(),
        message: format!("failed to load audio.npy: {err}"),
    })?;
    let hcqt: Array3<f32> = archive.by_name("hcqt.npy").map_err(|err| FrontendError::FeatureFixtureFormat {
        path: path.clone(),
        message: format!("failed to load hcqt.npy: {err}"),
    })?;
    let times: Array1<f32> = archive.by_name("times.npy").map_err(|err| FrontendError::FeatureFixtureFormat {
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
