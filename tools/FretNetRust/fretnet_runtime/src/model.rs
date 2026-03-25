use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::ValueType,
};

use crate::{
    config::DEFAULT_OUTPUT_NAMES,
    error::RuntimeError,
    inference::run_raw_session,
    postprocess::decode_conservatively,
    types::{DecodedOutput, FeatureBatch, ModelMetadata, ModelOutput, NamedTensorOutput, TensorMetadata},
};

pub struct FretNetRuntime {
    model_path: PathBuf,
    session: Session,
    metadata: ModelMetadata,
    load_time: Duration,
}

impl FretNetRuntime {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.exists() {
            return Err(RuntimeError::MissingModelFile(model_path));
        }

        let started = Instant::now();

        let session = Session::builder()
            .map_err(|err| RuntimeError::SessionCreation(err.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|err| RuntimeError::SessionCreation(err.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|err| RuntimeError::SessionCreation(err.to_string()))?;

        let metadata = ModelMetadata {
            model_path: model_path.display().to_string(),
            inputs: collect_input_metadata(session.inputs()),
            outputs: collect_output_metadata(session.outputs()),
        };

        if metadata.inputs.len() != 1 {
            return Err(RuntimeError::UnexpectedInputCount(metadata.inputs.len()));
        }

        if metadata.outputs.is_empty() {
            return Err(RuntimeError::UnexpectedOutputCount(0));
        }

        Ok(Self {
            model_path,
            session,
            metadata,
            load_time: started.elapsed(),
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    pub fn load_time(&self) -> Duration {
        self.load_time
    }

    pub fn infer_features(&mut self, features: &FeatureBatch) -> Result<ModelOutput, RuntimeError> {
        let outputs = run_raw_session(&mut self.session, &self.metadata, features)?;
        let mut tensors = Vec::with_capacity(self.metadata.outputs.len());

        for (index, meta) in self.metadata.outputs.iter().enumerate() {
            let value = &outputs[index];
            let extracted = value
                .try_extract_array::<f32>()
                .map_err(|err| RuntimeError::OutputExtraction {
                    name: meta.name.clone(),
                    message: err.to_string(),
                })?;

            tensors.push(NamedTensorOutput {
                name: meta.name.clone(),
                data: extracted.to_owned(),
            });
        }

        Ok(ModelOutput { tensors })
    }

    pub fn decode_output(&self, output: &ModelOutput) -> Result<DecodedOutput, RuntimeError> {
        decode_conservatively(output)
    }

    pub fn default_output_names(&self) -> Vec<String> {
        self.metadata
            .outputs
            .iter()
            .map(|item| item.name.clone())
            .collect()
    }

    pub fn output_name_fallbacks(&self) -> Vec<String> {
        DEFAULT_OUTPUT_NAMES.iter().map(|name| (*name).to_owned()).collect()
    }
}

fn collect_input_metadata(items: &[ort::value::Outlet]) -> Vec<TensorMetadata> {
    items.iter().map(tensor_metadata_from_input).collect()
}

fn tensor_metadata_from_input(item: &ort::value::Outlet) -> TensorMetadata {
    let (element_type, dimensions) = tensor_shape_from_value_type(item.dtype());
    TensorMetadata {
        name: item.name().to_owned(),
        element_type,
        dimensions,
    }
}

fn collect_output_metadata(items: &[ort::value::Outlet]) -> Vec<TensorMetadata> {
    items.iter().map(tensor_metadata_from_output).collect()
}

fn tensor_metadata_from_output(item: &ort::value::Outlet) -> TensorMetadata {
    let (element_type, dimensions) = tensor_shape_from_value_type(item.dtype());
    TensorMetadata {
        name: item.name().to_owned(),
        element_type,
        dimensions,
    }
}

fn tensor_shape_from_value_type(value_type: &ValueType) -> (String, Vec<Option<i64>>) {
    if let Some(shape) = value_type.tensor_shape() {
        let dims = shape
            .iter()
            .copied()
            .map(|dim| if dim < 0 { None } else { Some(dim) })
            .collect();
        let element_type = value_type
            .tensor_type()
            .map(|ty| format!("{ty:?}"))
            .unwrap_or_else(|| format!("{value_type:?}"));
        (element_type, dims)
    } else {
        (format!("{value_type:?}"), Vec::new())
    }
}
