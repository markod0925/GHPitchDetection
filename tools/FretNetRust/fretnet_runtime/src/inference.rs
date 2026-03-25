use ndarray::Array5;
use ort::{inputs, session::Session, value::TensorRef};

use crate::{
    error::RuntimeError,
    types::{FeatureBatch, ModelMetadata},
};

pub fn validate_feature_batch(features: &FeatureBatch, metadata: &ModelMetadata) -> Result<(), RuntimeError> {
    let shape = features.shape();
    let rank = shape.len();
    if rank != 5 {
        return Err(RuntimeError::InvalidFeatureRank {
            expected_rank: 5,
            actual_rank: rank,
        });
    }

    let Some(input) = metadata.inputs.first() else {
        return Err(RuntimeError::UnexpectedInputCount(0));
    };

    if input.dimensions.len() != 5 {
        return Err(RuntimeError::InvalidFeatureRank {
            expected_rank: 5,
            actual_rank: input.dimensions.len(),
        });
    }

    for (axis, expected) in input.dimensions.iter().enumerate() {
        if let Some(expected) = expected {
            if *expected >= 0 {
                let expected = *expected as usize;
                let actual = shape[axis];
                if expected != actual {
                    return Err(RuntimeError::FeatureDimensionMismatch {
                        axis,
                        expected,
                        actual,
                    });
                }
            }
        }
    }

    Ok(())
}

pub fn feature_array(features: &FeatureBatch) -> &Array5<f32> {
    features.data()
}

pub fn run_raw_session<'a>(
    session: &'a mut Session,
    metadata: &ModelMetadata,
    features: &FeatureBatch,
) -> Result<ort::session::SessionOutputs<'a>, RuntimeError> {
    let Some(input) = metadata.inputs.first() else {
        return Err(RuntimeError::UnexpectedInputCount(0));
    };

    validate_feature_batch(features, metadata)?;

    let tensor = TensorRef::from_array_view(feature_array(features).view())
        .map_err(|err| RuntimeError::Inference(err.to_string()))?;

    let inputs = inputs![input.name.as_str() => tensor];
    session.run(inputs).map_err(|err| RuntimeError::Inference(err.to_string()))
}
