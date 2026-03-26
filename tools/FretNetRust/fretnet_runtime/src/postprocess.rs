use ndarray::ArrayD;

use crate::{
    error::RuntimeError,
    types::{DecodedOutput, ModelOutput, TensorStats},
};

fn tensor_stats(name: &str, data: &ArrayD<f32>) -> TensorStats {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;

    for value in data.iter().copied() {
        min = min.min(value);
        max = max.max(value);
        sum += value as f64;
    }

    let mean = if data.is_empty() {
        0.0
    } else {
        (sum / data.len() as f64) as f32
    };

    TensorStats {
        name: name.to_owned(),
        shape: data.shape().to_vec(),
        min,
        max,
        mean,
    }
}

pub fn decode_conservatively(output: &ModelOutput) -> Result<DecodedOutput, RuntimeError> {
    if output.tensors.is_empty() {
        return Err(RuntimeError::Decode("model output is empty".to_owned()));
    }

    let tensors = output
        .tensors
        .iter()
        .map(|tensor| tensor_stats(&tensor.name, &tensor.data))
        .collect();

    Ok(DecodedOutput {
        tensors,
        notes: Some(
            "Conservative decoding only. Raw tensors are preserved; note/string/fret semantics remain TODO.".to_owned(),
        ),
    })
}
