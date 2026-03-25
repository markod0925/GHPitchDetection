use ndarray::Array5;

use fretnet_runtime::{default_model_path, FeatureBatch, FretNetRuntime, RuntimeError};

#[test]
fn shape_validation_rejects_wrong_channel_count() {
    let model_path = default_model_path();
    if !model_path.exists() {
        eprintln!("skipping shape test because model is missing at {}", model_path.display());
        return;
    }

    let mut runtime = FretNetRuntime::load(&model_path).expect("model should load");
    let wrong_shape = FeatureBatch::new(Array5::<f32>::zeros((1, 32, 5, 144, 9)));
    let error = runtime.infer_features(&wrong_shape).expect_err("shape validation should fail");

    match error {
        RuntimeError::FeatureDimensionMismatch { axis, expected, actual } => {
            assert_eq!(axis, 2);
            assert_eq!(expected, 6);
            assert_eq!(actual, 5);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

