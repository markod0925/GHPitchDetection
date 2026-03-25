use fretnet_runtime::{default_model_path, FeatureBatch, FretNetRuntime};

#[test]
fn smoke_loads_and_runs_inference_when_model_exists() {
    let model_path = default_model_path();
    if !model_path.exists() {
        eprintln!("skipping smoke test because model is missing at {}", model_path.display());
        return;
    }

    let mut runtime = FretNetRuntime::load(&model_path).expect("model should load");
    let features = FeatureBatch::synthetic(1, 32, 6, 144, 9);
    let output = runtime.infer_features(&features).expect("inference should succeed");
    assert!(!output.tensors.is_empty(), "expected at least one output tensor");
}

