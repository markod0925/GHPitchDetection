use std::path::PathBuf;

use fretnet_runtime::{
    default_existing_feature_fixture_paths, default_model_path, load_feature_reference_fixture,
    load_regression_fixture, FeatureExtractor, FretNetRuntime, FrontendConfig,
};

#[test]
fn rust_frontend_runs_end_to_end_into_onnx_runtime() {
    let model_path = default_model_path();
    if !model_path.exists() {
        eprintln!(
            "skipping end-to-end frontend test because model is missing at {}",
            model_path.display()
        );
        return;
    }

    let fixture_paths = default_existing_feature_fixture_paths();
    if fixture_paths.is_empty() {
        eprintln!(
            "skipping end-to-end frontend test because no Python feature fixtures are present"
        );
        return;
    }

    let extractor =
        FeatureExtractor::new(FrontendConfig::default()).expect("frontend should initialize");
    let mut runtime = FretNetRuntime::load(&model_path).expect("model should load");

    for feature_fixture_path in fixture_paths {
        let feature_fixture = load_feature_reference_fixture(&feature_fixture_path)
            .expect("feature fixture should load");
        let feature_name = feature_fixture_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("unknown_fixture");

        let hcqt_full = extractor
            .extract_hcqt(&feature_fixture.audio, feature_fixture.sample_rate)
            .expect("hcqt extraction should succeed");
        let hcqt = hcqt_full.truncated(feature_fixture.hcqt.frame_count());
        let batch = extractor
            .hcqt_to_batch(&hcqt, Some(feature_fixture.hcqt.frame_count()))
            .expect("feature batch conversion should succeed");
        let output = runtime
            .infer_features(&batch)
            .expect("inference should succeed");

        println!("Fixture: {feature_name}");
        println!("  HCQT shape: {:?}", hcqt.shape());
        println!("  Model input shape: {:?}", batch.shape());
        println!(
            "  Output tensors: {:?}",
            output
                .tensors
                .iter()
                .map(|tensor| tensor.name.clone())
                .collect::<Vec<_>>()
        );

        assert_eq!(batch.frame_count(), feature_fixture.hcqt.frame_count());
        assert!(
            !output.tensors.is_empty(),
            "expected at least one output tensor"
        );
        for tensor in &output.tensors {
            assert!(
                tensor.data.iter().all(|value| value.is_finite()),
                "tensor '{}' contains non-finite values",
                tensor.name
            );
        }

        if let Some(regression_path) = regression_fixture_for_feature_fixture(&feature_fixture_path)
        {
            let regression_fixture =
                load_regression_fixture(&regression_path).expect("regression fixture should load");
            assert_eq!(
                regression_fixture.features.shape(),
                batch.shape(),
                "model input shape mismatch for fixture '{}'",
                feature_name
            );
        }
    }
}

fn regression_fixture_for_feature_fixture(feature_fixture_path: &PathBuf) -> Option<PathBuf> {
    let file_name = feature_fixture_path.file_name()?.to_str()?;
    let regression_name = file_name.replace("_features_f130.npz", "_f130.npz");
    let regression_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/test_features")
        .join(regression_name);
    regression_path.exists().then_some(regression_path)
}
