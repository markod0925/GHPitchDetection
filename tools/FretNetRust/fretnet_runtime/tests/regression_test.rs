use fretnet_runtime::{
    default_fixture_paths, default_model_path, load_regression_fixture, FretNetRuntime,
};

#[test]
fn regression_matches_real_audio_reference_fixture() {
    let model_path = default_model_path();
    if !model_path.exists() {
        eprintln!(
            "skipping regression test because model is missing at {}",
            model_path.display()
        );
        return;
    }

    let fixture_paths: Vec<_> = default_fixture_paths()
        .into_iter()
        .filter(|path| path.exists())
        .collect();

    if fixture_paths.is_empty() {
        eprintln!("skipping regression test because no default fixtures are present");
        return;
    }

    let mut runtime = FretNetRuntime::load(&model_path).expect("model should load");

    for fixture_path in fixture_paths {
        let fixture = load_regression_fixture(&fixture_path).expect("fixture should load");
        let fixture_name = fixture_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("unknown_fixture");
        let output = runtime
            .infer_features(&fixture.features)
            .expect("inference should succeed");
        let reference_names: Vec<_> = fixture
            .reference_output
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect();
        let runtime_names: Vec<_> = output
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect();
        let shared_names: Vec<_> = reference_names
            .iter()
            .filter(|name| {
                runtime_names
                    .iter()
                    .any(|runtime_name| runtime_name == *name)
            })
            .cloned()
            .collect();

        println!("Fixture: {fixture_name}");
        println!("  Path: {}", fixture_path.display());
        println!("  Input shape: {:?}", fixture.features.shape());
        println!("  Reference tensors: {:?}", reference_names);
        println!("  Runtime tensors:   {:?}", runtime_names);
        println!("  Shared tensors:    {:?}", shared_names);

        for reference in &fixture.reference_output.tensors {
            let actual = output.tensor(&reference.name).unwrap_or_else(|| {
                panic!(
                    "missing output '{}' in fixture '{}' (path: {})",
                    reference.name,
                    fixture_name,
                    fixture_path.display()
                )
            });

            assert_eq!(
                actual.data.shape(),
                reference.data.shape(),
                "shape mismatch for tensor '{}' in fixture '{}' ({})",
                reference.name,
                fixture_name,
                fixture_path.display()
            );

            let mut max_abs_diff = 0.0f32;
            let mut mean_abs_diff = 0.0f64;
            let mut count = 0usize;

            for (actual_value, reference_value) in actual.data.iter().zip(reference.data.iter()) {
                let diff = (actual_value - reference_value).abs();
                max_abs_diff = max_abs_diff.max(diff);
                mean_abs_diff += diff as f64;
                count += 1;
            }

            if count > 0 {
                mean_abs_diff /= count as f64;
            }

            println!(
                "  - {} shape={:?} max_abs_diff={:.6e} mean_abs_diff={:.6e}",
                reference.name,
                reference.data.shape(),
                max_abs_diff,
                mean_abs_diff
            );

            assert!(
                max_abs_diff <= 1.0e-3,
                "max abs diff too large for tensor '{}' in fixture '{}' ({}): {}",
                reference.name,
                fixture_name,
                fixture_path.display(),
                max_abs_diff
            );
            assert!(
                mean_abs_diff <= 1.0e-4,
                "mean abs diff too large for tensor '{}' in fixture '{}' ({}): {}",
                reference.name,
                fixture_name,
                fixture_path.display(),
                mean_abs_diff
            );
        }
    }
}
