use fretnet_runtime::{
    default_existing_feature_fixture_paths, load_feature_reference_fixture, FeatureExtractor, FrontendConfig,
};

// The current Rust frontend matches the Python HCQT pipeline closely in shape and aggregate statistics,
// but it still uses a direct full-rate VQT construction instead of librosa's recursive octave/downsampling path.
// These tolerances are therefore provisional and intentionally documented rather than pretending exact parity.
const MAX_ABS_DIFF_THRESHOLD: f32 = 0.30;
const MEAN_ABS_DIFF_THRESHOLD: f64 = 0.005;

#[test]
fn rust_hcqt_matches_python_feature_fixture_within_documented_tolerance() {
    let fixture_paths = default_existing_feature_fixture_paths();
    if fixture_paths.is_empty() {
        eprintln!("skipping feature equivalence test because no Python feature fixtures are present");
        return;
    }

    let extractor = FeatureExtractor::new(FrontendConfig::default()).expect("frontend should initialize");

    for fixture_path in fixture_paths {
        let fixture = load_feature_reference_fixture(&fixture_path).expect("feature fixture should load");
        let fixture_name = fixture_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("unknown_fixture");
        let extracted_full = extractor
            .extract_hcqt(&fixture.audio, fixture.sample_rate)
            .expect("hcqt extraction should succeed");
        let extracted = extracted_full.truncated(fixture.hcqt.frame_count());

        assert_eq!(
            extracted.shape(),
            fixture.hcqt.shape(),
            "HCQT shape mismatch for fixture '{}'",
            fixture_name
        );

        let extracted_data = extracted.data();
        let reference_data = fixture.hcqt.data();

        let extracted_min = extracted_data.iter().copied().fold(f32::INFINITY, f32::min);
        let extracted_max = extracted_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let extracted_mean = extracted_data.iter().copied().map(f64::from).sum::<f64>() / extracted_data.len() as f64;

        let reference_min = reference_data.iter().copied().fold(f32::INFINITY, f32::min);
        let reference_max = reference_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let reference_mean = reference_data.iter().copied().map(f64::from).sum::<f64>() / reference_data.len() as f64;

        let mut max_abs_diff = 0.0f32;
        let mut mean_abs_diff = 0.0f64;
        let mut count = 0usize;
        let mut harmonic_max = vec![0.0f32; extracted.shape()[0]];
        let mut harmonic_sum = vec![0.0f64; extracted.shape()[0]];
        let harmonic_count = (extracted.shape()[1] * extracted.shape()[2]) as f64;

        for harmonic in 0..extracted.shape()[0] {
            for bin in 0..extracted.shape()[1] {
                for frame in 0..extracted.shape()[2] {
                    let actual = extracted_data[[harmonic, bin, frame]];
                    let reference = reference_data[[harmonic, bin, frame]];
                    let diff = (actual - reference).abs();
                    max_abs_diff = max_abs_diff.max(diff);
                    mean_abs_diff += diff as f64;
                    harmonic_max[harmonic] = harmonic_max[harmonic].max(diff);
                    harmonic_sum[harmonic] += diff as f64;
                    count += 1;
                }
            }
        }

        if count > 0 {
            mean_abs_diff /= count as f64;
        }

        println!("Fixture: {fixture_name}");
        println!("  Path: {}", fixture_path.display());
        println!("  Audio samples: {}", fixture.audio.len());
        println!("  HCQT shape: {:?}", extracted.shape());
        println!(
            "  Reference stats: min={:.6} max={:.6} mean={:.6}",
            reference_min, reference_max, reference_mean
        );
        println!(
            "  Extracted stats: min={:.6} max={:.6} mean={:.6}",
            extracted_min, extracted_max, extracted_mean
        );
        println!(
            "  Diff stats: max_abs_diff={:.6e} mean_abs_diff={:.6e}",
            max_abs_diff, mean_abs_diff
        );

        for harmonic in 0..extracted.shape()[0] {
            let harmonic_mean = harmonic_sum[harmonic] / harmonic_count;
            println!(
                "  - harmonic[{harmonic}] max_abs_diff={:.6e} mean_abs_diff={:.6e}",
                harmonic_max[harmonic],
                harmonic_mean
            );
        }

        assert!(
            max_abs_diff <= MAX_ABS_DIFF_THRESHOLD,
            "max abs diff too large for fixture '{}': {}",
            fixture_name,
            max_abs_diff
        );
        assert!(
            mean_abs_diff <= MEAN_ABS_DIFF_THRESHOLD,
            "mean abs diff too large for fixture '{}': {}",
            fixture_name,
            mean_abs_diff
        );
    }
}
