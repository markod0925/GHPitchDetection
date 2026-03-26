use std::path::Path;

use fretnet_runtime::{
    default_existing_stage_fixture_paths, load_audio_for_frontend, load_stage_reference_fixture,
    load_wav_mono, FeatureExtractor, FrontendConfig,
};
use ndarray::{Array2, Array3};

#[derive(Debug, Clone, Copy)]
struct DiffMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f64,
    rmse: f64,
}

#[derive(Debug, Clone, Copy)]
struct LagMetrics {
    best_lag: isize,
    best_mean_abs_diff: f64,
}

#[test]
fn rust_frontend_stage_diagnostics_against_python_reference() {
    let fixture_paths = default_existing_stage_fixture_paths();
    if fixture_paths.is_empty() {
        eprintln!("skipping stage equivalence test because no Python stage fixtures are present");
        return;
    }

    for fixture_path in fixture_paths {
        let fixture =
            load_stage_reference_fixture(&fixture_path).expect("stage fixture should load");
        let fixture_name = &fixture.metadata.fixture_name;
        let source_audio_path = Path::new(&fixture.metadata.source_audio_path);

        let raw_audio = load_wav_mono(source_audio_path).expect("source audio should load");
        let resampled_audio =
            load_audio_for_frontend(source_audio_path, fixture.metadata.frontend.sample_rate)
                .expect("frontend audio preparation should succeed");

        let extractor =
            FeatureExtractor::new(FrontendConfig::default()).expect("frontend should initialize");
        let extracted = extractor
            .extract_stages(
                &fixture.resampled_audio,
                fixture.metadata.frontend.sample_rate,
            )
            .expect("stage extraction should succeed");
        let extracted_from_wav = extractor
            .extract_stages(
                &resampled_audio.samples,
                fixture.metadata.frontend.sample_rate,
            )
            .expect("stage extraction from Rust-resampled audio should succeed");
        let target_frames = fixture.hcqt.frame_count();

        println!("Fixture: {fixture_name}");
        println!("  Root: {}", fixture.root.display());
        println!("  Source audio: {}", fixture.metadata.source_audio_path);
        println!(
            "  Stage 0 metadata: original_sr={} rust_sr={} original_samples={} rust_samples={} duration_s={:.3} channels={}",
            fixture.metadata.original_audio.sample_rate,
            raw_audio.sample_rate,
            fixture.metadata.original_audio.num_samples,
            raw_audio.samples.len(),
            fixture.metadata.original_audio.duration_seconds,
            fixture.metadata.original_audio.channels,
        );

        assert_eq!(
            raw_audio.sample_rate, fixture.metadata.original_audio.sample_rate,
            "original sample-rate mismatch for fixture '{}'",
            fixture_name
        );
        assert_eq!(
            raw_audio.samples.len(),
            fixture.metadata.original_audio.num_samples,
            "original sample-count mismatch for fixture '{}'",
            fixture_name
        );
        assert_eq!(
            resampled_audio.sample_rate, fixture.metadata.frontend.sample_rate,
            "resampled sample-rate mismatch for fixture '{}'",
            fixture_name
        );
        assert_eq!(
            resampled_audio.samples.len(),
            fixture.resampled_audio.len(),
            "resampled audio length mismatch for fixture '{}'",
            fixture_name
        );
        assert_eq!(
            extracted.harmonics.len(),
            fixture.harmonics.len(),
            "harmonic count mismatch for fixture '{}'",
            fixture_name
        );
        assert!(
            extracted.hcqt.frame_count() >= target_frames,
            "Rust HCQT has fewer frames than fixture '{}' expected at least {} got {}",
            fixture_name,
            target_frames,
            extracted.hcqt.frame_count()
        );
        assert_eq!(
            fixture.times.len(),
            fixture.metadata.max_frames.unwrap_or(fixture.times.len()),
            "time-axis length mismatch for fixture '{}'",
            fixture_name
        );

        let audio_metrics = diff_metrics_1d(&resampled_audio.samples, &fixture.resampled_audio);
        let audio_center_metrics =
            center_cropped_diff_1d(&resampled_audio.samples, &fixture.resampled_audio, 2048);
        let audio_lag = best_lag_1d(&resampled_audio.samples, &fixture.resampled_audio, 32);
        let audio_scale = rms_ratio_1d(&resampled_audio.samples, &fixture.resampled_audio);

        println!(
            "  Stage 1 resampled audio: shape=[{}] max_abs_diff={:.6e} mean_abs_diff={:.6e} rmse={:.6e}",
            fixture.resampled_audio.len(),
            audio_metrics.max_abs_diff,
            audio_metrics.mean_abs_diff,
            audio_metrics.rmse,
        );
        if let Some(center) = audio_center_metrics {
            println!(
                "    center-cropped: max_abs_diff={:.6e} mean_abs_diff={:.6e} rmse={:.6e}",
                center.max_abs_diff, center.mean_abs_diff, center.rmse,
            );
        }
        println!(
            "    best_lag={} samples best_mean_abs_diff={:.6e} rms_ratio={:.6e}",
            audio_lag.best_lag, audio_lag.best_mean_abs_diff, audio_scale,
        );
        if let Some(center) = audio_center_metrics {
            if center.mean_abs_diff < audio_metrics.mean_abs_diff * 0.75 {
                println!("    diagnostic: center crop improves audio agreement; boundary/resampling edges matter.");
            }
        }
        if audio_lag.best_mean_abs_diff < audio_metrics.mean_abs_diff * 0.75 {
            println!("    diagnostic: a small lag reduces waveform error noticeably.");
        }

        let mut worst_mag_mean = 0.0f64;
        let mut worst_mag_label = String::new();
        let mut worst_post_mean = 0.0f64;
        let mut worst_post_label = String::new();

        for (expected, actual) in fixture.harmonics.iter().zip(extracted.harmonics.iter()) {
            assert!(
                (expected.harmonic - actual.harmonic).abs() < 1.0e-6,
                "harmonic value mismatch for fixture '{}' expected {} got {}",
                fixture_name,
                expected.harmonic,
                actual.harmonic,
            );
            assert_eq!(
                actual.magnitude.shape()[0],
                expected.magnitude.shape()[0],
                "magnitude frequency-bin mismatch for fixture '{}' harmonic {}",
                fixture_name,
                expected.harmonic,
            );
            assert!(
                actual.magnitude.shape()[1] >= expected.magnitude.shape()[1],
                "magnitude frame mismatch for fixture '{}' harmonic {} expected at least {} got {}",
                fixture_name,
                expected.harmonic,
                expected.magnitude.shape()[1],
                actual.magnitude.shape()[1]
            );
            assert_eq!(
                actual.post_processed.shape()[0],
                expected.post_processed.shape()[0],
                "post-processed frequency-bin mismatch for fixture '{}' harmonic {}",
                fixture_name,
                expected.harmonic,
            );
            assert!(
                actual.post_processed.shape()[1] >= expected.post_processed.shape()[1],
                "post-processed frame mismatch for fixture '{}' harmonic {} expected at least {} got {}",
                fixture_name, expected.harmonic, expected.post_processed.shape()[1], actual.post_processed.shape()[1]
            );

            let actual_magnitude = actual
                .magnitude
                .slice(ndarray::s![.., ..expected.magnitude.shape()[1]])
                .to_owned();
            let actual_post = actual
                .post_processed
                .slice(ndarray::s![.., ..expected.post_processed.shape()[1]])
                .to_owned();

            let mag = diff_metrics_2d(&actual_magnitude, &expected.magnitude);
            let mag_center = center_cropped_diff_2d(&actual_magnitude, &expected.magnitude, 8);
            let mag_lag = best_lag_2d(&actual_magnitude, &expected.magnitude, 4);
            let mag_scale = rms_ratio_iter(
                actual_magnitude.iter().copied(),
                expected.magnitude.iter().copied(),
            );

            let post = diff_metrics_2d(&actual_post, &expected.post_processed);
            let post_center = center_cropped_diff_2d(&actual_post, &expected.post_processed, 8);
            let post_lag = best_lag_2d(&actual_post, &expected.post_processed, 4);
            let post_scale = rms_ratio_iter(
                actual_post.iter().copied(),
                expected.post_processed.iter().copied(),
            );

            println!(
                "  Stage 2 harmonic={} ({}) magnitude: shape={:?} max_abs_diff={:.6e} mean_abs_diff={:.6e} rmse={:.6e}",
                expected.harmonic,
                expected.label,
                expected.magnitude.shape(),
                mag.max_abs_diff,
                mag.mean_abs_diff,
                mag.rmse,
            );
            if let Some(center) = mag_center {
                println!(
                    "    center-cropped: max_abs_diff={:.6e} mean_abs_diff={:.6e} rmse={:.6e}",
                    center.max_abs_diff, center.mean_abs_diff, center.rmse,
                );
            }
            println!(
                "    best_lag={} frames best_mean_abs_diff={:.6e} rms_ratio={:.6e}",
                mag_lag.best_lag, mag_lag.best_mean_abs_diff, mag_scale,
            );

            println!(
                "  Stage 2 harmonic={} ({}) post_processed: shape={:?} max_abs_diff={:.6e} mean_abs_diff={:.6e} rmse={:.6e}",
                expected.harmonic,
                expected.label,
                expected.post_processed.shape(),
                post.max_abs_diff,
                post.mean_abs_diff,
                post.rmse,
            );
            if let Some(center) = post_center {
                println!(
                    "    center-cropped: max_abs_diff={:.6e} mean_abs_diff={:.6e} rmse={:.6e}",
                    center.max_abs_diff, center.mean_abs_diff, center.rmse,
                );
            }
            println!(
                "    best_lag={} frames best_mean_abs_diff={:.6e} rms_ratio={:.6e}",
                post_lag.best_lag, post_lag.best_mean_abs_diff, post_scale,
            );

            if mag.mean_abs_diff > worst_mag_mean {
                worst_mag_mean = mag.mean_abs_diff;
                worst_mag_label = expected.label.clone();
            }
            if post.mean_abs_diff > worst_post_mean {
                worst_post_mean = post.mean_abs_diff;
                worst_post_label = expected.label.clone();
            }
        }

        let actual_hcqt = extracted.hcqt.truncated(target_frames);
        let hcqt_metrics = diff_metrics_3d(actual_hcqt.data(), fixture.hcqt.data());
        let hcqt_center_metrics =
            center_cropped_diff_3d(actual_hcqt.data(), fixture.hcqt.data(), 8);
        let hcqt_lag = best_lag_3d(actual_hcqt.data(), fixture.hcqt.data(), 4);
        println!(
            "  Stage 3 HCQT: shape={:?} max_abs_diff={:.6e} mean_abs_diff={:.6e} rmse={:.6e}",
            fixture.hcqt.shape(),
            hcqt_metrics.max_abs_diff,
            hcqt_metrics.mean_abs_diff,
            hcqt_metrics.rmse,
        );
        if let Some(center) = hcqt_center_metrics {
            println!(
                "    center-cropped: max_abs_diff={:.6e} mean_abs_diff={:.6e} rmse={:.6e}",
                center.max_abs_diff, center.mean_abs_diff, center.rmse,
            );
            if center.mean_abs_diff < hcqt_metrics.mean_abs_diff * 0.75 {
                println!("    diagnostic: HCQT error drops on center crop; boundary/padding is a material factor.");
            }
        }
        println!(
            "    best_lag={} frames best_mean_abs_diff={:.6e}",
            hcqt_lag.best_lag, hcqt_lag.best_mean_abs_diff,
        );
        if hcqt_lag.best_mean_abs_diff < hcqt_metrics.mean_abs_diff * 0.75 {
            println!("    diagnostic: HCQT error drops after a small frame shift.");
        }

        if audio_metrics.mean_abs_diff * 10.0 < worst_post_mean {
            println!(
                "  Diagnostic hint: transform mismatch dominates resampling (audio mean_abs_diff={:.6e}, worst harmonic post mean_abs_diff={:.6e} at {}).",
                audio_metrics.mean_abs_diff,
                worst_post_mean,
                worst_post_label,
            );
        } else {
            println!(
                "  Diagnostic hint: resampling mismatch is already material (audio mean_abs_diff={:.6e}, worst harmonic post mean_abs_diff={:.6e} at {}).",
                audio_metrics.mean_abs_diff,
                worst_post_mean,
                worst_post_label,
            );
        }
        if worst_mag_mean > audio_metrics.mean_abs_diff * 1.1 {
            println!(
                "  Diagnostic hint: after resampling, the raw transform diverges further; inspect the {} harmonic magnitude path first.",
                worst_mag_label,
            );
        } else {
            println!(
                "  Diagnostic hint: raw transform error stays near the resampling baseline; fix resampling before deeper HCQT changes."
            );
        }
        let (pipeline_worst_mag_label, pipeline_worst_mag_mean) =
            worst_harmonic_magnitude_diff(&extracted_from_wav, &fixture);
        let pipeline_hcqt = diff_metrics_3d(
            extracted_from_wav.hcqt.truncated(target_frames).data(),
            fixture.hcqt.data(),
        );
        println!(
            "  Pipeline summary from WAV: stage1_mean_abs_diff={:.6e} stage2_worst_raw_magnitude={} mean_abs_diff={:.6e} stage3_hcqt_mean_abs_diff={:.6e}",
            audio_metrics.mean_abs_diff,
            pipeline_worst_mag_label,
            pipeline_worst_mag_mean,
            pipeline_hcqt.mean_abs_diff,
        );
        println!(
            "  Worst raw magnitude harmonic: {} mean_abs_diff={:.6e}",
            worst_mag_label, worst_mag_mean,
        );
        println!();

        assert!(resampled_audio
            .samples
            .iter()
            .all(|sample| sample.is_finite()));
        assert!(actual_hcqt.data().iter().all(|value| value.is_finite()));
    }
}

fn diff_metrics_1d(actual: &[f32], reference: &[f32]) -> DiffMetrics {
    assert_eq!(actual.len(), reference.len());
    diff_metrics_iter(actual.iter().copied(), reference.iter().copied())
}

fn diff_metrics_2d(actual: &Array2<f32>, reference: &Array2<f32>) -> DiffMetrics {
    assert_eq!(actual.shape(), reference.shape());
    diff_metrics_iter(actual.iter().copied(), reference.iter().copied())
}

fn diff_metrics_3d(actual: &Array3<f32>, reference: &Array3<f32>) -> DiffMetrics {
    assert_eq!(actual.shape(), reference.shape());
    diff_metrics_iter(actual.iter().copied(), reference.iter().copied())
}

fn diff_metrics_iter<I, J>(actual: I, reference: J) -> DiffMetrics
where
    I: IntoIterator<Item = f32>,
    J: IntoIterator<Item = f32>,
{
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;

    for (actual, reference) in actual.into_iter().zip(reference) {
        let diff = (actual - reference).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs += diff as f64;
        sum_sq += (diff as f64) * (diff as f64);
        count += 1;
    }

    DiffMetrics {
        max_abs_diff,
        mean_abs_diff: if count > 0 {
            sum_abs / count as f64
        } else {
            0.0
        },
        rmse: if count > 0 {
            (sum_sq / count as f64).sqrt()
        } else {
            0.0
        },
    }
}

fn center_cropped_diff_1d(
    actual: &[f32],
    reference: &[f32],
    max_trim: usize,
) -> Option<DiffMetrics> {
    let trim = max_trim.min(actual.len() / 4).min(reference.len() / 4);
    if trim == 0 || actual.len() <= 2 * trim || reference.len() <= 2 * trim {
        return None;
    }
    Some(diff_metrics_1d(
        &actual[trim..actual.len() - trim],
        &reference[trim..reference.len() - trim],
    ))
}

fn center_cropped_diff_2d(
    actual: &Array2<f32>,
    reference: &Array2<f32>,
    max_trim: usize,
) -> Option<DiffMetrics> {
    let frames = actual.shape()[1];
    let trim = max_trim.min(frames / 4);
    if trim == 0 || frames <= 2 * trim {
        return None;
    }
    Some(diff_metrics_iter(
        actual
            .slice(ndarray::s![.., trim..frames - trim])
            .iter()
            .copied(),
        reference
            .slice(ndarray::s![.., trim..frames - trim])
            .iter()
            .copied(),
    ))
}

fn center_cropped_diff_3d(
    actual: &Array3<f32>,
    reference: &Array3<f32>,
    max_trim: usize,
) -> Option<DiffMetrics> {
    let frames = actual.shape()[2];
    let trim = max_trim.min(frames / 4);
    if trim == 0 || frames <= 2 * trim {
        return None;
    }
    Some(diff_metrics_iter(
        actual
            .slice(ndarray::s![.., .., trim..frames - trim])
            .iter()
            .copied(),
        reference
            .slice(ndarray::s![.., .., trim..frames - trim])
            .iter()
            .copied(),
    ))
}

fn best_lag_1d(actual: &[f32], reference: &[f32], max_lag: isize) -> LagMetrics {
    best_lag_generic(max_lag, |lag| {
        overlap_mean_abs_diff_1d(actual, reference, lag)
    })
}

fn best_lag_2d(actual: &Array2<f32>, reference: &Array2<f32>, max_lag: isize) -> LagMetrics {
    best_lag_generic(max_lag, |lag| {
        overlap_mean_abs_diff_2d(actual, reference, lag)
    })
}

fn best_lag_3d(actual: &Array3<f32>, reference: &Array3<f32>, max_lag: isize) -> LagMetrics {
    best_lag_generic(max_lag, |lag| {
        overlap_mean_abs_diff_3d(actual, reference, lag)
    })
}

fn best_lag_generic<F>(max_lag: isize, mut scorer: F) -> LagMetrics
where
    F: FnMut(isize) -> Option<f64>,
{
    let mut best = LagMetrics {
        best_lag: 0,
        best_mean_abs_diff: f64::INFINITY,
    };

    for lag in -max_lag..=max_lag {
        if let Some(score) = scorer(lag) {
            if score < best.best_mean_abs_diff {
                best.best_lag = lag;
                best.best_mean_abs_diff = score;
            }
        }
    }

    best
}

fn overlap_mean_abs_diff_1d(actual: &[f32], reference: &[f32], lag: isize) -> Option<f64> {
    let (a_start, r_start, len) = overlap_offsets(actual.len(), reference.len(), lag)?;
    Some(
        diff_metrics_iter(
            actual[a_start..a_start + len].iter().copied(),
            reference[r_start..r_start + len].iter().copied(),
        )
        .mean_abs_diff,
    )
}

fn overlap_mean_abs_diff_2d(
    actual: &Array2<f32>,
    reference: &Array2<f32>,
    lag: isize,
) -> Option<f64> {
    let frames = actual.shape()[1];
    let (a_start, r_start, len) = overlap_offsets(frames, reference.shape()[1], lag)?;
    Some(
        diff_metrics_iter(
            actual
                .slice(ndarray::s![.., a_start..a_start + len])
                .iter()
                .copied(),
            reference
                .slice(ndarray::s![.., r_start..r_start + len])
                .iter()
                .copied(),
        )
        .mean_abs_diff,
    )
}

fn overlap_mean_abs_diff_3d(
    actual: &Array3<f32>,
    reference: &Array3<f32>,
    lag: isize,
) -> Option<f64> {
    let frames = actual.shape()[2];
    let (a_start, r_start, len) = overlap_offsets(frames, reference.shape()[2], lag)?;
    Some(
        diff_metrics_iter(
            actual
                .slice(ndarray::s![.., .., a_start..a_start + len])
                .iter()
                .copied(),
            reference
                .slice(ndarray::s![.., .., r_start..r_start + len])
                .iter()
                .copied(),
        )
        .mean_abs_diff,
    )
}

fn overlap_offsets(
    actual_len: usize,
    reference_len: usize,
    lag: isize,
) -> Option<(usize, usize, usize)> {
    let (a_start, r_start) = if lag >= 0 {
        (lag as usize, 0usize)
    } else {
        (0usize, (-lag) as usize)
    };
    if a_start >= actual_len || r_start >= reference_len {
        return None;
    }
    let len = (actual_len - a_start).min(reference_len - r_start);
    (len > 0).then_some((a_start, r_start, len))
}

fn rms_ratio_1d(actual: &[f32], reference: &[f32]) -> f64 {
    rms_ratio_iter(actual.iter().copied(), reference.iter().copied())
}

fn rms_ratio_iter<I, J>(actual: I, reference: J) -> f64
where
    I: IntoIterator<Item = f32>,
    J: IntoIterator<Item = f32>,
{
    let actual_rms = rms_iter(actual);
    let reference_rms = rms_iter(reference);
    if reference_rms <= f64::EPSILON {
        0.0
    } else {
        actual_rms / reference_rms
    }
}

fn rms_iter<I>(values: I) -> f64
where
    I: IntoIterator<Item = f32>,
{
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;
    for value in values {
        sum_sq += (value as f64) * (value as f64);
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt()
    }
}

fn worst_harmonic_magnitude_diff(
    actual: &fretnet_runtime::FrontendStageOutputs,
    fixture: &fretnet_runtime::StageReferenceFixture,
) -> (String, f64) {
    let mut worst_label = String::new();
    let mut worst_mean = 0.0f64;

    for (expected, actual) in fixture.harmonics.iter().zip(actual.harmonics.iter()) {
        let actual_magnitude = actual
            .magnitude
            .slice(ndarray::s![.., ..expected.magnitude.shape()[1]])
            .to_owned();
        let metrics = diff_metrics_2d(&actual_magnitude, &expected.magnitude);
        if metrics.mean_abs_diff > worst_mean {
            worst_mean = metrics.mean_abs_diff;
            worst_label = expected.label.clone();
        }
    }

    (worst_label, worst_mean)
}
