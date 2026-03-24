use std::path::Path;

use anyhow::Result;

use crate::config::load_app_config;
use crate::dataset::{load_mono_annotations, load_poly_events};
use crate::full_eval::{evaluate_full_dataset, write_full_artifacts};
use crate::metrics::evaluate_mono_dataset as evaluate_mono_summary;
use crate::report::{
    build_html_report, build_phase1_ab_comparison,
    evaluate_mono_dataset as evaluate_mono_diagnostics, evaluate_pitch_dataset,
    try_open_in_browser, write_mono_artifacts, write_phase1_ab_comparison, write_pitch_artifacts,
    write_pitch_artifacts_variant,
};
use crate::spectral_report::build_solo_string_spectrogram_report;
use crate::templates::load_templates;

pub fn run_eval_mono(
    config_path: &str,
    mono_dataset_path: &str,
    templates_path: &str,
    dataset_root: &str,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let templates = load_templates(templates_path)?;
    let dataset = load_mono_annotations(mono_dataset_path, Path::new(dataset_root), true)?;
    let report = evaluate_mono_summary(&dataset, &templates, &cfg)?;

    println!("mono evaluation summary");
    println!("records total: {}", report.total_records);
    println!("records evaluated: {}", report.evaluated_records);
    println!("records skipped: {}", report.skipped_records);
    println!("top-1 string accuracy: {:.4}", report.metrics.top1);
    println!("top-3 string accuracy: {:.4}", report.metrics.top3);
    if report.exact_string_fret_supported {
        println!(
            "exact string+fret accuracy: {:.4}",
            report.metrics.exact_string_fret
        );
    } else {
        println!(
            "exact string+fret accuracy: unsupported for template level '{}'",
            cfg.templates.level
        );
    }
    Ok(())
}

pub fn run_eval_full(
    config_path: &str,
    poly_dataset_path: &str,
    templates_path: &str,
    dataset_root: &str,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let templates = crate::templates::load_templates(templates_path)?;
    let dataset = load_poly_events(poly_dataset_path, Path::new(dataset_root), true)?;

    let artifact = evaluate_full_dataset(&dataset, &templates, &cfg)?;

    println!("full polyphonic evaluation summary");
    println!("events total: {}", artifact.metrics.total_events);
    println!("events evaluated: {}", artifact.metrics.evaluated_events);
    println!("events skipped: {}", artifact.metrics.skipped_events);
    println!("pitch f1: {:.4}", artifact.metrics.pitch.f1);
    println!("pitch+string f1: {:.4}", artifact.metrics.pitch_string.f1);
    println!(
        "pitch+string+fret f1: {:.4}",
        artifact.metrics.pitch_string_fret.f1
    );
    println!(
        "exact configuration accuracy: {:.4}",
        artifact.metrics.exact_configuration_accuracy
    );
    println!(
        "decode hit@1/@3/@5: {:.4} / {:.4} / {:.4}",
        artifact.metrics.decode_top1_hit_rate,
        artifact.metrics.decode_top3_hit_rate,
        artifact.metrics.decode_top5_hit_rate
    );
    println!(
        "avg candidates/pitch: {:.3}, avg global configs explored: {:.3}",
        artifact.metrics.avg_candidate_positions_per_pitch,
        artifact.metrics.avg_global_configurations_explored
    );
    println!(
        "failures: pitch-correct/string-wrong={}, string-correct/fret-wrong={}, full-wrong-despite-correct-pitch={}, full-decode-wrong-high-confidence={}",
        artifact.metrics.failure_pitch_correct_string_wrong_count,
        artifact.metrics.failure_string_correct_fret_wrong_count,
        artifact.metrics.failure_full_wrong_despite_correct_pitch_count,
        artifact.metrics
            .failure_full_decode_wrong_high_confidence_count
    );
    Ok(())
}

pub fn run_eval_full_report(
    config_path: &str,
    poly_dataset_path: &str,
    templates_path: &str,
    dataset_root: &str,
    out_dir: &str,
    open: bool,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let templates = crate::templates::load_templates(templates_path)?;
    let dataset = load_poly_events(poly_dataset_path, Path::new(dataset_root), true)?;

    let artifact = evaluate_full_dataset(&dataset, &templates, &cfg)?;
    write_full_artifacts(Path::new(out_dir), &artifact)?;

    let report_path = build_html_report(Path::new(out_dir))?;

    println!("phase3 full diagnostics complete");
    println!("events total: {}", artifact.metrics.total_events);
    println!("events evaluated: {}", artifact.metrics.evaluated_events);
    println!("events skipped: {}", artifact.metrics.skipped_events);
    println!("pitch f1: {:.4}", artifact.metrics.pitch.f1);
    println!("pitch+string f1: {:.4}", artifact.metrics.pitch_string.f1);
    println!(
        "pitch+string+fret f1: {:.4}",
        artifact.metrics.pitch_string_fret.f1
    );
    println!(
        "exact configuration accuracy: {:.4}",
        artifact.metrics.exact_configuration_accuracy
    );
    println!(
        "decode hit@1/@3/@5: {:.4} / {:.4} / {:.4}",
        artifact.metrics.decode_top1_hit_rate,
        artifact.metrics.decode_top3_hit_rate,
        artifact.metrics.decode_top5_hit_rate
    );
    println!(
        "avg candidates/pitch: {:.3}, avg global configs explored: {:.3}",
        artifact.metrics.avg_candidate_positions_per_pitch,
        artifact.metrics.avg_global_configurations_explored
    );
    println!(
        "artifacts: {}/phase3_full_metrics.json, {}/phase3_per_file.csv, {}/phase3_examples_topk.csv, {}/phase3_poly_scores/",
        out_dir, out_dir, out_dir, out_dir
    );
    println!("report index: {}", report_path.display());

    if open {
        try_open_in_browser(&report_path);
        println!("open requested: attempted to launch browser");
    }

    Ok(())
}

pub fn run_eval_pitch_report(
    config_path: &str,
    mono_dataset_path: &str,
    dataset_root: &str,
    out_dir: &str,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let dataset = load_mono_annotations(mono_dataset_path, Path::new(dataset_root), true)?;

    let mut stft_cfg = cfg.clone();
    stft_cfg.frontend.kind = "stft".to_string();
    let stft_artifact = evaluate_pitch_dataset(&dataset, &stft_cfg)?;
    write_pitch_artifacts(Path::new(out_dir), &stft_artifact)?;

    let mut qdft_cfg = cfg.clone();
    qdft_cfg.frontend.kind = "qdft".to_string();
    let qdft_artifact = evaluate_pitch_dataset(&dataset, &qdft_cfg)?;
    write_pitch_artifacts_variant(Path::new(out_dir), &qdft_artifact, "qdft")?;

    let comparison = build_phase1_ab_comparison("stft", &stft_artifact, "qdft", &qdft_artifact);
    write_phase1_ab_comparison(Path::new(out_dir), &comparison)?;

    println!("phase1 pitch diagnostics complete (A/B: stft vs qdft)");
    println!("records total: {}", stft_artifact.metrics.total_records);
    println!(
        "records evaluated: {}",
        stft_artifact.metrics.evaluated_records
    );
    println!("records skipped: {}", stft_artifact.metrics.skipped_records);
    println!(
        "stft precision/recall/f1: {:.4} / {:.4} / {:.4}",
        stft_artifact.metrics.precision, stft_artifact.metrics.recall, stft_artifact.metrics.f1
    );
    println!(
        "qdft precision/recall/f1: {:.4} / {:.4} / {:.4}",
        qdft_artifact.metrics.precision, qdft_artifact.metrics.recall, qdft_artifact.metrics.f1
    );
    println!(
        "delta qdft-stft: precision={:+.4} recall={:+.4} f1={:+.4}",
        comparison.delta_precision, comparison.delta_recall, comparison.delta_f1
    );
    println!(
        "artifacts: {}/phase1_pitch_metrics.json, {}/phase1_pitch_metrics_qdft.json, {}/phase1_ab_comparison.json, {}/phase1_per_file.csv, {}/phase1_per_file_qdft.csv, {}/phase1_pitch_scores/, {}/phase1_pitch_scores_qdft/",
        out_dir, out_dir, out_dir, out_dir, out_dir, out_dir, out_dir
    );

    // TODO(phase-3): add pitch diagnostics for polyphonic frame/event evaluation.

    Ok(())
}

pub fn run_eval_mono_report(
    config_path: &str,
    mono_dataset_path: &str,
    templates_path: &str,
    dataset_root: &str,
    out_dir: &str,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let templates = load_templates(templates_path)?;
    let dataset = load_mono_annotations(mono_dataset_path, Path::new(dataset_root), true)?;

    let artifact = evaluate_mono_diagnostics(&dataset, &templates, &cfg)?;
    write_mono_artifacts(Path::new(out_dir), &artifact)?;

    println!("phase2 mono diagnostics complete");
    println!("records total: {}", artifact.metrics.total_records);
    println!("records evaluated: {}", artifact.metrics.evaluated_records);
    println!("records skipped: {}", artifact.metrics.skipped_records);
    println!(
        "top-1 string accuracy: {:.4}",
        artifact.metrics.top1_string_accuracy
    );
    println!(
        "top-3 string accuracy: {:.4}",
        artifact.metrics.top3_string_accuracy
    );
    if let Some(exact) = artifact.metrics.exact_string_fret_accuracy {
        println!("exact string+fret accuracy: {:.4}", exact);
    } else {
        println!(
            "exact string+fret accuracy: not meaningful for template level '{}'",
            cfg.templates.level
        );
    }
    println!(
        "artifacts: {}/phase2_string_metrics.json, {}/phase2_confusion_matrix.csv, {}/phase2_examples_topk.csv, {}/phase2_mono_scores/",
        out_dir, out_dir, out_dir, out_dir
    );

    Ok(())
}

pub fn run_build_report(debug_dir: &str, open: bool) -> Result<()> {
    let report_path = build_html_report(Path::new(debug_dir))?;
    println!("report generated: {}", report_path.display());

    if open {
        try_open_in_browser(&report_path);
        println!("open requested: attempted to launch browser");
    }

    Ok(())
}

pub fn run_report(
    config_path: &str,
    mono_dataset_path: &str,
    templates_path: &str,
    dataset_root: &str,
    out_dir: &str,
    open: bool,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let templates = load_templates(templates_path)?;
    let dataset = load_mono_annotations(mono_dataset_path, Path::new(dataset_root), true)?;

    let mut stft_cfg = cfg.clone();
    stft_cfg.frontend.kind = "stft".to_string();
    let pitch_stft = evaluate_pitch_dataset(&dataset, &stft_cfg)?;
    write_pitch_artifacts(Path::new(out_dir), &pitch_stft)?;

    let mut qdft_cfg = cfg.clone();
    qdft_cfg.frontend.kind = "qdft".to_string();
    let pitch_qdft = evaluate_pitch_dataset(&dataset, &qdft_cfg)?;
    write_pitch_artifacts_variant(Path::new(out_dir), &pitch_qdft, "qdft")?;

    let comparison = build_phase1_ab_comparison("stft", &pitch_stft, "qdft", &pitch_qdft);
    write_phase1_ab_comparison(Path::new(out_dir), &comparison)?;

    let mono = evaluate_mono_diagnostics(&dataset, &templates, &cfg)?;
    write_mono_artifacts(Path::new(out_dir), &mono)?;

    let report_path = build_html_report(Path::new(out_dir))?;
    println!("phase2.5 report bundle generated under: {}", out_dir);
    println!("report index: {}", report_path.display());

    if open {
        try_open_in_browser(&report_path);
        println!("open requested: attempted to launch browser");
    }

    // TODO(phase-3): extend unified report command with polyphonic full-system diagnostics.

    Ok(())
}

pub fn run_solo_string_spectrogram_report(
    config_path: &str,
    mono_dataset_path: &str,
    dataset_root: &str,
    out_dir: &str,
    string_index: u8,
    midi_filter: Option<u8>,
    max_notes: Option<usize>,
    open: bool,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let summary = build_solo_string_spectrogram_report(
        &cfg,
        mono_dataset_path,
        Path::new(dataset_root),
        Path::new(out_dir),
        string_index,
        midi_filter,
        max_notes,
    )?;

    println!("solo string STFT+CQT report complete");
    println!("string: {}", summary.string_index);
    if let Some(midi) = midi_filter {
        println!("midi filter: {}", midi);
    }
    println!("notes requested: {}", summary.notes_requested);
    println!("notes rendered: {}", summary.notes_rendered);
    println!("notes skipped: {}", summary.notes_skipped);
    println!("figures dir: {}", summary.figures_dir.display());
    println!("report index: {}", summary.report_path.display());

    if open {
        try_open_in_browser(&summary.report_path);
        println!("open requested: attempted to launch browser");
    }

    Ok(())
}
