use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::audio::load_wav_mono;
use crate::config::load_app_config;
use crate::dataset::{load_mono_annotations, load_poly_events};
use crate::masp::{
    load_pretrain_artifacts, pretrain_from_guitarset, read_validation_requests,
    validate_expected_segment, write_pretrain_artifacts, write_validation_artifacts,
    MaspValidationResult,
};
use crate::tracking::{read_tablature_events, validate_masp_with_playhead};

pub fn run_pretrain_masp(
    config_path: &str,
    mono_dataset_path: &str,
    comp_dataset_path: &str,
    dataset_root: &str,
    out_dir: &str,
    trials: Option<usize>,
    target_mono_recall: f32,
    target_comp_recall: f32,
) -> Result<()> {
    let mut cfg = load_app_config(config_path)?;
    if let Some(t) = trials {
        cfg.masp.pretrain_trials = t.max(1);
    }

    let mono_dataset = load_mono_annotations(mono_dataset_path, Path::new(dataset_root), true)?;
    let comp_dataset = load_poly_events(comp_dataset_path, Path::new(dataset_root), true)?;

    let artifacts = pretrain_from_guitarset(
        &cfg,
        &mono_dataset,
        &comp_dataset,
        target_mono_recall,
        target_comp_recall,
    )?;
    write_pretrain_artifacts(Path::new(out_dir), &artifacts)?;

    println!("MASP pretraining complete");
    println!("mono records used: {}", mono_dataset.len());
    println!("comp events used: {}", comp_dataset.len());
    println!("note signatures: {}", artifacts.note_signatures.len());
    println!("joint signatures: {}", artifacts.joint_signatures.len());
    println!(
        "tuned b_exponent: {:.4}",
        artifacts.manifest.model_params.b_exponent
    );
    println!(
        "validation score weights (har/mbw/cent/rms): {:.4} / {:.4} / {:.4} / {:.4}",
        artifacts.manifest.validation_rule.score_weights.har,
        artifacts.manifest.validation_rule.score_weights.mbw,
        artifacts.manifest.validation_rule.score_weights.cent,
        artifacts.manifest.validation_rule.score_weights.rms
    );
    println!(
        "validation score threshold: {:.4}",
        artifacts.manifest.validation_rule.score_threshold
    );
    println!(
        "target recalls (mono/comp): {:.4} / {:.4}",
        artifacts.manifest.validation_rule.target_mono_recall,
        artifacts.manifest.validation_rule.target_comp_recall
    );
    println!(
        "calibrated recalls (mono/comp): {:.4} / {:.4}",
        artifacts
            .manifest
            .validation_rule
            .calibrated_mono_recall
            .unwrap_or(0.0),
        artifacts
            .manifest
            .validation_rule
            .calibrated_comp_recall
            .unwrap_or(0.0)
    );
    println!(
        "calibrated precision: {:.4}",
        artifacts
            .manifest
            .validation_rule
            .calibrated_precision
            .unwrap_or(0.0)
    );
    println!(
        "artifacts: {}/masp_manifest.json, {}/masp_note_signatures.jsonl, {}/masp_joint_signatures.jsonl",
        out_dir, out_dir, out_dir
    );

    Ok(())
}

pub fn run_validate_masp(
    config_path: &str,
    artifacts_dir: &str,
    audio_path: &str,
    start_sec: f32,
    end_sec: f32,
    expected_midis: &[u8],
    out_json: Option<&str>,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let artifacts = load_pretrain_artifacts(Path::new(artifacts_dir))?;
    let audio = load_wav_mono(audio_path)
        .with_context(|| format!("failed to load input audio for MASP validation: {audio_path}"))?;

    let started = Instant::now();
    let mut result =
        validate_expected_segment(&audio, start_sec, end_sec, expected_midis, &cfg, &artifacts)?;
    // Keep per-instance timing to support latency analysis in reports.
    result.execution_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
    result.audio_path = audio_path.to_string();

    let raw = serde_json::to_string_pretty(&result)?;
    println!("{raw}");

    if let Some(path) = out_json {
        std::fs::write(path, raw)
            .with_context(|| format!("failed to write MASP validation json: {path}"))?;
        println!("saved validation json: {}", path);
    }

    Ok(())
}

pub fn run_validate_masp_batch(
    config_path: &str,
    artifacts_dir: &str,
    requests_jsonl: &str,
    out_dir: &str,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let artifacts = load_pretrain_artifacts(Path::new(artifacts_dir))?;
    let requests = read_validation_requests(Path::new(requests_jsonl))?;

    let mut audio_cache: BTreeMap<String, crate::types::AudioBuffer> = BTreeMap::new();
    let mut results = Vec::<MaspValidationResult>::with_capacity(requests.len());

    for req in requests {
        if req.expected_midis.is_empty() {
            continue;
        }

        if !audio_cache.contains_key(&req.audio_path) {
            let audio = load_wav_mono(&req.audio_path).with_context(|| {
                format!(
                    "failed to load input audio for MASP batch validation: {}",
                    req.audio_path
                )
            })?;
            audio_cache.insert(req.audio_path.clone(), audio);
        }

        let audio = audio_cache
            .get(&req.audio_path)
            .with_context(|| format!("missing cache entry for {}", req.audio_path))?;

        let started = Instant::now();
        let mut result = validate_expected_segment(
            audio,
            req.start_sec,
            req.end_sec,
            &req.expected_midis,
            &cfg,
            &artifacts,
        )?;
        result.execution_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
        result.id = req.id.clone();
        result.audio_path = req.audio_path.clone();
        result.expected_valid = req.expected_valid;
        if let Some(expected_valid) = req.expected_valid {
            result.false_accept = !expected_valid && result.pass;
            result.false_reject = expected_valid && !result.pass;
        }
        results.push(result);
    }

    let summary = write_validation_artifacts(Path::new(out_dir), &results)?;

    println!("MASP batch validation complete");
    println!("total: {}", summary.total);
    println!("passed: {}", summary.passed);
    println!("failed: {}", summary.failed);
    println!("pass rate: {:.4}", summary.pass_rate);
    if let Some(acc) = summary.accuracy {
        println!("labeled accuracy: {:.4}", acc);
    }
    println!(
        "false accepts/rejects: {} / {}",
        summary.false_accept_count, summary.false_reject_count
    );
    println!(
        "artifacts: {}/masp_validation_results.jsonl, {}/masp_validation_summary.json",
        out_dir, out_dir
    );

    Ok(())
}

pub fn run_validate_masp_playhead(
    config_path: &str,
    artifacts_dir: &str,
    audio_path: &str,
    tablature_path: &str,
    playhead_sec: f32,
    out_json: Option<&str>,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let artifacts = load_pretrain_artifacts(Path::new(artifacts_dir))?;
    let audio = load_wav_mono(audio_path)
        .with_context(|| format!("failed to load input audio for MASP validation: {audio_path}"))?;
    let tablature = read_tablature_events(Path::new(tablature_path))?;

    let started = Instant::now();
    let mut result =
        validate_masp_with_playhead(&audio, playhead_sec, &tablature, &cfg, &artifacts)?;
    result.validation.execution_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
    result.validation.audio_path = audio_path.to_string();

    let raw = serde_json::to_string_pretty(&result)?;
    println!("{raw}");

    if let Some(path) = out_json {
        std::fs::write(path, raw)
            .with_context(|| format!("failed to write MASP playhead validation json: {path}"))?;
        println!("saved validation json: {}", path);
    }

    Ok(())
}
