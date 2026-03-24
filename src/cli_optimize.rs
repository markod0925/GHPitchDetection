use std::fs::create_dir_all;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::config::{load_app_config, AppConfig};
use crate::dataset::{
    filename_contains_solo, load_mono_annotations, load_poly_events, PolyEvalEvent,
};
use crate::full_eval::{evaluate_full_dataset, write_full_artifacts};
use crate::optimize::{
    build_tuning_report, run_stage_a_random_search, run_stage_c_random_search,
    StageAObjectiveWeights, StageCObjectiveWeights,
};
use crate::report::{
    evaluate_mono_dataset, evaluate_pitch_dataset, try_open_in_browser, write_mono_artifacts,
    write_pitch_artifacts,
};
use crate::templates::{load_templates, save_templates, train_templates_from_mono_dataset};

const DEFAULT_STAGE_A_TRIALS: usize = 500;
const DEFAULT_STAGE_C_TRIALS: usize = 500;

pub fn run_optimize_phase1(
    config_path: &str,
    poly_dataset_path: &str,
    dataset_root: &str,
    out_dir: &str,
    trials: Option<usize>,
    seed: Option<u64>,
    objective_weights: StageAObjectiveWeights,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let dataset = load_stage_a_solo_poly_dataset(poly_dataset_path, dataset_root)?;

    let trial_count = trials.unwrap_or(DEFAULT_STAGE_A_TRIALS);
    let trial_seed = seed.unwrap_or(cfg.optimization.seed);
    let tuning_dir = tuning_dir_from_out(out_dir);

    let summary = run_stage_a_random_search(
        &cfg,
        &dataset,
        trial_count,
        trial_seed,
        &objective_weights,
        &tuning_dir,
    )?;
    let report_path = build_tuning_report(&tuning_dir)?;

    println!("stage A random search complete");
    println!("solo events used: {}", dataset.len());
    println!("trials: {}", trial_count);
    println!("seed: {}", trial_seed);
    println!("best trial id: {}", summary.best_trial.trial_id);
    println!("best objective: {:.6}", summary.best_trial.objective);
    println!(
        "best pitch metrics: precision={:.4} recall={:.4} f1={:.4}",
        summary.best_trial.metrics.pitch_precision.unwrap_or(0.0),
        summary.best_trial.metrics.pitch_recall.unwrap_or(0.0),
        summary.best_trial.metrics.pitch_f1.unwrap_or(0.0)
    );
    println!("trials json: {}", summary.trials_json_path.display());
    println!("trials csv: {}", summary.trials_csv_path.display());
    println!(
        "best stage A config: {}",
        summary.best_config_path.display()
    );
    println!("tuning report: {}", report_path.display());

    Ok(())
}

pub fn run_optimize_phase3(
    config_path: &str,
    poly_dataset_path: &str,
    templates_path: &str,
    dataset_root: &str,
    out_dir: &str,
    trials: Option<usize>,
    seed: Option<u64>,
    phase1_config: Option<&str>,
    objective_weights: StageCObjectiveWeights,
) -> Result<()> {
    let cfg_base = load_app_config(config_path)?;
    let tuning_dir = tuning_dir_from_out(out_dir);

    let cfg_for_stage3 = resolve_stage_a_seed_config(&cfg_base, &tuning_dir, phase1_config)?;
    let dataset = load_poly_events(poly_dataset_path, Path::new(dataset_root), true)?;
    let templates = load_templates(templates_path)?;

    let trial_count = trials.unwrap_or(DEFAULT_STAGE_C_TRIALS);
    let trial_seed = seed.unwrap_or(cfg_base.optimization.seed.saturating_add(1));

    let summary = run_stage_c_random_search(
        &cfg_for_stage3,
        &dataset,
        &templates,
        trial_count,
        trial_seed,
        &objective_weights,
        &tuning_dir,
    )?;
    let report_path = build_tuning_report(&tuning_dir)?;

    println!("stage C random search complete");
    println!("comp events used: {}", dataset.len());
    println!("trials: {}", trial_count);
    println!("seed: {}", trial_seed);
    println!("best trial id: {}", summary.best_trial.trial_id);
    println!("best objective: {:.6}", summary.best_trial.objective);
    println!(
        "best full metrics: pitch_f1={:.4} pitch+string_f1={:.4} pitch+string+fret_f1={:.4} exact={:.4}",
        summary.best_trial.metrics.pitch_f1.unwrap_or(0.0),
        summary.best_trial.metrics.pitch_string_f1.unwrap_or(0.0),
        summary.best_trial.metrics.pitch_string_fret_f1.unwrap_or(0.0),
        summary
            .best_trial
            .metrics
            .exact_configuration_accuracy
            .unwrap_or(0.0),
    );
    println!(
        "decode hit@1/@3/@5: {:.4} / {:.4} / {:.4}",
        summary
            .best_trial
            .metrics
            .decode_top1_hit_rate
            .unwrap_or(0.0),
        summary
            .best_trial
            .metrics
            .decode_top3_hit_rate
            .unwrap_or(0.0),
        summary
            .best_trial
            .metrics
            .decode_top5_hit_rate
            .unwrap_or(0.0),
    );
    println!("trials json: {}", summary.trials_json_path.display());
    println!("trials csv: {}", summary.trials_csv_path.display());
    println!(
        "best stage C config: {}",
        summary.best_config_path.display()
    );
    println!("tuning report: {}", report_path.display());

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run_optimize_all(
    config_path: &str,
    mono_dataset_path: &str,
    poly_dataset_path: &str,
    dataset_root: &str,
    out_dir: &str,
    templates_out: Option<&str>,
    stage_a_trials: Option<usize>,
    stage_c_trials: Option<usize>,
    stage_a_seed: Option<u64>,
    stage_c_seed: Option<u64>,
    stage_a_weights: StageAObjectiveWeights,
    stage_c_weights: StageCObjectiveWeights,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let tuning_dir = tuning_dir_from_out(out_dir);
    create_dir_all(&tuning_dir)?;

    let solo_mono_dataset =
        load_mono_annotations(mono_dataset_path, Path::new(dataset_root), true)?;
    let solo_poly_dataset = load_stage_a_solo_poly_dataset(mono_dataset_path, dataset_root)?;
    let comp_poly_dataset = load_poly_events(poly_dataset_path, Path::new(dataset_root), true)?;

    let a_trials = stage_a_trials.unwrap_or(DEFAULT_STAGE_A_TRIALS);
    let c_trials = stage_c_trials.unwrap_or(DEFAULT_STAGE_C_TRIALS);
    let a_seed = stage_a_seed.unwrap_or(cfg.optimization.seed);
    let c_seed = stage_c_seed.unwrap_or(cfg.optimization.seed.saturating_add(1));

    let stage_a_summary = run_stage_a_random_search(
        &cfg,
        &solo_poly_dataset,
        a_trials,
        a_seed,
        &stage_a_weights,
        &tuning_dir,
    )?;

    let stage_a_report_dir = stage_report_dir(&tuning_dir, "1_solo_pitch_tuning")?;
    let stage_a_pitch = evaluate_pitch_dataset(&solo_mono_dataset, &stage_a_summary.best_config)?;
    write_pitch_artifacts(&stage_a_report_dir, &stage_a_pitch)?;

    let stage_b_templates =
        train_templates_from_mono_dataset(&solo_mono_dataset, &stage_a_summary.best_config)?;

    let default_templates_path = tuning_dir.join("best_stage_b_templates.bin");
    save_templates(
        &default_templates_path.to_string_lossy(),
        &stage_b_templates,
    )?;
    let templates_path = if let Some(out_path) = templates_out {
        let out = PathBuf::from(out_path);
        if out != default_templates_path {
            save_templates(&out.to_string_lossy(), &stage_b_templates)?;
        }
        out
    } else {
        default_templates_path.clone()
    };

    let stage_b_report_dir = stage_report_dir(&tuning_dir, "2_solo_mono_template")?;
    let stage_b_mono = evaluate_mono_dataset(
        &solo_mono_dataset,
        &stage_b_templates,
        &stage_a_summary.best_config,
    )?;
    write_mono_artifacts(&stage_b_report_dir, &stage_b_mono)?;

    let stage_c_summary = run_stage_c_random_search(
        &stage_a_summary.best_config,
        &comp_poly_dataset,
        &stage_b_templates,
        c_trials,
        c_seed,
        &stage_c_weights,
        &tuning_dir,
    )?;

    let stage_c_report_dir = stage_report_dir(&tuning_dir, "3_comp_poly_decode")?;
    let stage_c_full = evaluate_full_dataset(
        &comp_poly_dataset,
        &stage_b_templates,
        &stage_c_summary.best_config,
    )?;
    write_full_artifacts(&stage_c_report_dir, &stage_c_full)?;

    let stage_d_report_dir = stage_report_dir(&tuning_dir, "4_solo_control_after_decoder")?;
    let stage_d_control = evaluate_full_dataset(
        &solo_poly_dataset,
        &stage_b_templates,
        &stage_c_summary.best_config,
    )?;
    write_full_artifacts(&stage_d_report_dir, &stage_d_control)?;
    std::fs::write(
        stage_d_report_dir.join("CONTROL_NOTE.txt"),
        "This report is a solo control/generalization check after decoder tuning on comp. It is not the main optimization target.",
    )?;

    let report_path = build_tuning_report(&tuning_dir)?;

    println!("staged optimization complete");
    println!("stage A (solo pitch) events: {}", solo_poly_dataset.len());
    println!(
        "stage B (solo mono templates) records: {}",
        solo_mono_dataset.len()
    );
    println!("stage C (comp decode) events: {}", comp_poly_dataset.len());
    println!("stage D (solo control) events: {}", solo_poly_dataset.len());
    println!(
        "stage A best: trial={} objective={:.6}",
        stage_a_summary.best_trial.trial_id, stage_a_summary.best_trial.objective
    );
    println!(
        "stage C best: trial={} objective={:.6}",
        stage_c_summary.best_trial.trial_id, stage_c_summary.best_trial.objective
    );
    println!(
        "stage C best metrics: pitch+string_f1={:.4} pitch+string+fret_f1={:.4} exact={:.4} top1/top3/top5={:.4}/{:.4}/{:.4}",
        stage_c_summary.best_trial.metrics.pitch_string_f1.unwrap_or(0.0),
        stage_c_summary
            .best_trial
            .metrics
            .pitch_string_fret_f1
            .unwrap_or(0.0),
        stage_c_summary
            .best_trial
            .metrics
            .exact_configuration_accuracy
            .unwrap_or(0.0),
        stage_c_summary.best_trial.metrics.decode_top1_hit_rate.unwrap_or(0.0),
        stage_c_summary.best_trial.metrics.decode_top3_hit_rate.unwrap_or(0.0),
        stage_c_summary.best_trial.metrics.decode_top5_hit_rate.unwrap_or(0.0),
    );
    println!(
        "stage D control metrics: pitch+string_f1={:.4} pitch+string+fret_f1={:.4} exact={:.4}",
        stage_d_control.metrics.pitch_string.f1,
        stage_d_control.metrics.pitch_string_fret.f1,
        stage_d_control.metrics.exact_configuration_accuracy,
    );
    println!(
        "best stage A config: {}",
        stage_a_summary.best_config_path.display()
    );
    println!("best stage B templates: {}", templates_path.display());
    println!(
        "best stage C config: {}",
        stage_c_summary.best_config_path.display()
    );
    println!(
        "stage A trial log: {}",
        stage_a_summary.trials_json_path.display()
    );
    println!(
        "stage C trial log: {}",
        stage_c_summary.trials_json_path.display()
    );
    println!("solo pitch report dir: {}", stage_a_report_dir.display());
    println!("solo mono report dir: {}", stage_b_report_dir.display());
    println!("comp decode report dir: {}", stage_c_report_dir.display());
    println!("solo control report dir: {}", stage_d_report_dir.display());
    println!("tuning report: {}", report_path.display());

    Ok(())
}

pub fn run_report_tuning(out_dir: &str, open: bool) -> Result<()> {
    let tuning_dir = tuning_dir_from_out(out_dir);
    let report_path = build_tuning_report(&tuning_dir)?;
    println!("tuning report generated: {}", report_path.display());

    if open {
        try_open_in_browser(&report_path);
        println!("open requested: attempted to launch browser");
    }

    Ok(())
}

fn load_stage_a_solo_poly_dataset(
    dataset_path: &str,
    dataset_root: &str,
) -> Result<Vec<PolyEvalEvent>> {
    let mut dataset = load_poly_events(dataset_path, Path::new(dataset_root), false)?;
    dataset.retain(|e| filename_contains_solo(&e.audio_path));
    if dataset.is_empty() {
        bail!(
            "stage A requires solo events; no filenames containing 'solo' were found in {}",
            dataset_path
        );
    }
    Ok(dataset)
}

fn resolve_stage_a_seed_config(
    fallback_cfg: &AppConfig,
    tuning_dir: &Path,
    explicit_stage_a_cfg: Option<&str>,
) -> Result<AppConfig> {
    if let Some(path) = explicit_stage_a_cfg {
        return load_app_config(path);
    }

    let candidates = [
        tuning_dir.join("best_stage_a_pitch_config.toml"),
        tuning_dir.join("best_phase1_config.toml"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return load_app_config(&candidate.to_string_lossy());
        }
    }

    Ok(fallback_cfg.clone())
}

fn tuning_dir_from_out(out_dir: &str) -> PathBuf {
    Path::new(out_dir).join("tuning")
}

fn stage_report_dir(tuning_dir: &Path, stage_dir_name: &str) -> Result<PathBuf> {
    let root = tuning_dir.join("reports");
    let dir = root.join(stage_dir_name);
    create_dir_all(&dir)?;
    Ok(dir)
}
