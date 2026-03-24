use std::path::Path;

use anyhow::Result;

use crate::config::load_app_config;
use crate::report::try_open_in_browser;
use crate::solo_string::{
    build_string_solo_report, eval_string_solo, optimize_string_solo, SoloStringEvalThresholds,
    SoloStringObjectiveWeights,
};

pub fn run_optimize_string_solo(
    config_path: &str,
    mono_dataset_path: &str,
    dataset_root: &str,
    out_dir: &str,
    trials: Option<usize>,
    seed: Option<u64>,
    weights: SoloStringObjectiveWeights,
    thresholds: SoloStringEvalThresholds,
    open: bool,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let effective_trials = trials.unwrap_or(500);
    let effective_seed = seed.unwrap_or(cfg.optimization.seed);
    let summary = optimize_string_solo(
        &cfg,
        mono_dataset_path,
        Path::new(dataset_root),
        Path::new(out_dir),
        trials,
        seed,
        &weights,
        &thresholds,
    )?;

    println!("phase S1 SOLO optimization complete");
    println!("phase dir: {}", summary.phase_dir.display());
    println!("trials: {}", effective_trials);
    println!("seed: {}", effective_seed);
    println!("best trial id: {}", summary.best_trial.trial_id);
    println!("best objective: {:.6}", summary.best_trial.objective);
    println!(
        "best metrics: top1={:.4} top2={:.4} top3={:.4} avg_margin={:.4} high_conf_wrong={:.4}",
        summary.best_trial.metrics.top1_string_accuracy,
        summary.best_trial.metrics.top2_string_accuracy,
        summary.best_trial.metrics.top3_string_accuracy,
        summary.best_trial.metrics.average_margin_top1_top2,
        summary.best_trial.metrics.high_confidence_wrong_rate,
    );
    println!("trials json: {}", summary.trials_json_path.display());
    println!("trials csv: {}", summary.trials_csv_path.display());
    println!("best config: {}", summary.best_config_path.display());
    println!(
        "best top1 config: {}",
        summary.best_top1_config_path.display()
    );
    println!(
        "best balanced config: {}",
        summary.best_balanced_config_path.display()
    );
    println!(
        "best robust config: {}",
        summary.best_robust_config_path.display()
    );
    println!("string metrics: {}", summary.metrics_json_path.display());
    println!("confusion csv: {}", summary.confusion_csv_path.display());
    println!(
        "example rankings csv: {}",
        summary.examples_csv_path.display()
    );
    println!("report index: {}", summary.report_path.display());

    if open {
        try_open_in_browser(&summary.report_path);
        println!("open requested: attempted to launch browser");
    }

    Ok(())
}

pub fn run_eval_string_solo(
    config_path: &str,
    mono_dataset_path: &str,
    dataset_root: &str,
    out_dir: &str,
    thresholds: SoloStringEvalThresholds,
    open: bool,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let summary = eval_string_solo(
        &cfg,
        mono_dataset_path,
        Path::new(dataset_root),
        Path::new(out_dir),
        &thresholds,
    )?;

    println!("phase S1 SOLO evaluation complete");
    println!("phase dir: {}", summary.phase_dir.display());
    println!("string metrics: {}", summary.metrics_json_path.display());
    println!("confusion csv: {}", summary.confusion_csv_path.display());
    println!(
        "example rankings csv: {}",
        summary.examples_csv_path.display()
    );
    println!("report index: {}", summary.report_path.display());

    if open {
        try_open_in_browser(&summary.report_path);
        println!("open requested: attempted to launch browser");
    }

    Ok(())
}

pub fn run_report_string_solo(out_dir: &str, open: bool) -> Result<()> {
    let report_path = build_string_solo_report(Path::new(out_dir))?;
    println!("phase S1 SOLO report generated: {}", report_path.display());

    if open {
        try_open_in_browser(&report_path);
        println!("open requested: attempted to launch browser");
    }

    Ok(())
}
