use std::path::Path;

use anyhow::Result;

use crate::config::load_app_config;
use crate::dataset::load_mono_annotations;
use crate::metrics::evaluate_mono_dataset;
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
    let report = evaluate_mono_dataset(&dataset, &templates, &cfg)?;

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
