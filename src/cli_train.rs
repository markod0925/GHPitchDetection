use std::path::Path;

use anyhow::Result;

use crate::config::load_app_config;
use crate::dataset::load_mono_annotations;
use crate::templates::{save_templates, train_templates_from_mono_dataset};

pub fn run_train_templates(
    config_path: &str,
    mono_dataset_path: &str,
    out_path: &str,
    dataset_root: &str,
) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let dataset = load_mono_annotations(mono_dataset_path, Path::new(dataset_root), true)?;
    let templates = train_templates_from_mono_dataset(&dataset, &cfg)?;
    save_templates(out_path, &templates)?;

    let total_by_string: u32 = templates.by_string.iter().map(|x| x.count).sum();
    println!("trained templates saved to {}", out_path);
    println!("records used (solo-filtered): {}", dataset.len());
    println!("string-level template samples: {}", total_by_string);
    println!(
        "string-region template cells: {}x{}",
        templates.by_string_region.len(),
        templates
            .by_string_region
            .first()
            .map(|x| x.len())
            .unwrap_or(0)
    );
    Ok(())
}
