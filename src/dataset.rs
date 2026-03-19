use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::types::MonoAnnotation;

pub fn load_mono_annotations(
    path: &str,
    dataset_root: &Path,
    only_solo: bool,
) -> Result<Vec<MonoAnnotation>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read mono annotations from: {path}"))?;
    let first_non_ws = data.chars().find(|c| !c.is_whitespace());

    let mut records = if first_non_ws == Some('[') {
        serde_json::from_str::<Vec<MonoAnnotation>>(&data)
            .with_context(|| format!("failed to parse JSON array: {path}"))?
    } else {
        load_mono_annotations_jsonl(path)?
    };

    let annotation_path = Path::new(path);
    for rec in &mut records {
        rec.audio_path = resolve_audio_path(&rec.audio_path, dataset_root, annotation_path)
            .to_string_lossy()
            .into();
    }

    if only_solo {
        records.retain(|r| filename_contains_solo(&r.audio_path));
    }
    if records.is_empty() {
        bail!("no mono annotation records loaded from {path}");
    }
    Ok(records)
}

pub fn resolve_audio_path(
    audio_path: &str,
    dataset_root: &Path,
    annotation_file: &Path,
) -> PathBuf {
    let p = Path::new(audio_path);
    if p.is_absolute() {
        return p.to_path_buf();
    }

    let mut candidates = Vec::with_capacity(4);
    candidates.push(dataset_root.join(p));
    candidates.push(Path::new("input").join(p));
    if let Some(parent) = annotation_file.parent() {
        candidates.push(parent.join(p));
    }
    candidates.push(p.to_path_buf());

    for candidate in &candidates {
        if candidate.exists() {
            return candidate.clone();
        }
    }
    candidates
        .into_iter()
        .next()
        .unwrap_or_else(|| p.to_path_buf())
}

pub fn filename_contains_solo(path: &str) -> bool {
    let filename = Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path);
    filename.to_ascii_lowercase().contains("solo")
}

fn load_mono_annotations_jsonl(path: &str) -> Result<Vec<MonoAnnotation>> {
    let file = File::open(path).with_context(|| format!("failed to open JSONL: {path}"))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (line_idx, line) in reader.lines().enumerate() {
        let line =
            line.with_context(|| format!("failed to read line {} in {path}", line_idx + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: MonoAnnotation = serde_json::from_str(trimmed).with_context(|| {
            format!("failed to parse JSONL record at {}:{}", path, line_idx + 1)
        })?;
        out.push(rec);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::{filename_contains_solo, load_mono_annotations};

    #[test]
    fn solo_filter_is_case_insensitive() {
        assert!(filename_contains_solo("abc/track_SOLO_01.wav"));
        assert!(!filename_contains_solo("abc/track_comp_01.wav"));
    }

    #[test]
    fn loads_jsonl_and_filters_non_solo_records() {
        let mut path = std::env::temp_dir();
        path.push(format!("mono_{}.jsonl", std::process::id()));
        let mut file = std::fs::File::create(&path).expect("create jsonl");
        writeln!(
            file,
            "{{\"audio_path\":\"a/solo_take.wav\",\"sample_rate\":44100,\"midi\":60,\"string\":1,\"fret\":5,\"onset_sec\":0.0,\"offset_sec\":0.3,\"player_id\":null,\"guitar_id\":null,\"style\":null,\"pickup_mode\":null}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"audio_path\":\"a/chord_take.wav\",\"sample_rate\":44100,\"midi\":62,\"string\":2,\"fret\":7,\"onset_sec\":0.4,\"offset_sec\":0.7,\"player_id\":null,\"guitar_id\":null,\"style\":null,\"pickup_mode\":null}}"
        )
        .expect("write");

        let dataset = load_mono_annotations(path.to_str().expect("utf8"), Path::new("input"), true)
            .expect("load dataset");
        assert_eq!(dataset.len(), 1);
        assert!(dataset[0].audio_path.contains("solo_take.wav"));
        let _ = std::fs::remove_file(path);
    }

    use std::path::Path;
}
