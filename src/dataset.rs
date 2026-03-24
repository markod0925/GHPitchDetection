use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::types::{MonoAnnotation, PolyAnnotation, PolyEvent};

const EVENT_ONSET_TOLERANCE_SEC: f32 = 0.05;

#[derive(Debug, Clone)]
pub struct PolyEvalEvent {
    pub audio_path: String,
    pub sample_rate: u32,
    pub onset_sec: f32,
    pub offset_sec: f32,
    pub notes: Vec<PolyEvent>,
}

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

pub fn load_poly_events(
    path: &str,
    dataset_root: &Path,
    only_comp: bool,
) -> Result<Vec<PolyEvalEvent>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read poly annotations from: {path}"))?;
    let first_non_ws = data.chars().find(|c| !c.is_whitespace());

    let mut grouped_notes: BTreeMap<(String, u32), Vec<PolyEvent>> = BTreeMap::new();
    let annotation_path = Path::new(path);

    if first_non_ws == Some('[') {
        let value: Value = serde_json::from_str(&data)
            .with_context(|| format!("failed to parse JSON array payload: {path}"))?;
        let Some(items) = value.as_array() else {
            bail!("expected JSON array in {path}");
        };
        for (idx, item) in items.iter().enumerate() {
            ingest_poly_like_record(
                item,
                dataset_root,
                annotation_path,
                idx + 1,
                &mut grouped_notes,
            )
            .with_context(|| format!("failed to parse array record {} in {path}", idx + 1))?;
        }
    } else {
        let file = File::open(path).with_context(|| format!("failed to open JSONL: {path}"))?;
        let reader = BufReader::new(file);
        for (line_idx, line) in reader.lines().enumerate() {
            let line =
                line.with_context(|| format!("failed to read line {} in {path}", line_idx + 1))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(trimmed).with_context(|| {
                format!("failed to parse JSONL record at {}:{}", path, line_idx + 1)
            })?;
            ingest_poly_like_record(
                &value,
                dataset_root,
                annotation_path,
                line_idx + 1,
                &mut grouped_notes,
            )
            .with_context(|| format!("failed to ingest record at {}:{}", path, line_idx + 1))?;
        }
    }

    let mut out = Vec::new();
    for ((audio_path, sample_rate), mut notes) in grouped_notes {
        if only_comp && !filename_contains_comp(&audio_path) {
            continue;
        }
        notes.retain(valid_poly_note);
        if notes.is_empty() {
            continue;
        }
        notes.sort_by(|a, b| {
            a.onset_sec
                .total_cmp(&b.onset_sec)
                .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
                .then_with(|| a.midi.cmp(&b.midi))
                .then_with(|| a.string.cmp(&b.string))
                .then_with(|| a.fret.cmp(&b.fret))
        });

        let grouped = group_notes_into_events(notes, EVENT_ONSET_TOLERANCE_SEC);
        for (onset_sec, offset_sec, mut event_notes) in grouped {
            event_notes.sort_by(|a, b| {
                a.midi
                    .cmp(&b.midi)
                    .then_with(|| a.string.cmp(&b.string))
                    .then_with(|| a.fret.cmp(&b.fret))
                    .then_with(|| a.onset_sec.total_cmp(&b.onset_sec))
            });
            out.push(PolyEvalEvent {
                audio_path: audio_path.clone(),
                sample_rate,
                onset_sec,
                offset_sec,
                notes: event_notes,
            });
        }
    }

    out.sort_by(|a, b| {
        a.audio_path
            .cmp(&b.audio_path)
            .then_with(|| a.onset_sec.total_cmp(&b.onset_sec))
            .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
    });

    if out.is_empty() {
        bail!("no poly events loaded from {path}");
    }
    Ok(out)
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

pub fn filename_contains_comp(path: &str) -> bool {
    let filename = Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path);
    filename.to_ascii_lowercase().contains("comp")
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

fn ingest_poly_like_record(
    value: &Value,
    dataset_root: &Path,
    annotation_file: &Path,
    row_number: usize,
    grouped_notes: &mut BTreeMap<(String, u32), Vec<PolyEvent>>,
) -> Result<()> {
    if value.get("events").is_some() {
        let mut record: PolyAnnotation = serde_json::from_value(value.clone())
            .with_context(|| format!("failed to parse PolyAnnotation at record {row_number}"))?;
        record.audio_path = resolve_audio_path(&record.audio_path, dataset_root, annotation_file)
            .to_string_lossy()
            .into();
        grouped_notes
            .entry((record.audio_path, record.sample_rate))
            .or_default()
            .extend(record.events.into_iter());
        return Ok(());
    }

    let mut mono: MonoAnnotation = serde_json::from_value(value.clone())
        .with_context(|| format!("failed to parse MonoAnnotation at record {row_number}"))?;
    mono.audio_path = resolve_audio_path(&mono.audio_path, dataset_root, annotation_file)
        .to_string_lossy()
        .into();

    grouped_notes
        .entry((mono.audio_path, mono.sample_rate))
        .or_default()
        .push(PolyEvent {
            onset_sec: mono.onset_sec,
            offset_sec: mono.offset_sec,
            midi: mono.midi,
            string: Some(mono.string),
            fret: Some(mono.fret),
        });
    Ok(())
}

fn group_notes_into_events(
    notes: Vec<PolyEvent>,
    onset_tolerance_sec: f32,
) -> Vec<(f32, f32, Vec<PolyEvent>)> {
    if notes.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut current_notes = Vec::new();
    let mut current_onset = 0.0_f32;
    let mut current_offset = 0.0_f32;

    for note in notes {
        if current_notes.is_empty() {
            current_onset = note.onset_sec;
            current_offset = note.offset_sec;
            current_notes.push(note);
            continue;
        }

        if note.onset_sec - current_onset <= onset_tolerance_sec {
            current_onset = current_onset.min(note.onset_sec);
            current_offset = current_offset.max(note.offset_sec);
            current_notes.push(note);
            continue;
        }

        out.push((
            current_onset,
            current_offset,
            std::mem::take(&mut current_notes),
        ));
        current_onset = note.onset_sec;
        current_offset = note.offset_sec;
        current_notes.push(note);
    }

    if !current_notes.is_empty() {
        out.push((current_onset, current_offset, current_notes));
    }

    out
}

fn valid_poly_note(note: &PolyEvent) -> bool {
    note.onset_sec.is_finite()
        && note.offset_sec.is_finite()
        && note.offset_sec >= note.onset_sec
        && note.midi <= 127
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::Path;

    use super::{
        filename_contains_comp, filename_contains_solo, load_mono_annotations, load_poly_events,
    };

    #[test]
    fn split_filter_is_case_insensitive() {
        assert!(filename_contains_solo("abc/track_SOLO_01.wav"));
        assert!(!filename_contains_solo("abc/track_comp_01.wav"));
        assert!(filename_contains_comp("abc/track_COMP_01.wav"));
        assert!(!filename_contains_comp("abc/track_solo_01.wav"));
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

    #[test]
    fn loads_flat_comp_notes_as_grouped_poly_events() {
        let mut path = std::env::temp_dir();
        path.push(format!("poly_{}.jsonl", std::process::id()));
        let mut file = std::fs::File::create(&path).expect("create jsonl");

        writeln!(
            file,
            "{{\"audio_path\":\"a/song_comp.wav\",\"sample_rate\":44100,\"midi\":60,\"string\":0,\"fret\":20,\"onset_sec\":0.000,\"offset_sec\":0.300,\"player_id\":null,\"guitar_id\":null,\"style\":null,\"pickup_mode\":null}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"audio_path\":\"a/song_comp.wav\",\"sample_rate\":44100,\"midi\":64,\"string\":1,\"fret\":19,\"onset_sec\":0.020,\"offset_sec\":0.280,\"player_id\":null,\"guitar_id\":null,\"style\":null,\"pickup_mode\":null}}"
        )
        .expect("write");
        writeln!(
            file,
            "{{\"audio_path\":\"a/song_comp.wav\",\"sample_rate\":44100,\"midi\":67,\"string\":2,\"fret\":17,\"onset_sec\":0.420,\"offset_sec\":0.700,\"player_id\":null,\"guitar_id\":null,\"style\":null,\"pickup_mode\":null}}"
        )
        .expect("write");

        let events = load_poly_events(path.to_str().expect("utf8"), Path::new("input"), true)
            .expect("load poly events");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].notes.len(), 2);
        assert_eq!(events[1].notes.len(), 1);

        let _ = std::fs::remove_file(path);
    }
}
