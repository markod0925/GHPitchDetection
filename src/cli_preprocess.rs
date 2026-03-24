use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

use crate::types::{MonoAnnotation, OPEN_MIDI};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetSplit {
    Solo,
    Comp,
}

#[derive(Debug, Default)]
struct ParseStats {
    jams_files_total: usize,
    jams_files_processed: usize,
    jams_files_skipped: usize,
    notes_total: usize,
    notes_written: usize,
    notes_dropped_invalid: usize,
    notes_dropped_unplayable: usize,
}

pub fn run_preprocess_jams(
    dataset_root: &str,
    annotation_dir: &str,
    audio_dir: &str,
    out_mono: &str,
    out_comp: &str,
    fret_max: u8,
) -> Result<()> {
    let dataset_root = Path::new(dataset_root);
    let annotation_dir = Path::new(annotation_dir);
    let audio_dir = Path::new(audio_dir);

    let jams_files = collect_jams_files(annotation_dir)?;
    if jams_files.is_empty() {
        anyhow::bail!("no .jams files found under {}", annotation_dir.display());
    }

    let mut sample_rate_cache: BTreeMap<PathBuf, u32> = BTreeMap::new();
    let mut mono_records = Vec::new();
    let mut comp_records = Vec::new();
    let mut stats = ParseStats {
        jams_files_total: jams_files.len(),
        ..ParseStats::default()
    };

    for jams_path in jams_files {
        let raw = std::fs::read_to_string(&jams_path)
            .with_context(|| format!("failed to read JAMS file: {}", jams_path.display()))?;
        let jams: Value = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse JAMS JSON: {}", jams_path.display()))?;

        let title = extract_title(&jams).unwrap_or_else(|| {
            jams_path
                .file_stem()
                .and_then(|x| x.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

        let Some(split) = split_from_title(&title) else {
            stats.jams_files_skipped += 1;
            eprintln!(
                "warning: skipping JAMS with unknown split (expected _solo or _comp): {}",
                jams_path.display()
            );
            continue;
        };

        let audio_path_abs = resolve_audio_path(audio_dir, &title).with_context(|| {
            format!(
                "failed to resolve audio for title '{}' using directory {}",
                title,
                audio_dir.display()
            )
        })?;

        let sample_rate = if let Some(sr) = sample_rate_cache.get(&audio_path_abs) {
            *sr
        } else {
            let sr = read_wav_sample_rate(&audio_path_abs)?;
            sample_rate_cache.insert(audio_path_abs.clone(), sr);
            sr
        };

        let audio_path_for_record = path_for_dataset_record(&audio_path_abs, dataset_root);

        let parsed = parse_jams_note_midi(
            &jams,
            &audio_path_for_record,
            sample_rate,
            fret_max,
            &mut stats,
        );

        match split {
            DatasetSplit::Solo => mono_records.extend(parsed),
            DatasetSplit::Comp => comp_records.extend(parsed),
        }
        stats.jams_files_processed += 1;
    }

    mono_records.sort_by(|a, b| {
        a.audio_path
            .cmp(&b.audio_path)
            .then_with(|| a.onset_sec.total_cmp(&b.onset_sec))
            .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
            .then_with(|| a.string.cmp(&b.string))
            .then_with(|| a.midi.cmp(&b.midi))
    });

    comp_records.sort_by(|a, b| {
        a.audio_path
            .cmp(&b.audio_path)
            .then_with(|| a.onset_sec.total_cmp(&b.onset_sec))
            .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
            .then_with(|| a.string.cmp(&b.string))
            .then_with(|| a.midi.cmp(&b.midi))
    });

    write_jsonl(out_mono, &mono_records)?;
    write_jsonl(out_comp, &comp_records)?;

    println!("JAMS preprocessing complete");
    println!("dataset root: {}", dataset_root.display());
    println!("annotation dir: {}", annotation_dir.display());
    println!("audio dir: {}", audio_dir.display());
    println!("JAMS files total: {}", stats.jams_files_total);
    println!("JAMS files processed: {}", stats.jams_files_processed);
    println!("JAMS files skipped: {}", stats.jams_files_skipped);
    println!("notes seen: {}", stats.notes_total);
    println!("notes written: {}", stats.notes_written);
    println!("notes dropped (invalid): {}", stats.notes_dropped_invalid);
    println!(
        "notes dropped (unplayable): {}",
        stats.notes_dropped_unplayable
    );
    println!(
        "mono annotations (solo): {} records -> {}",
        mono_records.len(),
        out_mono
    );
    println!(
        "comp annotations: {} records -> {}",
        comp_records.len(),
        out_comp
    );

    Ok(())
}

fn collect_jams_files(annotation_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(annotation_dir)
        .with_context(|| format!("failed to list directory: {}", annotation_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("jams"))
        {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn extract_title(jams: &Value) -> Option<String> {
    jams.pointer("/file_metadata/title")
        .and_then(|x| x.as_str())
        .map(|x| x.to_string())
}

fn split_from_title(title: &str) -> Option<DatasetSplit> {
    let title = title.to_ascii_lowercase();
    if title.contains("_solo") {
        Some(DatasetSplit::Solo)
    } else if title.contains("_comp") {
        Some(DatasetSplit::Comp)
    } else {
        None
    }
}

fn resolve_audio_path(audio_dir: &Path, title: &str) -> Result<PathBuf> {
    let direct = audio_dir.join(format!("{title}_mic.wav"));
    if direct.exists() {
        return Ok(direct);
    }

    let plain = audio_dir.join(format!("{title}.wav"));
    if plain.exists() {
        return Ok(plain);
    }

    let mut fallback = None;
    for entry in std::fs::read_dir(audio_dir)
        .with_context(|| format!("failed to list audio directory: {}", audio_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
            continue;
        };
        if name.starts_with(title) && name.ends_with(".wav") {
            fallback = Some(path);
            break;
        }
    }

    fallback.with_context(|| {
        format!(
            "no audio WAV found for title '{}' under {}",
            title,
            audio_dir.display()
        )
    })
}

fn read_wav_sample_rate(path: &Path) -> Result<u32> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV: {}", path.display()))?;
    Ok(reader.spec().sample_rate)
}

fn path_for_dataset_record(audio_abs: &Path, dataset_root: &Path) -> String {
    let relative = audio_abs
        .strip_prefix(dataset_root)
        .unwrap_or(audio_abs)
        .to_string_lossy()
        .to_string();
    relative.replace('\\', "/")
}

fn parse_jams_note_midi(
    jams: &Value,
    audio_path: &str,
    sample_rate: u32,
    fret_max: u8,
    stats: &mut ParseStats,
) -> Vec<MonoAnnotation> {
    let mut records = Vec::new();

    let Some(annotations) = jams.get("annotations").and_then(|x| x.as_array()) else {
        return records;
    };

    let mut note_tracks: Vec<&Value> = annotations
        .iter()
        .filter(|ann| {
            ann.get("namespace")
                .and_then(|x| x.as_str())
                .is_some_and(|ns| ns == "note_midi")
        })
        .collect();
    if note_tracks.is_empty() {
        note_tracks = annotations
            .iter()
            .filter(|ann| {
                ann.get("namespace")
                    .and_then(|x| x.as_str())
                    .is_some_and(|ns| ns == "pitch_midi")
            })
            .collect();
    }

    for (track_idx, ann) in note_tracks.iter().enumerate() {
        // Mirror GuitarSet visualize/interpreter.py::tablaturize_jams:
        // string index comes from traversal order (s = 0..5).
        let Ok(string_idx) = u8::try_from(track_idx) else {
            continue;
        };
        if string_idx >= OPEN_MIDI.len() as u8 {
            continue;
        }

        let Some(notes) = ann.get("data").and_then(|x| x.as_array()) else {
            continue;
        };

        for note in notes {
            stats.notes_total += 1;

            let Some(onset) = note.get("time").and_then(number_as_f32) else {
                stats.notes_dropped_invalid += 1;
                continue;
            };
            let Some(duration) = note.get("duration").and_then(number_as_f32) else {
                stats.notes_dropped_invalid += 1;
                continue;
            };
            let Some(midi_raw) = note.get("value").and_then(number_as_f32) else {
                stats.notes_dropped_invalid += 1;
                continue;
            };

            if !onset.is_finite() || !duration.is_finite() || !midi_raw.is_finite() {
                stats.notes_dropped_invalid += 1;
                continue;
            }

            let midi_i32 = midi_raw.round() as i32;
            if !(0..=127).contains(&midi_i32) {
                stats.notes_dropped_invalid += 1;
                continue;
            }
            let midi = midi_i32 as u8;

            let open = OPEN_MIDI[string_idx as usize] as i32;
            let fret_i32 = midi_i32 - open;
            if fret_i32 < 0 || fret_i32 > i32::from(fret_max) {
                stats.notes_dropped_unplayable += 1;
                continue;
            }

            let onset_sec = onset.max(0.0);
            let offset_sec = (onset + duration.max(0.0)).max(onset_sec);

            records.push(MonoAnnotation {
                audio_path: audio_path.to_string(),
                sample_rate,
                midi,
                string: string_idx,
                fret: fret_i32 as u8,
                onset_sec,
                offset_sec,
                player_id: None,
                guitar_id: None,
                style: None,
                pickup_mode: None,
            });
            stats.notes_written += 1;
        }
    }

    records
}

fn number_as_f32(value: &Value) -> Option<f32> {
    value.as_f64().map(|x| x as f32)
}

fn write_jsonl(path: &str, records: &[MonoAnnotation]) -> Result<()> {
    let out_path = Path::new(path);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory: {}", parent.display()))?;
    }

    let file = File::create(out_path)
        .with_context(|| format!("failed to create output file: {}", out_path.display()))?;
    let mut writer = BufWriter::new(file);

    for record in records {
        let line = serde_json::to_string(record)
            .with_context(|| format!("failed to serialize JSONL record for {}", path))?;
        writeln!(writer, "{line}")
            .with_context(|| format!("failed to write JSONL output: {}", path))?;
    }

    writer
        .flush()
        .with_context(|| format!("failed to flush output file: {}", path))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{parse_jams_note_midi, ParseStats};

    #[test]
    fn extracts_note_midi_records() {
        let jams = json!({
            "annotations": [
                {
                    "namespace": "note_midi",
                    "annotation_metadata": {"data_source": "5"},
                    "data": [
                        {"time": 0.1, "duration": 0.2, "value": 60.1}
                    ]
                },
                {
                    "namespace": "note_midi",
                    "annotation_metadata": {"data_source": "0"},
                    "data": [
                        {"time": 0.5, "duration": 0.3, "value": 46.0}
                    ]
                }
            ]
        });

        let mut stats = ParseStats::default();
        let out = parse_jams_note_midi(&jams, "guitarset/audio/x.wav", 44100, 20, &mut stats);
        assert_eq!(out.len(), 2);
        // Track order determines the string index, matching tablaturize_jams.
        assert_eq!(out[0].string, 0);
        assert_eq!(out[0].midi, 60);
        assert_eq!(out[0].fret, 20);
        assert_eq!(out[1].string, 1);
        assert_eq!(out[1].midi, 46);
        assert_eq!(out[1].fret, 1);
    }
}
