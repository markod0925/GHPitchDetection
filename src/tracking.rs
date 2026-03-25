use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::AppConfig;
use crate::masp::{validate_expected_segment, MaspPretrainArtifacts, MaspValidationResult};
use crate::types::{AudioBuffer, OPEN_MIDI};

const PLAYHEAD_EPS_SEC: f32 = 1e-4;
const STRING_COUNT: u8 = 6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TablatureNoteEvent {
    pub string: u8,
    pub fret: u8,
    pub onset_sec: f32,
    pub offset_sec: f32,
    pub midi: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedTabSegment {
    pub playhead_sec: f32,
    pub start_sec: f32,
    pub end_sec: f32,
    pub expected_midis: Vec<u8>,
    pub notes: Vec<TablatureNoteEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaspPlayheadValidationResult {
    pub segment: ResolvedTabSegment,
    pub validation: MaspValidationResult,
}

pub fn read_tablature_events(path: &Path) -> Result<Vec<TablatureNoteEvent>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read tablature events from: {}", path.display()))?;
    let first_non_ws = data.chars().find(|c| !c.is_whitespace());
    let mut out = if first_non_ws == Some('[') {
        serde_json::from_str::<Vec<TablatureNoteEvent>>(&data)
            .with_context(|| format!("failed to parse tablature JSON array: {}", path.display()))?
    } else {
        read_tablature_events_jsonl(path)?
    };
    normalize_tablature_events(&mut out)?;
    Ok(out)
}

pub fn resolve_tablature_playhead(
    events: &[TablatureNoteEvent],
    playhead_sec: f32,
) -> Result<ResolvedTabSegment> {
    if !playhead_sec.is_finite() || playhead_sec < 0.0 {
        bail!("playhead_sec must be finite and >= 0");
    }
    if events.is_empty() {
        bail!("tablature event list must not be empty");
    }

    let mut normalized_events = events.to_vec();
    normalize_tablature_events(&mut normalized_events)?;

    let mut active_by_string = BTreeMap::<u8, TablatureNoteEvent>::new();
    for event in &normalized_events {
        if !event_contains_playhead(event, playhead_sec) {
            continue;
        }
        match active_by_string.get(&event.string) {
            None => {
                active_by_string.insert(event.string, event.clone());
            }
            Some(previous) => {
                if prefer_playhead_event(event, previous) {
                    active_by_string.insert(event.string, event.clone());
                }
            }
        }
    }

    if active_by_string.is_empty() {
        bail!("no tablature note is active at playhead_sec={playhead_sec:.6}");
    }

    let mut notes = active_by_string.into_values().collect::<Vec<_>>();
    notes.sort_by(|a, b| {
        a.string
            .cmp(&b.string)
            .then_with(|| a.onset_sec.total_cmp(&b.onset_sec))
            .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
            .then_with(|| a.fret.cmp(&b.fret))
    });

    let start_sec = notes
        .iter()
        .map(|x| x.onset_sec)
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(playhead_sec);
    let end_sec = notes
        .iter()
        .map(|x| x.offset_sec)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(playhead_sec);
    let mut expected_midis = notes.iter().map(tab_event_midi).collect::<Vec<_>>();
    expected_midis.sort_unstable();
    expected_midis.dedup();

    Ok(ResolvedTabSegment {
        playhead_sec,
        start_sec,
        end_sec,
        expected_midis,
        notes,
    })
}

pub fn validate_masp_with_playhead(
    audio: &AudioBuffer,
    playhead_sec: f32,
    tablature: &[TablatureNoteEvent],
    cfg: &AppConfig,
    artifacts: &MaspPretrainArtifacts,
) -> Result<MaspPlayheadValidationResult> {
    let segment = resolve_tablature_playhead(tablature, playhead_sec)?;
    let mut validation = validate_expected_segment(
        audio,
        segment.start_sec,
        segment.end_sec,
        &segment.expected_midis,
        cfg,
        artifacts,
    )?;
    validation.expected_midis = segment.expected_midis.clone();
    Ok(MaspPlayheadValidationResult {
        segment,
        validation,
    })
}

fn read_tablature_events_jsonl(path: &Path) -> Result<Vec<TablatureNoteEvent>> {
    let file =
        File::open(path).with_context(|| format!("failed to open JSONL: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!("failed to read line {} in {}", line_idx + 1, path.display())
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: TablatureNoteEvent = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "failed to parse tablature JSONL record at {}:{}",
                path.display(),
                line_idx + 1
            )
        })?;
        out.push(rec);
    }
    Ok(out)
}

fn normalize_tablature_events(events: &mut [TablatureNoteEvent]) -> Result<()> {
    for event in events.iter_mut() {
        validate_tablature_event(event)?;
        if event.midi.is_none() {
            event.midi = Some(open_string_midi(event.string)? + event.fret);
        }
    }
    events.sort_by(|a, b| {
        a.onset_sec
            .total_cmp(&b.onset_sec)
            .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
            .then_with(|| a.string.cmp(&b.string))
            .then_with(|| a.fret.cmp(&b.fret))
    });
    Ok(())
}

fn validate_tablature_event(event: &TablatureNoteEvent) -> Result<()> {
    if event.string >= STRING_COUNT {
        bail!("tablature string index must be in 0..6 (got {})", event.string);
    }
    if !event.onset_sec.is_finite() || !event.offset_sec.is_finite() {
        bail!("tablature onset/offset must be finite");
    }
    if event.onset_sec < 0.0 || event.offset_sec < event.onset_sec {
        bail!(
            "tablature onset/offset must satisfy 0 <= onset <= offset (got {}..{})",
            event.onset_sec,
            event.offset_sec
        );
    }
    if let Some(midi) = event.midi {
        let expected = open_string_midi(event.string)? + event.fret;
        if midi != expected {
            bail!(
                "tablature midi mismatch for string={} fret={}: got {}, expected {}",
                event.string,
                event.fret,
                midi,
                expected
            );
        }
    }
    Ok(())
}

fn event_contains_playhead(event: &TablatureNoteEvent, playhead_sec: f32) -> bool {
    event.onset_sec - PLAYHEAD_EPS_SEC <= playhead_sec
        && playhead_sec <= event.offset_sec + PLAYHEAD_EPS_SEC
}

fn prefer_playhead_event(candidate: &TablatureNoteEvent, previous: &TablatureNoteEvent) -> bool {
    candidate.onset_sec > previous.onset_sec
        || (candidate.onset_sec == previous.onset_sec && candidate.offset_sec > previous.offset_sec)
        || (candidate.onset_sec == previous.onset_sec
            && candidate.offset_sec == previous.offset_sec
            && candidate.fret > previous.fret)
}

fn tab_event_midi(event: &TablatureNoteEvent) -> u8 {
    event
        .midi
        .unwrap_or_else(|| OPEN_MIDI[event.string as usize] + event.fret)
}

fn open_string_midi(string: u8) -> Result<u8> {
    OPEN_MIDI
        .get(string as usize)
        .copied()
        .with_context(|| format!("invalid string index: {string}"))
}

#[cfg(test)]
mod tests {
    use super::{
        resolve_tablature_playhead, tab_event_midi, validate_tablature_event, TablatureNoteEvent,
    };
    use serde::Deserialize;

    const CONSISTENCY_FIXTURE_JSON: &str = include_str!("../ts/masp/consistencyFixtures.json");

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TsTrackingFixture {
        playhead_resolution: TsPlayheadCase,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TsPlayheadCase {
        playhead_sec: f32,
        events: Vec<TsTabEvent>,
        resolved: TsResolvedSegment,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TsResolvedSegment {
        playhead_sec: f32,
        start_sec: f32,
        end_sec: f32,
        expected_midis: Vec<u8>,
        notes: Vec<TsTabEvent>,
    }

    #[derive(Debug, Clone, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct TsTabEvent {
        string: u8,
        fret: u8,
        onset_sec: f32,
        offset_sec: f32,
        midi: Option<u8>,
    }

    fn load_tracking_fixture() -> TsTrackingFixture {
        serde_json::from_str(CONSISTENCY_FIXTURE_JSON).expect("valid TS tracking fixture")
    }

    fn to_tab_event(event: &TsTabEvent) -> TablatureNoteEvent {
        TablatureNoteEvent {
            string: event.string,
            fret: event.fret,
            onset_sec: event.onset_sec,
            offset_sec: event.offset_sec,
            midi: event.midi,
        }
    }

    #[test]
    fn resolve_playhead_picks_active_single_note() {
        let events = vec![TablatureNoteEvent {
            string: 1,
            fret: 7,
            onset_sec: 1.0,
            offset_sec: 1.5,
            midi: None,
        }];
        let resolved = resolve_tablature_playhead(&events, 1.2).expect("resolve");
        assert_eq!(resolved.expected_midis, vec![52]);
        assert_eq!(resolved.start_sec, 1.0);
        assert_eq!(resolved.end_sec, 1.5);
    }

    #[test]
    fn resolve_playhead_builds_chord_from_concurrent_strings() {
        let events = vec![
            TablatureNoteEvent {
                string: 0,
                fret: 3,
                onset_sec: 2.0,
                offset_sec: 2.4,
                midi: None,
            },
            TablatureNoteEvent {
                string: 1,
                fret: 2,
                onset_sec: 2.02,
                offset_sec: 2.35,
                midi: None,
            },
            TablatureNoteEvent {
                string: 2,
                fret: 0,
                onset_sec: 2.01,
                offset_sec: 2.3,
                midi: None,
            },
        ];
        let resolved = resolve_tablature_playhead(&events, 2.1).expect("resolve");
        assert_eq!(resolved.expected_midis, vec![43, 47, 50]);
        assert_eq!(resolved.notes.len(), 3);
        assert_eq!(resolved.start_sec, 2.0);
        assert_eq!(resolved.end_sec, 2.4);
    }

    #[test]
    fn resolve_playhead_prefers_latest_note_on_same_string() {
        let events = vec![
            TablatureNoteEvent {
                string: 2,
                fret: 2,
                onset_sec: 1.0,
                offset_sec: 1.4,
                midi: None,
            },
            TablatureNoteEvent {
                string: 2,
                fret: 5,
                onset_sec: 1.2,
                offset_sec: 1.6,
                midi: None,
            },
        ];
        let resolved = resolve_tablature_playhead(&events, 1.25).expect("resolve");
        assert_eq!(resolved.notes.len(), 1);
        assert_eq!(resolved.notes[0].fret, 5);
        assert_eq!(resolved.expected_midis, vec![55]);
    }

    #[test]
    fn tablature_midi_must_match_string_and_fret() {
        let err = validate_tablature_event(&TablatureNoteEvent {
            string: 0,
            fret: 3,
            onset_sec: 0.0,
            offset_sec: 0.3,
            midi: Some(99),
        })
        .expect_err("invalid midi should fail");
        assert!(err.to_string().contains("midi mismatch"));
    }

    #[test]
    fn tab_event_midi_uses_open_string_when_missing() {
        let midi = tab_event_midi(&TablatureNoteEvent {
            string: 5,
            fret: 3,
            onset_sec: 0.0,
            offset_sec: 0.1,
            midi: None,
        });
        assert_eq!(midi, 67);
    }

    #[test]
    fn ts_consistency_fixture_matches_rust_playhead_resolution() {
        let fixture = load_tracking_fixture();
        let playhead = fixture.playhead_resolution;
        let events = playhead.events.iter().map(to_tab_event).collect::<Vec<_>>();
        let resolved = resolve_tablature_playhead(&events, playhead.playhead_sec).expect("resolve");

        assert_eq!(resolved.playhead_sec, playhead.resolved.playhead_sec);
        assert_eq!(resolved.start_sec, playhead.resolved.start_sec);
        assert_eq!(resolved.end_sec, playhead.resolved.end_sec);
        assert_eq!(resolved.expected_midis, playhead.resolved.expected_midis);
        assert_eq!(resolved.notes.len(), playhead.resolved.notes.len());

        for (actual, expected) in resolved.notes.iter().zip(playhead.resolved.notes.iter()) {
            assert_eq!(actual.string, expected.string);
            assert_eq!(actual.fret, expected.fret);
            assert_eq!(actual.onset_sec, expected.onset_sec);
            assert_eq!(actual.offset_sec, expected.offset_sec);
            assert_eq!(actual.midi, expected.midi);
        }
    }
}
