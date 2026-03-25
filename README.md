# GHPitchDetection

Phase 2.5 adds built-in diagnostics for the existing Phase 1 (pitch-only) and Phase 2 (mono string classification) pipeline, plus a self-contained HTML report generator.

## Diagnostics Commands

Build derived annotation files from GuitarSet JAMS (`mono_annotations` from `_solo`, `comp_annotations` from `_comp`):

```bash
guitar_pitch preprocess-jams \
  --dataset-root input \
  --annotation-dir input/guitarset/annotation \
  --audio-dir input/guitarset/audio \
  --out-mono input/guitarset/derived/mono_annotations.jsonl \
  --out-comp input/guitarset/derived/comp_annotations.jsonl \
  --fret-max 20
```

The fret extraction follows the GuitarSet visualization helper logic (`tablaturize_jams`): for each string track `s`, `fret = round(midi - open_midi[s])`.

Pretrain MASP signatures and tune both the MASP `b_exponent` and the validation-rule weights from SOLO/COMP GuitarSet-derived annotations:

```bash
guitar_pitch pretrain-masp \
  --config config.toml \
  --mono-dataset input/guitarset/derived/mono_annotations.jsonl \
  --comp-dataset input/guitarset/derived/comp_annotations.jsonl \
  --dataset-root input \
  --out-dir output/debug
```

Validate one expected note/chord window with MASP:

```bash
guitar_pitch validate-masp \
  --config config.toml \
  --artifacts-dir output/debug \
  --audio input/guitarset/audio/00_BN1-129-Eb_comp_mic.wav \
  --start-sec 1.0 \
  --end-sec 1.3 \
  --expected-midis 51,58,63
```

Validate MASP from a time-aligned tablature and a moving playhead:

```bash
guitar_pitch validate-masp-playhead \
  --config config.toml \
  --artifacts-dir output/debug \
  --audio input/guitarset/audio/00_BN1-129-Eb_comp_mic.wav \
  --tablature input/example_tab.json \
  --playhead-sec 1.24
```

Batch MASP validation from JSONL requests:

```bash
guitar_pitch validate-masp-batch \
  --config config.toml \
  --artifacts-dir output/debug \
  --requests-jsonl input/guitarset/derived/masp_requests.jsonl \
  --out-dir output/debug
```

Train templates first (existing Phase 2 flow):

```bash
guitar_pitch train-templates \
  --config config.toml \
  --mono-dataset input/guitarset/derived/mono_annotations.jsonl \
  --out templates.bin
```

Generate only pitch diagnostics artifacts:

```bash
guitar_pitch eval-pitch-report \
  --config config.toml \
  --mono-dataset input/guitarset/derived/mono_annotations.jsonl \
  --dataset-root input \
  --out-dir output/debug
```

`eval-pitch-report` now runs Phase 1 as an A/B comparison:
- baseline `stft` (written to the existing `phase1_*` artifacts)
- variant `qdft` (written to `phase1_*_qdft` artifacts)
- metric deltas in `phase1_ab_comparison.json`

Generate only mono string-classification diagnostics artifacts:

```bash
guitar_pitch eval-mono-report \
  --config config.toml \
  --mono-dataset input/guitarset/derived/mono_annotations.jsonl \
  --templates templates.bin \
  --dataset-root input \
  --out-dir output/debug
```

Generate a per-note STFT/CQT comparison report for all SOLO notes played on one string:

```bash
guitar_pitch solo-string-spectrogram-report \
  --config config.toml \
  --mono-dataset input/guitarset/derived/mono_annotations.jsonl \
  --string 2 \
  --dataset-root input \
  --out-dir output/debug
```

Optional cap for quick previews:

```bash
guitar_pitch solo-string-spectrogram-report ... --max-notes 50 --open
```

The report now compares STFT and CQT (`qdft`) on each note.
CQT parameters are fixed to:

- min frequency: `75 Hz`
- max frequency: `4000 Hz`
- bins per octave: `48`
- sample rate: `44100 Hz`
- target window length: `4096` samples (mapped to QDFT quality)

STFT visualization in that report is limited to `0-4000 Hz` for direct comparison with CQT.

Build HTML report from already-generated artifacts:

```bash
guitar_pitch build-report --debug-dir output/debug
```

Run full Phase 2.5 flow in one command (pitch + mono + report):

```bash
guitar_pitch report \
  --config config.toml \
  --mono-dataset input/guitarset/derived/mono_annotations.jsonl \
  --templates templates.bin \
  --dataset-root input \
  --out-dir output/debug
```

Optional browser open:

```bash
guitar_pitch report ... --open
```

## Artifact Layout

Main outputs are written under `output/debug/`:

- `phase1_pitch_metrics.json`
- `phase1_pitch_metrics_qdft.json`
- `phase1_ab_comparison.json`
- `phase1_per_file.csv`
- `phase1_per_file_qdft.csv`
- `phase1_pitch_scores/<example>.json`
- `phase1_pitch_scores_qdft/<example>.json`
- `phase2_string_metrics.json`
- `phase2_confusion_matrix.csv`
- `phase2_confusion_matrix.json`
- `phase2_examples_topk.csv`
- `phase2_examples_topk.json`
- `phase2_feature_stats.json`
- `phase2_per_file.csv`
- `phase2_mono_scores/<example>.json`
- `phase2_error_highlights.json`
- `masp_manifest.json`
- `masp_note_signatures.jsonl`
- `masp_joint_signatures.jsonl`
- `masp_validation_results.jsonl`
- `masp_validation_summary.json`
- `report/index.html`
- `solo_string_<string>_spectrogram_report/index.html`
- `solo_string_<string>_spectrogram_report/figures/*.svg`

## Metric Notes

- `masp_manifest.json` now separates MASP backend settings (`model_params`) from the final pass/fail rule (`validation_rule`).
- `validation_rule.score_weights` and `validation_rule.score_threshold` affect only the final validation decision; they do not change MASP spectrum construction.
- `validate-masp-playhead` resolves the active tablature event at the given playhead and then calls the same MASP backend using the derived `expected_midis`, `start_sec`, and `end_sec`.
- `ts/masp/maspCore.ts` is the TypeScript-portable MASP core for GuitarHelio integration. It mirrors the current Rust formulas and embeds the tuned parameters:
  - `bExponent = 0.7726795`
  - `scoreThreshold = 0.4370982`
  - `scoreWeights.har = 0.16453995`
  - `scoreWeights.mbw = 0.5508079`
  - `scoreWeights.cent = 0.038484424`
  - `scoreWeights.rms = 0.24616772`
- `ts/masp/checkConsistency.mjs` must be run with Node v22 and validates the TypeScript module against the shared fixture file `ts/masp/consistencyFixtures.json`.
- Rust unit tests read that same fixture file through `include_str!`, so the Rust and TypeScript implementations are checked against one identical source of truth.
- Tablature runtime input is a JSON array or JSONL of timed note events:

```json
[
  { "string": 1, "fret": 3, "onset_sec": 1.00, "offset_sec": 1.30 },
  { "string": 2, "fret": 0, "onset_sec": 1.02, "offset_sec": 1.28 },
  { "string": 3, "fret": 0, "onset_sec": 1.01, "offset_sec": 1.25 }
]
```

- `midi` is optional in tablature input; if omitted, it is derived from `(string, fret)` using standard guitar tuning.
- If multiple overlapping notes exist on the same string at the playhead, the runtime resolver keeps the most recent one on that string.

- Pitch precision/recall/F1 is computed event-wise using one representative frame per annotated note event (midpoint frame in the note segment).
- False positives are predicted pitches not present in ground truth for the evaluated event.
- Octave and sub-octave errors are tracked when the true pitch is missed and `midi +/- 12` appears in predictions.
- Top-1 / Top-3 string accuracy measures whether true string appears in the top 1 / top 3 predicted candidates.
- Exact string+fret accuracy is reported only when meaningful (`templates.level = "string_fret"`).
- Confusion matrix is true string vs predicted top-1 string, with both count and normalized row values.
- Top-k table includes score margin (`top1 - top2`) to highlight ambiguity and miscalibration patterns.

## Phase 3+ TODOs

- Add polyphonic pitch diagnostics and full-system (pitch+string+fret) decoding evaluation in the report flow.
- Extend diagnostics to global decode/tracking outputs after Phase 3 implementation.

## Phase 3 Commands

Evaluate the full polyphonic pipeline (pitch + string + fret decode):

```bash
guitar_pitch eval-full \
  --config config.toml \
  --poly-dataset input/guitarset/derived/comp_annotations.jsonl \
  --templates templates.bin \
  --dataset-root input
```

Generate Phase 3 debug artifacts and refresh HTML report:

```bash
guitar_pitch eval-full-report \
  --config config.toml \
  --poly-dataset input/guitarset/derived/comp_annotations.jsonl \
  --templates templates.bin \
  --dataset-root input \
  --out-dir output/debug
```
