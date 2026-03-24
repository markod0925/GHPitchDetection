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
- `report/index.html`
- `solo_string_<string>_spectrogram_report/index.html`
- `solo_string_<string>_spectrogram_report/figures/*.svg`

## Metric Notes

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
