# FretNet ONNX Runtime

Inference-focused Rust runtime for the exported FretNet ONNX model used by GuitarHelio.

The crate now includes:

- an ONNX Runtime backend for FretNet inference
- a Rust HCQT frontend intended to match the Python reference pipeline as closely as possible
- Python-generated reference fixtures for both model-input regression and feature-level validation
- stage-by-stage Python-vs-Rust frontend diagnostics for localizing mismatch

It still does not train models, export checkpoints, or depend on Python at runtime.

## Scope

- Load `output/export/model.onnx`
- Extract HCQT features from mono audio in Rust
- Validate feature tensor shape before inference
- Reuse a single ONNX Runtime session
- Return raw named tensors faithfully
- Provide a conservative decoding layer with shape and numeric summaries
- Offer examples, regression tests, and a lightweight latency benchmark

## Required Artifact

The runtime expects an external ONNX model. The default path is:

```text
../guitar-transcription-continuous/output/export/model.onnx
```

relative to this crate directory.

The validated TorchScript artifact also exists at:

```text
../guitar-transcription-continuous/output/export/model_traced.ts
```

but TorchScript is not used by this crate yet. It is only a future comparison target.

## Real-Audio Regression Fixtures

The crate also supports a fixture generated from a real GuitarSet wave:

```text
assets/test_features/00_BN1-129-Eb_comp_mic_f130.npz
assets/test_features/00_BN1-129-Eb_comp_mic_f130.json
assets/test_features/00_Jazz1-130-D_comp_mic_f130.npz
assets/test_features/00_Jazz1-130-D_comp_mic_f130.json
```

These fixtures were generated from:

```text
../../../input/guitarset/audio/00_BN1-129-Eb_comp_mic.wav
../../../input/guitarset/audio/00_Jazz1-130-D_comp_mic.wav
```

using the existing Python 3.11 environment at:

```text
../guitar-transcription-continuous/.venv311
```

The fixture stores:

- the ONNX-ready input tensor returned by `model.pre_proc(...)`
- raw PyTorch reference outputs for `tablature`, `tablature_rel`, and `onsets`
- metadata describing the source audio and HCQT configuration

Regenerate it with:

```bash
../guitar-transcription-continuous/.venv311/bin/python tools/generate_reference_fixture.py
../guitar-transcription-continuous/.venv311/bin/python tools/generate_reference_fixture.py --audio-path ../../../input/guitarset/audio/00_Jazz1-130-D_comp_mic.wav --output-prefix assets/test_features/00_Jazz1-130-D_comp_mic_f130
```

## Python Feature Reference Fixtures

Feature-equivalence validation uses Python-generated HCQT fixtures:

```text
assets/test_features_python/00_BN1-129-Eb_comp_mic_features_f130.npz
assets/test_features_python/00_BN1-129-Eb_comp_mic_features_f130.json
assets/test_features_python/00_Jazz1-130-D_comp_mic_features_f130.npz
assets/test_features_python/00_Jazz1-130-D_comp_mic_features_f130.json
```

These fixtures store:

- normalized and resampled mono audio
- raw HCQT features from the canonical Python frontend
- frame times
- frontend metadata in JSON

Generate them with the existing Python 3.11 environment:

```bash
../guitar-transcription-continuous/.venv311/bin/python tools/generate_feature_fixture.py
../guitar-transcription-continuous/.venv311/bin/python tools/generate_feature_fixture.py --audio-path ../../../input/guitarset/audio/00_Jazz1-130-D_comp_mic.wav --output-prefix assets/test_features_python/00_Jazz1-130-D_comp_mic_features_f130
```

Inspect a saved feature fixture with:

```bash
../guitar-transcription-continuous/.venv311/bin/python tools/inspect_feature_fixture.py --fixture-path assets/test_features_python/00_BN1-129-Eb_comp_mic_features_f130.npz
```

## Python Stage Fixtures

Stage-by-stage frontend diagnostics use structured fixture directories under:

```text
assets/test_features_python_stages/
```

Generate them with the canonical Python frontend:

```bash
../guitar-transcription-continuous/.venv311/bin/python tools/generate_stage_fixtures.py
../guitar-transcription-continuous/.venv311/bin/python tools/generate_stage_fixtures.py --audio-path ../../../input/guitarset/audio/00_Jazz1-130-D_comp_mic.wav --output-dir assets/test_features_python_stages/00_Jazz1-130-D_comp_mic
```

Inspect a stage fixture directory with:

```bash
../guitar-transcription-continuous/.venv311/bin/python tools/inspect_stage_fixture.py assets/test_features_python_stages/00_BN1-129-Eb_comp_mic
```

## Current Input Assumptions

Two input paths are supported:

- audio input for Rust HCQT extraction
- precomputed HCQT model-input tensors for direct ONNX regression/inference

Current observed ONNX input shape is:

```text
[batch, frames, channels, freq_bins, frame_width]
```

with static non-dynamic dimensions:

- `channels = 6`
- `freq_bins = 144`
- `frame_width = 9`

Dynamic axes are currently `batch` and `frames`.

## Resampling

The Rust audio path now uses an embedded `kaiser_best` filter compatible with the Python reference path:

- Python reference: `librosa.load(..., mono=True, res_type='kaiser_best')`
- Effective backend in the reference environment: `resampy`
- Rust implementation: a local `resampy`-style interpolator using the packaged `kaiser_best` filter coefficients

Audio preparation order is aligned with the Python path:

1. load mono WAV
2. resample to the frontend sample rate
3. apply RMS normalization

## Public API

```rust
let mut runtime = FretNetRuntime::load(model_path)?;
let output = runtime.infer_features(&features)?;
let decoded = runtime.decode_output(&output)?;
```

The API intentionally separates:

- model/session loading
- input validation and tensor preparation
- raw inference execution
- conservative postprocessing

## Commands

Inspect model I/O:

```bash
cargo run --release --example inspect_model
```

Run inference on synthetic or fixture-backed model input tensors:

```bash
cargo run --release --example run_inference
```

If the default fixture exists, `run_inference` automatically uses it instead of synthetic data. To force a specific fixture path:

```bash
cargo run --release --example run_inference -- --fixture-path assets/test_features/00_BN1-129-Eb_comp_mic_f130.npz
```

Extract Rust HCQT features from real audio:

```bash
cargo run --release --example extract_features -- --audio-path ../../../input/guitarset/audio/00_BN1-129-Eb_comp_mic.wav
```

Run the full Rust path `audio -> HCQT -> ONNX`:

```bash
cargo run --release --example run_end_to_end -- --audio-path ../../../input/guitarset/audio/00_BN1-129-Eb_comp_mic.wav
```

Run the full real-audio end-to-end benchmark:

```bash
cargo run --release --example benchmark_end_to_end -- --iterations 10
```

This benchmark uses the real audio paths referenced by the existing stage fixtures by default. You can override them explicitly:

```bash
cargo run --release --example benchmark_end_to_end -- --audio-path ../../../input/guitarset/audio/00_BN1-129-Eb_comp_mic.wav --audio-path ../../../input/guitarset/audio/00_Jazz1-130-D_comp_mic.wav
```

It reports one-time model load cost separately, then per-input timings for:

- stage A: WAV load / decode
- stage B: audio preparation, with resample and normalize sub-timings
- stage C: frontend extraction and HCQT-to-model-input conversion
- stage D: ONNX inference
- stage E: total pipeline time from WAV input to model outputs

Run the detailed frontend profiler on the default real-audio set:

```bash
cargo run --release --example benchmark_frontend_profile -- --iterations 10 --json-output output/frontend_profile.json
```

This profiler uses the same real audio discovery path as the existing validation fixtures. It separates:

- extractor initialization / basis construction
- per-run frontend total
- per-harmonic time
- per-octave transform, FFT, dot-product, downsample, and materialization time
- final HCQT assembly
- HCQT-to-batch conversion

Use it before optimization work to identify whether the frontend is dominated by downsampling, transform execution, repeated allocations, or other setup costs.

Run the end-to-end Python-vs-Rust consistency benchmark on a small subset:

```bash
../guitar-transcription-continuous/.venv311/bin/python tools/benchmark_python_vs_rust.py --limit 5 --max-frames 130
```

Run it on the full inferred GuitarSet audio directory:

```bash
../guitar-transcription-continuous/.venv311/bin/python tools/benchmark_python_vs_rust.py --full-dataset
```

This benchmark uses the real Python reference path (`audio -> Python frontend -> Python model`) and the real Rust path (`audio -> Rust frontend -> ONNX`). It writes:

- per-file JSON results and pipeline artifacts under `output/python_vs_rust_consistency/per_file/`
- `results.jsonl` and `results.csv`
- `summary.json`
- `worst_cases.json`

It reports raw tensor agreement, optional semantic agreement using the existing Python post-processing semantics, aggregate per-tensor metrics, and worst-case files. Use it before frontend performance optimization work.

Run the annotation-level Python-vs-Rust benchmark on a small subset:

```bash
../guitar-transcription-continuous/.venv311/bin/python tools/benchmark_against_guitarset.py --limit 5 --max-frames 130
```

Run it on the full inferred GuitarSet audio directory:

```bash
../guitar-transcription-continuous/.venv311/bin/python tools/benchmark_against_guitarset.py --full-dataset
```

This benchmark uses:

- the real Python reference path against GuitarSet JAMS annotations
- the real Rust `audio -> frontend -> ONNX` path against the same annotations
- the same Python semantic post-processing for both pipelines before annotation scoring

It reports:

- optional raw Python-vs-Rust tensor consistency
- onset precision / recall / F1
- tablature precision / recall / F1 and TDR
- pitch-only precision / recall / F1
- exact tablature frame accuracy
- Rust-minus-Python deltas for each evaluation metric
- per-file Python and Rust pipeline wall-clock timings
- worst-case files by metric delta and overall degradation score

Artifacts are written under `output/annotation_eval/` by default, including per-file JSON results, `results.jsonl`, `results.csv`, `summary.json`, and `worst_cases.json`. The summary includes one-time Python model load cost, one-time Rust extractor/runtime load cost, and per-file pipeline timings. The annotation benchmark now uses a persistent Rust batch runner, so the reported per-file Rust timings no longer include reloading the ONNX session for every audio file. Run this benchmark before frontend performance optimization work so you know whether the remaining frontend mismatch is still musically meaningful.

Run latency benchmark:

```bash
cargo run --release --example benchmark_inference -- --iterations 50
```

When default real fixtures are present, the benchmark runs on all of them automatically and prints:

- per-fixture input shape and load time
- first / mean / min / max / p95 inference latency
- a final summary with fastest fixture, slowest fixture, global mean of means, and global max latency

If no real fixture is found, the benchmark falls back explicitly to synthetic input.

Run tests:

```bash
cargo test --release
```

The regression test now compares Rust ONNX output against saved PyTorch reference tensors generated from both real GuitarSet `comp` waves above.

To print feature-equivalence diagnostics:

```bash
cargo test --release rust_hcqt_matches_python_feature_fixture_within_documented_tolerance -- --nocapture
```

This prints fixture name, feature shape, global `max_abs_diff` / `mean_abs_diff`, and per-harmonic diffs.

To compare the frontend stage by stage:

```bash
cargo test --release rust_frontend_stage_diagnostics_against_python_reference -- --nocapture
```

This prints, per fixture:

- stage 0 audio metadata checks
- stage 1 resampled waveform diffs
- stage 2 per-harmonic VQT magnitude and post-processed diffs
- stage 3 final HCQT diffs
- center-cropped and small-lag diagnostics to reveal boundary or alignment issues

To print per-fixture and per-tensor ONNX regression diagnostics:

```bash
cargo test --release regression_matches_real_audio_reference_fixture -- --nocapture
```

This prints fixture name, input shape, shared tensor names, and `max_abs_diff` / `mean_abs_diff` for each compared output tensor.

## Limitations

- The Rust frontend now uses recursive octave/downsampling VQT construction, but it is still not exact librosa parity
- Resampling is now very close to the Python reference on the real fixtures; the remaining frontend mismatch is smaller and still concentrated in the transform stage, especially the `0.5` harmonic branch
- Feature-equivalence tolerances are therefore provisional and documented in the test rather than claimed as exact parity
- Current mismatch should be reduced by improving frontend fidelity stage by stage, not by loosening tolerances
- No quantized ONNX support yet
- Output semantic decoding is intentionally conservative
- No TorchScript runtime path yet
- Python remains the source of truth for frontend equivalence validation

## Roadmap

- Tighten Rust HCQT fidelity until feature tolerances can be reduced
- Python-vs-Rust numerical comparison harness for full frontend and model output parity
- Realtime streaming/windowed inference wrapper
- INT8 model loading and benchmark comparison
