# FretNet ONNX Runtime

Inference-focused Rust runtime for the exported FretNet ONNX model used by GuitarHelio.

The crate now includes:

- an ONNX Runtime backend for FretNet inference
- a Rust HCQT frontend intended to match the Python reference pipeline as closely as possible
- Python-generated reference fixtures for both model-input regression and feature-level validation

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

To print per-fixture and per-tensor ONNX regression diagnostics:

```bash
cargo test --release regression_matches_real_audio_reference_fixture -- --nocapture
```

This prints fixture name, input shape, shared tensor names, and `max_abs_diff` / `mean_abs_diff` for each compared output tensor.

## Limitations

- The Rust frontend currently uses a direct full-rate VQT construction, not librosa's recursive octave/downsampling path
- Feature-equivalence tolerances are therefore provisional and documented in the test rather than claimed as exact parity
- Live audio loading currently uses a simple linear resampler instead of the Python pipeline's higher-fidelity resampling path
- No quantized ONNX support yet
- Output semantic decoding is intentionally conservative
- No TorchScript runtime path yet
- Python remains the source of truth for frontend equivalence validation

## Roadmap

- Tighten Rust HCQT fidelity until feature tolerances can be reduced
- Python-vs-Rust numerical comparison harness for full frontend and model output parity
- Realtime streaming/windowed inference wrapper
- INT8 model loading and benchmark comparison
