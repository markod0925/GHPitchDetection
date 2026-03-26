# FretNet ONNX Runtime (Rust)

Inference-focused Rust runtime for the exported FretNet ONNX model used by GuitarHelio.

## Core Facts

- Developed in **Rust** for deployment and runtime use.
- Inference-only crate (no training, no checkpoint export).
- Includes a Rust rewrite of:
  - HCQT feature extraction
  - the Python preprocessing path needed before model inference (audio load, resample, normalize, HCQT-to-model-input conversion)
- No Python dependency at runtime.

## Runtime Versions

- **Batch (offline)**:
  - Full-file pipeline (`audio -> Rust preprocessing/HCQT -> ONNX`).
  - Supports multi-file batch execution and dataset-level evaluation.
- **Live (streaming/realtime)**:
  - Callback-driven microphone-style ingestion at 44.1 kHz.
  - Includes `legacy` and `incremental` streaming frontend modes.

## 1:1 Python Comparison

- Includes direct Python-vs-Rust comparison harnesses on the same inputs.
- Includes stage-by-stage fixtures to compare preprocessing/HCQT outputs and localize mismatches.
- Includes annotation-level evaluation with Rust-minus-Python deltas.
- Python remains the reference implementation for equivalence validation.

## Runtime Optimization (Batch + Live)

- **Batch optimization**:
  - Reuses a single ONNX Runtime session across files.
  - End-to-end and frontend profilers identify dominant cost centers.
- **Live optimization**:
  - Incremental streaming path keeps persistent downsampling/ring-buffer state.
  - Dedicated streaming benchmark and sweep tools measure deadline misses, backlog, and latency.
  - Frozen incremental presets are available (`production-safe`, `balanced`).

## Required Model Artifact

Default ONNX path (relative to this crate):

```text
../guitar-transcription-continuous/output/export/model.onnx
```

## Key Commands (Release Only)

```bash
cargo run --release --example run_end_to_end -- --audio-path ../../../input/guitarset/audio/00_BN1-129-Eb_comp_mic.wav
cargo run --release --example run_end_to_end_batch_dump -- --help
cargo run --release --example benchmark_streaming -- --frontend-mode legacy,incremental --callback-sizes 128,256,512,1024
cargo run --release --example sweep_streaming -- --preset production-safe,balanced
../guitar-transcription-continuous/.venv311/bin/python tools/benchmark_python_vs_rust.py --full-dataset
../guitar-transcription-continuous/.venv311/bin/python tools/benchmark_against_guitarset.py --full-dataset
cargo test --release
```
