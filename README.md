# GHPitchDetection

This repository contains three pitch/fret detection approaches for guitar audio.

## 1) Native Rust Multi-Phase Pipeline (`src/`, main CLI)
- Hand-crafted implementation (no neural model), developed in 3+ phases.
- Goal: general pitch/string/fret detection.
- Status: baseline/reference pipeline with mixed-but-usable results.

## 2) MASP Validator (Rust + TypeScript)
- Rust backend: `src/masp.rs`, CLI commands (`pretrain-masp`, `validate-masp`, `validate-masp-playhead`, `validate-masp-batch`).
- TypeScript backend: `ts/masp/maspCore.ts` (+ benchmark/consistency scripts).
- Goal: score-informed validation of expected notes/chords (not open-ended transcription), intended for GuitarHelio gameplay validation.

## 3) FretNet Rust Runtime (`tools/FretNetRust/fretnet_runtime`)
- Rust runtime/front-end for exported FretNet (see https://github.com/cwitkowitz/guitar-transcription-continuous) neural model (ONNX inference path).
- Goal: general neural pitch/fret transcription path for GuitarHelio (practice-oriented usage).
- Current Rust post-processing is intentionally conservative; annotation benchmarks still evaluate the path against GuitarSet.
- Includes a streaming/realtime benchmark harness (`benchmark_streaming`) for callback/deadline/backlog/latency evaluation at microphone-like 44.1 kHz ingestion, with `legacy` vs `incremental` frontend-mode comparison; use this as the primary performance benchmark for real-time viability decisions.
- Includes an incremental streaming sweep harness (`sweep_streaming`) for systematic callback/context/inference-hop tuning with GO/BORDERLINE/NO-GO classification and ranked JSON/CSV outputs before further realtime architecture changes.
- Frozen incremental presets are now available: `production-safe = (1024, 0.5s, hops=1)` and `balanced = (1024, 0.5s, hops=2)`.

## Common Ground
- All three approaches are intended for polyphonic contexts.
- All have been validated against GuitarSet `*_mic` audio (directly or through dedicated benchmark/evaluation flows in this repo).
