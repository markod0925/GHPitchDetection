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
- Rust runtime/front-end for exported FretNet neural model (ONNX inference path).
- Goal: general neural pitch/fret transcription path for GuitarHelio (practice-oriented usage).
- Current Rust post-processing is intentionally conservative; annotation benchmarks still evaluate the path against GuitarSet.

## Common Ground
- All three approaches are intended for polyphonic contexts.
- All have been validated against GuitarSet `*_mic` audio (directly or through dedicated benchmark/evaluation flows in this repo).
