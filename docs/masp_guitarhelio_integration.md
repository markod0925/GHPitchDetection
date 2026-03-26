# MASP Integration Guide for GuitarHelio

## Purpose

This document describes how to integrate the MASP score-informed note/chord validator into GuitarHelio using two interchangeable backends:

1. A pure TypeScript implementation
2. A Rust implementation invoked from TypeScript

The intent is to let GuitarHelio test both paths with the same expected-note input model and compare:

- raw validation quality
- per-instance latency
- end-to-end runtime overhead
- additional TS -> Rust call overhead versus the additive slowness of the pure TS path

## Current Status

The project currently provides:

- a Rust MASP backend with pretraining, batch validation, and tablature/playhead resolution
- a TypeScript MASP core with the same scoring logic and the same tuned parameters
- a TypeScript end-to-end benchmark runner for GuitarSet
- Rust/TS consistency checks for the shared MASP math layer and tablature/playhead resolution

Important limitation:

- The pure TS implementation is not yet end-to-end bit-identical to Rust.
- The shared MASP math and playhead logic are aligned, but the TS audio frontend/STFT path is numerically different from the Rust frontend.
- In practice, the benchmarked TS metrics are close to Rust and already usable for GuitarHelio integration.

## Recommended Integration Strategy

Use one common application-side contract in GuitarHelio:

1. Resolve the active tablature event set at the current playhead
2. Convert it to `expected_midis + [start_sec, end_sec]`
3. Send that request to either:
   - the pure TS validator
   - the Rust validator
4. Compare:
   - `pass`
   - `weighted_score`
   - `execution_ms`
   - wall-clock latency seen by GuitarHelio

This keeps the UI/state-management code identical while only swapping the validation backend.

## Tuned Parameters

These are the tuned parameters currently embedded in the TypeScript MASP core and reflected in the Rust benchmark report used for comparison.

### Model Parameters

- `mode = "strict"`
- `strict_sample_rate = 22050`
- `bins_per_octave = 36`
- `max_harmonics = 8`
- `b_exponent = 0.7726795`
- `cent_tolerance = 50.0`
- `rms_window_ms = 50`
- `rms_h_relax = 0.25`
- `local_aggregation = "max"`
- `midi_min = 40`

### Validation Rule Parameters

- `score_threshold = 0.4370982`
- `score_weights.har = 0.16453995`
- `score_weights.mbw = 0.5508079`
- `score_weights.cent = 0.038484424`
- `score_weights.rms = 0.24616772`

## Shared Runtime Model

Both backends use the same conceptual runtime inputs:

- `audio`
- `playhead_sec` or explicit `start_sec` and `end_sec`
- `expected_midis`

The validator is score-informed:

- it does not perform blind transcription first
- it validates whether the expected note or chord is present in the requested interval

The current decision rule is:

1. compute per-frame MASP-derived metrics in the interval
2. keep the best frame in the interval using the Rust/TS shared interval-selection heuristic
3. compute the final score:

```text
weighted_score =
  w_har  * HAR +
  w_mbw  * MBW +
  w_cent * cent_score +
  w_rms  * rms_score
```

4. accept if:

```text
weighted_score >= score_threshold
```

Important:

- `cent_gate` and `h_gate` are computed and reported
- but the current implementation does not hard-block on them
- they affect diagnostics, not the final `pass` decision

## Data Contracts

### Tablature Note Event

This is the runtime unit used to resolve playhead state.

Rust:

```rust
pub struct TablatureNoteEvent {
    pub string: u8,
    pub fret: u8,
    pub onset_sec: f32,
    pub offset_sec: f32,
    pub midi: Option<u8>,
}
```

TypeScript runtime shape:

```ts
type TablatureNoteEvent = {
  string: number;
  fret: number;
  onsetSec: number;
  offsetSec: number;
  midi?: number | null;
};
```

Notes:

- string numbering is zero-based
- standard tuning is assumed through `OPEN_MIDI = [40, 45, 50, 55, 59, 64]`
- if `midi` is omitted, it is derived from `open_midi[string] + fret`

### Resolved Playhead Segment

Rust:

```rust
pub struct ResolvedTabSegment {
    pub playhead_sec: f32,
    pub start_sec: f32,
    pub end_sec: f32,
    pub expected_midis: Vec<u8>,
    pub notes: Vec<TablatureNoteEvent>,
}
```

TypeScript runtime shape:

```ts
type ResolvedTabSegment = {
  playheadSec: number;
  startSec: number;
  endSec: number;
  expectedMidis: number[];
  notes: TablatureNoteEvent[];
};
```

Resolution rules:

- select all notes active at `playhead`
- if multiple notes overlap on the same string, keep the most recent one
- `start_sec` is the earliest active onset
- `end_sec` is the latest active offset
- `expected_midis` is sorted and deduplicated

### Validation Request

Rust:

```rust
pub struct MaspValidationRequest {
    pub id: Option<String>,
    pub audio_path: String,
    pub start_sec: f32,
    pub end_sec: f32,
    pub expected_midis: Vec<u8>,
    pub expected_valid: Option<bool>,
}
```

Suggested TypeScript shape:

```ts
type MaspValidationRequest = {
  id?: string | null;
  audio_path: string;
  start_sec: number;
  end_sec: number;
  expected_midis: number[];
  expected_valid?: boolean | null;
};
```

### Validation Result

Rust:

```rust
pub struct MaspValidationResult {
    pub id: Option<String>,
    pub audio_path: String,
    pub expected_midis: Vec<u8>,
    pub pass: bool,
    pub reason: String,
    pub h_real: f32,
    pub h_target: f32,
    pub h_dynamic_threshold: f32,
    pub har: f32,
    pub mbw: f32,
    pub cent_error: f32,
    pub cent_tolerance: f32,
    pub noise_rms: f32,
    pub weighted_score: f32,
    pub score_threshold: f32,
    pub execution_ms: Option<f64>,
    pub expected_valid: Option<bool>,
    pub false_accept: bool,
    pub false_reject: bool,
}
```

Suggested TypeScript shape:

```ts
type MaspValidationResult = {
  id?: string | null;
  audio_path: string;
  expected_midis: number[];
  pass: boolean;
  reason: string;
  h_real: number;
  h_target: number;
  h_dynamic_threshold: number;
  har: number;
  mbw: number;
  cent_error: number;
  cent_tolerance: number;
  noise_rms: number;
  weighted_score: number;
  score_threshold: number;
  execution_ms?: number | null;
  expected_valid?: boolean | null;
  false_accept: boolean;
  false_reject: boolean;
};
```

## Rust Implementation

### Main Public Modules

- [src/masp.rs](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/src/masp.rs)
- [src/tracking.rs](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/src/tracking.rs)
- [src/cli_masp.rs](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/src/cli_masp.rs)

### Main Public Rust APIs

#### Pretraining and Artifacts

- `pretrain_from_guitarset(...) -> Result<MaspPretrainArtifacts>`
- `write_pretrain_artifacts(...)`
- `load_pretrain_artifacts(...)`

Use these when the Rust process should own the pretrained signatures and manifest.

#### Validation

- `validate_expected_segment(...) -> Result<MaspValidationResult>`
- `write_validation_artifacts(...)`
- `read_validation_requests(...)`

This is the direct score-informed entry point when GuitarHelio already knows:

- the audio buffer
- the time interval
- the expected MIDI set

#### Tablature + Playhead

- `read_tablature_events(...)`
- `resolve_tablature_playhead(...)`
- `validate_masp_with_playhead(...) -> Result<MaspPlayheadValidationResult>`

This is the most natural Rust entry point if GuitarHelio wants Rust to own the playhead resolution logic as well.

### Rust CLI Entry Points

Useful if GuitarHelio wants process-based integration first:

- `pretrain-masp`
- `validate-masp`
- `validate-masp-batch`
- `validate-masp-playhead`

Example:

```bash
cargo run --release -- validate-masp-playhead \
  --config config.toml \
  --artifacts-dir output/debug \
  --audio input/guitarset/audio/00_BN1-129-Eb_comp_mic.wav \
  --tablature input/example_tab.json \
  --playhead-sec 1.24
```

### Rust Audio Path

Rust currently includes the complete end-to-end path:

1. WAV load
2. mono conversion
3. optional downsample to `22050` in strict mode
4. RMS normalization
5. STFT or QDFT frontend
6. harmonic map construction
7. MASP pitch-spectrum scoring
8. final validation decision

Relevant modules:

- [src/audio.rs](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/src/audio.rs)
- [src/stft.rs](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/src/stft.rs)
- [src/harmonic_map.rs](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/src/harmonic_map.rs)
- [src/masp.rs](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/src/masp.rs)

### Rust Integration Options for GuitarHelio

#### Option A: Rust as a child process

Pros:

- fastest path to integration
- no FFI work required
- reuses current CLI and artifact formats

Cons:

- process startup overhead
- JSON serialization overhead
- harder to do ultra-frequent per-frame or per-playhead calls

Best for:

- benchmarking
- offline batch validation
- early product integration

#### Option B: Rust as an embedded native backend

Pros:

- lower runtime overhead than child processes
- direct function calls after bridge setup

Cons:

- requires a Node native bridge strategy
- packaging complexity is higher

Best for:

- final production integration if Rust latency is clearly better than pure TS

## TypeScript Implementation

### Main Files

- [ts/masp/maspCore.ts](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/ts/masp/maspCore.ts)
- [ts/masp/benchmarkGuitarSet.ts](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/ts/masp/benchmarkGuitarSet.ts)
- [ts/masp/checkConsistency.mjs](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/ts/masp/checkConsistency.mjs)

### Node Version

The current TS validation scripts are intended to run with Node v22.

### TS Core Scope

`maspCore.ts` currently provides:

- tuned parameters
- MASP frame scoring math
- scalar metrics:
  - `H`
  - `HAR`
  - `MBW`
  - `cent_error`
- final validation rule
- tablature normalization
- tablature/playhead resolution

It is the right file to embed directly into GuitarHelio if GuitarHelio wants the MASP logic in-process and in-language.

### Exported TS API

From [maspCore.ts](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/ts/masp/maspCore.ts):

- `OPEN_MIDI`
- `MASP_TUNED_PARAMS`
- `pitchToHz(midi)`
- `clamp(value, lo, hi)`
- `smoothingFactor(k, bExponent)`
- `centScore(centError, tolerance)`
- `aggregateLocalMasp(mag, start, end, mode = "max")`
- `scoreMaspMidiFrame(mag, maps, params = MASP_TUNED_PARAMS)`
- `computeHarmonicityH(mag, pitchSpectrum, sampleRate, nfft, midiMin = MASP_TUNED_PARAMS.midiMin)`
- `computeHar(pitchSpectrum, expectedMidis, midiMin = MASP_TUNED_PARAMS.midiMin)`
- `computeMbw(mag, expectedMidis, maps)`
- `computeCentError(pitchSpectrum, expectedMidis, midiMin = MASP_TUNED_PARAMS.midiMin)`
- `weightedMetricsScore(metrics, weights = MASP_TUNED_PARAMS.scoreWeights, centTolerance = MASP_TUNED_PARAMS.centTolerance)`
- `computeValidationDecision({ metrics, hTarget, params = MASP_TUNED_PARAMS })`
- `normalizeTablatureEvents(events)`
- `resolveTablaturePlayhead(events, playheadSec)`

### TS End-to-End Benchmark Scope

`benchmarkGuitarSet.ts` adds the missing end-to-end runtime components:

- WAV parsing
- mono reduction
- linear resampling
- RMS normalization
- Hann window generation
- FFT/STFT
- harmonic map generation
- full GuitarSet request evaluation

This file is primarily a benchmark/reference implementation, not the recommended API surface for GuitarHelio.

If GuitarHelio uses the pure TS route, the preferred architecture is:

1. keep `maspCore.ts` as the stable MASP math module
2. move the benchmark-only audio frontend code into a dedicated runtime wrapper
3. expose one high-level validator function inside GuitarHelio

Suggested wrapper signature:

```ts
type MaspPlayheadRequest = {
  audioSamples: Float32Array;
  sampleRate: number;
  tablature: TablatureNoteEvent[];
  playheadSec: number;
};

type MaspPlayheadValidationResult = {
  segment: ResolvedTabSegment;
  validation: MaspValidationResult;
};
```

## Artifact Contracts

### Pretraining Artifacts

- `masp_manifest.json`
- `masp_note_signatures.jsonl`
- `masp_joint_signatures.jsonl`

Use these if GuitarHelio wants to load pretrained signatures produced by Rust.

Manifest structure:

- `model_params`
- `validation_rule`
- legacy flat fields for compatibility

### Validation Artifacts

- `masp_validation_results.jsonl`
- `masp_validation_summary.json`

These are useful as a shared reporting format for both backends.

### TS Benchmark Artifacts

- `ts_masp_precision_recall_report.json`
- `ts_masp_precision_recall_report.md`
- `ts_masp_eval_mono/ts_masp_validation_results.jsonl`
- `ts_masp_eval_comp/ts_masp_validation_results.jsonl`

These are benchmark/report artifacts, not required at runtime in GuitarHelio.

## Integration Patterns for GuitarHelio

### Pattern 1: Pure TypeScript

Use when:

- fastest integration is needed
- keeping everything in the existing TS runtime is preferable
- the observed TS latency is already acceptable

Recommended flow:

1. resolve tablature state in GuitarHelio
2. derive `expected_midis`, `start_sec`, `end_sec`
3. run the TS audio frontend
4. call the MASP math from `maspCore.ts`
5. store `pass`, `weighted_score`, and `execution_ms`

### Pattern 2: TS UI + Rust Validator

Use when:

- Rust per-request latency is materially better after including bridge overhead
- you want Rust to own the full audio frontend

Recommended flow:

1. keep tablature and playhead state in GuitarHelio
2. send a request object to Rust:
   - either `playhead + tablature`
   - or `start_sec/end_sec + expected_midis`
3. collect `MaspValidationResult`
4. compare:
   - Rust internal `execution_ms`
   - GuitarHelio wall-clock elapsed time

### Pattern 3: A/B Backend Switch

Recommended for initial rollout.

In GuitarHelio, define one backend enum:

```ts
type MaspBackendMode = "ts" | "rust";
```

Then keep one common request shape and one common response shape.

This lets GuitarHelio compare:

- identical playhead events
- identical expected notes
- identical UI behavior
- only backend-dependent latency and score differences

## Validation Reasons

The current backends return the following main `reason` values:

- `validated`
- `validated_soft_score_without_cent_or_h_gate`
- `validated_soft_score_without_cent_gate`
- `validated_soft_score_without_h_gate`
- `cent_gate_failed`
- `harmonicity_threshold_failed`
- `weighted_score_failed`

These are useful for UI diagnostics and developer telemetry.

## Benchmark Snapshot

These benchmark numbers are currently available with the tuned parameters listed above.

### Rust Benchmark

Source:

- [masp_precision_recall_report.json](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/output/debug/masp_precision_recall_report.json)

Rust results:

- MONO
  - Precision: `0.904940`
  - Recall: `0.989740`
  - F1: `0.945442`
  - Accuracy: `0.942886`
  - execution_ms mean / median / p95 / p99 / max: `0.569758 / 0.290873 / 1.880176 / 4.808544 / 17.324330`
- COMP
  - Precision: `0.605612`
  - Recall: `0.961508`
  - F1: `0.743148`
  - Accuracy: `0.667676`
  - execution_ms mean / median / p95 / p99 / max: `1.013531 / 0.669810 / 3.018689 / 6.090968 / 20.503486`
- OVERALL
  - Precision: `0.718632`
  - Recall: `0.974728`
  - F1: `0.827315`
  - Accuracy: `0.796545`
  - execution_ms mean / median / p95 / p99 / max: `0.805731 / 0.441305 / 2.585988 / 5.504567 / 20.503486`

### TypeScript Benchmark

Source:

- [ts_masp_precision_recall_report.json](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/output/debug/ts_masp_precision_recall_report.json)

TS results:

- MONO
  - Precision: `0.908260`
  - Recall: `0.989977`
  - F1: `0.947359`
  - Accuracy: `0.944991`
  - execution_ms mean / median / p95 / p99 / max: `2.022693 / 0.528786 / 7.728453 / 20.440156 / 535.763789`
- COMP
  - Precision: `0.603748`
  - Recall: `0.962553`
  - F1: `0.742053`
  - Accuracy: `0.665405`
  - execution_ms mean / median / p95 / p99 / max: `2.897960 / 1.321504 / 10.060590 / 19.886994 / 145.872925`
- OVERALL
  - Precision: `0.718183`
  - Recall: `0.975394`
  - F1: `0.827256`
  - Accuracy: `0.796323`
  - execution_ms mean / median / p95 / p99 / max: `2.488110 / 0.905412 / 9.270727 / 20.244064 / 535.763789`

### Practical Interpretation

- Accuracy and precision/recall are close enough for integration work in GuitarHelio.
- Rust is materially faster per instance.
- TS remains fully viable if the observed wall-clock latency inside GuitarHelio is acceptable.
- The final architecture decision should be based on:
  - GuitarHelio wall-clock latency
  - bridge overhead for Rust calls
  - operational complexity

## Consistency Checks

Shared fixture validation exists for the MASP math layer and for tablature/playhead resolution.

Files:

- [ts/masp/consistencyFixtures.json](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/ts/masp/consistencyFixtures.json)
- [ts/masp/checkConsistency.mjs](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/ts/masp/checkConsistency.mjs)
- [src/masp.rs](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/src/masp.rs)
- [src/tracking.rs](/mnt/c/Dati/Marco/GameDev/GHPitchDetection/src/tracking.rs)

Run:

```bash
source "$HOME/.nvm/nvm.sh" && nvm use 22 >/dev/null && node ts/masp/checkConsistency.mjs
cargo test --release
```

What is covered:

- smoothing factor
- cent score
- MASP frame scoring
- `H`
- `HAR`
- `MBW`
- `cent_error`
- weighted decision score
- validation decision structure
- tablature/playhead resolution

What is not yet fully 1:1:

- end-to-end audio frontend numerical behavior

## Suggested GuitarHelio Measurement Plan

To decide between pure TS and Rust-backed MASP, measure three separate timings:

1. `backend execution_ms`
   - returned by the validator itself
2. `bridge overhead`
   - wall-clock in GuitarHelio minus backend `execution_ms`
3. `UI-visible latency`
   - total elapsed time between playhead trigger and result availability in the UI loop

Recommended experiment matrix:

1. pure TS in-process
2. Rust as child process
3. Rust embedded bridge, if later implemented

Log at least:

- playhead time
- note/chord cardinality
- backend mode
- `pass`
- `weighted_score`
- `execution_ms`
- total wall-clock elapsed time

## Recommended Next Step in GuitarHelio

Implement a backend adapter interface first, then plug both implementations behind it.

Suggested interface:

```ts
type MaspBackendMode = "ts" | "rust";

type MaspBackend = {
  validatePlayhead(
    audio: Float32Array,
    sampleRate: number,
    tablature: TablatureNoteEvent[],
    playheadSec: number
  ): Promise<MaspPlayheadValidationResult>;
};
```

This isolates the integration decision from the rest of GuitarHelio and makes A/B testing straightforward.
