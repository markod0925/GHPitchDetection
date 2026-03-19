````md
# SPEC.md

# Guitar Pitch + String Attribution Engine
## Technical Specification (Rust)

## 1. Scope

Implement a Rust engine for guitar audio analysis that estimates, from recorded audio:

1. active pitches
2. likely string for each pitch
3. likely fret / playable position
4. confidence scores
5. optional temporal tracking across frames/events

The system must support:

- **monophonic** and **polyphonic** guitar audio
- training from a large annotated dataset
- efficient offline optimization
- eventual reuse in a low-latency runtime

This is **not** a generic music transcription engine. It is a **guitar-structured** estimator that exploits:

- standard guitar tuning
- limited number of strings
- playable fretboard constraints
- timbral differences between strings
- pitch-conditioned template matching

The first production target is **offline training/evaluation**.  
The final architecture must remain compatible with later **real-time inference**.

---

## 2. High-level design

The system is composed of 6 major stages:

1. **Dataset I/O**
2. **Frontend DSP**
3. **Pitch candidate scoring**
4. **Pitch-conditioned feature extraction**
5. **String/fret template scoring**
6. **Global decoding + optional tracking**

Pipeline:

\[
audio \rightarrow STFT \rightarrow whitening \rightarrow pitch\ scoring \rightarrow NMS \rightarrow candidate\ note\ features \rightarrow string/fret\ scoring \rightarrow global\ decode
\]

Training pipeline:

\[
annotated\ mono/poly\ dataset \rightarrow feature\ extraction \rightarrow template\ statistics \rightarrow hyperparameter\ optimization \rightarrow saved\ config + templates
\]

---

## 3. Design principles

### 3.1 Rust-first
Implementation must be designed for Rust performance characteristics, not translated from MATLAB.

### 3.2 Streaming-oriented
Do not load the whole dataset into RAM. Process file-by-file, event-by-event.

### 3.3 Precomputation-heavy
Anything depending only on configuration should be precomputed once:
- FFT window
- harmonic bin maps
- MIDI lookup tables
- fretboard mappings

### 3.4 Hot-path simplicity
In performance-critical paths:
- prefer fixed-size arrays over heap vectors where practical
- avoid dynamic dispatch
- reuse buffers
- parallelize across files, not across bins

### 3.5 Separation of concerns
Keep training logic and runtime inference clearly separated.

---

## 4. Repository structure

Recommended layout:

```text
guitar_pitch/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── config.rs
│   ├── types.rs
│   ├── audio.rs
│   ├── dataset.rs
│   ├── stft.rs
│   ├── whitening.rs
│   ├── harmonic_map.rs
│   ├── pitch.rs
│   ├── features.rs
│   ├── fretboard.rs
│   ├── templates.rs
│   ├── decode.rs
│   ├── tracking.rs
│   ├── metrics.rs
│   ├── optimize.rs
│   ├── cli_train.rs
│   ├── cli_eval.rs
│   ├── cli_infer.rs
│   └── cli_optimize.rs
└── tests/
````

---

## 5. Dependencies

Recommended crates:

```toml
[dependencies]
anyhow = "1"
thiserror = "1"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
bincode = "1"
csv = "1"
hound = "3"
rayon = "1"
rustfft = "6"
ndarray = "0.15"
indicatif = "0.17"
```

Optional:

* `memmap2` for large auxiliary files
* `nalgebra` if matrix algebra becomes useful
* `rand` for random search

---

## 6. Tuning and notation assumptions

Standard tuning:

* 6th string: E2 = MIDI 40
* 5th string: A2 = MIDI 45
* 4th string: D3 = MIDI 50
* 3rd string: G3 = MIDI 55
* 2nd string: B3 = MIDI 59
* 1st string: E4 = MIDI 64

Internal convention:

* `string = 0` means 6th string
* `string = 5` means 1st string

Open MIDI array:

```rust
pub const OPEN_MIDI: [u8; 6] = [40, 45, 50, 55, 59, 64];
```

Playable fret range must be configurable, default:

```rust
fret_max = 20
```

---

## 7. Dataset location and annotation extraction

### 7.1 Dataset root

The dataset is expected to be stored under the repository-local `input/` directory.

Recommended layout:

```text
input/
├── guitarset/
│   ├── audio/
│   ├── annotation/
│   ├── derived/
│   ├── splits/
│   └── ...
```

At minimum, the implementation must assume that:

* raw dataset assets live under `input/`
* audio paths in annotation files may be either:

  * absolute paths, or
  * relative to `input/`

The code must expose a configurable dataset root, but the default must be:

```text
input/
```

### 7.2 GuitarSet as annotation source

The implementation must support GuitarSet as a primary source dataset. The GuitarSet repository states that it is a dataset for guitar transcription, that the dataset can be downloaded separately, and that examples/helper functions for reading annotations are provided in `visualize/plot_demo.ipynb`. The repository also explicitly warns users to check open issues for known annotation errors. ([GitHub][1])

### 7.3 Annotation format: JAMS

Annotation extraction for GuitarSet must be based on the dataset’s **JAMS** files. JAMS is a JSON-based music annotation format designed to store multiple annotation types in one file. ([GitHub][2])

For GuitarSet specifically, the annotation structure exposed by `mirdata` indicates that each excerpt has an accompanying JAMS file containing 16 annotations, including:

* 6 `pitch_contour` annotations, one per string
* 6 `midi_note` annotations, one per string
* beat and tempo annotations
* chord-related annotations. ([mirdata.readthedocs.io][3])

### 7.4 Source of truth for annotation parsing

The implementation must treat the GuitarSet JAMS files under `input/` as the source of truth for annotations.

Instructions and reference examples for extracting those annotations are specified by the GuitarSet project in its repository, including the notebook mentioned above for reading annotations. ([GitHub][1])

When implementing the parser:

1. read the `.jams` file associated with each recording
2. extract note-level annotations per string
3. convert them into the internal `MonoAnnotation` / `PolyAnnotation` structures
4. preserve:

   * onset time
   * offset time
   * MIDI pitch
   * string index
   * fret when derivable or available
5. store additional metadata useful for stratified train/validation/test splitting

### 7.5 Derived annotation outputs

The dataset preparation step should generate normalized intermediate annotation files, for example:

```text
input/
├── guitarset/
│   ├── audio/
│   ├── annotation/
│   ├── derived/
│   │   ├── mono_annotations.jsonl
│   │   ├── poly_annotations.jsonl
│   │   └── splits.json
```

Where:

* `mono_annotations.jsonl` contains note-isolated or note-event records for template training
* `poly_annotations.jsonl` contains full multi-note event annotations for polyphonic evaluation/decoding
* `splits.json` defines train/val/test partitions

### 7.6 Preprocessing rule

The JAMS parser must be implemented as a dedicated preprocessing step, not embedded in the runtime inference path. The runtime engine must consume normalized derived annotation files only.

### 7.7 Known annotation cautions

The GuitarSet repository points to known annotation issues in its issue tracker, including incorrect timings in two annotations and a duplicated note in one MIDI annotation. The preprocessing pipeline must therefore support file-level blacklisting, patching, or override rules for problematic examples. ([GitHub][1])

---

## 8. Dataset formats

### 8.1 Mono dataset annotation

Use JSONL or JSON arrays. JSONL is preferable for large datasets.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonoAnnotation {
    pub audio_path: String,
    pub sample_rate: u32,
    pub midi: u8,
    pub string: u8,
    pub fret: u8,
    pub onset_sec: f32,
    pub offset_sec: f32,
    pub player_id: Option<String>,
    pub guitar_id: Option<String>,
    pub style: Option<String>,
    pub pickup_mode: Option<String>,
}
```

### 8.2 Poly dataset annotation

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyEvent {
    pub onset_sec: f32,
    pub offset_sec: f32,
    pub midi: u8,
    pub string: Option<u8>,
    pub fret: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyAnnotation {
    pub audio_path: String,
    pub sample_rate: u32,
    pub events: Vec<PolyEvent>,
    pub player_id: Option<String>,
    pub guitar_id: Option<String>,
    pub style: Option<String>,
    pub pickup_mode: Option<String>,
}
```

### 8.3 Split files

Explicit train/val/test split files must be supported:

```json
{
  "train": ["file1", "file2"],
  "val": ["file3"],
  "test": ["file4"]
}
```

Avoid random split leakage across:

* same recording session
* same player
* same guitar
* same musical phrase

---

## 9. Configuration model

Use TOML for configuration.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub sample_rate_target: u32,
    pub frontend: FrontendConfig,
    pub whitening: WhiteningConfig,
    pub pitch: PitchConfig,
    pub nms: NmsConfig,
    pub templates: TemplateConfig,
    pub decode: DecodeConfig,
    pub tracking: TrackingConfig,
    pub optimization: OptimizationConfig,
}
```

### 9.1 Frontend config

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontendConfig {
    pub kind: String,           // "stft"
    pub win_length: usize,
    pub hop_length: usize,
    pub nfft: usize,
    pub window: String,         // "hann"
}
```

### 9.2 Whitening config

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteningConfig {
    pub method: String,         // "moving_average"
    pub domain: String,         // "logmag"
    pub span_bins: usize,
    pub rectify_positive: bool,
    pub floor_eps: f32,
}
```

### 9.3 Pitch config

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchConfig {
    pub midi_min: u8,
    pub midi_max: u8,
    pub max_harmonics: usize,
    pub harmonic_alpha: f32,
    pub local_bandwidth_bins: usize,
    pub local_aggregation: String,   // "max" | "mean"
    pub use_subharmonic_penalty: bool,
    pub lambda_subharmonic: f32,
    pub use_missing_penalty: bool,
    pub lambda_missing: f32,
}
```

### 9.4 NMS config

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NmsConfig {
    pub threshold_mode: String,     // "relative" | "absolute"
    pub threshold_value: f32,
    pub radius_midi_steps: usize,
    pub max_polyphony: usize,
}
```

### 9.5 Template config

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConfig {
    pub level: String,              // "string" | "string_region" | "string_fret"
    pub num_harmonics: usize,
    pub include_harmonic_ratios: bool,
    pub include_centroid: bool,
    pub include_rolloff: bool,
    pub include_flux: bool,
    pub include_attack_ratio: bool,
    pub include_noise_ratio: bool,
    pub include_inharmonicity: bool,
    pub score_type: String,         // "diag_mahalanobis" | "cosine"
    pub region_bounds: Vec<u8>,     // e.g. [0,5,10,15,21]
}
```

### 9.6 Decode config

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeConfig {
    pub lambda_conflict: f32,
    pub lambda_fret_spread: f32,
    pub lambda_position_prior: f32,
    pub max_candidates_per_pitch: usize,
    pub max_global_solutions: usize,
}
```

### 9.7 Tracking config

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingConfig {
    pub mode: String,               // "none" | "event"
    pub smooth_frames: usize,
    pub lambda_switch_string: f32,
}
```

### 9.8 Optimization config

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub search_strategy: String,    // "random"
    pub trials: usize,
    pub seed: u64,
}
```

---

## 10. Core data structures

### 10.1 Audio segment

```rust
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}
```

### 10.2 STFT frame

```rust
pub struct SpectrumFrame {
    pub frame_index: usize,
    pub time_sec: f32,
    pub mag: Vec<f32>,
    pub white: Vec<f32>,
}
```

For performance-sensitive runtime, avoid storing all frames if not needed.

### 10.3 Harmonic map

```rust
#[derive(Debug, Clone)]
pub struct HarmonicBinRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct MidiHarmonicMap {
    pub midi: u8,
    pub f0_hz: f32,
    pub ranges: Vec<HarmonicBinRange>,
}
```

### 10.4 Pitch score result

```rust
pub struct PitchScoreFrame {
    pub time_sec: f32,
    pub midi_scores: Vec<f32>,
}
```

### 10.5 Note features

Use fixed-size arrays where possible.

```rust
pub const MAX_HARMONICS: usize = 8;

#[derive(Debug, Clone)]
pub struct NoteFeatures {
    pub midi: u8,
    pub f0_hz: f32,
    pub harmonic_amps: [f32; MAX_HARMONICS],
    pub harmonic_ratios: [f32; MAX_HARMONICS],
    pub centroid: f32,
    pub rolloff: f32,
    pub flux: f32,
    pub attack_ratio: f32,
    pub noise_ratio: f32,
    pub inharmonicity: f32,
}
```

### 10.6 Candidate position

```rust
#[derive(Debug, Clone)]
pub struct CandidatePosition {
    pub midi: u8,
    pub string: u8,
    pub fret: u8,
    pub local_score: f32,
}
```

### 10.7 Decoded solution

```rust
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub time_sec: f32,
    pub positions: Vec<CandidatePosition>,
    pub total_score: f32,
}
```

---

## 11. Fretboard mapping

Implement a dedicated fretboard module.

### Required functions

```rust
pub fn midi_to_fret_positions(midi: u8, open_midi: &[u8; 6], fret_max: u8) -> Vec<(u8, u8)>;
pub fn pitch_to_hz(midi: u8) -> f32;
pub fn hz_to_midi_float(freq_hz: f32) -> f32;
pub fn fret_region(fret: u8, region_bounds: &[u8]) -> usize;
```

### Behavior

`midi_to_fret_positions` must return all playable `(string, fret)` pairs such that:

[
fret = midi - openMidi[string]
]

and:

[
0 \le fret \le fret_{max}
]

---

## 12. Audio loading

Implement WAV loading first. Later extension to other formats can be added separately.

### Required functions

```rust
pub fn load_wav_mono(path: &str) -> anyhow::Result<AudioBuffer>;
pub fn normalize_audio_in_place(samples: &mut [f32]);
pub fn trim_audio_region(samples: &[f32], sample_rate: u32, start_sec: f32, end_sec: f32) -> Vec<f32>;
```

### Notes

* Convert all WAV formats to `f32`
* If stereo, average channels for first version
* Optional later: choose left/right/channel strategy explicitly

---

## 13. STFT frontend

### 13.1 Rationale

Use STFT for first implementation because it is:

* easy to optimize
* easy to precompute
* easier than CQT in Rust
* sufficient for harmonic template matching

### 13.2 Required capabilities

* Hann window
* configurable window/hop/nfft
* one-sided magnitude spectrum
* reusable FFT buffers

### Required structs

```rust
pub struct StftPlan {
    pub win_length: usize,
    pub hop_length: usize,
    pub nfft: usize,
    pub window: Vec<f32>,
}
```

### Required functions

```rust
pub fn build_stft_plan(cfg: &FrontendConfig) -> StftPlan;
pub fn compute_stft_frames(samples: &[f32], sample_rate: u32, plan: &StftPlan) -> Vec<SpectrumFrame>;
```

### Performance note

Later add a frame iterator version to avoid collecting all frames in memory.

---

## 14. Whitening

### 14.1 Definition

For each frequency bin:

[
L[k] = \log(M[k] + \epsilon)
]

Estimate local envelope:

[
E[k] = movingAverage(L[k])
]

Whitened spectrum:

[
W[k] = \max(L[k] - E[k], 0)
]

### Required functions

```rust
pub fn whiten_log_spectrum(
    mag: &[f32],
    cfg: &WhiteningConfig,
    tmp_log: &mut [f32],
    envelope: &mut [f32],
    white: &mut [f32],
);
```

### Behavior

* no allocation inside the hot path
* moving average implemented with cumulative sum or rolling window
* `rectify_positive = true` means clamp negative values to zero

---

## 15. Harmonic map precomputation

This is essential.

### Goal

Precompute, for each MIDI candidate and harmonic index, which STFT bins to inspect.

### Required function

```rust
pub fn build_harmonic_maps(
    sample_rate: u32,
    nfft: usize,
    midi_min: u8,
    midi_max: u8,
    max_harmonics: usize,
    local_bandwidth_bins: usize,
) -> Vec<MidiHarmonicMap>;
```

### Behavior

For each MIDI note:

* compute `f0_hz`
* for each harmonic:

  * compute central bin
  * build `[start, end]` bin range
* discard harmonics above Nyquist

---

## 16. Pitch scoring

### 16.1 Score model

For each MIDI note (m):

[
Score(m) = \sum_{h=1}^{H} w_h \cdot A_h
]

where:

* (A_h) is local energy near harmonic (h f_0)
* (w_h = 1/h^\alpha)

Optional penalties:

* subharmonic penalty
* missing harmonic penalty

### Required functions

```rust
pub fn harmonic_weights(max_harmonics: usize, alpha: f32) -> Vec<f32>;

pub fn score_pitch_frame(
    white: &[f32],
    maps: &[MidiHarmonicMap],
    cfg: &PitchConfig,
    harmonic_weights: &[f32],
) -> PitchScoreFrame;
```

### Local aggregation

Support:

* `max`
* `mean`

`max` is the first recommended implementation.

---

## 17. Peak picking and NMS

### 17.1 Purpose

Convert raw MIDI score vector into a small set of pitch hypotheses.

### Steps

1. threshold scores
2. keep local maxima
3. sort descending
4. suppress neighbors within configured radius
5. keep up to `max_polyphony`

### Required functions

```rust
pub fn select_pitch_candidates(
    midi_scores: &[f32],
    midi_min: u8,
    cfg: &NmsConfig,
) -> Vec<(u8, f32)>;
```

Return:

* MIDI note
* pitch score

---

## 18. Feature extraction

### 18.1 Purpose

For each pitch candidate, extract a feature vector used for string/fret discrimination.

### 18.2 Recommended features for version 1

* `harmonic_amps[0..H]`
* `harmonic_ratios[0..H]`
* spectral centroid in local pitch neighborhood
* spectral rolloff in local pitch neighborhood
* spectral flux
* attack ratio
* noise ratio
* inharmonicity estimate

### 18.3 Required functions

```rust
pub fn extract_note_features(
    current_white: &[f32],
    previous_white: Option<&[f32]>,
    midi: u8,
    map: &MidiHarmonicMap,
    sample_rate: u32,
    nfft: usize,
    cfg: &TemplateConfig,
) -> NoteFeatures;
```

### 18.4 Definitions

#### Harmonic amplitudes

Local whitened energy around each harmonic.

#### Harmonic ratios

[
r_h = \frac{a_h}{a_1 + \epsilon}
]

#### Centroid

Computed within a frequency region around the candidate note harmonics.

#### Flux

Difference between current and previous local spectrum.

#### Attack ratio

Energy ratio between an onset-proximal frame and a later frame.
For first version this may be approximated or deferred to event mode.

#### Noise ratio

Energy outside harmonic neighborhoods divided by energy inside harmonic neighborhoods.

#### Inharmonicity

Estimate deviation of observed local peaks from ideal harmonic positions.

---

## 19. Template training

### 19.1 Levels

Support three levels:

1. `string`
2. `string_region`
3. `string_fret`

The first recommended trainable production level is `string_region`.

### 19.2 Statistics

For each class, store:

* count
* mean feature vector
* standard deviation feature vector

### Required structs

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateStats {
    pub count: u32,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuitarTemplates {
    pub open_midi: [u8; 6],
    pub fret_max: u8,
    pub feature_names: Vec<String>,
    pub by_string: Vec<TemplateStats>,
    pub by_string_region: Vec<Vec<Option<TemplateStats>>>,
    pub by_string_fret: Vec<Vec<Option<TemplateStats>>>,
}
```

### 19.3 Online accumulation

Use Welford-style running statistics.

```rust
pub struct RunningStats {
    pub count: u64,
    pub mean: Vec<f64>,
    pub m2: Vec<f64>,
}
```

Required functions:

```rust
pub fn running_stats_new(dim: usize) -> RunningStats;
pub fn running_stats_update(stats: &mut RunningStats, x: &[f32]);
pub fn running_stats_finalize(stats: RunningStats) -> TemplateStats;
```

### 19.4 Training entry point

```rust
pub fn train_templates_from_mono_dataset(
    dataset: &[MonoAnnotation],
    cfg: &AppConfig,
) -> anyhow::Result<GuitarTemplates>;
```

---

## 20. Feature vector flattening

The template scorer should operate on flat feature vectors.

### Required function

```rust
pub fn flatten_note_features(features: &NoteFeatures, cfg: &TemplateConfig) -> Vec<f32>;
```

### Order example

If all features enabled:

```text
harmonic_amps[0..H]
harmonic_ratios[0..H]
centroid
rolloff
flux
attack_ratio
noise_ratio
inharmonicity
```

This order must be fixed and stored in `feature_names`.

---

## 21. Template scoring

### 21.1 Supported scorers

#### Diagonal Mahalanobis

Recommended default.

[
S(z) = - \sum_i \frac{(z_i-\mu_i)^2}{\sigma_i^2+\epsilon}
]

#### Cosine similarity

Optional baseline.

### Required functions

```rust
pub fn score_diag_mahalanobis(z: &[f32], mean: &[f32], std: &[f32], eps: f32) -> f32;
pub fn score_cosine(z: &[f32], mean: &[f32], eps: f32) -> f32;
```

### 21.2 Candidate scoring

Given a detected pitch, enumerate playable `(string, fret)` positions and score each.

```rust
pub fn score_string_fret_candidates(
    midi: u8,
    z: &[f32],
    templates: &GuitarTemplates,
    cfg: &AppConfig,
) -> Vec<CandidatePosition>;
```

Behavior:

* generate positions from fretboard mapping
* select correct template level
* score each candidate
* return sorted descending by `local_score`

---

## 22. Global decoding

### 22.1 Problem

Multiple notes detected in the same frame may be assigned to the same string if scored independently. This is physically impossible.

### 22.2 Objective

Choose one candidate per detected pitch maximizing:

[
J = \sum_i s_i - \lambda_c C - \lambda_s S - \lambda_p P
]

where:

* (s_i): local template score
* (C): string conflict penalty
* (S): fret spread penalty
* (P): position prior penalty

### 22.3 Penalties

#### String conflict

Strong penalty or hard rejection if two notes use same string.

#### Fret spread

Penalty proportional to range of active frets.

#### Position prior

Optional penalty for unlikely high frets or awkward shapes.

### 22.4 Required function

```rust
pub fn decode_global_configuration(
    per_pitch_candidates: &[Vec<CandidatePosition>],
    cfg: &DecodeConfig,
) -> Option<DecodedFrame>;
```

### 22.5 Search strategy

For version 1:

* brute-force combinations after pruning top-K per pitch

This is feasible because:

* max polyphony is small
* candidate positions per pitch are small
* top-K pruning reduces combinatorics sharply

---

## 23. Tracking

### 23.1 First implementation

Use event-based or local smoothing, not full HMM.

### 23.2 Event mode

* decide string/fret mainly near onset
* keep assignment stable through sustain unless score collapses

### Required function

```rust
pub fn smooth_decoded_sequence(
    frames: &[DecodedFrame],
    cfg: &TrackingConfig,
) -> Vec<DecodedFrame>;
```

---

## 24. Metrics

### 24.1 Pitch metrics

For mono and poly:

* precision
* recall
* F1
* framewise pitch F1
* onset-aware pitch F1

### 24.2 String metrics

Conditioned on correct pitch:

* top-1 string accuracy
* top-3 string accuracy
* exact string+fret accuracy

### 24.3 End-to-end metrics

* exact frame configuration accuracy
* exact event configuration accuracy

### Required structs

```rust
pub struct PitchMetrics {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

pub struct StringMetrics {
    pub top1: f32,
    pub top3: f32,
    pub exact_string_fret: f32,
}
```

### Required functions

```rust
pub fn evaluate_pitch_predictions(... ) -> PitchMetrics;
pub fn evaluate_string_predictions(... ) -> StringMetrics;
```

Implementation details can remain flexible, but the metrics module must be separate.

---

## 25. Optimization strategy

### 25.1 Search strategy

Use staged random search.

Do **not** optimize everything jointly at first.

### 25.2 Stage A: pitch-only optimization

Optimize:

* window length
* hop length
* nfft
* whitening span
* max harmonics
* harmonic alpha
* local bandwidth bins
* NMS threshold
* NMS radius

Objective:

* maximize pitch F1 on validation data

### 25.3 Stage B: template optimization

Keep best pitch config fixed, then optimize:

* template level
* enabled features
* scorer type
* region boundaries

Objective:

* maximize string top-1 or top-3 on mono validation set

### 25.4 Stage C: decode optimization

Optimize:

* lambda conflict
* lambda fret spread
* lambda position prior
* max candidates per pitch

Objective:

* maximize end-to-end string+pitch accuracy on poly validation set

---

## 26. Optimization API

```rust
pub struct TrialResult {
    pub config: AppConfig,
    pub pitch_metrics: Option<PitchMetrics>,
    pub string_metrics: Option<StringMetrics>,
    pub objective: f32,
}

pub fn run_random_search(
    train_split: &str,
    val_split: &str,
    base_cfg: &AppConfig,
) -> anyhow::Result<Vec<TrialResult>>;
```

---

## 27. Serialization

### 27.1 Config

Use TOML or JSON.

### 27.2 Templates

Use `bincode`.

Required functions:

```rust
pub fn save_templates(path: &str, templates: &GuitarTemplates) -> anyhow::Result<()>;
pub fn load_templates(path: &str) -> anyhow::Result<GuitarTemplates>;
```

---

## 28. Command-line interface

Implement these subcommands.

### 28.1 Train templates

```bash
guitar_pitch train-templates \
  --config config.toml \
  --mono-dataset input/guitarset/derived/mono_annotations.jsonl \
  --out templates.bin
```

### 28.2 Evaluate pitch

```bash
guitar_pitch eval-pitch \
  --config config.toml \
  --dataset input/guitarset/derived/poly_annotations.jsonl
```

### 28.3 Evaluate full system

```bash
guitar_pitch eval-full \
  --config config.toml \
  --templates templates.bin \
  --dataset input/guitarset/derived/poly_annotations.jsonl
```

### 28.4 Optimize

```bash
guitar_pitch optimize \
  --config config.toml \
  --mono-train input/guitarset/derived/mono_train.jsonl \
  --mono-val input/guitarset/derived/mono_val.jsonl \
  --poly-train input/guitarset/derived/poly_train.jsonl \
  --poly-val input/guitarset/derived/poly_val.jsonl \
  --out best_trials.json
```

### 28.5 Infer on one file

```bash
guitar_pitch infer \
  --config config.toml \
  --templates templates.bin \
  --audio input.wav
```

---

## 29. Performance requirements

### 29.1 Mandatory

* no repeated allocation in per-frame DSP hot path
* use `rayon` for file-level parallelism
* reuse FFT plans and temporary buffers
* precompute harmonic maps
* avoid storing unnecessary intermediate data

### 29.2 Recommended

* per-worker scratch buffers
* frame iterator mode for large files
* prune candidate positions early
* top-K restriction before global decoding

---

## 30. Minimum viable implementation order

### Phase 1

Pitch-only engine:

* WAV loader
* STFT
* whitening
* harmonic maps
* pitch score
* NMS
* pitch evaluation

### Phase 2

Mono template training:

* note feature extraction
* running statistics
* string / string-region templates
* string classification metrics

### Phase 3

Poly decoding:

* candidate position scoring
* top-K pruning
* global decode
* full evaluation

### Phase 4

Tracking:

* event-based smoothing
* stability metrics

---

## 31. Default initial values

Use these as starting defaults.

```toml
sample_rate_target = 44100

[frontend]
kind = "stft"
win_length = 4096
hop_length = 512
nfft = 4096
window = "hann"

[whitening]
method = "moving_average"
domain = "logmag"
span_bins = 31
rectify_positive = true
floor_eps = 1e-8

[pitch]
midi_min = 40
midi_max = 88
max_harmonics = 8
harmonic_alpha = 0.8
local_bandwidth_bins = 2
local_aggregation = "max"
use_subharmonic_penalty = true
lambda_subharmonic = 0.15
use_missing_penalty = false
lambda_missing = 0.0

[nms]
threshold_mode = "relative"
threshold_value = 0.30
radius_midi_steps = 1
max_polyphony = 6

[templates]
level = "string_region"
num_harmonics = 8
include_harmonic_ratios = true
include_centroid = true
include_rolloff = true
include_flux = true
include_attack_ratio = false
include_noise_ratio = true
include_inharmonicity = true
score_type = "diag_mahalanobis"
region_bounds = [0, 5, 10, 15, 21]

[decode]
lambda_conflict = 1000.0
lambda_fret_spread = 0.1
lambda_position_prior = 0.02
max_candidates_per_pitch = 3
max_global_solutions = 256

[tracking]
mode = "event"
smooth_frames = 3
lambda_switch_string = 0.2

[optimization]
search_strategy = "random"
trials = 100
seed = 12345
```

---

## 32. Non-goals for version 1

Do not implement yet:

* neural network models
* end-to-end differentiable training
* full HMM/Viterbi sequence decoding
* CQT frontend
* support for non-standard tunings
* distortion-specific adaptation
* multi-instrument mixed-source separation

These can come later.

---

## 33. Expected outputs

### 33.1 Inference output

For each detected event/frame:

```json
{
  "time_sec": 1.248,
  "notes": [
    {
      "midi": 64,
      "string": 5,
      "fret": 0,
      "score": 2.14
    },
    {
      "midi": 59,
      "string": 4,
      "fret": 0,
      "score": 1.77
    }
  ]
}
```

### 33.2 Evaluation output

Report:

* pitch metrics
* string metrics
* confusion matrix by string
* top failure cases

---

## 34. Implementation guidance for Codex

Codex should implement the project in this order:

1. config loading
2. WAV audio loading
3. STFT engine with reusable plan
4. whitening
5. harmonic map precomputation
6. pitch scoring
7. NMS selection
8. GuitarSet JAMS preprocessing into normalized derived JSONL annotations under `input/guitarset/derived/` ([GitHub][1])
9. mono note feature extraction
10. running statistics and template training
11. template serialization
12. string/fret candidate scoring
13. global decode
14. evaluation metrics
15. CLI commands
16. staged random search

The implementation must prioritize:

* correctness
* clean modular structure
* no unnecessary allocations in hot loops
* clear separation between offline training and inference

---

## 35. Final engineering note

The key architectural idea is this:

* **pitch detection** should be treated as a robust harmonic salience problem
* **string attribution** should be treated as a pitch-conditioned timbre matching problem
* **polyphonic decoding** should be treated as a constrained fretboard assignment problem

Those are three distinct inference layers. The codebase should preserve that separation explicitly.

```
::contentReference[oaicite:6]{index=6}
```

[1]: https://github.com/marl/GuitarSet "GitHub - marl/GuitarSet: GuitarSet: a dataset for guitar transcription · GitHub"
[2]: https://github.com/marl/jams?utm_source=chatgpt.com "marl/jams: A JSON Annotated Music Specification ..."
[3]: https://mirdata.readthedocs.io/en/0.3.8/_modules/mirdata/datasets/guitarset.html?utm_source=chatgpt.com "Source code for mirdata.datasets.guitarset"
