use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use guitar_pitch::audio::load_wav_mono;
use guitar_pitch::cli_eval::{
    run_build_report, run_eval_full, run_eval_full_report, run_eval_mono, run_eval_mono_report,
    run_eval_pitch_report, run_report, run_solo_string_spectrogram_report,
};
use guitar_pitch::cli_masp::{
    run_pretrain_masp, run_validate_masp, run_validate_masp_batch, run_validate_masp_playhead,
};
use guitar_pitch::cli_optimize::{
    run_optimize_all, run_optimize_phase1, run_optimize_phase3, run_report_tuning,
};
use guitar_pitch::cli_preprocess::run_preprocess_jams;
use guitar_pitch::cli_solo_string::{
    run_eval_string_solo, run_optimize_string_solo, run_report_string_solo,
};
use guitar_pitch::cli_train::run_train_templates;
use guitar_pitch::config::load_app_config;
use guitar_pitch::infer::run_pitch_only_inference;
use guitar_pitch::optimize::{StageAObjectiveWeights, StageCObjectiveWeights};
use guitar_pitch::report::DEFAULT_DEBUG_DIR;
use guitar_pitch::solo_string::{SoloStringEvalThresholds, SoloStringObjectiveWeights};
use std::io::{self, Write};

#[derive(Debug, Parser)]
#[command(name = "guitar_pitch")]
#[command(about = "Guitar pitch + string attribution engine (Phase 3 diagnostics)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    InferPitch {
        #[arg(long)]
        config: String,
        #[arg(long)]
        audio: String,
    },
    TrainTemplates {
        #[arg(long)]
        config: String,
        #[arg(long)]
        mono_dataset: String,
        #[arg(long)]
        out: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
    },
    PreprocessJams {
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = "input/guitarset/annotation")]
        annotation_dir: String,
        #[arg(long, default_value = "input/guitarset/audio")]
        audio_dir: String,
        #[arg(long, default_value = "input/guitarset/derived/mono_annotations.jsonl")]
        out_mono: String,
        #[arg(long, default_value = "input/guitarset/derived/comp_annotations.jsonl")]
        out_comp: String,
        #[arg(long, default_value_t = 20)]
        fret_max: u8,
    },
    PretrainMasp {
        #[arg(long)]
        config: String,
        #[arg(long, default_value = "input/guitarset/derived/mono_annotations.jsonl")]
        mono_dataset: String,
        #[arg(long, default_value = "input/guitarset/derived/comp_annotations.jsonl")]
        comp_dataset: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        out_dir: String,
        #[arg(long)]
        trials: Option<usize>,
        #[arg(long, default_value_t = 0.99)]
        target_mono_recall: f32,
        #[arg(long, default_value_t = 0.70)]
        target_comp_recall: f32,
    },
    ValidateMasp {
        #[arg(long)]
        config: String,
        #[arg(long)]
        artifacts_dir: String,
        #[arg(long)]
        audio: String,
        #[arg(long)]
        start_sec: f32,
        #[arg(long)]
        end_sec: f32,
        #[arg(long, value_delimiter = ',')]
        expected_midis: Vec<u8>,
        #[arg(long)]
        out_json: Option<String>,
    },
    ValidateMaspBatch {
        #[arg(long)]
        config: String,
        #[arg(long)]
        artifacts_dir: String,
        #[arg(long)]
        requests_jsonl: String,
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        out_dir: String,
    },
    ValidateMaspPlayhead {
        #[arg(long)]
        config: String,
        #[arg(long)]
        artifacts_dir: String,
        #[arg(long)]
        audio: String,
        #[arg(long)]
        tablature: String,
        #[arg(long)]
        playhead_sec: f32,
        #[arg(long)]
        out_json: Option<String>,
    },
    EvalMono {
        #[arg(long)]
        config: String,
        #[arg(long)]
        mono_dataset: String,
        #[arg(long)]
        templates: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
    },
    EvalFull {
        #[arg(long)]
        config: String,
        #[arg(long)]
        poly_dataset: String,
        #[arg(long)]
        templates: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
    },
    EvalFullReport {
        #[arg(long)]
        config: String,
        #[arg(long)]
        poly_dataset: String,
        #[arg(long)]
        templates: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        out_dir: String,
        #[arg(long, default_value_t = false)]
        open: bool,
    },
    EvalPitchReport {
        #[arg(long)]
        config: String,
        #[arg(long)]
        mono_dataset: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        out_dir: String,
    },
    EvalMonoReport {
        #[arg(long)]
        config: String,
        #[arg(long)]
        mono_dataset: String,
        #[arg(long)]
        templates: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        out_dir: String,
    },
    SoloStringSpectrogramReport {
        #[arg(long)]
        config: String,
        #[arg(long)]
        mono_dataset: String,
        #[arg(long)]
        string: u8,
        #[arg(long)]
        midi: Option<u8>,
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        out_dir: String,
        #[arg(long)]
        max_notes: Option<usize>,
        #[arg(long, default_value_t = false)]
        open: bool,
    },
    OptimizeStringSolo {
        #[arg(long)]
        config: String,
        #[arg(long)]
        mono_dataset: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = "output/debug/phase_s1")]
        out_dir: String,
        #[arg(long)]
        trials: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, default_value_t = 0.70)]
        w_top1: f32,
        #[arg(long, default_value_t = 0.10)]
        w_top3: f32,
        #[arg(long, default_value_t = 0.20)]
        w_margin: f32,
        #[arg(long, default_value_t = 0.25)]
        w_high_confidence_wrong_rate: f32,
        #[arg(long, default_value_t = 0.40)]
        high_confidence_margin_threshold: f32,
        #[arg(long, default_value_t = 0.08)]
        low_margin_threshold: f32,
        #[arg(long, default_value_t = false)]
        open: bool,
    },
    EvalStringSolo {
        #[arg(long)]
        config: String,
        #[arg(long)]
        mono_dataset: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = "output/debug/phase_s1")]
        out_dir: String,
        #[arg(long, default_value_t = 0.40)]
        high_confidence_margin_threshold: f32,
        #[arg(long, default_value_t = 0.08)]
        low_margin_threshold: f32,
        #[arg(long, default_value_t = false)]
        open: bool,
    },
    ReportStringSolo {
        #[arg(long, default_value = "output/debug/phase_s1")]
        out_dir: String,
        #[arg(long, default_value_t = false)]
        open: bool,
    },
    BuildReport {
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        debug_dir: String,
        #[arg(long, default_value_t = false)]
        open: bool,
    },
    Report {
        #[arg(long)]
        config: String,
        #[arg(long)]
        mono_dataset: String,
        #[arg(long)]
        templates: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        out_dir: String,
        #[arg(long, default_value_t = false)]
        open: bool,
    },
    OptimizePhase1 {
        #[arg(long)]
        config: String,
        #[arg(long)]
        poly_dataset: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        out_dir: String,
        #[arg(long)]
        trials: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, default_value_t = 0.45)]
        w_precision: f32,
        #[arg(long, default_value_t = 0.35)]
        w_false_positive_rate: f32,
        #[arg(long, default_value_t = 0.15)]
        w_count_penalty: f32,
        #[arg(long, default_value_t = 2.0)]
        target_avg_predicted_pitches: f32,
        #[arg(long, default_value_t = 0.70)]
        recall_floor: f32,
        #[arg(long, default_value_t = 0.30)]
        w_recall_drop: f32,
    },
    OptimizePhase3 {
        #[arg(long)]
        config: String,
        #[arg(long)]
        poly_dataset: String,
        #[arg(long)]
        templates: String,
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        out_dir: String,
        #[arg(long)]
        trials: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        phase1_config: Option<String>,
        #[arg(long, default_value_t = 0.50)]
        w_pitch_string_f1: f32,
        #[arg(long, default_value_t = 0.25)]
        w_pitch_string_fret_f1: f32,
        #[arg(long, default_value_t = 0.15)]
        w_exact_configuration: f32,
        #[arg(long, default_value_t = 0.15)]
        w_decode_top1: f32,
        #[arg(long, default_value_t = 0.08)]
        w_decode_top3: f32,
        #[arg(long, default_value_t = 0.04)]
        w_decode_top5: f32,
        #[arg(long, default_value_t = 0.03)]
        w_pitch_f1: f32,
        #[arg(long, default_value_t = 0.25)]
        w_high_confidence_wrong_decode_penalty: f32,
    },
    OptimizeAll {
        #[arg(long)]
        config: String,
        #[arg(long)]
        mono_dataset: String,
        #[arg(long)]
        poly_dataset: String,
        #[arg(long)]
        templates_out: Option<String>,
        #[arg(long, default_value = "input")]
        dataset_root: String,
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        out_dir: String,
        #[arg(long)]
        stage_a_trials: Option<usize>,
        #[arg(long)]
        stage_c_trials: Option<usize>,
        #[arg(long)]
        stage_a_seed: Option<u64>,
        #[arg(long)]
        stage_c_seed: Option<u64>,
        #[arg(long, default_value_t = 0.45)]
        a_w_precision: f32,
        #[arg(long, default_value_t = 0.35)]
        a_w_false_positive_rate: f32,
        #[arg(long, default_value_t = 0.15)]
        a_w_count_penalty: f32,
        #[arg(long, default_value_t = 2.0)]
        a_target_avg_predicted_pitches: f32,
        #[arg(long, default_value_t = 0.70)]
        a_recall_floor: f32,
        #[arg(long, default_value_t = 0.30)]
        a_w_recall_drop: f32,
        #[arg(long, default_value_t = 0.50)]
        c_w_pitch_string_f1: f32,
        #[arg(long, default_value_t = 0.25)]
        c_w_pitch_string_fret_f1: f32,
        #[arg(long, default_value_t = 0.15)]
        c_w_exact_configuration: f32,
        #[arg(long, default_value_t = 0.15)]
        c_w_decode_top1: f32,
        #[arg(long, default_value_t = 0.08)]
        c_w_decode_top3: f32,
        #[arg(long, default_value_t = 0.04)]
        c_w_decode_top5: f32,
        #[arg(long, default_value_t = 0.03)]
        c_w_pitch_f1: f32,
        #[arg(long, default_value_t = 0.25)]
        c_w_high_confidence_wrong_decode_penalty: f32,
    },
    ReportTuning {
        #[arg(long, default_value = DEFAULT_DEBUG_DIR)]
        out_dir: String,
        #[arg(long, default_value_t = false)]
        open: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::InferPitch { config, audio } => run_infer_pitch(&config, &audio),
        Command::TrainTemplates {
            config,
            mono_dataset,
            out,
            dataset_root,
        } => run_train_templates(&config, &mono_dataset, &out, &dataset_root),
        Command::PreprocessJams {
            dataset_root,
            annotation_dir,
            audio_dir,
            out_mono,
            out_comp,
            fret_max,
        } => run_preprocess_jams(
            &dataset_root,
            &annotation_dir,
            &audio_dir,
            &out_mono,
            &out_comp,
            fret_max,
        ),
        Command::PretrainMasp {
            config,
            mono_dataset,
            comp_dataset,
            dataset_root,
            out_dir,
            trials,
            target_mono_recall,
            target_comp_recall,
        } => run_pretrain_masp(
            &config,
            &mono_dataset,
            &comp_dataset,
            &dataset_root,
            &out_dir,
            trials,
            target_mono_recall,
            target_comp_recall,
        ),
        Command::ValidateMasp {
            config,
            artifacts_dir,
            audio,
            start_sec,
            end_sec,
            expected_midis,
            out_json,
        } => run_validate_masp(
            &config,
            &artifacts_dir,
            &audio,
            start_sec,
            end_sec,
            &expected_midis,
            out_json.as_deref(),
        ),
        Command::ValidateMaspBatch {
            config,
            artifacts_dir,
            requests_jsonl,
            out_dir,
        } => run_validate_masp_batch(&config, &artifacts_dir, &requests_jsonl, &out_dir),
        Command::ValidateMaspPlayhead {
            config,
            artifacts_dir,
            audio,
            tablature,
            playhead_sec,
            out_json,
        } => run_validate_masp_playhead(
            &config,
            &artifacts_dir,
            &audio,
            &tablature,
            playhead_sec,
            out_json.as_deref(),
        ),
        Command::EvalMono {
            config,
            mono_dataset,
            templates,
            dataset_root,
        } => run_eval_mono(&config, &mono_dataset, &templates, &dataset_root),
        Command::EvalFull {
            config,
            poly_dataset,
            templates,
            dataset_root,
        } => run_eval_full(&config, &poly_dataset, &templates, &dataset_root),
        Command::EvalFullReport {
            config,
            poly_dataset,
            templates,
            dataset_root,
            out_dir,
            open,
        } => run_eval_full_report(
            &config,
            &poly_dataset,
            &templates,
            &dataset_root,
            &out_dir,
            open,
        ),
        Command::EvalPitchReport {
            config,
            mono_dataset,
            dataset_root,
            out_dir,
        } => run_eval_pitch_report(&config, &mono_dataset, &dataset_root, &out_dir),
        Command::EvalMonoReport {
            config,
            mono_dataset,
            templates,
            dataset_root,
            out_dir,
        } => run_eval_mono_report(&config, &mono_dataset, &templates, &dataset_root, &out_dir),
        Command::SoloStringSpectrogramReport {
            config,
            mono_dataset,
            string,
            midi,
            dataset_root,
            out_dir,
            max_notes,
            open,
        } => run_solo_string_spectrogram_report(
            &config,
            &mono_dataset,
            &dataset_root,
            &out_dir,
            string,
            midi,
            max_notes,
            open,
        ),
        Command::OptimizeStringSolo {
            config,
            mono_dataset,
            dataset_root,
            out_dir,
            trials,
            seed,
            w_top1,
            w_top3,
            w_margin,
            w_high_confidence_wrong_rate,
            high_confidence_margin_threshold,
            low_margin_threshold,
            open,
        } => run_optimize_string_solo(
            &config,
            &mono_dataset,
            &dataset_root,
            &out_dir,
            trials,
            seed,
            SoloStringObjectiveWeights {
                w_top1,
                w_top3,
                w_margin,
                w_high_confidence_wrong_rate,
            },
            SoloStringEvalThresholds {
                high_confidence_margin_threshold,
                low_margin_threshold,
            },
            open,
        ),
        Command::EvalStringSolo {
            config,
            mono_dataset,
            dataset_root,
            out_dir,
            high_confidence_margin_threshold,
            low_margin_threshold,
            open,
        } => run_eval_string_solo(
            &config,
            &mono_dataset,
            &dataset_root,
            &out_dir,
            SoloStringEvalThresholds {
                high_confidence_margin_threshold,
                low_margin_threshold,
            },
            open,
        ),
        Command::ReportStringSolo { out_dir, open } => run_report_string_solo(&out_dir, open),
        Command::BuildReport { debug_dir, open } => run_build_report(&debug_dir, open),
        Command::Report {
            config,
            mono_dataset,
            templates,
            dataset_root,
            out_dir,
            open,
        } => run_report(
            &config,
            &mono_dataset,
            &templates,
            &dataset_root,
            &out_dir,
            open,
        ),
        Command::OptimizePhase1 {
            config,
            poly_dataset,
            dataset_root,
            out_dir,
            trials,
            seed,
            w_precision,
            w_false_positive_rate,
            w_count_penalty,
            target_avg_predicted_pitches,
            recall_floor,
            w_recall_drop,
        } => run_optimize_phase1(
            &config,
            &poly_dataset,
            &dataset_root,
            &out_dir,
            trials,
            seed,
            StageAObjectiveWeights {
                w_precision,
                w_false_positive_rate,
                w_count_penalty,
                target_avg_predicted_pitches,
                recall_floor,
                w_recall_drop,
            },
        ),
        Command::OptimizePhase3 {
            config,
            poly_dataset,
            templates,
            dataset_root,
            out_dir,
            trials,
            seed,
            phase1_config,
            w_pitch_string_f1,
            w_pitch_string_fret_f1,
            w_exact_configuration,
            w_decode_top1,
            w_decode_top3,
            w_decode_top5,
            w_pitch_f1,
            w_high_confidence_wrong_decode_penalty,
        } => run_optimize_phase3(
            &config,
            &poly_dataset,
            &templates,
            &dataset_root,
            &out_dir,
            trials,
            seed,
            phase1_config.as_deref(),
            StageCObjectiveWeights {
                w_pitch_string_f1,
                w_pitch_string_fret_f1,
                w_exact_configuration,
                w_decode_top1,
                w_decode_top3,
                w_decode_top5,
                w_pitch_f1,
                w_high_confidence_wrong_decode_penalty,
            },
        ),
        Command::OptimizeAll {
            config,
            mono_dataset,
            poly_dataset,
            templates_out,
            dataset_root,
            out_dir,
            stage_a_trials,
            stage_c_trials,
            stage_a_seed,
            stage_c_seed,
            a_w_precision,
            a_w_false_positive_rate,
            a_w_count_penalty,
            a_target_avg_predicted_pitches,
            a_recall_floor,
            a_w_recall_drop,
            c_w_pitch_string_f1,
            c_w_pitch_string_fret_f1,
            c_w_exact_configuration,
            c_w_decode_top1,
            c_w_decode_top3,
            c_w_decode_top5,
            c_w_pitch_f1,
            c_w_high_confidence_wrong_decode_penalty,
        } => run_optimize_all(
            &config,
            &mono_dataset,
            &poly_dataset,
            &dataset_root,
            &out_dir,
            templates_out.as_deref(),
            stage_a_trials,
            stage_c_trials,
            stage_a_seed,
            stage_c_seed,
            StageAObjectiveWeights {
                w_precision: a_w_precision,
                w_false_positive_rate: a_w_false_positive_rate,
                w_count_penalty: a_w_count_penalty,
                target_avg_predicted_pitches: a_target_avg_predicted_pitches,
                recall_floor: a_recall_floor,
                w_recall_drop: a_w_recall_drop,
            },
            StageCObjectiveWeights {
                w_pitch_string_f1: c_w_pitch_string_f1,
                w_pitch_string_fret_f1: c_w_pitch_string_fret_f1,
                w_exact_configuration: c_w_exact_configuration,
                w_decode_top1: c_w_decode_top1,
                w_decode_top3: c_w_decode_top3,
                w_decode_top5: c_w_decode_top5,
                w_pitch_f1: c_w_pitch_f1,
                w_high_confidence_wrong_decode_penalty: c_w_high_confidence_wrong_decode_penalty,
            },
        ),
        Command::ReportTuning { out_dir, open } => run_report_tuning(&out_dir, open),
    }
}

fn run_infer_pitch(config_path: &str, audio_path: &str) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let audio = load_wav_mono(audio_path)?;

    if cfg.pitch.backend != "masp" && audio.sample_rate != cfg.sample_rate_target {
        // TODO(phase-1.5): add high-quality resampling when sample rates do not match.
        bail!(
            "sample rate mismatch: file={}Hz config.sample_rate_target={}Hz",
            audio.sample_rate,
            cfg.sample_rate_target
        );
    }
    if cfg.pitch.backend == "masp"
        && cfg.masp.mode == "strict"
        && audio.sample_rate > cfg.masp.strict_sample_rate
    {
        eprintln!(
            "warning: MASP strict mode will downsample {}Hz -> {}Hz",
            audio.sample_rate, cfg.masp.strict_sample_rate
        );
    }

    let frames = run_pitch_only_inference(&audio, &cfg);
    let mut stdout = io::stdout().lock();
    for frame in frames.into_iter().filter(|f| !f.notes.is_empty()) {
        let line = serde_json::to_string(&frame)?;
        if let Err(err) = writeln!(stdout, "{line}") {
            if err.kind() == io::ErrorKind::BrokenPipe {
                return Ok(());
            }
            return Err(err.into());
        }
    }

    Ok(())
}
