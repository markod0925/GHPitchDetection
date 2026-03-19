use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use guitar_pitch::audio::load_wav_mono;
use guitar_pitch::config::load_app_config;
use guitar_pitch::infer::run_pitch_only_inference;
use std::io::{self, Write};

#[derive(Debug, Parser)]
#[command(name = "guitar_pitch")]
#[command(about = "Guitar pitch + string attribution engine (Phase 1: pitch-only)")]
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::InferPitch { config, audio } => run_infer_pitch(&config, &audio),
    }
}

fn run_infer_pitch(config_path: &str, audio_path: &str) -> Result<()> {
    let cfg = load_app_config(config_path)?;
    let audio = load_wav_mono(audio_path)?;

    if audio.sample_rate != cfg.sample_rate_target {
        // TODO(phase-1.5): add high-quality resampling when sample rates do not match.
        bail!(
            "sample rate mismatch: file={}Hz config.sample_rate_target={}Hz",
            audio.sample_rate,
            cfg.sample_rate_target
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
