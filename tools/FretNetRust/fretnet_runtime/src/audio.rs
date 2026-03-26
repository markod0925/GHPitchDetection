use std::{io::Cursor, path::Path, sync::OnceLock};

use hound::{SampleFormat, WavReader};
use ndarray::{Array0, Array1};
use ndarray_npy::NpzReader;

use crate::error::FrontendError;

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

struct ResampyFilter {
    half_window: Vec<f64>,
    precision: usize,
}

static KAISER_BEST_FILTER: OnceLock<Result<ResampyFilter, String>> = OnceLock::new();

pub fn load_wav_mono(path: impl AsRef<Path>) -> Result<AudioBuffer, FrontendError> {
    let path = path.as_ref().to_path_buf();
    if !path.exists() {
        return Err(FrontendError::MissingAudioFile(path));
    }

    let mut reader = WavReader::open(&path).map_err(|err| FrontendError::AudioIo {
        path: path.clone(),
        message: err.to_string(),
    })?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(FrontendError::InvalidAudio(format!(
            "only mono WAV is supported for now, got {} channels",
            spec.channels
        )));
    }

    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|sample| sample.unwrap_or(0.0))
            .collect(),
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.unwrap_or(0) as f32 / i16::MAX as f32)
            .collect(),
        (SampleFormat::Int, 24) | (SampleFormat::Int, 32) => {
            let scale = ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.unwrap_or(0) as f32 / scale)
                .collect()
        }
        _ => {
            return Err(FrontendError::InvalidAudio(format!(
                "unsupported WAV format: {:?} / {} bits",
                spec.sample_format, spec.bits_per_sample
            )))
        }
    };

    Ok(AudioBuffer {
        samples,
        sample_rate: spec.sample_rate,
    })
}

pub fn rms_normalize(samples: &mut [f32]) -> Result<(), FrontendError> {
    if samples.is_empty() {
        return Err(FrontendError::InvalidAudio(
            "cannot normalize empty audio".to_owned(),
        ));
    }

    let rms = (samples
        .iter()
        .map(|sample| (*sample as f64) * (*sample as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt() as f32;

    if !rms.is_finite() || rms <= 0.0 {
        return Err(FrontendError::InvalidAudio(
            "audio RMS is zero or non-finite; cannot perform RMS normalization".to_owned(),
        ));
    }

    for sample in samples.iter_mut() {
        *sample /= rms;
    }

    Ok(())
}

pub fn resample_kaiser_best(
    samples: &[f32],
    original_sample_rate: u32,
    target_sample_rate: u32,
) -> Result<Vec<f32>, FrontendError> {
    if samples.is_empty() {
        return Err(FrontendError::InvalidAudio(
            "cannot resample empty audio".to_owned(),
        ));
    }

    if original_sample_rate == target_sample_rate {
        return Ok(samples.to_vec());
    }

    let filter = load_kaiser_best_filter()?;
    let sample_ratio = target_sample_rate as f64 / original_sample_rate as f64;
    let resampy_output_len =
        ((samples.len() as f64) * target_sample_rate as f64 / original_sample_rate as f64) as usize;

    if resampy_output_len < 1 {
        return Err(FrontendError::InvalidAudio(format!(
            "input signal length={} is too small to resample from {} to {}",
            samples.len(),
            original_sample_rate,
            target_sample_rate
        )));
    }

    let desired_len = ((samples.len() as f64) * sample_ratio).ceil() as usize;
    let mut output = vec![0.0f32; resampy_output_len];

    let mut interp_win = filter.half_window.clone();
    if sample_ratio < 1.0 {
        for value in &mut interp_win {
            *value *= sample_ratio;
        }
    }

    let mut interp_delta = Vec::with_capacity(interp_win.len());
    for values in interp_win.windows(2) {
        interp_delta.push(values[1] - values[0]);
    }
    interp_delta.push(0.0);

    let scale = sample_ratio.min(1.0);
    let num_table = filter.precision;
    let index_step = ((scale * num_table as f64) as usize).max(1);
    let time_increment = 1.0 / sample_ratio;
    let nwin = interp_win.len();
    let n_orig = samples.len();

    for (t, value) in output.iter_mut().enumerate() {
        let time_register = t as f64 * time_increment;
        let n = time_register as usize;

        let mut frac = scale * (time_register - n as f64);
        let mut index_frac = frac * num_table as f64;
        let mut offset = index_frac as usize;
        let mut eta = index_frac - offset as f64;

        let i_max = (n + 1).min((nwin.saturating_sub(offset)) / index_step);
        let mut accumulator = 0.0f64;
        for i in 0..i_max {
            let filter_index = offset + i * index_step;
            let weight = interp_win[filter_index] + eta * interp_delta[filter_index];
            accumulator += weight * samples[n - i] as f64;
        }

        frac = scale - frac;
        index_frac = frac * num_table as f64;
        offset = index_frac as usize;
        eta = index_frac - offset as f64;

        let k_max = n_orig
            .saturating_sub(n + 1)
            .min((nwin.saturating_sub(offset)) / index_step);
        for k in 0..k_max {
            let filter_index = offset + k * index_step;
            let weight = interp_win[filter_index] + eta * interp_delta[filter_index];
            accumulator += weight * samples[n + k + 1] as f64;
        }

        *value = accumulator as f32;
    }

    if output.len() < desired_len {
        output.resize(desired_len, 0.0);
    } else if output.len() > desired_len {
        output.truncate(desired_len);
    }

    Ok(output)
}

pub fn load_audio_for_frontend(
    path: impl AsRef<Path>,
    target_sample_rate: u32,
) -> Result<AudioBuffer, FrontendError> {
    let mut audio = load_wav_mono(path)?;

    if audio.sample_rate != target_sample_rate {
        audio.samples =
            resample_kaiser_best(&audio.samples, audio.sample_rate, target_sample_rate)?;
        audio.sample_rate = target_sample_rate;
    }

    rms_normalize(&mut audio.samples)?;

    Ok(audio)
}

fn load_kaiser_best_filter() -> Result<&'static ResampyFilter, FrontendError> {
    let result = KAISER_BEST_FILTER.get_or_init(|| {
        let bytes = include_bytes!("../assets/resample/kaiser_best.npz");
        let cursor = Cursor::new(&bytes[..]);
        let mut archive = NpzReader::new(cursor).map_err(|err| err.to_string())?;
        let half_window: Array1<f64> = archive
            .by_name("half_window.npy")
            .map_err(|err| err.to_string())?;
        let precision: Array0<i64> = archive
            .by_name("precision.npy")
            .map_err(|err| err.to_string())?;

        Ok(ResampyFilter {
            half_window: half_window.to_vec(),
            precision: precision.into_scalar() as usize,
        })
    });

    result.as_ref().map_err(|message| {
        FrontendError::ConfigMismatch(format!(
            "failed to load embedded kaiser_best resample filter: {message}"
        ))
    })
}
