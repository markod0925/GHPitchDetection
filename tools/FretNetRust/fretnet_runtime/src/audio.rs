use std::path::Path;

use hound::{SampleFormat, WavReader};

use crate::error::FrontendError;

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

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
        return Err(FrontendError::InvalidAudio("cannot normalize empty audio".to_owned()));
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

pub fn resample_linear(samples: &[f32], original_sample_rate: u32, target_sample_rate: u32) -> Result<Vec<f32>, FrontendError> {
    if samples.is_empty() {
        return Err(FrontendError::InvalidAudio("cannot resample empty audio".to_owned()));
    }

    if original_sample_rate == target_sample_rate {
        return Ok(samples.to_vec());
    }

    let ratio = target_sample_rate as f64 / original_sample_rate as f64;
    let output_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    let mut output = vec![0.0f32; output_len];

    for (index, value) in output.iter_mut().enumerate() {
        let position = (index as f64) / ratio;
        let left = position.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let frac = (position - left as f64) as f32;
        *value = samples[left] * (1.0 - frac) + samples[right] * frac;
    }

    Ok(output)
}

pub fn load_audio_for_frontend(
    path: impl AsRef<Path>,
    target_sample_rate: u32,
) -> Result<AudioBuffer, FrontendError> {
    let mut audio = load_wav_mono(path)?;
    rms_normalize(&mut audio.samples)?;

    if audio.sample_rate != target_sample_rate {
        audio.samples = resample_linear(&audio.samples, audio.sample_rate, target_sample_rate)?;
        audio.sample_rate = target_sample_rate;
    }

    Ok(audio)
}
