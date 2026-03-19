use anyhow::{Context, Result};

use crate::types::AudioBuffer;

pub fn load_wav_mono(path: &str) -> Result<AudioBuffer> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("failed to open WAV file: {path}"))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));

    let mut samples = Vec::with_capacity(reader.duration() as usize);

    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut acc = 0.0_f32;
            let mut ch = 0usize;
            for sample in reader.samples::<f32>() {
                acc +=
                    sample.with_context(|| format!("failed to read float sample from: {path}"))?;
                ch += 1;
                if ch == channels {
                    samples.push(acc / channels as f32);
                    acc = 0.0;
                    ch = 0;
                }
            }
            if ch != 0 {
                samples.push(acc / ch as f32);
            }
        }
        hound::SampleFormat::Int => {
            let max_amplitude = if spec.bits_per_sample == 0 {
                1.0
            } else {
                ((1_i64 << (u32::from(spec.bits_per_sample) - 1)) - 1) as f32
            };

            let mut acc = 0.0_f32;
            let mut ch = 0usize;
            for sample in reader.samples::<i32>() {
                let v = sample.with_context(|| format!("failed to read int sample from: {path}"))?
                    as f32
                    / max_amplitude;
                acc += v;
                ch += 1;
                if ch == channels {
                    samples.push(acc / channels as f32);
                    acc = 0.0;
                    ch = 0;
                }
            }
            if ch != 0 {
                samples.push(acc / ch as f32);
            }
        }
    }

    Ok(AudioBuffer {
        sample_rate: spec.sample_rate,
        samples,
    })
}

pub fn normalize_audio_in_place(samples: &mut [f32]) {
    let mut max_abs = 0.0_f32;
    for &x in samples.iter() {
        max_abs = max_abs.max(x.abs());
    }
    if max_abs > 0.0 {
        let inv = 1.0 / max_abs;
        for x in samples.iter_mut() {
            *x *= inv;
        }
    }
}

pub fn trim_audio_region(
    samples: &[f32],
    sample_rate: u32,
    start_sec: f32,
    end_sec: f32,
) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let sr = sample_rate as f32;
    let start_idx = (start_sec.max(0.0) * sr).floor() as usize;
    let end_idx = (end_sec.max(start_sec).max(0.0) * sr).ceil() as usize;

    let start_idx = start_idx.min(samples.len());
    let end_idx = end_idx.min(samples.len());

    if end_idx <= start_idx {
        return Vec::new();
    }

    samples[start_idx..end_idx].to_vec()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{load_wav_mono, normalize_audio_in_place, trim_audio_region};

    fn temp_path(filename: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "guitar_pitch_{filename}_{}_{}.wav",
            std::process::id(),
            1
        ));
        path
    }

    #[test]
    fn normalize_scales_to_unit_peak() {
        let mut x = vec![0.5, -2.0, 1.0];
        normalize_audio_in_place(&mut x);
        let peak = x.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!((peak - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trim_returns_expected_region() {
        let samples: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let out = trim_audio_region(&samples, 10, 0.2, 0.6);
        assert_eq!(out, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn wav_loader_averages_stereo_to_mono() {
        let path = temp_path("stereo");
        {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 8_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
            let data = [1000_i16, -1000, 2000, -2000];
            for s in data {
                writer.write_sample(s).expect("write sample");
            }
            writer.finalize().expect("finalize wav");
        }

        let audio = load_wav_mono(path.to_str().expect("path utf8")).expect("load wav");
        assert_eq!(audio.sample_rate, 8_000);
        assert_eq!(audio.samples.len(), 2);
        assert!(audio.samples[0].abs() < 1e-6);
        assert!(audio.samples[1].abs() < 1e-6);

        let _ = std::fs::remove_file(path);
    }
}
