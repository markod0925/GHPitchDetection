use std::{f32::consts::PI, sync::Arc};

use ndarray::Array3;
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::{
    config::{
        FRONTEND_BINS_PER_OCTAVE, FRONTEND_FMIN_HZ, FRONTEND_HARMONICS, FRONTEND_HOP_LENGTH, FRONTEND_N_BINS,
        FRONTEND_SAMPLE_RATE, FRONTEND_SPARSITY, FRONTEND_TOP_DB,
    },
    error::FrontendError,
    types::HcqtFeatures,
};

#[derive(Debug, Clone)]
pub struct FrontendConfig {
    pub sample_rate: u32,
    pub hop_length: usize,
    pub fmin_hz: f32,
    pub harmonics: Vec<f32>,
    pub n_bins: usize,
    pub bins_per_octave: usize,
    pub top_db: f32,
    pub sparsity: f32,
}

impl Default for FrontendConfig {
    fn default() -> Self {
        Self {
            sample_rate: FRONTEND_SAMPLE_RATE,
            hop_length: FRONTEND_HOP_LENGTH,
            fmin_hz: FRONTEND_FMIN_HZ,
            harmonics: FRONTEND_HARMONICS.to_vec(),
            n_bins: FRONTEND_N_BINS,
            bins_per_octave: FRONTEND_BINS_PER_OCTAVE,
            top_db: FRONTEND_TOP_DB,
            sparsity: FRONTEND_SPARSITY,
        }
    }
}

struct HarmonicBasis {
    n_fft: usize,
    positive_bins: usize,
    lengths: Vec<f32>,
    fft_basis: Vec<Complex32>,
    fft: Arc<dyn Fft<f32>>,
}

pub struct HcqtExtractor {
    config: FrontendConfig,
    harmonic_bases: Vec<HarmonicBasis>,
}

impl HcqtExtractor {
    pub fn new(config: FrontendConfig) -> Result<Self, FrontendError> {
        if config.sample_rate == 0 {
            return Err(FrontendError::ConfigMismatch("sample_rate must be positive".to_owned()));
        }
        if config.hop_length == 0 {
            return Err(FrontendError::ConfigMismatch("hop_length must be positive".to_owned()));
        }
        if config.n_bins == 0 || config.bins_per_octave == 0 {
            return Err(FrontendError::ConfigMismatch(
                "n_bins and bins_per_octave must be positive".to_owned(),
            ));
        }
        if config.harmonics.is_empty() {
            return Err(FrontendError::ConfigMismatch("at least one harmonic is required".to_owned()));
        }

        let mut planner = FftPlanner::<f32>::new();
        let mut harmonic_bases = Vec::with_capacity(config.harmonics.len());
        for harmonic in &config.harmonics {
            let basis = build_harmonic_basis(&config, *harmonic, &mut planner)?;
            harmonic_bases.push(basis);
        }

        Ok(Self {
            config,
            harmonic_bases,
        })
    }

    pub fn config(&self) -> &FrontendConfig {
        &self.config
    }

    pub fn extract(&self, audio: &[f32], sample_rate: u32) -> Result<HcqtFeatures, FrontendError> {
        if sample_rate != self.config.sample_rate {
            return Err(FrontendError::UnsupportedSampleRate {
                expected: self.config.sample_rate,
                actual: sample_rate,
            });
        }
        if audio.is_empty() {
            return Err(FrontendError::InvalidAudio("audio is empty".to_owned()));
        }

        let frame_count = 1 + audio.len() / self.config.hop_length;
        let mut hcqt = Array3::<f32>::zeros((self.config.harmonics.len(), self.config.n_bins, frame_count));

        for (channel_index, basis) in self.harmonic_bases.iter().enumerate() {
            let magnitudes = harmonic_response(audio, self.config.hop_length, basis);
            apply_post_processing(&mut hcqt, channel_index, &magnitudes, self.config.top_db);
        }

        Ok(HcqtFeatures::new(
            hcqt,
            self.config.sample_rate,
            self.config.hop_length,
        ))
    }
}

fn build_harmonic_basis(
    config: &FrontendConfig,
    harmonic: f32,
    planner: &mut FftPlanner<f32>,
) -> Result<HarmonicBasis, FrontendError> {
    let frequencies = cqt_frequencies(config.fmin_hz * harmonic, config.n_bins, config.bins_per_octave);
    let alpha = relative_bandwidth(&frequencies)?;
    let lengths = wavelet_lengths(config.sample_rate as f32, &frequencies, &alpha);
    let max_len = lengths
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .ceil()
        .max(1.0) as usize;
    let n_fft = max_len.next_power_of_two();
    let positive_bins = n_fft / 2 + 1;
    let fft = planner.plan_fft_forward(n_fft);

    let mut fft_basis = vec![Complex32::new(0.0, 0.0); config.n_bins * positive_bins];
    for (row, (&freq, &length)) in frequencies.iter().zip(lengths.iter()).enumerate() {
        let filter = build_wavelet_filter(freq, config.sample_rate as f32, length, n_fft);
        let mut spectrum = filter;
        fft.process(&mut spectrum);

        let row_slice = &mut fft_basis[row * positive_bins..(row + 1) * positive_bins];
        let scale = length / n_fft as f32;
        for (index, value) in row_slice.iter_mut().enumerate() {
            *value = spectrum[index] * scale;
        }
        sparsify_row(row_slice, config.sparsity);
    }

    Ok(HarmonicBasis {
        n_fft,
        positive_bins,
        lengths,
        fft_basis,
        fft,
    })
}

fn cqt_frequencies(fmin_hz: f32, n_bins: usize, bins_per_octave: usize) -> Vec<f32> {
    (0..n_bins)
        .map(|index| fmin_hz * 2.0_f32.powf(index as f32 / bins_per_octave as f32))
        .collect()
}

fn relative_bandwidth(frequencies: &[f32]) -> Result<Vec<f32>, FrontendError> {
    if frequencies.len() < 2 {
        return Err(FrontendError::ConfigMismatch(
            "at least two frequencies are required to compute relative bandwidth".to_owned(),
        ));
    }

    let mut bpo = vec![0.0f32; frequencies.len()];
    let logf: Vec<f32> = frequencies.iter().map(|frequency| frequency.log2()).collect();

    bpo[0] = 1.0 / (logf[1] - logf[0]);
    let last = frequencies.len() - 1;
    bpo[last] = 1.0 / (logf[last] - logf[last - 1]);
    for index in 1..last {
        bpo[index] = 2.0 / (logf[index + 1] - logf[index - 1]);
    }

    Ok(bpo
        .into_iter()
        .map(|value| {
            let r = 2.0_f32.powf(2.0 / value);
            (r - 1.0) / (r + 1.0)
        })
        .collect())
}

fn wavelet_lengths(sample_rate: f32, frequencies: &[f32], alpha: &[f32]) -> Vec<f32> {
    frequencies
        .iter()
        .zip(alpha.iter())
        .map(|(&frequency, &alpha)| (sample_rate / frequency) * (1.0 / alpha))
        .collect()
}

fn build_wavelet_filter(frequency: f32, sample_rate: f32, length: f32, n_fft: usize) -> Vec<Complex32> {
    let start = (-length / 2.0).floor() as i32;
    let stop = (length / 2.0).floor() as i32;
    let filter_len = (stop - start).max(1) as usize;
    let mut filter = vec![Complex32::new(0.0, 0.0); filter_len];

    for (index, sample_index) in (start..stop).enumerate() {
        let phase = 2.0 * PI * frequency * sample_index as f32 / sample_rate;
        let window = hann_window(index, filter_len);
        filter[index] = Complex32::new(phase.cos(), phase.sin()) * window;
    }

    let norm = filter.iter().map(|value| value.norm()).sum::<f32>().max(f32::EPSILON);
    for value in &mut filter {
        *value /= norm;
    }

    pad_center_complex(&filter, n_fft)
}

fn hann_window(index: usize, length: usize) -> f32 {
    if length <= 1 {
        return 1.0;
    }
    0.5 - 0.5 * (2.0 * PI * index as f32 / length as f32).cos()
}

fn pad_center_complex(data: &[Complex32], size: usize) -> Vec<Complex32> {
    let mut padded = vec![Complex32::new(0.0, 0.0); size];
    let left_pad = (size - data.len()) / 2;
    padded[left_pad..left_pad + data.len()].copy_from_slice(data);
    padded
}

fn sparsify_row(row: &mut [Complex32], quantile: f32) {
    if !(0.0..1.0).contains(&quantile) {
        return;
    }

    let mut magnitudes: Vec<f32> = row.iter().map(|value| value.norm()).collect();
    let norm = magnitudes.iter().sum::<f32>();
    if norm <= f32::EPSILON {
        return;
    }

    magnitudes.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let mut cumulative = 0.0f32;
    let mut threshold = magnitudes[0];
    for magnitude in &magnitudes {
        cumulative += *magnitude / norm;
        threshold = *magnitude;
        if cumulative >= quantile {
            break;
        }
    }

    for value in row.iter_mut() {
        if value.norm() < threshold {
            *value = Complex32::new(0.0, 0.0);
        }
    }
}

fn harmonic_response(audio: &[f32], hop_length: usize, basis: &HarmonicBasis) -> Vec<f32> {
    let frame_count = 1 + audio.len() / hop_length;
    let mut padded = vec![0.0f32; audio.len() + basis.n_fft];
    let pad_length = basis.n_fft / 2;
    padded[pad_length..pad_length + audio.len()].copy_from_slice(audio);

    let mut response = vec![0.0f32; basis.lengths.len() * frame_count];
    let mut spectrum = vec![Complex32::new(0.0, 0.0); basis.n_fft];

    for frame_index in 0..frame_count {
        let start = frame_index * hop_length;
        for (offset, value) in spectrum.iter_mut().enumerate() {
            value.re = padded[start + offset];
            value.im = 0.0;
        }

        basis.fft.process(&mut spectrum);
        for row in 0..basis.lengths.len() {
            let basis_row = &basis.fft_basis[row * basis.positive_bins..(row + 1) * basis.positive_bins];
            let mut accumulator = Complex32::new(0.0, 0.0);
            for (basis_value, spectrum_value) in basis_row.iter().zip(spectrum.iter().take(basis.positive_bins)) {
                accumulator += *basis_value * *spectrum_value;
            }

            let magnitude = accumulator.norm() / basis.lengths[row].sqrt();
            response[row * frame_count + frame_index] = magnitude;
        }
    }

    response
}

fn apply_post_processing(output: &mut Array3<f32>, channel_index: usize, magnitudes: &[f32], top_db: f32) {
    let frame_count = output.shape()[2];
    let n_bins = output.shape()[1];
    let reference = magnitudes
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1.0e-5);

    let mut max_db = f32::NEG_INFINITY;
    let mut db_values = vec![0.0f32; magnitudes.len()];
    for (index, &magnitude) in magnitudes.iter().enumerate() {
        let magnitude = magnitude.max(1.0e-5);
        let db = 20.0 * magnitude.log10() - 20.0 * reference.log10();
        db_values[index] = db;
        max_db = max_db.max(db);
    }

    let floor = max_db - top_db;
    for bin in 0..n_bins {
        for frame in 0..frame_count {
            let index = bin * frame_count + frame;
            let db = db_values[index].max(floor);
            output[[channel_index, bin, frame]] = db / top_db + 1.0;
        }
    }
}
