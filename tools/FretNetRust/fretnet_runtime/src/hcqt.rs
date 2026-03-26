use std::{f32::consts::PI, mem::size_of, ops::Range, sync::Arc, time::Instant};

use ndarray::{Array2, Array3};
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::{
    audio::downsample_kaiser_best_by_2_profiled,
    config::{
        FRONTEND_BINS_PER_OCTAVE, FRONTEND_FMIN_HZ, FRONTEND_HARMONICS, FRONTEND_HOP_LENGTH,
        FRONTEND_N_BINS, FRONTEND_SAMPLE_RATE, FRONTEND_SPARSITY, FRONTEND_TOP_DB,
    },
    error::FrontendError,
    profile::{
        FrontendInitProfile, FrontendRunProfile, HarmonicInitProfile, HarmonicRunProfile,
        OctaveInitProfile, OctavePyramidProfile, OctaveRunProfile,
    },
    types::{FrontendStageOutputs, HarmonicStageOutput, HcqtFeatures},
};

const OCTAVE_DOWNSAMPLE_ORIG_SR: u32 = 2;
const OCTAVE_DOWNSAMPLE_TARGET_SR: u32 = 1;

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

struct OctaveBasis {
    n_fft: usize,
    positive_bins: usize,
    row_count: usize,
    storage: BasisStorage,
    fft: Arc<dyn Fft<f32>>,
}

enum BasisStorage {
    Dense {
        fft_basis: Vec<Complex32>,
    },
    Sparse {
        row_offsets: Vec<usize>,
        indices: Vec<usize>,
        values: Vec<Complex32>,
    },
}

struct HarmonicBasis {
    lengths: Vec<f32>,
    octave_slices: Vec<Range<usize>>,
    octaves: Vec<OctaveBasis>,
}

struct OctaveAudioLevel {
    samples: Vec<f32>,
    hop_length: usize,
}

struct OctaveAudioPyramid {
    levels: Vec<OctaveAudioLevel>,
    profile: Vec<OctavePyramidProfile>,
    allocation_bytes: usize,
    total_seconds: f64,
}

pub struct HcqtExtractor {
    config: FrontendConfig,
    harmonic_bases: Vec<HarmonicBasis>,
}

impl HcqtExtractor {
    pub fn new(config: FrontendConfig) -> Result<Self, FrontendError> {
        Ok(Self::new_internal(config)?.0)
    }

    pub fn new_profiled(
        config: FrontendConfig,
    ) -> Result<(Self, FrontendInitProfile), FrontendError> {
        Self::new_internal(config)
    }

    fn new_internal(config: FrontendConfig) -> Result<(Self, FrontendInitProfile), FrontendError> {
        let init_start = Instant::now();
        if config.sample_rate == 0 {
            return Err(FrontendError::ConfigMismatch(
                "sample_rate must be positive".to_owned(),
            ));
        }
        if config.hop_length == 0 {
            return Err(FrontendError::ConfigMismatch(
                "hop_length must be positive".to_owned(),
            ));
        }
        if config.n_bins == 0 || config.bins_per_octave == 0 {
            return Err(FrontendError::ConfigMismatch(
                "n_bins and bins_per_octave must be positive".to_owned(),
            ));
        }
        if config.harmonics.is_empty() {
            return Err(FrontendError::ConfigMismatch(
                "at least one harmonic is required".to_owned(),
            ));
        }

        let mut planner = FftPlanner::<f32>::new();
        let mut harmonic_bases = Vec::with_capacity(config.harmonics.len());
        let mut harmonic_profiles = Vec::with_capacity(config.harmonics.len());
        for harmonic in &config.harmonics {
            let (basis, profile) = build_harmonic_basis(&config, *harmonic, &mut planner)?;
            harmonic_bases.push(basis);
            harmonic_profiles.push(profile);
        }

        Ok((
            Self {
                config,
                harmonic_bases,
            },
            FrontendInitProfile {
                total_seconds: elapsed_seconds(init_start),
                harmonic_profiles,
            },
        ))
    }

    pub fn config(&self) -> &FrontendConfig {
        &self.config
    }

    pub fn extract(&self, audio: &[f32], sample_rate: u32) -> Result<HcqtFeatures, FrontendError> {
        Ok(self.extract_stages(audio, sample_rate)?.hcqt)
    }

    pub fn extract_profiled(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<(HcqtFeatures, FrontendRunProfile), FrontendError> {
        let (stages, profile) = self.extract_stages_profiled(audio, sample_rate)?;
        Ok((stages.hcqt, profile))
    }

    pub fn extract_stages(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<FrontendStageOutputs, FrontendError> {
        Ok(self.extract_stages_profiled(audio, sample_rate)?.0)
    }

    pub fn extract_stages_profiled(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<(FrontendStageOutputs, FrontendRunProfile), FrontendError> {
        let total_start = Instant::now();
        if sample_rate != self.config.sample_rate {
            return Err(FrontendError::UnsupportedSampleRate {
                expected: self.config.sample_rate,
                actual: sample_rate,
            });
        }
        if audio.is_empty() {
            return Err(FrontendError::InvalidAudio("audio is empty".to_owned()));
        }

        let octave_pyramid =
            build_octave_audio_pyramid(audio, self.config.hop_length, self.max_octave_count())?;
        let mut harmonics = Vec::with_capacity(self.harmonic_bases.len());
        let mut harmonic_profiles = Vec::with_capacity(self.harmonic_bases.len());

        for (harmonic, basis) in self
            .config
            .harmonics
            .iter()
            .copied()
            .zip(self.harmonic_bases.iter())
        {
            let (magnitude, mut profile) = harmonic_response_recursive(&octave_pyramid, basis)?;
            let frame_count = magnitude.shape()[1];
            let flatten_start = Instant::now();
            let magnitudes = array2_to_vec(&magnitude);
            profile.flatten_seconds = elapsed_seconds(flatten_start);
            profile.temporary_allocation_bytes += magnitudes.len() * size_of::<f32>();

            let post_start = Instant::now();
            let post_processed = post_process_magnitudes(
                &magnitudes,
                self.config.n_bins,
                frame_count,
                self.config.top_db,
            )?;
            profile.post_process_seconds = elapsed_seconds(post_start);
            profile.temporary_allocation_bytes +=
                2 * self.config.n_bins * frame_count * size_of::<f32>();
            profile.harmonic = harmonic;
            profile.total_seconds = profile.audio_clone_seconds
                + profile
                    .octave_profiles
                    .iter()
                    .map(|octave| octave.total_seconds)
                    .sum::<f64>()
                + profile.trim_stack_seconds
                + profile.normalization_seconds
                + profile.flatten_seconds
                + profile.post_process_seconds;

            harmonics.push(HarmonicStageOutput {
                harmonic,
                magnitude,
                post_processed,
            });
            harmonic_profiles.push(profile);
        }

        let frame_count = harmonics
            .first()
            .map(|harmonic| harmonic.post_processed.shape()[1])
            .unwrap_or(0);
        let assembly_start = Instant::now();
        let mut hcqt =
            Array3::<f32>::zeros((self.config.harmonics.len(), self.config.n_bins, frame_count));
        for (channel_index, harmonic) in harmonics.iter().enumerate() {
            for bin in 0..self.config.n_bins {
                for frame in 0..frame_count {
                    hcqt[[channel_index, bin, frame]] = harmonic.post_processed[[bin, frame]];
                }
            }
        }

        let hcqt_shape = [self.config.harmonics.len(), self.config.n_bins, frame_count];
        let profile = FrontendRunProfile {
            sample_rate: self.config.sample_rate,
            input_samples: audio.len(),
            total_seconds: elapsed_seconds(total_start),
            octave_pyramid_seconds: octave_pyramid.total_seconds,
            octave_pyramid_allocation_bytes: octave_pyramid.allocation_bytes,
            octave_pyramid_profiles: octave_pyramid.profile,
            harmonic_profiles,
            hcqt_assembly_seconds: elapsed_seconds(assembly_start),
            hcqt_assembly_allocation_bytes: self.config.harmonics.len()
                * self.config.n_bins
                * frame_count
                * size_of::<f32>(),
            hcqt_shape,
            batch_conversion_seconds: 0.0,
            batch_conversion_allocation_bytes: 0,
            batch_shape: [0; 5],
        };

        Ok((
            FrontendStageOutputs {
                hcqt: HcqtFeatures::new(hcqt, self.config.sample_rate, self.config.hop_length),
                harmonics,
            },
            profile,
        ))
    }

    fn max_octave_count(&self) -> usize {
        self.harmonic_bases
            .iter()
            .map(|basis| basis.octaves.len())
            .max()
            .unwrap_or(0)
    }
}

fn build_harmonic_basis(
    config: &FrontendConfig,
    harmonic: f32,
    planner: &mut FftPlanner<f32>,
) -> Result<(HarmonicBasis, HarmonicInitProfile), FrontendError> {
    let harmonic_start = Instant::now();
    let frequencies = cqt_frequencies(
        config.fmin_hz * harmonic,
        config.n_bins,
        config.bins_per_octave,
    );
    let alpha = relative_bandwidth(&frequencies)?;
    let lengths = wavelet_lengths(config.sample_rate as f32, &frequencies, &alpha);
    let n_octaves = ((config.n_bins as f32) / config.bins_per_octave as f32).ceil() as usize;
    let n_filters = config.bins_per_octave.min(config.n_bins);
    let mut octaves = Vec::with_capacity(n_octaves);
    let mut octave_slices = Vec::with_capacity(n_octaves);
    let mut octave_profiles = Vec::with_capacity(n_octaves);

    for octave_index in 0..n_octaves {
        let slice = if octave_index == 0 {
            config.n_bins.saturating_sub(n_filters)..config.n_bins
        } else {
            config.n_bins.saturating_sub(n_filters * (octave_index + 1))
                ..config.n_bins.saturating_sub(n_filters * octave_index)
        };

        let octave_frequencies = &frequencies[slice.clone()];
        let octave_alpha = &alpha[slice.clone()];
        let octave_sr = config.sample_rate as f32 / 2_f32.powi(octave_index as i32);
        let octave_lengths = wavelet_lengths(octave_sr, octave_frequencies, octave_alpha);
        octave_slices.push(slice);
        let (octave_basis, mut octave_profile) = build_octave_basis(
            octave_sr,
            config.hop_length >> octave_index,
            octave_frequencies,
            &octave_lengths,
            config.sparsity,
            (config.sample_rate as f32 / octave_sr).sqrt(),
            planner,
        )?;
        octave_profile.octave_index = octave_index;
        octave_profile.bin_start = octave_slices.last().map(|range| range.start).unwrap_or(0);
        octave_profile.bin_end = octave_slices.last().map(|range| range.end).unwrap_or(0);
        octaves.push(octave_basis);
        octave_profiles.push(octave_profile);
    }

    Ok((
        HarmonicBasis {
            lengths,
            octave_slices,
            octaves,
        },
        HarmonicInitProfile {
            harmonic,
            total_seconds: elapsed_seconds(harmonic_start),
            octave_profiles,
        },
    ))
}

fn build_octave_basis(
    sample_rate: f32,
    hop_length: usize,
    frequencies: &[f32],
    lengths: &[f32],
    sparsity: f32,
    octave_scale: f32,
    planner: &mut FftPlanner<f32>,
) -> Result<(OctaveBasis, OctaveInitProfile), FrontendError> {
    let octave_start = Instant::now();
    let max_len = lengths
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .ceil()
        .max(1.0) as usize;
    let mut n_fft = max_len.next_power_of_two();
    let min_n_fft = (hop_length.max(1) * 2).next_power_of_two();
    if n_fft < min_n_fft {
        n_fft = min_n_fft;
    }
    let positive_bins = n_fft / 2 + 1;
    let fft_plan_start = Instant::now();
    let fft = planner.plan_fft_forward(n_fft);
    let fft_plan_seconds = elapsed_seconds(fft_plan_start);

    let mut fft_basis = vec![Complex32::new(0.0, 0.0); frequencies.len() * positive_bins];
    let mut row_nonzero_counts = vec![0usize; frequencies.len()];
    let mut filter_build_seconds = 0.0f64;
    let mut sparsify_seconds = 0.0f64;
    for (row, (&freq, &length)) in frequencies.iter().zip(lengths.iter()).enumerate() {
        let filter_start = Instant::now();
        let filter = build_wavelet_filter(freq, sample_rate, length, n_fft);
        filter_build_seconds += elapsed_seconds(filter_start);
        let mut spectrum = filter;
        fft.process(&mut spectrum);

        let row_slice = &mut fft_basis[row * positive_bins..(row + 1) * positive_bins];
        let scale = length / n_fft as f32;
        for (index, value) in row_slice.iter_mut().enumerate() {
            *value = spectrum[index] * scale * octave_scale;
        }
        let sparsify_start = Instant::now();
        sparsify_row(row_slice, sparsity);
        sparsify_seconds += elapsed_seconds(sparsify_start);
        row_nonzero_counts[row] = row_slice.iter().filter(|value| value.norm() > 0.0).count();
    }

    let basis_nonzero_count = row_nonzero_counts.iter().sum::<usize>();
    let average_row_nonzero = if row_nonzero_counts.is_empty() {
        0.0
    } else {
        basis_nonzero_count as f64 / row_nonzero_counts.len() as f64
    };
    let max_row_nonzero = row_nonzero_counts.iter().copied().max().unwrap_or(0);
    let dense_bytes = frequencies.len() * positive_bins * size_of::<Complex32>();
    let sparse_bytes = basis_nonzero_count * (size_of::<usize>() + size_of::<Complex32>())
        + (frequencies.len() + 1) * size_of::<usize>();
    let (storage, basis_storage, fft_basis_bytes) = if sparse_bytes < dense_bytes {
        (
            pack_sparse_basis(&fft_basis, frequencies.len(), positive_bins),
            "sparse".to_owned(),
            sparse_bytes,
        )
    } else {
        (
            BasisStorage::Dense { fft_basis },
            "dense".to_owned(),
            dense_bytes,
        )
    };

    Ok((
        OctaveBasis {
            n_fft,
            positive_bins,
            row_count: lengths.len(),
            storage,
            fft,
        },
        OctaveInitProfile {
            octave_index: 0,
            sample_rate,
            hop_length,
            bin_start: 0,
            bin_end: frequencies.len(),
            row_count: lengths.len(),
            n_fft,
            positive_bins,
            fft_basis_bytes,
            basis_storage,
            basis_nonzero_count,
            average_row_nonzero,
            max_row_nonzero,
            fft_plan_seconds,
            filter_build_seconds,
            sparsify_seconds,
            total_seconds: elapsed_seconds(octave_start),
        },
    ))
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
    let logf: Vec<f32> = frequencies
        .iter()
        .map(|frequency| frequency.log2())
        .collect();

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

fn build_wavelet_filter(
    frequency: f32,
    sample_rate: f32,
    length: f32,
    n_fft: usize,
) -> Vec<Complex32> {
    let start = (-length / 2.0).floor() as i32;
    let stop = (length / 2.0).floor() as i32;
    let filter_len = (stop - start).max(1) as usize;
    let mut filter = vec![Complex32::new(0.0, 0.0); filter_len];

    for (index, sample_index) in (start..stop).enumerate() {
        let phase = 2.0 * PI * frequency * sample_index as f32 / sample_rate;
        let window = hann_window(index, filter_len);
        filter[index] = Complex32::new(phase.cos(), phase.sin()) * window;
    }

    let norm = filter
        .iter()
        .map(|value| value.norm())
        .sum::<f32>()
        .max(f32::EPSILON);
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

fn pack_sparse_basis(fft_basis: &[Complex32], row_count: usize, row_stride: usize) -> BasisStorage {
    let mut row_offsets = Vec::with_capacity(row_count + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0usize);

    for row in 0..row_count {
        let row_slice = &fft_basis[row * row_stride..(row + 1) * row_stride];
        for (index, value) in row_slice.iter().copied().enumerate() {
            if value.norm() > 0.0 {
                indices.push(index);
                values.push(value);
            }
        }
        row_offsets.push(indices.len());
    }

    BasisStorage::Sparse {
        row_offsets,
        indices,
        values,
    }
}

fn harmonic_response(
    audio: &[f32],
    hop_length: usize,
    basis: &OctaveBasis,
    octave_index: usize,
) -> (Vec<f32>, OctaveRunProfile) {
    let total_start = Instant::now();
    let frame_count = 1 + audio.len() / hop_length;
    let alloc_start = Instant::now();
    let mut padded = vec![0.0f32; audio.len() + basis.n_fft];
    let pad_length = basis.n_fft / 2;
    padded[pad_length..pad_length + audio.len()].copy_from_slice(audio);

    let mut response = vec![0.0f32; basis.row_count * frame_count];
    let mut spectrum = vec![Complex32::new(0.0, 0.0); basis.n_fft];
    let buffer_allocation_seconds = elapsed_seconds(alloc_start);
    let buffer_allocation_bytes = padded.len() * size_of::<f32>()
        + response.len() * size_of::<f32>()
        + spectrum.len() * size_of::<Complex32>();
    let mut frame_copy_seconds = 0.0f64;
    let mut fft_execution_seconds = 0.0f64;
    let mut dot_product_seconds = 0.0f64;
    let profile_frame_index = frame_count / 2;
    let mut sampled_basis_lookup_seconds = 0.0f64;
    let mut sampled_multiply_accumulate_seconds = 0.0f64;
    let mut sampled_output_write_seconds = 0.0f64;
    let basis_coefficients_total_per_frame = basis.row_count * basis.positive_bins;

    let active_basis_coefficients_per_frame = match &basis.storage {
        BasisStorage::Dense { fft_basis } => {
            for frame_index in 0..frame_count {
                let start = frame_index * hop_length;
                let copy_start = Instant::now();
                for (offset, value) in spectrum.iter_mut().enumerate() {
                    value.re = padded[start + offset];
                    value.im = 0.0;
                }
                frame_copy_seconds += elapsed_seconds(copy_start);

                let fft_start = Instant::now();
                basis.fft.process(&mut spectrum);
                fft_execution_seconds += elapsed_seconds(fft_start);

                let dot_start = Instant::now();
                let spectrum_bins = &spectrum[..basis.positive_bins];
                if frame_index == profile_frame_index {
                    for row in 0..basis.row_count {
                        let lookup_start = Instant::now();
                        let basis_row =
                            &fft_basis[row * basis.positive_bins..(row + 1) * basis.positive_bins];
                        sampled_basis_lookup_seconds += elapsed_seconds(lookup_start);

                        let mac_start = Instant::now();
                        let mut accumulator = Complex32::new(0.0, 0.0);
                        for (basis_value, spectrum_value) in
                            basis_row.iter().zip(spectrum_bins.iter())
                        {
                            accumulator += *basis_value * *spectrum_value;
                        }
                        sampled_multiply_accumulate_seconds += elapsed_seconds(mac_start);

                        let output_start = Instant::now();
                        response[row * frame_count + frame_index] = accumulator.norm();
                        sampled_output_write_seconds += elapsed_seconds(output_start);
                    }
                } else {
                    for row in 0..basis.row_count {
                        let basis_row =
                            &fft_basis[row * basis.positive_bins..(row + 1) * basis.positive_bins];
                        let mut accumulator = Complex32::new(0.0, 0.0);
                        for (basis_value, spectrum_value) in
                            basis_row.iter().zip(spectrum_bins.iter())
                        {
                            accumulator += *basis_value * *spectrum_value;
                        }
                        response[row * frame_count + frame_index] = accumulator.norm();
                    }
                }
                dot_product_seconds += elapsed_seconds(dot_start);
            }
            basis.row_count * basis.positive_bins
        }
        BasisStorage::Sparse {
            row_offsets,
            indices,
            values,
        } => {
            for frame_index in 0..frame_count {
                let start = frame_index * hop_length;
                let copy_start = Instant::now();
                for (offset, value) in spectrum.iter_mut().enumerate() {
                    value.re = padded[start + offset];
                    value.im = 0.0;
                }
                frame_copy_seconds += elapsed_seconds(copy_start);

                let fft_start = Instant::now();
                basis.fft.process(&mut spectrum);
                fft_execution_seconds += elapsed_seconds(fft_start);

                let dot_start = Instant::now();
                let spectrum_bins = &spectrum[..basis.positive_bins];
                if frame_index == profile_frame_index {
                    for row in 0..basis.row_count {
                        let lookup_start = Instant::now();
                        let row_start = row_offsets[row];
                        let row_end = row_offsets[row + 1];
                        sampled_basis_lookup_seconds += elapsed_seconds(lookup_start);

                        let mac_start = Instant::now();
                        let row_indices = &indices[row_start..row_end];
                        let row_values = &values[row_start..row_end];
                        let mut accumulator = Complex32::new(0.0, 0.0);
                        for (&basis_index, &basis_value) in
                            row_indices.iter().zip(row_values.iter())
                        {
                            accumulator += basis_value * spectrum_bins[basis_index];
                        }
                        sampled_multiply_accumulate_seconds += elapsed_seconds(mac_start);

                        let output_start = Instant::now();
                        response[row * frame_count + frame_index] = accumulator.norm();
                        sampled_output_write_seconds += elapsed_seconds(output_start);
                    }
                } else {
                    for row in 0..basis.row_count {
                        let row_start = row_offsets[row];
                        let row_end = row_offsets[row + 1];
                        let row_indices = &indices[row_start..row_end];
                        let row_values = &values[row_start..row_end];
                        let mut accumulator = Complex32::new(0.0, 0.0);
                        for (&basis_index, &basis_value) in
                            row_indices.iter().zip(row_values.iter())
                        {
                            accumulator += basis_value * spectrum_bins[basis_index];
                        }
                        response[row * frame_count + frame_index] = accumulator.norm();
                    }
                }
                dot_product_seconds += elapsed_seconds(dot_start);
            }
            *row_offsets.last().unwrap_or(&0)
        }
    };

    let frame_scale = frame_count as f64;
    let mut basis_lookup_seconds = sampled_basis_lookup_seconds * frame_scale;
    let mut multiply_accumulate_seconds = sampled_multiply_accumulate_seconds * frame_scale;
    let mut output_write_seconds = sampled_output_write_seconds * frame_scale;
    let substage_sum = basis_lookup_seconds + multiply_accumulate_seconds + output_write_seconds;
    if substage_sum > dot_product_seconds && substage_sum > 0.0 {
        let rescale = dot_product_seconds / substage_sum;
        basis_lookup_seconds *= rescale;
        multiply_accumulate_seconds *= rescale;
        output_write_seconds *= rescale;
    }
    let dot_loop_overhead_seconds = (dot_product_seconds
        - basis_lookup_seconds
        - multiply_accumulate_seconds
        - output_write_seconds)
        .max(0.0);
    let active_basis_coefficients = active_basis_coefficients_per_frame * frame_count;
    let basis_coefficients_total = basis_coefficients_total_per_frame * frame_count;

    (
        response,
        OctaveRunProfile {
            octave_index,
            hop_length,
            input_samples: audio.len(),
            output_frames: frame_count,
            total_seconds: elapsed_seconds(total_start),
            transform_seconds: elapsed_seconds(total_start),
            buffer_allocation_seconds,
            buffer_allocation_bytes,
            frame_copy_seconds,
            fft_execution_seconds,
            dot_product_seconds,
            dot_loop_overhead_seconds,
            basis_lookup_seconds,
            multiply_accumulate_seconds,
            output_write_seconds,
            active_basis_coefficients,
            basis_coefficients_total,
            array_materialization_seconds: 0.0,
            downsample_seconds: 0.0,
            downsampled_samples: None,
        },
    )
}

fn harmonic_response_recursive(
    octave_pyramid: &OctaveAudioPyramid,
    basis: &HarmonicBasis,
) -> Result<(Array2<f32>, HarmonicRunProfile), FrontendError> {
    let mut octave_responses = Vec::with_capacity(basis.octaves.len());
    let mut octave_profiles = Vec::with_capacity(basis.octaves.len());
    let mut temporary_allocation_bytes = 0usize;

    for (octave_index, octave_basis) in basis.octaves.iter().enumerate() {
        let octave_audio = octave_pyramid.levels.get(octave_index).ok_or_else(|| {
            FrontendError::ShapeMismatch(format!("octave pyramid is missing level {octave_index}"))
        })?;
        let (response, mut octave_profile) = harmonic_response(
            &octave_audio.samples,
            octave_audio.hop_length,
            octave_basis,
            octave_index,
        );
        temporary_allocation_bytes += octave_profile.buffer_allocation_bytes;
        let array_start = Instant::now();
        let response_array = magnitudes_to_array2(
            response,
            octave_basis.row_count,
            1 + octave_audio.samples.len() / octave_audio.hop_length,
        )?;
        octave_profile.array_materialization_seconds = elapsed_seconds(array_start);
        temporary_allocation_bytes += response_array.len() * size_of::<f32>();
        octave_responses.push(response_array);

        octave_profile.total_seconds = octave_profile.transform_seconds
            + octave_profile.array_materialization_seconds
            + octave_profile.downsample_seconds;
        octave_profiles.push(octave_profile);
    }

    let trim_start = Instant::now();
    let mut output = trim_stack(&octave_responses, basis.lengths.len(), &basis.octave_slices)?;
    let trim_stack_seconds = elapsed_seconds(trim_start);
    temporary_allocation_bytes += output.len() * size_of::<f32>();
    let norm_start = Instant::now();
    for (bin, length) in basis.lengths.iter().enumerate() {
        let scale = length.sqrt().max(f32::EPSILON);
        for frame in 0..output.shape()[1] {
            output[[bin, frame]] /= scale;
        }
    }
    let normalization_seconds = elapsed_seconds(norm_start);

    Ok((
        output,
        HarmonicRunProfile {
            harmonic: 0.0,
            total_seconds: 0.0,
            audio_clone_seconds: 0.0,
            trim_stack_seconds,
            normalization_seconds,
            flatten_seconds: 0.0,
            post_process_seconds: 0.0,
            temporary_allocation_bytes,
            octave_profiles,
        },
    ))
}

fn build_octave_audio_pyramid(
    audio: &[f32],
    hop_length: usize,
    octave_count: usize,
) -> Result<OctaveAudioPyramid, FrontendError> {
    let total_start = Instant::now();
    if octave_count == 0 {
        return Ok(OctaveAudioPyramid {
            levels: Vec::new(),
            profile: Vec::new(),
            allocation_bytes: 0,
            total_seconds: 0.0,
        });
    }

    let mut levels = Vec::with_capacity(octave_count);
    let mut profile = Vec::with_capacity(octave_count);
    let base_samples = audio.to_vec();
    let mut allocation_bytes = base_samples.len() * size_of::<f32>();
    levels.push(OctaveAudioLevel {
        samples: base_samples,
        hop_length,
    });
    profile.push(OctavePyramidProfile {
        octave_index: 0,
        hop_length,
        input_samples: audio.len(),
        output_samples: audio.len(),
        total_seconds: 0.0,
        downsample_seconds: 0.0,
        scale_seconds: 0.0,
        output_allocation_seconds: 0.0,
        kernel_prepare_seconds: 0.0,
        convolution_seconds: 0.0,
        resize_seconds: 0.0,
        specialized_path: false,
        left_tap_count: 0,
        right_tap_count: 0,
    });

    for octave_index in 1..octave_count {
        let previous_level = levels.last().ok_or_else(|| {
            FrontendError::ShapeMismatch("octave pyramid lost its previous level".to_owned())
        })?;
        let next_hop = previous_level.hop_length / 2;
        if next_hop == 0 {
            return Err(FrontendError::ConfigMismatch(format!(
                "hop_length became zero while building octave pyramid at octave {octave_index}"
            )));
        }

        let (mut next_samples, downsample_profile) =
            downsample_kaiser_best_by_2_profiled(&previous_level.samples)?;
        let downsample_seconds = downsample_profile.total_seconds;

        let scale_start = Instant::now();
        let scale = (OCTAVE_DOWNSAMPLE_ORIG_SR as f32 / OCTAVE_DOWNSAMPLE_TARGET_SR as f32).sqrt();
        for sample in &mut next_samples {
            *sample *= scale;
        }
        let scale_seconds = elapsed_seconds(scale_start);

        allocation_bytes += next_samples.len() * size_of::<f32>();
        profile.push(OctavePyramidProfile {
            octave_index,
            hop_length: next_hop,
            input_samples: previous_level.samples.len(),
            output_samples: next_samples.len(),
            total_seconds: downsample_seconds + scale_seconds,
            downsample_seconds,
            scale_seconds,
            output_allocation_seconds: downsample_profile.output_allocation_seconds,
            kernel_prepare_seconds: downsample_profile.kernel_prepare_seconds,
            convolution_seconds: downsample_profile.convolution_seconds,
            resize_seconds: downsample_profile.resize_seconds,
            specialized_path: downsample_profile.specialized_path,
            left_tap_count: downsample_profile.left_tap_count,
            right_tap_count: downsample_profile.right_tap_count,
        });
        levels.push(OctaveAudioLevel {
            samples: next_samples,
            hop_length: next_hop,
        });
    }

    Ok(OctaveAudioPyramid {
        levels,
        profile,
        allocation_bytes,
        total_seconds: elapsed_seconds(total_start),
    })
}

fn trim_stack(
    octave_responses: &[Array2<f32>],
    n_bins: usize,
    octave_slices: &[Range<usize>],
) -> Result<Array2<f32>, FrontendError> {
    let max_col = octave_responses
        .iter()
        .map(|response| response.shape()[1])
        .min()
        .ok_or_else(|| {
            FrontendError::ShapeMismatch("no octave responses were generated".to_owned())
        })?;
    let mut output = Array2::<f32>::zeros((n_bins, max_col));
    if octave_responses.len() != octave_slices.len() {
        return Err(FrontendError::ShapeMismatch(format!(
            "octave response count {} does not match octave slice count {}",
            octave_responses.len(),
            octave_slices.len()
        )));
    }

    for (response, slice) in octave_responses.iter().zip(octave_slices.iter()) {
        if response.shape()[0] != slice.len() {
            return Err(FrontendError::ShapeMismatch(format!(
                "octave response rows {} do not match target slice length {}",
                response.shape()[0],
                slice.len()
            )));
        }

        for (row_offset, bin) in slice.clone().enumerate() {
            for frame in 0..max_col {
                output[[bin, frame]] = response[[row_offset, frame]];
            }
        }
    }

    Ok(output)
}

fn magnitudes_to_array2(
    magnitudes: Vec<f32>,
    n_bins: usize,
    frame_count: usize,
) -> Result<Array2<f32>, FrontendError> {
    if magnitudes.len() != n_bins * frame_count {
        return Err(FrontendError::ShapeMismatch(format!(
            "harmonic magnitude length mismatch: expected {} values for ({n_bins}, {frame_count}), got {}",
            n_bins * frame_count,
            magnitudes.len()
        )));
    }

    Array2::from_shape_vec((n_bins, frame_count), magnitudes).map_err(|error| {
        FrontendError::ShapeMismatch(format!(
            "failed to materialize harmonic magnitudes as ndarray: {error}"
        ))
    })
}

fn array2_to_vec(array: &Array2<f32>) -> Vec<f32> {
    let shape = array.shape();
    let mut output = Vec::with_capacity(shape[0] * shape[1]);
    for bin in 0..shape[0] {
        for frame in 0..shape[1] {
            output.push(array[[bin, frame]]);
        }
    }
    output
}

fn post_process_magnitudes(
    magnitudes: &[f32],
    n_bins: usize,
    frame_count: usize,
    top_db: f32,
) -> Result<Array2<f32>, FrontendError> {
    let mut output = Array2::<f32>::zeros((n_bins, frame_count));
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
            output[[bin, frame]] = db / top_db + 1.0;
        }
    }

    Ok(output)
}

fn elapsed_seconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64()
}
