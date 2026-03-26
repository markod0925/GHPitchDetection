use std::{f32::consts::PI, mem::size_of, ops::Range, sync::Arc, time::Instant};

use ndarray::{Array2, Array3};
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::{
    audio::resample_kaiser_best,
    config::{
        FRONTEND_BINS_PER_OCTAVE, FRONTEND_FMIN_HZ, FRONTEND_HARMONICS, FRONTEND_HOP_LENGTH,
        FRONTEND_N_BINS, FRONTEND_SAMPLE_RATE, FRONTEND_SPARSITY, FRONTEND_TOP_DB,
    },
    error::FrontendError,
    profile::{
        FrontendInitProfile, FrontendRunProfile, HarmonicInitProfile, HarmonicRunProfile,
        OctaveInitProfile, OctaveRunProfile,
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
    fft_basis: Vec<Complex32>,
    fft: Arc<dyn Fft<f32>>,
}

struct HarmonicBasis {
    lengths: Vec<f32>,
    octave_slices: Vec<Range<usize>>,
    octaves: Vec<OctaveBasis>,
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

        let mut harmonics = Vec::with_capacity(self.harmonic_bases.len());
        let mut harmonic_profiles = Vec::with_capacity(self.harmonic_bases.len());

        for (harmonic, basis) in self
            .config
            .harmonics
            .iter()
            .copied()
            .zip(self.harmonic_bases.iter())
        {
            let (magnitude, mut profile) =
                harmonic_response_recursive(audio, self.config.hop_length, basis)?;
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
    }

    Ok((
        OctaveBasis {
            n_fft,
            positive_bins,
            row_count: lengths.len(),
            fft_basis,
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
            fft_basis_bytes: frequencies.len() * positive_bins * size_of::<Complex32>(),
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
        for row in 0..basis.row_count {
            let basis_row =
                &basis.fft_basis[row * basis.positive_bins..(row + 1) * basis.positive_bins];
            let mut accumulator = Complex32::new(0.0, 0.0);
            for (basis_value, spectrum_value) in basis_row
                .iter()
                .zip(spectrum.iter().take(basis.positive_bins))
            {
                accumulator += *basis_value * *spectrum_value;
            }

            let magnitude = accumulator.norm();
            response[row * frame_count + frame_index] = magnitude;
        }
        dot_product_seconds += elapsed_seconds(dot_start);
    }

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
            array_materialization_seconds: 0.0,
            downsample_seconds: 0.0,
            downsampled_samples: None,
        },
    )
}

fn harmonic_response_recursive(
    audio: &[f32],
    hop_length: usize,
    basis: &HarmonicBasis,
) -> Result<(Array2<f32>, HarmonicRunProfile), FrontendError> {
    let mut octave_responses = Vec::with_capacity(basis.octaves.len());
    let clone_start = Instant::now();
    let mut octave_audio = audio.to_vec();
    let audio_clone_seconds = elapsed_seconds(clone_start);
    let mut octave_hop = hop_length;
    let mut octave_profiles = Vec::with_capacity(basis.octaves.len());
    let mut temporary_allocation_bytes = octave_audio.len() * size_of::<f32>();

    for (octave_index, octave_basis) in basis.octaves.iter().enumerate() {
        let (response, mut octave_profile) =
            harmonic_response(&octave_audio, octave_hop, octave_basis, octave_index);
        temporary_allocation_bytes += octave_profile.buffer_allocation_bytes;
        let array_start = Instant::now();
        let response_array = magnitudes_to_array2(
            &response,
            octave_basis.row_count,
            1 + octave_audio.len() / octave_hop,
        )?;
        octave_profile.array_materialization_seconds = elapsed_seconds(array_start);
        temporary_allocation_bytes += response_array.len() * size_of::<f32>();
        octave_responses.push(response_array);

        if octave_index + 1 < basis.octaves.len() && octave_hop % 2 == 0 {
            octave_hop /= 2;
            let downsample_start = Instant::now();
            octave_audio = resample_kaiser_best(
                &octave_audio,
                OCTAVE_DOWNSAMPLE_ORIG_SR,
                OCTAVE_DOWNSAMPLE_TARGET_SR,
            )?;
            octave_profile.downsample_seconds = elapsed_seconds(downsample_start);
            octave_profile.downsampled_samples = Some(octave_audio.len());
            let scale =
                (OCTAVE_DOWNSAMPLE_ORIG_SR as f32 / OCTAVE_DOWNSAMPLE_TARGET_SR as f32).sqrt();
            for sample in &mut octave_audio {
                *sample *= scale;
            }
        }

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
            audio_clone_seconds,
            trim_stack_seconds,
            normalization_seconds,
            flatten_seconds: 0.0,
            post_process_seconds: 0.0,
            temporary_allocation_bytes,
            octave_profiles,
        },
    ))
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
    magnitudes: &[f32],
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

    let mut output = Array2::<f32>::zeros((n_bins, frame_count));
    for bin in 0..n_bins {
        for frame in 0..frame_count {
            output[[bin, frame]] = magnitudes[bin * frame_count + frame];
        }
    }

    Ok(output)
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
