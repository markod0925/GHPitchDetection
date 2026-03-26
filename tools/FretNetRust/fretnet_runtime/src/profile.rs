use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct FrontendInitProfile {
    pub total_seconds: f64,
    pub harmonic_profiles: Vec<HarmonicInitProfile>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct HarmonicInitProfile {
    pub harmonic: f32,
    pub total_seconds: f64,
    pub octave_profiles: Vec<OctaveInitProfile>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct OctaveInitProfile {
    pub octave_index: usize,
    pub sample_rate: f32,
    pub hop_length: usize,
    pub bin_start: usize,
    pub bin_end: usize,
    pub row_count: usize,
    pub n_fft: usize,
    pub positive_bins: usize,
    pub fft_basis_bytes: usize,
    pub fft_plan_seconds: f64,
    pub filter_build_seconds: f64,
    pub sparsify_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FrontendRunProfile {
    pub sample_rate: u32,
    pub input_samples: usize,
    pub total_seconds: f64,
    pub harmonic_profiles: Vec<HarmonicRunProfile>,
    pub hcqt_assembly_seconds: f64,
    pub hcqt_assembly_allocation_bytes: usize,
    pub hcqt_shape: [usize; 3],
    pub batch_conversion_seconds: f64,
    pub batch_conversion_allocation_bytes: usize,
    pub batch_shape: [usize; 5],
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct HarmonicRunProfile {
    pub harmonic: f32,
    pub total_seconds: f64,
    pub audio_clone_seconds: f64,
    pub trim_stack_seconds: f64,
    pub normalization_seconds: f64,
    pub flatten_seconds: f64,
    pub post_process_seconds: f64,
    pub temporary_allocation_bytes: usize,
    pub octave_profiles: Vec<OctaveRunProfile>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct OctaveRunProfile {
    pub octave_index: usize,
    pub hop_length: usize,
    pub input_samples: usize,
    pub output_frames: usize,
    pub total_seconds: f64,
    pub transform_seconds: f64,
    pub buffer_allocation_seconds: f64,
    pub buffer_allocation_bytes: usize,
    pub frame_copy_seconds: f64,
    pub fft_execution_seconds: f64,
    pub dot_product_seconds: f64,
    pub array_materialization_seconds: f64,
    pub downsample_seconds: f64,
    pub downsampled_samples: Option<usize>,
}
