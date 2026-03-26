use ndarray::{Array2, Array3, Array5, ArrayD};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub element_type: String,
    pub dimensions: Vec<Option<i64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_path: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

#[derive(Debug, Clone)]
pub struct FeatureBatch {
    data: Array5<f32>,
}

impl FeatureBatch {
    pub fn new(data: Array5<f32>) -> Self {
        Self { data }
    }

    pub fn synthetic(
        batch: usize,
        frames: usize,
        channels: usize,
        freq_bins: usize,
        frame_width: usize,
    ) -> Self {
        let mut data = Array5::<f32>::zeros((batch, frames, channels, freq_bins, frame_width));

        for b in 0..batch {
            for t in 0..frames {
                for c in 0..channels {
                    for f in 0..freq_bins {
                        for w in 0..frame_width {
                            let phase = (b as f32 * 0.13)
                                + (t as f32 * 0.017)
                                + (c as f32 * 0.11)
                                + (f as f32 * 0.003)
                                + (w as f32 * 0.07);
                            data[[b, t, c, f, w]] = phase.sin().abs().min(1.0);
                        }
                    }
                }
            }
        }

        Self { data }
    }

    pub fn data(&self) -> &Array5<f32> {
        &self.data
    }

    pub fn shape(&self) -> [usize; 5] {
        let shape = self.data.shape();
        [shape[0], shape[1], shape[2], shape[3], shape[4]]
    }

    pub fn batch_size(&self) -> usize {
        self.shape()[0]
    }

    pub fn frame_count(&self) -> usize {
        self.shape()[1]
    }

    pub fn channel_count(&self) -> usize {
        self.shape()[2]
    }

    pub fn freq_bins(&self) -> usize {
        self.shape()[3]
    }

    pub fn frame_width(&self) -> usize {
        self.shape()[4]
    }

    pub fn from_hcqt(hcqt: &Array3<f32>, frame_width: usize, max_frames: Option<usize>) -> Self {
        let shape = hcqt.shape();
        let channels = shape[0];
        let freq_bins = shape[1];
        let total_frames = shape[2];
        let frame_count = max_frames.unwrap_or(total_frames).min(total_frames);
        let pad_length = frame_width / 2;
        let padded_frames = frame_count + 2 * pad_length;

        let mut padded = Array3::<f32>::zeros((channels, freq_bins, padded_frames));
        for c in 0..channels {
            for f in 0..freq_bins {
                for t in 0..frame_count {
                    padded[[c, f, t + pad_length]] = hcqt[[c, f, t]];
                }
            }
        }

        let mut batch = Array5::<f32>::zeros((1, frame_count, channels, freq_bins, frame_width));
        for t in 0..frame_count {
            for c in 0..channels {
                for f in 0..freq_bins {
                    for w in 0..frame_width {
                        batch[[0, t, c, f, w]] = padded[[c, f, t + w]];
                    }
                }
            }
        }

        Self { data: batch }
    }
}

#[derive(Debug, Clone)]
pub struct HcqtFeatures {
    data: Array3<f32>,
    sample_rate: u32,
    hop_length: usize,
}

#[derive(Debug, Clone)]
pub struct HarmonicStageOutput {
    pub harmonic: f32,
    pub magnitude: Array2<f32>,
    pub post_processed: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct FrontendStageOutputs {
    pub hcqt: HcqtFeatures,
    pub harmonics: Vec<HarmonicStageOutput>,
}

impl HcqtFeatures {
    pub fn new(data: Array3<f32>, sample_rate: u32, hop_length: usize) -> Self {
        Self {
            data,
            sample_rate,
            hop_length,
        }
    }

    pub fn data(&self) -> &Array3<f32> {
        &self.data
    }

    pub fn shape(&self) -> [usize; 3] {
        let shape = self.data.shape();
        [shape[0], shape[1], shape[2]]
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    pub fn frame_count(&self) -> usize {
        self.shape()[2]
    }

    pub fn truncated(&self, max_frames: usize) -> Self {
        let frame_count = max_frames.min(self.frame_count());
        let mut data = Array3::<f32>::zeros((self.shape()[0], self.shape()[1], frame_count));
        for channel in 0..self.shape()[0] {
            for bin in 0..self.shape()[1] {
                for frame in 0..frame_count {
                    data[[channel, bin, frame]] = self.data[[channel, bin, frame]];
                }
            }
        }

        Self {
            data,
            sample_rate: self.sample_rate,
            hop_length: self.hop_length,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NamedTensorOutput {
    pub name: String,
    pub data: ArrayD<f32>,
}

#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub tensors: Vec<NamedTensorOutput>,
}

impl ModelOutput {
    pub fn tensor(&self, name: &str) -> Option<&NamedTensorOutput> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStats {
    pub name: String,
    pub shape: Vec<usize>,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedOutput {
    pub tensors: Vec<TensorStats>,
    pub notes: Option<String>,
}
