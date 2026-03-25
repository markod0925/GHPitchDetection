use crate::{
    config::FRONTEND_FRAME_WIDTH,
    error::FrontendError,
    hcqt::{FrontendConfig, HcqtExtractor},
    types::{FeatureBatch, HcqtFeatures},
};

pub struct FeatureExtractor {
    config: FrontendConfig,
    hcqt: HcqtExtractor,
}

impl FeatureExtractor {
    pub fn new(config: FrontendConfig) -> Result<Self, FrontendError> {
        let hcqt = HcqtExtractor::new(config.clone())?;
        Ok(Self { config, hcqt })
    }

    pub fn extract_hcqt(&self, audio: &[f32], sample_rate: u32) -> Result<HcqtFeatures, FrontendError> {
        self.hcqt.extract(audio, sample_rate)
    }

    pub fn hcqt_to_batch(&self, hcqt: &HcqtFeatures, max_frames: Option<usize>) -> Result<FeatureBatch, FrontendError> {
        if hcqt.sample_rate() != self.config.sample_rate {
            return Err(FrontendError::UnsupportedSampleRate {
                expected: self.config.sample_rate,
                actual: hcqt.sample_rate(),
            });
        }
        Ok(FeatureBatch::from_hcqt(
            hcqt.data(),
            FRONTEND_FRAME_WIDTH,
            max_frames,
        ))
    }

    pub fn extract(&self, audio: &[f32], sample_rate: u32) -> Result<FeatureBatch, FrontendError> {
        let hcqt = self.extract_hcqt(audio, sample_rate)?;
        self.hcqt_to_batch(&hcqt, None)
    }

    pub fn config(&self) -> &FrontendConfig {
        &self.config
    }
}
