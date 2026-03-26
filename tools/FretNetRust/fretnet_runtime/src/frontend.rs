use std::mem::size_of;

use crate::{
    config::FRONTEND_FRAME_WIDTH,
    error::FrontendError,
    hcqt::{FrontendConfig, HcqtExtractor},
    profile::{FrontendInitProfile, FrontendRunProfile},
    types::{FeatureBatch, FrontendStageOutputs, HcqtFeatures},
};

pub struct FeatureExtractor {
    config: FrontendConfig,
    hcqt: HcqtExtractor,
}

impl FeatureExtractor {
    pub fn new(config: FrontendConfig) -> Result<Self, FrontendError> {
        Ok(Self::new_profiled(config)?.0)
    }

    pub fn new_profiled(
        config: FrontendConfig,
    ) -> Result<(Self, FrontendInitProfile), FrontendError> {
        let (hcqt, profile) = HcqtExtractor::new_profiled(config.clone())?;
        Ok((Self { config, hcqt }, profile))
    }

    pub fn extract_hcqt(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<HcqtFeatures, FrontendError> {
        self.hcqt.extract(audio, sample_rate)
    }

    pub fn extract_hcqt_profiled(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<(HcqtFeatures, FrontendRunProfile), FrontendError> {
        self.hcqt.extract_profiled(audio, sample_rate)
    }

    pub fn extract_stages(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<FrontendStageOutputs, FrontendError> {
        self.hcqt.extract_stages(audio, sample_rate)
    }

    pub fn extract_stages_profiled(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<(FrontendStageOutputs, FrontendRunProfile), FrontendError> {
        self.hcqt.extract_stages_profiled(audio, sample_rate)
    }

    pub fn hcqt_to_batch_profiled(
        &self,
        hcqt: &HcqtFeatures,
        max_frames: Option<usize>,
    ) -> Result<(FeatureBatch, f64, usize), FrontendError> {
        let start = std::time::Instant::now();
        let batch = self.hcqt_to_batch(hcqt, max_frames)?;
        let frame_count = batch.frame_count();
        let channel_count = batch.channel_count();
        let freq_bins = batch.freq_bins();
        let frame_width = batch.frame_width();
        let pad_frames = frame_count + 2 * (frame_width / 2);
        let allocation_bytes = channel_count * freq_bins * pad_frames * size_of::<f32>()
            + batch.data().len() * size_of::<f32>();
        Ok((batch, start.elapsed().as_secs_f64(), allocation_bytes))
    }

    pub fn extract_profiled(
        &self,
        audio: &[f32],
        sample_rate: u32,
        max_frames: Option<usize>,
    ) -> Result<(FeatureBatch, FrontendRunProfile), FrontendError> {
        let (hcqt, mut profile) = self.extract_hcqt_profiled(audio, sample_rate)?;
        let (batch, batch_conversion_seconds, batch_conversion_allocation_bytes) =
            self.hcqt_to_batch_profiled(&hcqt, max_frames)?;
        profile.batch_conversion_seconds = batch_conversion_seconds;
        profile.batch_conversion_allocation_bytes = batch_conversion_allocation_bytes;
        profile.batch_shape = batch.shape();
        profile.total_seconds += batch_conversion_seconds;
        Ok((batch, profile))
    }

    pub fn hcqt_to_batch(
        &self,
        hcqt: &HcqtFeatures,
        max_frames: Option<usize>,
    ) -> Result<FeatureBatch, FrontendError> {
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
