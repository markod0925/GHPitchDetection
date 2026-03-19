use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::config::FrontendConfig;
use crate::types::SpectrumFrame;

pub struct StftPlan {
    pub win_length: usize,
    pub hop_length: usize,
    pub nfft: usize,
    pub window: Vec<f32>,
}

pub fn build_stft_plan(cfg: &FrontendConfig) -> StftPlan {
    let mut window = Vec::with_capacity(cfg.win_length);
    if cfg.win_length <= 1 {
        window.push(1.0);
    } else {
        let denom = (cfg.win_length - 1) as f32;
        for n in 0..cfg.win_length {
            let phase = 2.0 * std::f32::consts::PI * (n as f32) / denom;
            window.push(0.5 - 0.5 * phase.cos());
        }
    }

    StftPlan {
        win_length: cfg.win_length,
        hop_length: cfg.hop_length,
        nfft: cfg.nfft,
        window,
    }
}

pub fn compute_stft_frames(
    samples: &[f32],
    sample_rate: u32,
    plan: &StftPlan,
) -> Vec<SpectrumFrame> {
    if plan.win_length == 0
        || plan.hop_length == 0
        || plan.nfft == 0
        || samples.len() < plan.win_length
    {
        return Vec::new();
    }

    let frame_count = 1 + (samples.len() - plan.win_length) / plan.hop_length;
    let bins = plan.nfft / 2 + 1;
    let mut out = Vec::with_capacity(frame_count);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(plan.nfft);
    let mut fft_buf = vec![Complex::<f32>::new(0.0, 0.0); plan.nfft];

    for frame_index in 0..frame_count {
        let start = frame_index * plan.hop_length;

        for c in fft_buf.iter_mut() {
            c.re = 0.0;
            c.im = 0.0;
        }

        for i in 0..plan.win_length {
            fft_buf[i].re = samples[start + i] * plan.window[i];
        }

        fft.process(&mut fft_buf);

        let mut mag = vec![0.0_f32; bins];
        for (bin, dst) in mag.iter_mut().enumerate() {
            *dst = fft_buf[bin].norm();
        }

        out.push(SpectrumFrame {
            frame_index,
            time_sec: start as f32 / sample_rate as f32,
            mag,
            white: vec![0.0_f32; bins],
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use crate::config::FrontendConfig;

    use super::{build_stft_plan, compute_stft_frames};

    #[test]
    fn stft_returns_expected_frame_and_bin_count() {
        let cfg = FrontendConfig {
            kind: "stft".to_string(),
            win_length: 8,
            hop_length: 4,
            nfft: 8,
            window: "hann".to_string(),
        };
        let plan = build_stft_plan(&cfg);
        let samples = vec![0.0_f32; 16];
        let frames = compute_stft_frames(&samples, 8_000, &plan);

        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].mag.len(), 5);
        assert_eq!(frames[0].white.len(), 5);
    }
}
