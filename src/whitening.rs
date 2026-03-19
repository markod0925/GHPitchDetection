use crate::config::WhiteningConfig;

pub fn whiten_log_spectrum(
    mag: &[f32],
    cfg: &WhiteningConfig,
    tmp_log: &mut [f32],
    envelope: &mut [f32],
    white: &mut [f32],
) {
    debug_assert_eq!(mag.len(), tmp_log.len());
    debug_assert_eq!(mag.len(), envelope.len());
    debug_assert_eq!(mag.len(), white.len());

    let n = mag.len();
    if n == 0 {
        return;
    }

    let eps = cfg.floor_eps.max(1e-12);
    for (i, &m) in mag.iter().enumerate() {
        tmp_log[i] = (m + eps).ln();
    }

    let span = cfg.span_bins.max(1);
    let half = span / 2;

    let mut left = 0usize;
    let mut right = 0usize;
    let mut sum = 0.0f32;

    for k in 0..n {
        let target_left = k.saturating_sub(half);
        let target_right = (k + half + 1).min(n);

        while left > target_left {
            left -= 1;
            sum += tmp_log[left];
        }
        while left < target_left {
            sum -= tmp_log[left];
            left += 1;
        }
        while right < target_right {
            sum += tmp_log[right];
            right += 1;
        }
        while right > target_right {
            right -= 1;
            sum -= tmp_log[right];
        }

        let len = (right - left).max(1);
        envelope[k] = sum / len as f32;
    }

    if cfg.rectify_positive {
        for i in 0..n {
            white[i] = (tmp_log[i] - envelope[i]).max(0.0);
        }
    } else {
        for i in 0..n {
            white[i] = tmp_log[i] - envelope[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::WhiteningConfig;

    use super::whiten_log_spectrum;

    #[test]
    fn constant_spectrum_whitens_to_near_zero() {
        let cfg = WhiteningConfig::default();
        let mag = vec![1.0_f32; 32];
        let mut tmp_log = vec![0.0; mag.len()];
        let mut envelope = vec![0.0; mag.len()];
        let mut white = vec![0.0; mag.len()];

        whiten_log_spectrum(&mag, &cfg, &mut tmp_log, &mut envelope, &mut white);

        let peak = white.iter().copied().fold(0.0_f32, f32::max);
        assert!(peak < 1e-4, "peak was {peak}");
    }
}
