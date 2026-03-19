use crate::fretboard::pitch_to_hz;
use crate::types::{HarmonicBinRange, MidiHarmonicMap};

pub fn build_harmonic_maps(
    sample_rate: u32,
    nfft: usize,
    midi_min: u8,
    midi_max: u8,
    max_harmonics: usize,
    local_bandwidth_bins: usize,
) -> Vec<MidiHarmonicMap> {
    if nfft == 0 || midi_min > midi_max || max_harmonics == 0 {
        return Vec::new();
    }

    let nyquist = sample_rate as f32 * 0.5;
    let hz_per_bin = sample_rate as f32 / nfft as f32;
    let last_bin = nfft / 2;

    let mut out = Vec::with_capacity((midi_max - midi_min + 1) as usize);
    for midi in midi_min..=midi_max {
        let f0_hz = pitch_to_hz(midi);
        let mut ranges = Vec::with_capacity(max_harmonics);

        for h in 1..=max_harmonics {
            let harmonic_hz = f0_hz * h as f32;
            if harmonic_hz >= nyquist {
                break;
            }

            let center = (harmonic_hz / hz_per_bin).round() as isize;
            let start = center.saturating_sub(local_bandwidth_bins as isize).max(0) as usize;
            let end = (center + local_bandwidth_bins as isize).min(last_bin as isize) as usize;

            ranges.push(HarmonicBinRange { start, end });
        }

        out.push(MidiHarmonicMap {
            midi,
            f0_hz,
            ranges,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::build_harmonic_maps;

    #[test]
    fn harmonic_ranges_are_in_bounds() {
        let maps = build_harmonic_maps(44_100, 4096, 40, 41, 8, 2);
        assert_eq!(maps.len(), 2);
        for map in maps {
            assert!(!map.ranges.is_empty());
            for r in map.ranges {
                assert!(r.start <= r.end);
                assert!(r.end <= 4096 / 2);
            }
        }
    }
}
