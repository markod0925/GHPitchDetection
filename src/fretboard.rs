pub fn midi_to_fret_positions(midi: u8, open_midi: &[u8; 6], fret_max: u8) -> Vec<(u8, u8)> {
    let mut out = Vec::new();
    for (string, &open) in open_midi.iter().enumerate() {
        if midi >= open {
            let fret = midi - open;
            if fret <= fret_max {
                out.push((string as u8, fret));
            }
        }
    }
    out
}

pub fn pitch_to_hz(midi: u8) -> f32 {
    440.0 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0)
}

pub fn hz_to_midi_float(freq_hz: f32) -> f32 {
    69.0 + 12.0 * (freq_hz / 440.0).log2()
}

pub fn fret_region(fret: u8, region_bounds: &[u8]) -> usize {
    if region_bounds.len() < 2 {
        return 0;
    }
    for i in 0..region_bounds.len() - 1 {
        if fret >= region_bounds[i] && fret < region_bounds[i + 1] {
            return i;
        }
    }
    region_bounds.len() - 2
}

#[cfg(test)]
mod tests {
    use crate::types::OPEN_MIDI;

    use super::midi_to_fret_positions;

    #[test]
    fn midi_positions_cover_playable_strings() {
        let pos = midi_to_fret_positions(64, &OPEN_MIDI, 20);
        assert!(pos.contains(&(5, 0)));
        assert!(pos.contains(&(4, 5)));
        assert!(pos.contains(&(0, 24)) == false);
    }
}
