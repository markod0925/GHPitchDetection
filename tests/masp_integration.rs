use std::f32::consts::PI;

use guitar_pitch::config::AppConfig;
use guitar_pitch::dataset::PolyEvalEvent;
use guitar_pitch::masp::{
    load_pretrain_artifacts, pretrain_from_guitarset, score_pitch_frames_masp,
    validate_expected_segment, write_pretrain_artifacts, write_validation_artifacts, MaspManifest,
    MaspModelParams, MaspNoteSignature, MaspPretrainArtifacts, MaspValidationResult,
    MaspValidationRule,
};
use guitar_pitch::tracking::{validate_masp_with_playhead, TablatureNoteEvent};
use guitar_pitch::types::{AudioBuffer, MonoAnnotation, PolyEvent};

fn synth_sine(midi: u8, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    let freq = 440.0_f32 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0);
    let len = (sample_rate as f32 * duration_sec).round() as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let t = i as f32 / sample_rate as f32;
        out.push((2.0 * PI * freq * t).sin() * 0.2);
    }
    out
}

fn write_wav(path: &std::path::Path, sample_rate: u32, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
    for &s in samples {
        writer.write_sample(s).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
}

fn temp_file(name: &str, ext: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "ghpitch_masp_{}_{}_{}.{}",
        name,
        std::process::id(),
        1,
        ext
    ));
    p
}

#[test]
fn validate_masp_accepts_detected_and_rejects_suboctave_decoy() {
    let sample_rate = 44_100;
    let audio = AudioBuffer {
        sample_rate,
        samples: synth_sine(60, sample_rate, 0.8),
    };

    let mut cfg = AppConfig::default();
    cfg.pitch.backend = "masp".to_string();
    cfg.masp.mode = "strict".to_string();
    cfg.masp.strict_sample_rate = 22_050;
    cfg.masp.cent_tolerance = 200.0;

    let dim = usize::from(cfg.pitch.midi_max - cfg.pitch.midi_min) + 1;
    let artifacts = MaspPretrainArtifacts {
        manifest: MaspManifest {
            version: 1,
            config_hash: 0,
            model_params: MaspModelParams {
                mode: "strict".to_string(),
                strict_sample_rate: cfg.masp.strict_sample_rate,
                bins_per_octave: cfg.masp.bins_per_octave,
                max_harmonics: cfg.masp.max_harmonics,
                b_exponent: cfg.masp.b_exponent,
                cent_tolerance: cfg.masp.cent_tolerance,
                rms_window_ms: cfg.masp.rms_window_ms,
                rms_h_relax: cfg.masp.rms_h_relax,
                pretrain_trials: 1,
            },
            validation_rule: MaspValidationRule {
                score_threshold: 0.0,
                score_weights: cfg.masp.weights.clone(),
                target_mono_recall: 0.0,
                target_comp_recall: 0.0,
                calibrated_precision: None,
                calibrated_mono_recall: None,
                calibrated_comp_recall: None,
            },
            mode: "strict".to_string(),
            strict_sample_rate: cfg.masp.strict_sample_rate,
            bins_per_octave: cfg.masp.bins_per_octave,
            max_harmonics: cfg.masp.max_harmonics,
            b_exponent: cfg.masp.b_exponent,
            cent_tolerance: cfg.masp.cent_tolerance,
            rms_window_ms: cfg.masp.rms_window_ms,
            rms_h_relax: cfg.masp.rms_h_relax,
            validation_score_threshold: 0.0,
            pretrain_trials: 1,
            target_mono_recall: 0.0,
            target_comp_recall: 0.0,
            calibrated_precision: None,
            calibrated_mono_recall: None,
            calibrated_comp_recall: None,
            weights: cfg.masp.weights.clone(),
        },
        note_signatures: vec![MaspNoteSignature {
            key: "s0_f0_m60".to_string(),
            midi: 60,
            string: 0,
            fret: 20,
            count: 1,
            mean_pitch_spectrum: vec![0.0; dim],
            h_target: 0.0,
            har_target: 0.5,
            mbw_target: 0.5,
        }],
        joint_signatures: vec![],
    };

    let scored = score_pitch_frames_masp(&audio, &cfg);
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for frame in &scored {
        for (idx, score) in frame.midi_scores.iter().copied().enumerate() {
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
    }
    let detected_midi = cfg.pitch.midi_min + best_idx as u8;
    let mut decoy_midi = detected_midi.saturating_sub(12).max(cfg.pitch.midi_min);
    if decoy_midi == detected_midi {
        decoy_midi = detected_midi.saturating_add(12).min(cfg.pitch.midi_max);
    }

    let ok = validate_expected_segment(&audio, 0.0, 0.8, &[detected_midi], &cfg, &artifacts)
        .expect("validate expected");
    assert!(ok.pass, "detected note should pass");

    let bad = validate_expected_segment(&audio, 0.0, 0.8, &[decoy_midi], &cfg, &artifacts)
        .expect("validate wrong");
    assert!(
        ok.weighted_score > bad.weighted_score,
        "detected note should score above sub-octave decoy"
    );
}

#[test]
fn validate_masp_with_playhead_resolves_tablature_note() {
    let sample_rate = 44_100;
    let audio = AudioBuffer {
        sample_rate,
        samples: synth_sine(60, sample_rate, 0.8),
    };

    let mut cfg = AppConfig::default();
    cfg.pitch.backend = "masp".to_string();
    cfg.masp.mode = "strict".to_string();
    cfg.masp.strict_sample_rate = 22_050;
    cfg.masp.cent_tolerance = 200.0;

    let dim = usize::from(cfg.pitch.midi_max - cfg.pitch.midi_min) + 1;
    let artifacts = MaspPretrainArtifacts {
        manifest: MaspManifest {
            version: 1,
            config_hash: 0,
            model_params: MaspModelParams {
                mode: "strict".to_string(),
                strict_sample_rate: cfg.masp.strict_sample_rate,
                bins_per_octave: cfg.masp.bins_per_octave,
                max_harmonics: cfg.masp.max_harmonics,
                b_exponent: cfg.masp.b_exponent,
                cent_tolerance: cfg.masp.cent_tolerance,
                rms_window_ms: cfg.masp.rms_window_ms,
                rms_h_relax: cfg.masp.rms_h_relax,
                pretrain_trials: 1,
            },
            validation_rule: MaspValidationRule {
                score_threshold: 0.0,
                score_weights: cfg.masp.weights.clone(),
                target_mono_recall: 0.0,
                target_comp_recall: 0.0,
                calibrated_precision: None,
                calibrated_mono_recall: None,
                calibrated_comp_recall: None,
            },
            mode: "strict".to_string(),
            strict_sample_rate: cfg.masp.strict_sample_rate,
            bins_per_octave: cfg.masp.bins_per_octave,
            max_harmonics: cfg.masp.max_harmonics,
            b_exponent: cfg.masp.b_exponent,
            cent_tolerance: cfg.masp.cent_tolerance,
            rms_window_ms: cfg.masp.rms_window_ms,
            rms_h_relax: cfg.masp.rms_h_relax,
            validation_score_threshold: 0.0,
            pretrain_trials: 1,
            target_mono_recall: 0.0,
            target_comp_recall: 0.0,
            calibrated_precision: None,
            calibrated_mono_recall: None,
            calibrated_comp_recall: None,
            weights: cfg.masp.weights.clone(),
        },
        note_signatures: vec![MaspNoteSignature {
            key: "s1_f15_m60".to_string(),
            midi: 60,
            string: 1,
            fret: 15,
            count: 1,
            mean_pitch_spectrum: vec![0.0; dim],
            h_target: 0.0,
            har_target: 0.5,
            mbw_target: 0.5,
        }],
        joint_signatures: vec![],
    };

    let tab = vec![TablatureNoteEvent {
        string: 1,
        fret: 15,
        onset_sec: 0.1,
        offset_sec: 0.6,
        midi: Some(60),
    }];
    let result =
        validate_masp_with_playhead(&audio, 0.3, &tab, &cfg, &artifacts).expect("validate");
    assert_eq!(result.segment.expected_midis, vec![60]);
    assert_eq!(result.segment.start_sec, 0.1);
    assert_eq!(result.segment.end_sec, 0.6);
    assert!(result.validation.pass);
}

#[test]
fn pretrain_roundtrip_writes_and_loads_json_artifacts() {
    let wav_path = temp_file("pretrain", "wav");
    let out_dir = temp_file("pretrain_out", "dir");
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let sample_rate = 44_100;
    let samples = synth_sine(60, sample_rate, 0.8);
    write_wav(&wav_path, sample_rate, &samples);

    let mono = vec![MonoAnnotation {
        audio_path: wav_path.to_string_lossy().to_string(),
        sample_rate,
        midi: 60,
        string: 1,
        fret: 15,
        onset_sec: 0.0,
        offset_sec: 0.7,
        player_id: None,
        guitar_id: None,
        style: None,
        pickup_mode: None,
    }];
    let comp = vec![PolyEvalEvent {
        audio_path: wav_path.to_string_lossy().to_string(),
        sample_rate,
        onset_sec: 0.0,
        offset_sec: 0.7,
        notes: vec![PolyEvent {
            onset_sec: 0.0,
            offset_sec: 0.7,
            midi: 60,
            string: Some(1),
            fret: Some(15),
        }],
    }];

    let cfg = AppConfig::default();
    let trained = pretrain_from_guitarset(&cfg, &mono, &comp, 0.99, 0.70).expect("pretrain");
    assert!(!trained.note_signatures.is_empty());
    assert!(!trained.joint_signatures.is_empty());

    write_pretrain_artifacts(&out_dir, &trained).expect("write artifacts");
    let loaded = load_pretrain_artifacts(&out_dir).expect("load artifacts");
    assert_eq!(loaded.note_signatures.len(), trained.note_signatures.len());
    assert_eq!(
        loaded.joint_signatures.len(),
        trained.joint_signatures.len()
    );

    let _ = std::fs::remove_file(wav_path);
    let _ = std::fs::remove_file(out_dir.join("masp_manifest.json"));
    let _ = std::fs::remove_file(out_dir.join("masp_note_signatures.jsonl"));
    let _ = std::fs::remove_file(out_dir.join("masp_joint_signatures.jsonl"));
    let _ = std::fs::remove_dir(out_dir);
}

#[test]
fn validation_summary_counts_false_accept_and_false_reject() {
    let out_dir = temp_file("summary", "dir");
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let results = vec![
        MaspValidationResult {
            id: Some("a".to_string()),
            audio_path: "a.wav".to_string(),
            expected_midis: vec![60],
            pass: true,
            reason: "validated".to_string(),
            h_real: 1.0,
            h_target: 1.0,
            h_dynamic_threshold: 0.8,
            har: 0.8,
            mbw: 0.6,
            cent_error: 0.0,
            cent_tolerance: 50.0,
            noise_rms: 0.1,
            weighted_score: 0.7,
            score_threshold: 0.5,
            execution_ms: Some(1.2),
            expected_valid: Some(false),
            false_accept: true,
            false_reject: false,
        },
        MaspValidationResult {
            id: Some("b".to_string()),
            audio_path: "a.wav".to_string(),
            expected_midis: vec![62],
            pass: false,
            reason: "cent_gate_failed".to_string(),
            h_real: 0.4,
            h_target: 1.0,
            h_dynamic_threshold: 0.8,
            har: 0.2,
            mbw: 0.1,
            cent_error: 1200.0,
            cent_tolerance: 50.0,
            noise_rms: 0.1,
            weighted_score: 0.1,
            score_threshold: 0.5,
            execution_ms: Some(1.8),
            expected_valid: Some(true),
            false_accept: false,
            false_reject: true,
        },
    ];

    let summary = write_validation_artifacts(&out_dir, &results).expect("write summary");
    assert_eq!(summary.total, 2);
    assert_eq!(summary.false_accept_count, 1);
    assert_eq!(summary.false_reject_count, 1);
    assert_eq!(summary.labeled_total, 2);

    let _ = std::fs::remove_file(out_dir.join("masp_validation_results.jsonl"));
    let _ = std::fs::remove_file(out_dir.join("masp_validation_summary.json"));
    let _ = std::fs::remove_dir(out_dir);
}
