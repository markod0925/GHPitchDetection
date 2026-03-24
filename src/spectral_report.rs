use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use num::complex::Complex64;
use qdft::QDFT;

use crate::audio::{load_wav_mono, trim_audio_region};
use crate::config::AppConfig;
use crate::dataset::load_mono_annotations;
use crate::stft::{build_stft_plan, compute_stft_frames, StftPlan};
use crate::types::{MonoAnnotation, SpectrumFrame};

const NOTE_CONTEXT_MARGIN_SEC: f32 = 0.005;
const FFT_EPSILON: f32 = 1e-8;

const MAX_SPECTROGRAM_TIME_BINS: usize = 160;
const MAX_STFT_FREQ_BINS: usize = 120;
const MAX_CQT_FREQ_BINS: usize = 140;
const STFT_MAX_FREQ_HZ: f32 = 4000.0;

const CQT_MIN_FREQ_HZ: f64 = 75.0;
const CQT_MAX_FREQ_HZ: f64 = 4000.0;
const CQT_BINS_PER_OCTAVE: f64 = 24.0;
const CQT_SAMPLE_RATE_HZ: f64 = 44_100.0;
const CQT_WINDOW_LENGTH_SAMPLES: f64 = 4096.0;
const CQT_WINDOW: Option<(f64, f64)> = Some((0.5, -0.5)); // Hann

#[derive(Debug, Clone)]
pub struct SoloStringSpectrogramReportSummary {
    pub string_index: u8,
    pub notes_requested: usize,
    pub notes_rendered: usize,
    pub notes_skipped: usize,
    pub report_path: PathBuf,
    pub figures_dir: PathBuf,
}

#[derive(Debug, Clone)]
struct IndexedNote {
    sequence_id: usize,
    note: MonoAnnotation,
}

#[derive(Debug, Clone, Copy)]
struct PanelGeometry {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

#[derive(Debug, Clone)]
struct CqtSpectrogram {
    db: Vec<Vec<f32>>,
    freqs_hz: Vec<f32>,
    raw_bin_count: usize,
}

pub fn build_solo_string_spectrogram_report(
    cfg: &AppConfig,
    mono_dataset_path: &str,
    dataset_root: &Path,
    out_dir: &Path,
    string_index: u8,
    midi_filter: Option<u8>,
    max_notes: Option<usize>,
) -> Result<SoloStringSpectrogramReportSummary> {
    let mut notes = load_mono_annotations(mono_dataset_path, dataset_root, true)?;
    notes.retain(|note| {
        note.string == string_index && midi_filter.is_none_or(|midi| note.midi == midi)
    });
    if notes.is_empty() {
        if let Some(midi) = midi_filter {
            bail!(
                "no SOLO notes found for string {} and midi {} in {}",
                string_index,
                midi,
                mono_dataset_path
            );
        }
        bail!(
            "no SOLO notes found for string {} in {}",
            string_index,
            mono_dataset_path
        );
    }

    notes.sort_by(|a, b| {
        a.audio_path
            .cmp(&b.audio_path)
            .then_with(|| a.onset_sec.total_cmp(&b.onset_sec))
            .then_with(|| a.offset_sec.total_cmp(&b.offset_sec))
            .then_with(|| a.midi.cmp(&b.midi))
            .then_with(|| a.fret.cmp(&b.fret))
    });

    if let Some(limit) = max_notes {
        notes.truncate(limit);
    }

    let mut grouped: BTreeMap<String, Vec<IndexedNote>> = BTreeMap::new();
    for (idx, note) in notes.into_iter().enumerate() {
        grouped
            .entry(note.audio_path.clone())
            .or_default()
            .push(IndexedNote {
                sequence_id: idx + 1,
                note,
            });
    }

    let report_dir_name = if let Some(midi) = midi_filter {
        format!(
            "solo_string_{}_midi_{}_spectrogram_report",
            usize::from(string_index),
            midi
        )
    } else {
        format!(
            "solo_string_{}_spectrogram_report",
            usize::from(string_index)
        )
    };
    let report_dir = out_dir.join(report_dir_name);
    let figures_dir = report_dir.join("figures");
    create_dir_all(&figures_dir).with_context(|| {
        format!(
            "failed to create report figures directory: {}",
            figures_dir.display()
        )
    })?;

    let stft_plan = build_stft_plan(&cfg.frontend);

    let mut notes_html = String::new();
    let mut notes_requested = 0usize;
    let mut notes_rendered = 0usize;
    let mut notes_skipped = 0usize;

    for (audio_path, mut file_notes) in grouped {
        file_notes.sort_by(|a, b| {
            a.note
                .onset_sec
                .total_cmp(&b.note.onset_sec)
                .then_with(|| a.note.offset_sec.total_cmp(&b.note.offset_sec))
                .then_with(|| a.note.midi.cmp(&b.note.midi))
                .then_with(|| a.note.fret.cmp(&b.note.fret))
        });

        let audio = load_wav_mono(&audio_path)
            .with_context(|| format!("failed to load note source audio: {audio_path}"))?;

        for indexed_note in file_notes {
            notes_requested += 1;

            match render_note_spectrogram_card(
                &indexed_note,
                &audio_path,
                &audio.samples,
                audio.sample_rate,
                cfg,
                &stft_plan,
                &figures_dir,
            )? {
                Some(card_html) => {
                    notes_rendered += 1;
                    notes_html.push_str(&card_html);
                }
                None => {
                    notes_skipped += 1;
                }
            }
        }
    }

    let report_html = render_html_report(
        string_index,
        midi_filter,
        notes_requested,
        notes_rendered,
        notes_skipped,
        &notes_html,
    );
    let report_path = report_dir.join("index.html");
    std::fs::write(&report_path, report_html)
        .with_context(|| format!("failed to write report html: {}", report_path.display()))?;

    Ok(SoloStringSpectrogramReportSummary {
        string_index,
        notes_requested,
        notes_rendered,
        notes_skipped,
        report_path,
        figures_dir,
    })
}

fn render_note_spectrogram_card(
    indexed_note: &IndexedNote,
    audio_path: &str,
    audio_samples: &[f32],
    sample_rate: u32,
    cfg: &AppConfig,
    stft_plan: &StftPlan,
    figures_dir: &Path,
) -> Result<Option<String>> {
    if sample_rate != cfg.sample_rate_target {
        eprintln!(
            "warning: skipped note {} due to sample rate mismatch (audio={}Hz config={}Hz)",
            indexed_note.sequence_id, sample_rate, cfg.sample_rate_target
        );
        return Ok(None);
    }
    if sample_rate as f64 != CQT_SAMPLE_RATE_HZ {
        eprintln!(
            "warning: skipped note {} because CQT is configured for {}Hz and audio is {}Hz",
            indexed_note.sequence_id, CQT_SAMPLE_RATE_HZ as u32, sample_rate
        );
        return Ok(None);
    }

    let start_sec = (indexed_note.note.onset_sec - NOTE_CONTEXT_MARGIN_SEC).max(0.0);
    let end_sec = (indexed_note.note.offset_sec + NOTE_CONTEXT_MARGIN_SEC).max(start_sec);

    let mut segment = trim_audio_region(audio_samples, sample_rate, start_sec, end_sec);
    if segment.is_empty() {
        return Ok(None);
    }
    if segment.len() < stft_plan.win_length {
        segment.resize(stft_plan.win_length, 0.0);
    }

    let stft_frames = compute_stft_frames(&segment, sample_rate, stft_plan);
    if stft_frames.is_empty() {
        return Ok(None);
    }

    let segment_duration_sec = segment.len() as f32 / sample_rate as f32;
    let stft_time_bins = stft_frames.len().min(MAX_SPECTROGRAM_TIME_BINS).max(1);
    let cqt = compute_cqt_spectrogram_db(&segment, sample_rate, stft_time_bins, MAX_CQT_FREQ_BINS)?;

    let figure_file_name = note_figure_file_name(
        indexed_note.sequence_id,
        &indexed_note.note,
        audio_path,
        "svg",
    );
    let figure_path = figures_dir.join(&figure_file_name);
    write_note_svg(
        &figure_path,
        &indexed_note.note,
        indexed_note.sequence_id,
        audio_path,
        sample_rate,
        stft_plan,
        &stft_frames,
        &cqt,
        segment_duration_sec,
    )?;

    let mut html = String::new();
    write!(
        &mut html,
        "<section class=\"card\"><h2>Note #{}</h2>\
         <p><strong>File:</strong> {}<br>\
         <strong>String:</strong> {} | <strong>MIDI:</strong> {} | <strong>Fret:</strong> {}<br>\
         <strong>Onset/Offset:</strong> {:.6}s / {:.6}s<br>\
         <strong>STFT frames/bins:</strong> {} / {} | \
         <strong>CQT bins (raw/downsampled):</strong> {} / {}</p>\
         <img loading=\"lazy\" src=\"figures/{}\" alt=\"Spectrogram note {}\">\
         </section>",
        indexed_note.sequence_id,
        escape_html(audio_path),
        indexed_note.note.string,
        indexed_note.note.midi,
        indexed_note.note.fret,
        indexed_note.note.onset_sec,
        indexed_note.note.offset_sec,
        stft_frames.len(),
        stft_frames[0].mag.len(),
        cqt.raw_bin_count,
        cqt.db.len(),
        escape_html(&figure_file_name),
        indexed_note.sequence_id
    )
    .ok();

    Ok(Some(html))
}

#[allow(clippy::too_many_arguments)]
fn write_note_svg(
    path: &Path,
    note: &MonoAnnotation,
    sequence_id: usize,
    audio_path: &str,
    sample_rate: u32,
    stft_plan: &StftPlan,
    stft_frames: &[SpectrumFrame],
    cqt: &CqtSpectrogram,
    segment_duration_sec: f32,
) -> Result<()> {
    let width = 1280.0_f32;
    let height = 1360.0_f32;

    let fft_panel = PanelGeometry {
        x: 80.0,
        y: 110.0,
        w: 1120.0,
        h: 230.0,
    };
    let stft_panel = PanelGeometry {
        x: 80.0,
        y: 420.0,
        w: 1120.0,
        h: 360.0,
    };
    let cqt_panel = PanelGeometry {
        x: 80.0,
        y: 860.0,
        w: 1120.0,
        h: 360.0,
    };

    let stft_bin_hz = sample_rate as f32 / stft_plan.nfft as f32;
    let stft_max_bin = ((STFT_MAX_FREQ_HZ / stft_bin_hz).floor() as usize)
        .min(stft_frames[0].mag.len().saturating_sub(1));
    let stft_freq_axis_max_hz = stft_max_bin as f32 * stft_bin_hz;

    let fft_db = average_fft_db_limited(stft_frames, stft_max_bin);
    let cqt_fft_db = mean_spectrum_over_time(&cqt.db);
    let (stft_fft_min, stft_fft_max) = finite_min_max(&fft_db).unwrap_or((-120.0, 0.0));
    let (cqt_fft_min, cqt_fft_max) = finite_min_max(&cqt_fft_db).unwrap_or((-120.0, 0.0));
    let fft_min = stft_fft_min.min(cqt_fft_min);
    let fft_max = stft_fft_max.max(cqt_fft_max);
    let fft_range = (fft_max - fft_min).max(1e-6);

    let stft_time_bins = stft_frames.len().min(MAX_SPECTROGRAM_TIME_BINS).max(1);
    let stft_freq_bins = (stft_max_bin + 1).min(MAX_STFT_FREQ_BINS).max(1);
    let stft_spectrogram_db = downsample_spectrogram_db_limited(
        stft_frames,
        stft_time_bins,
        stft_freq_bins,
        stft_max_bin,
    );
    let (stft_min, stft_max) = finite_min_max_2d(&stft_spectrogram_db).unwrap_or((-120.0, 0.0));
    let stft_range = (stft_max - stft_min).max(1e-6);

    let (cqt_min, cqt_max) = finite_min_max_2d(&cqt.db).unwrap_or((-120.0, 0.0));
    let cqt_range = (cqt_max - cqt_min).max(1e-6);

    let mut svg = String::new();
    write!(
        &mut svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0}\" height=\"{:.0}\" viewBox=\"0 0 {:.0} {:.0}\">",
        width, height, width, height
    )
    .ok();
    svg.push_str(
        "<style>\
         .title{font:700 26px 'Segoe UI',sans-serif;fill:#0f172a;}\
         .subtitle{font:500 14px 'Segoe UI',sans-serif;fill:#334155;}\
         .axis{font:500 12px 'Segoe UI',sans-serif;fill:#334155;}\
         .box{fill:#ffffff;stroke:#cbd5e1;stroke-width:1;}\
         .grid{stroke:#e2e8f0;stroke-width:1;}\
         .line-stft{fill:none;stroke:#0369a1;stroke-width:2;}\
         .line-cqt{fill:none;stroke:#dc2626;stroke-width:2;}\
         </style>",
    );
    svg.push_str("<rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"#f8fafc\"/>");

    write!(
        &mut svg,
        "<text class=\"title\" x=\"{}\" y=\"45\">String {} note #{} (STFT vs CQT)</text>",
        fft_panel.x, note.string, sequence_id
    )
    .ok();
    write!(
        &mut svg,
        "<text class=\"subtitle\" x=\"{}\" y=\"70\">file={} | midi={} | fret={} | onset={:.6}s | offset={:.6}s | sample_rate={}Hz</text>",
        fft_panel.x,
        escape_xml(audio_path),
        note.midi,
        note.fret,
        note.onset_sec,
        note.offset_sec,
        sample_rate
    )
    .ok();
    write!(
        &mut svg,
        "<text class=\"subtitle\" x=\"{}\" y=\"92\">STFT: win={} hop={} nfft={} (Hann, 0-{}Hz) | CQT(qdft): min={}Hz max={}Hz bins/oct={} fs={}Hz win={} (Hann)</text>",
        fft_panel.x,
        stft_plan.win_length,
        stft_plan.hop_length,
        stft_plan.nfft,
        stft_freq_axis_max_hz.round() as u32,
        CQT_MIN_FREQ_HZ as u32,
        CQT_MAX_FREQ_HZ as u32,
        CQT_BINS_PER_OCTAVE as u32,
        CQT_SAMPLE_RATE_HZ as u32,
        CQT_WINDOW_LENGTH_SAMPLES as u32
    )
    .ok();

    draw_box(&mut svg, fft_panel);
    for tick in 0..=5 {
        let ratio = tick as f32 / 5.0;
        let x = fft_panel.x + ratio * fft_panel.w;
        draw_grid_vline(&mut svg, x, fft_panel.y, fft_panel.y + fft_panel.h);
        let freq = ratio * stft_freq_axis_max_hz;
        draw_axis_text(
            &mut svg,
            x,
            fft_panel.y + fft_panel.h + 18.0,
            "middle",
            &format!("{freq:.0} Hz"),
        );
    }
    for tick in 0..=4 {
        let ratio = tick as f32 / 4.0;
        let y = fft_panel.y + ratio * fft_panel.h;
        draw_grid_hline(&mut svg, fft_panel.x, fft_panel.x + fft_panel.w, y);
        let db = fft_max - ratio * fft_range;
        draw_axis_text(
            &mut svg,
            fft_panel.x - 8.0,
            y + 4.0,
            "end",
            &format!("{db:.1} dB"),
        );
    }

    let mut stft_line_path = String::new();
    let fft_len = fft_db.len().max(1);
    for (i, value_db) in fft_db.iter().enumerate() {
        let ratio_x = if fft_len > 1 {
            i as f32 / (fft_len - 1) as f32
        } else {
            0.0
        };
        let x = fft_panel.x + ratio_x * fft_panel.w;
        let norm = ((value_db - fft_min) / fft_range).clamp(0.0, 1.0);
        let y = fft_panel.y + (1.0 - norm) * fft_panel.h;
        if i == 0 {
            write!(&mut stft_line_path, "M{:.2},{:.2}", x, y).ok();
        } else {
            write!(&mut stft_line_path, " L{:.2},{:.2}", x, y).ok();
        }
    }
    write!(
        &mut svg,
        "<path class=\"line-stft\" d=\"{}\"/>",
        stft_line_path
    )
    .ok();

    if !cqt_fft_db.is_empty() && !cqt.freqs_hz.is_empty() {
        let n = cqt_fft_db.len().min(cqt.freqs_hz.len());
        let mut cqt_line_path = String::new();
        for i in 0..n {
            let ratio_x = (cqt.freqs_hz[i] / stft_freq_axis_max_hz).clamp(0.0, 1.0);
            let x = fft_panel.x + ratio_x * fft_panel.w;
            let norm = ((cqt_fft_db[i] - fft_min) / fft_range).clamp(0.0, 1.0);
            let y = fft_panel.y + (1.0 - norm) * fft_panel.h;
            if i == 0 {
                write!(&mut cqt_line_path, "M{:.2},{:.2}", x, y).ok();
            } else {
                write!(&mut cqt_line_path, " L{:.2},{:.2}", x, y).ok();
            }
        }
        write!(
            &mut svg,
            "<path class=\"line-cqt\" d=\"{}\"/>",
            cqt_line_path
        )
        .ok();
    }

    write!(
        &mut svg,
        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#0369a1\" stroke-width=\"2\"/>",
        fft_panel.x + 8.0,
        fft_panel.y - 24.0,
        fft_panel.x + 28.0,
        fft_panel.y - 24.0
    )
    .ok();
    draw_axis_text(
        &mut svg,
        fft_panel.x + 34.0,
        fft_panel.y - 20.0,
        "start",
        "Mean FFT from STFT",
    );
    write!(
        &mut svg,
        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#dc2626\" stroke-width=\"2\"/>",
        fft_panel.x + 220.0,
        fft_panel.y - 24.0,
        fft_panel.x + 240.0,
        fft_panel.y - 24.0
    )
    .ok();
    draw_axis_text(
        &mut svg,
        fft_panel.x + 246.0,
        fft_panel.y - 20.0,
        "start",
        "Mean FFT from CQT",
    );

    draw_label(
        &mut svg,
        fft_panel.x,
        fft_panel.y - 10.0,
        "Mean FFT magnitude in dB (STFT + CQT, 0-4000 Hz)",
    );
    draw_axis_text(
        &mut svg,
        fft_panel.x + fft_panel.w / 2.0,
        fft_panel.y + fft_panel.h + 40.0,
        "middle",
        "Frequency (Hz)",
    );

    draw_box(&mut svg, stft_panel);
    draw_spectrogram_cells(
        &mut svg,
        stft_panel,
        &stft_spectrogram_db,
        stft_min,
        stft_range,
    );
    draw_time_axis(&mut svg, stft_panel, segment_duration_sec);
    draw_linear_frequency_axis(&mut svg, stft_panel, stft_freq_axis_max_hz);
    draw_label(
        &mut svg,
        stft_panel.x,
        stft_panel.y - 10.0,
        "STFT spectrogram (log-magnitude dB, 0-4000 Hz)",
    );
    draw_axis_text(
        &mut svg,
        stft_panel.x + stft_panel.w / 2.0,
        stft_panel.y + stft_panel.h + 40.0,
        "middle",
        "Time (s)",
    );

    draw_box(&mut svg, cqt_panel);
    draw_spectrogram_cells(&mut svg, cqt_panel, &cqt.db, cqt_min, cqt_range);
    draw_time_axis(&mut svg, cqt_panel, segment_duration_sec);
    draw_frequency_axis_from_values(&mut svg, cqt_panel, &cqt.freqs_hz);
    draw_label(
        &mut svg,
        cqt_panel.x,
        cqt_panel.y - 10.0,
        "CQT spectrogram via qdft (log-magnitude dB)",
    );
    draw_axis_text(
        &mut svg,
        cqt_panel.x + cqt_panel.w / 2.0,
        cqt_panel.y + cqt_panel.h + 40.0,
        "middle",
        "Time (s)",
    );

    svg.push_str("</svg>");
    std::fs::write(path, svg)
        .with_context(|| format!("failed to write note spectrogram: {}", path.display()))?;
    Ok(())
}

fn compute_cqt_spectrogram_db(
    samples: &[f32],
    sample_rate: u32,
    out_time_bins: usize,
    out_freq_bins: usize,
) -> Result<CqtSpectrogram> {
    if sample_rate as f64 != CQT_SAMPLE_RATE_HZ {
        bail!(
            "CQT is configured for {}Hz, but segment is {}Hz",
            CQT_SAMPLE_RATE_HZ as u32,
            sample_rate
        );
    }
    if samples.is_empty() {
        bail!("cannot compute CQT on empty sample segment");
    }

    // Approximate the requested window length (4096 samples) at the minimum frequency.
    let quality = CQT_MIN_FREQ_HZ * CQT_WINDOW_LENGTH_SAMPLES / CQT_SAMPLE_RATE_HZ;

    let mut qdft = QDFT::<f32, f64>::new(
        CQT_SAMPLE_RATE_HZ,
        (CQT_MIN_FREQ_HZ, CQT_MAX_FREQ_HZ),
        CQT_BINS_PER_OCTAVE,
        quality,
        CQT_WINDOW,
    );

    let src_freq_bins = qdft.size();
    if src_freq_bins == 0 {
        bail!("CQT returned zero bins");
    }

    let time_bins = out_time_bins.max(1);
    let freq_bins = out_freq_bins.min(src_freq_bins).max(1);

    let mut sum = vec![vec![0.0_f64; time_bins]; freq_bins];
    let mut count = vec![vec![0usize; time_bins]; freq_bins];
    let mut frame = vec![Complex64::new(0.0, 0.0); src_freq_bins];
    let sample_count = samples.len();

    for (sample_idx, &sample) in samples.iter().enumerate() {
        qdft.qdft_scalar(&sample, &mut frame);

        let out_t = (sample_idx * time_bins / sample_count).min(time_bins - 1);
        for (src_f, coeff) in frame.iter().enumerate() {
            let out_f = (src_f * freq_bins / src_freq_bins).min(freq_bins - 1);
            sum[out_f][out_t] += coeff.norm();
            count[out_f][out_t] += 1;
        }
    }

    let mut db = vec![vec![-120.0_f32; time_bins]; freq_bins];
    for out_f in 0..freq_bins {
        for out_t in 0..time_bins {
            let avg = if count[out_f][out_t] > 0 {
                sum[out_f][out_t] / count[out_f][out_t] as f64
            } else {
                0.0
            };
            db[out_f][out_t] = 20.0 * (avg as f32).max(FFT_EPSILON).log10();
        }
    }

    let freqs_hz = downsample_frequency_axis(qdft.frequencies(), freq_bins);

    Ok(CqtSpectrogram {
        db,
        freqs_hz,
        raw_bin_count: src_freq_bins,
    })
}

fn draw_box(svg: &mut String, panel: PanelGeometry) {
    write!(
        svg,
        "<rect class=\"box\" x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\"/>",
        panel.x, panel.y, panel.w, panel.h
    )
    .ok();
}

fn draw_grid_vline(svg: &mut String, x: f32, y1: f32, y2: f32) {
    write!(
        svg,
        "<line class=\"grid\" x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\"/>",
        x, y1, x, y2
    )
    .ok();
}

fn draw_grid_hline(svg: &mut String, x1: f32, x2: f32, y: f32) {
    write!(
        svg,
        "<line class=\"grid\" x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\"/>",
        x1, y, x2, y
    )
    .ok();
}

fn draw_axis_text(svg: &mut String, x: f32, y: f32, anchor: &str, label: &str) {
    write!(
        svg,
        "<text class=\"axis\" x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"{}\">{}</text>",
        x,
        y,
        anchor,
        escape_xml(label)
    )
    .ok();
}

fn draw_label(svg: &mut String, x: f32, y: f32, label: &str) {
    write!(
        svg,
        "<text class=\"axis\" x=\"{:.2}\" y=\"{:.2}\">{}</text>",
        x,
        y,
        escape_xml(label)
    )
    .ok();
}

fn draw_time_axis(svg: &mut String, panel: PanelGeometry, segment_duration_sec: f32) {
    for tick in 0..=5 {
        let ratio = tick as f32 / 5.0;
        let x = panel.x + ratio * panel.w;
        let time_sec = ratio * segment_duration_sec;
        draw_grid_vline(svg, x, panel.y, panel.y + panel.h);
        draw_axis_text(
            svg,
            x,
            panel.y + panel.h + 18.0,
            "middle",
            &format!("{time_sec:.3}s"),
        );
    }
}

fn draw_linear_frequency_axis(svg: &mut String, panel: PanelGeometry, max_hz: f32) {
    for tick in 0..=4 {
        let ratio = tick as f32 / 4.0;
        let y = panel.y + ratio * panel.h;
        let hz = max_hz * (1.0 - ratio);
        draw_grid_hline(svg, panel.x, panel.x + panel.w, y);
        draw_axis_text(svg, panel.x - 8.0, y + 4.0, "end", &format!("{hz:.0} Hz"));
    }
}

fn draw_frequency_axis_from_values(svg: &mut String, panel: PanelGeometry, freqs_hz: &[f32]) {
    if freqs_hz.is_empty() {
        draw_linear_frequency_axis(svg, panel, CQT_MAX_FREQ_HZ as f32);
        return;
    }

    for tick in 0..=4 {
        let ratio = tick as f32 / 4.0;
        let y = panel.y + ratio * panel.h;
        let idx_f = (1.0 - ratio) * (freqs_hz.len() - 1) as f32;
        let idx = idx_f.round().clamp(0.0, (freqs_hz.len() - 1) as f32) as usize;
        let hz = freqs_hz[idx];
        draw_grid_hline(svg, panel.x, panel.x + panel.w, y);
        draw_axis_text(svg, panel.x - 8.0, y + 4.0, "end", &format!("{hz:.0} Hz"));
    }
}

fn draw_spectrogram_cells(
    svg: &mut String,
    panel: PanelGeometry,
    spectrogram_db: &[Vec<f32>],
    min_db: f32,
    range_db: f32,
) {
    let freq_bins = spectrogram_db.len().max(1);
    let time_bins = spectrogram_db
        .first()
        .map(|row| row.len())
        .unwrap_or(1)
        .max(1);

    let cell_w = panel.w / time_bins as f32;
    let cell_h = panel.h / freq_bins as f32;

    for time_idx in 0..time_bins {
        for freq_idx in 0..freq_bins {
            let db = spectrogram_db
                .get(freq_idx)
                .and_then(|row| row.get(time_idx))
                .copied()
                .unwrap_or(min_db);
            let norm = ((db - min_db) / range_db).clamp(0.0, 1.0);
            let (r, g, b) = gradient_color(norm);
            let x = panel.x + time_idx as f32 * cell_w;
            let y = panel.y + (freq_bins - 1 - freq_idx) as f32 * cell_h;
            write!(
                svg,
                "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"rgb({},{},{})\"/>",
                x,
                y,
                cell_w.ceil().max(1.0),
                cell_h.ceil().max(1.0),
                r,
                g,
                b
            )
            .ok();
        }
    }
}

fn mean_spectrum_over_time(spectrogram_db: &[Vec<f32>]) -> Vec<f32> {
    if spectrogram_db.is_empty() {
        return Vec::new();
    }

    let time_bins = spectrogram_db[0].len().max(1);
    let mut out = vec![0.0_f32; spectrogram_db.len()];
    for (freq_idx, row) in spectrogram_db.iter().enumerate() {
        if row.is_empty() {
            continue;
        }
        let sum: f32 = row.iter().copied().sum();
        out[freq_idx] = sum / time_bins as f32;
    }
    out
}

fn average_fft_db_limited(frames: &[SpectrumFrame], max_bin_inclusive: usize) -> Vec<f32> {
    let bins = frames
        .first()
        .map(|f| f.mag.len().min(max_bin_inclusive + 1))
        .unwrap_or(0);
    if bins == 0 {
        return Vec::new();
    }

    let mut mean = vec![0.0_f32; bins];
    for frame in frames {
        for (bin, value) in frame.mag.iter().take(bins).enumerate() {
            mean[bin] += *value;
        }
    }
    let inv_n = 1.0 / frames.len() as f32;
    for value in &mut mean {
        *value = 20.0 * ((*value * inv_n).max(FFT_EPSILON)).log10();
    }
    mean
}

fn downsample_spectrogram_db_limited(
    frames: &[SpectrumFrame],
    out_time_bins: usize,
    out_freq_bins: usize,
    max_bin_inclusive: usize,
) -> Vec<Vec<f32>> {
    let src_time_bins = frames.len();
    let src_freq_bins = frames
        .first()
        .map(|f| f.mag.len().min(max_bin_inclusive + 1))
        .unwrap_or(0);
    let mut out = vec![vec![-120.0_f32; out_time_bins]; out_freq_bins];

    if src_time_bins == 0 || src_freq_bins == 0 {
        return out;
    }

    for out_f in 0..out_freq_bins {
        let src_f_start = out_f * src_freq_bins / out_freq_bins;
        let src_f_end = ((out_f + 1) * src_freq_bins / out_freq_bins)
            .max(src_f_start + 1)
            .min(src_freq_bins);

        for out_t in 0..out_time_bins {
            let src_t_start = out_t * src_time_bins / out_time_bins;
            let src_t_end = ((out_t + 1) * src_time_bins / out_time_bins)
                .max(src_t_start + 1)
                .min(src_time_bins);

            let mut sum = 0.0_f32;
            let mut count = 0usize;
            for frame in frames.iter().take(src_t_end).skip(src_t_start) {
                for value in frame.mag.iter().take(src_f_end).skip(src_f_start) {
                    sum += *value;
                    count += 1;
                }
            }

            let avg = if count > 0 { sum / count as f32 } else { 0.0 };
            out[out_f][out_t] = 20.0 * avg.max(FFT_EPSILON).log10();
        }
    }

    out
}

fn downsample_frequency_axis(src_hz: &[f64], out_bins: usize) -> Vec<f32> {
    if src_hz.is_empty() || out_bins == 0 {
        return Vec::new();
    }

    let mut sum = vec![0.0_f64; out_bins];
    let mut count = vec![0usize; out_bins];
    for (src_idx, &freq_hz) in src_hz.iter().enumerate() {
        let out_idx = (src_idx * out_bins / src_hz.len()).min(out_bins - 1);
        sum[out_idx] += freq_hz;
        count[out_idx] += 1;
    }

    let mut out = vec![0.0_f32; out_bins];
    for i in 0..out_bins {
        out[i] = if count[i] > 0 {
            (sum[i] / count[i] as f64) as f32
        } else if i > 0 {
            out[i - 1]
        } else {
            src_hz[0] as f32
        };
    }
    out
}

fn finite_min_max(values: &[f32]) -> Option<(f32, f32)> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &value in values {
        if !value.is_finite() {
            continue;
        }
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }
    if min.is_finite() && max.is_finite() {
        Some((min, max))
    } else {
        None
    }
}

fn finite_min_max_2d(values: &[Vec<f32>]) -> Option<(f32, f32)> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for row in values {
        for &value in row {
            if !value.is_finite() {
                continue;
            }
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }
    }
    if min.is_finite() && max.is_finite() {
        Some((min, max))
    } else {
        None
    }
}

fn gradient_color(value: f32) -> (u8, u8, u8) {
    let v = value.clamp(0.0, 1.0);
    let stops: [(f32, [u8; 3]); 5] = [
        (0.0, [9, 17, 83]),
        (0.25, [25, 104, 171]),
        (0.5, [43, 170, 138]),
        (0.75, [253, 231, 37]),
        (1.0, [180, 4, 38]),
    ];

    for i in 0..(stops.len() - 1) {
        let (left_pos, left_color) = stops[i];
        let (right_pos, right_color) = stops[i + 1];
        if v >= left_pos && v <= right_pos {
            let span = (right_pos - left_pos).max(1e-6);
            let t = (v - left_pos) / span;
            let r = lerp_u8(left_color[0], right_color[0], t);
            let g = lerp_u8(left_color[1], right_color[1], t);
            let b = lerp_u8(left_color[2], right_color[2], t);
            return (r, g, b);
        }
    }

    let last = stops.last().map(|(_, c)| *c).unwrap_or([255, 255, 255]);
    (last[0], last[1], last[2])
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let af = a as f32;
    let bf = b as f32;
    (af + (bf - af) * t.clamp(0.0, 1.0)).round() as u8
}

fn note_figure_file_name(
    sequence_id: usize,
    note: &MonoAnnotation,
    audio_path: &str,
    ext: &str,
) -> String {
    let file_id = Path::new(audio_path)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("audio");
    let clean_id = sanitize_filename(file_id);
    format!(
        "note_{:05}_s{}_m{}_f{}_{}.{}",
        sequence_id, note.string, note.midi, note.fret, clean_id, ext
    )
}

fn sanitize_filename(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_was_sep = false;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

fn render_html_report(
    string_index: u8,
    midi_filter: Option<u8>,
    notes_requested: usize,
    notes_rendered: usize,
    notes_skipped: usize,
    notes_html: &str,
) -> String {
    let mut html = String::new();
    html.push_str("<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
    html.push_str("<title>SOLO STFT vs CQT Spectrogram Report</title>");
    html.push_str(
        "<style>\
         body{font-family:'Segoe UI',sans-serif;background:#f1f5f9;color:#0f172a;margin:0;}\
         main{max-width:1320px;margin:0 auto;padding:28px 20px 40px;}\
         h1{margin:0 0 10px;font-size:32px;}\
         .summary{background:#e2e8f0;border:1px solid #cbd5e1;border-radius:14px;padding:14px 16px;margin:0 0 20px;}\
         .summary p{margin:6px 0;}\
         .grid{display:grid;grid-template-columns:1fr;gap:18px;}\
         .card{background:#ffffff;border:1px solid #dbeafe;border-radius:14px;padding:16px;box-shadow:0 1px 3px rgba(15,23,42,.08);}\
         .card h2{margin:0 0 8px;font-size:22px;}\
         .card p{margin:0 0 12px;line-height:1.45;}\
         img{width:100%;height:auto;border-radius:10px;border:1px solid #cbd5e1;background:#fff;}\
         </style>",
    );
    html.push_str("</head><body><main>");
    if let Some(midi) = midi_filter {
        write!(
            &mut html,
            "<h1>SOLO STFT vs CQT Report - String {} - MIDI {}</h1>",
            string_index, midi
        )
        .ok();
    } else {
        write!(
            &mut html,
            "<h1>SOLO STFT vs CQT Report - String {}</h1>",
            string_index
        )
        .ok();
    }
    write!(
        &mut html,
        "<section class=\"summary\"><p><strong>Notes requested:</strong> {}</p>\
         <p><strong>Notes rendered:</strong> {}</p>\
         <p><strong>Notes skipped:</strong> {}</p>\
         <p><strong>Filter:</strong> string={} {}</p>\
         <p><strong>STFT display range:</strong> 0-{}Hz.</p>\
         <p><strong>CQT parameters:</strong> min={}Hz, max={}Hz, bins/oct={}, sample_rate={}Hz, window={} samples.</p>\
         <p><strong>Each figure contains:</strong> mean STFT FFT + mean CQT FFT + STFT spectrogram + CQT spectrogram (qdft) for the same note.</p></section>",
        notes_requested,
        notes_rendered,
        notes_skipped,
        string_index,
        midi_filter
            .map(|midi| format!("and midi={midi}"))
            .unwrap_or_else(|| "(all midi)".to_string()),
        STFT_MAX_FREQ_HZ as u32,
        CQT_MIN_FREQ_HZ as u32,
        CQT_MAX_FREQ_HZ as u32,
        CQT_BINS_PER_OCTAVE as u32,
        CQT_SAMPLE_RATE_HZ as u32,
        CQT_WINDOW_LENGTH_SAMPLES as u32
    )
    .ok();
    html.push_str("<section class=\"grid\">");
    html.push_str(notes_html);
    html.push_str("</section></main></body></html>");
    html
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn escape_xml(input: &str) -> String {
    escape_html(input)
}
