#!/usr/bin/env python3
"""Generate stage-by-stage Python frontend fixtures for Rust diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CRATE_ROOT = SCRIPT_DIR.parent
PYTHON_REPO_ROOT = CRATE_ROOT.parent / "guitar-transcription-continuous"
PYTHON_TOOLS_DIR = PYTHON_REPO_ROOT / "tools"

if str(PYTHON_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_TOOLS_DIR))

from common import HCQT_CONFIG, ensure_repo_on_path  # noqa: E402

DEFAULT_AUDIO_PATH = CRATE_ROOT.parent.parent.parent / "input" / "guitarset" / "audio" / "00_BN1-129-Eb_comp_mic.wav"
DEFAULT_OUTPUT_DIR = CRATE_ROOT / "assets" / "test_features_python_stages" / "00_BN1-129-Eb_comp_mic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Python frontend stage fixtures for Rust diagnostics.")
    parser.add_argument("--audio-path", default=str(DEFAULT_AUDIO_PATH), help="Path to a real input wave file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for the stage fixture.")
    parser.add_argument("--max-frames", type=int, default=130, help="Optional frame limit applied after transform extraction.")
    return parser.parse_args()


def resolve_audio_path(audio_path: str) -> Path:
    candidate = Path(audio_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    for path in [Path.cwd() / candidate, CRATE_ROOT / candidate, PYTHON_REPO_ROOT / candidate]:
        if path.exists():
            return path.resolve()

    return (Path.cwd() / candidate).resolve()


def summarize(arr: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
    }


def harmonic_label(value: float) -> str:
    return str(value).replace(".", "_").replace("-", "neg_")


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> int:
    args = parse_args()
    ensure_repo_on_path()

    import librosa
    import soundfile as sf
    import amt_tools.tools as tools
    from amt_tools.features import HCQT

    audio_path = resolve_audio_path(args.audio_path)
    output_dir = Path(args.output_dir)

    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        return 1

    audio_info = sf.info(str(audio_path))
    audio, sample_rate = tools.load_normalize_audio(str(audio_path), HCQT_CONFIG["sample_rate"])

    data_proc = HCQT(
        sample_rate=HCQT_CONFIG["sample_rate"],
        hop_length=HCQT_CONFIG["hop_length"],
        fmin=librosa.note_to_hz(HCQT_CONFIG["fmin_note"]),
        harmonics=HCQT_CONFIG["harmonics"],
        n_bins=HCQT_CONFIG["n_bins"],
        bins_per_octave=HCQT_CONFIG["bins_per_octave"],
    )

    num_frames = data_proc.get_expected_frames(audio)
    if args.max_frames is not None and args.max_frames > 0:
        num_frames = min(num_frames, args.max_frames)

    output_dir.mkdir(parents=True, exist_ok=True)

    resampled_audio_path = output_dir / "resampled_audio.npz"
    save_npz(resampled_audio_path, audio=np.asarray(audio, dtype=np.float32))

    harmonic_entries: list[dict[str, object]] = []
    harmonic_summaries: list[dict[str, object]] = []
    hcqt_channels = []

    for harmonic, module in zip(HCQT_CONFIG["harmonics"], data_proc.modules):
        magnitude = np.abs(
            librosa.vqt(
                y=audio,
                sr=module.sample_rate,
                hop_length=module.hop_length,
                fmin=module.fmin,
                n_bins=module.n_bins,
                bins_per_octave=module.bins_per_octave,
                gamma=module.gamma,
            )
        )[..., :num_frames]
        post_processed = module.process_audio(audio)[0, :, :num_frames]
        hcqt_channels.append(post_processed)

        label = harmonic_label(harmonic)
        magnitude_name = f"harmonic_{label}_magnitude.npz"
        post_name = f"harmonic_{label}_post.npz"
        save_npz(output_dir / magnitude_name, magnitude=np.asarray(magnitude, dtype=np.float32))
        save_npz(output_dir / post_name, post_processed=np.asarray(post_processed, dtype=np.float32))

        harmonic_entries.append(
            {
                "harmonic": float(harmonic),
                "label": label,
                "magnitude": {
                    "path": magnitude_name,
                    "shape": list(magnitude.shape),
                    "dtype": "float32",
                    "axis_semantics": ["freq_bins", "frames"],
                },
                "post_processed": {
                    "path": post_name,
                    "shape": list(post_processed.shape),
                    "dtype": "float32",
                    "axis_semantics": ["freq_bins", "frames"],
                },
            }
        )
        harmonic_summaries.append(
            {
                "harmonic": float(harmonic),
                "magnitude": summarize(np.asarray(magnitude, dtype=np.float32)),
                "post_processed": summarize(np.asarray(post_processed, dtype=np.float32)),
            }
        )

    hcqt = np.stack(hcqt_channels, axis=0).astype(np.float32)
    times = data_proc.get_times(audio)[:num_frames].astype(np.float32)
    hcqt_path = output_dir / "hcqt.npz"
    times_path = output_dir / "times.npz"
    save_npz(hcqt_path, hcqt=hcqt)
    save_npz(times_path, times=times)

    metadata = {
        "fixture_name": output_dir.name,
        "source_audio_path": str(audio_path.resolve()),
        "original_audio": {
            "sample_rate": int(audio_info.samplerate),
            "num_samples": int(audio_info.frames),
            "duration_seconds": float(audio_info.duration),
            "channels": int(audio_info.channels),
        },
        "resampled_audio": {
            "path": resampled_audio_path.name,
            "shape": [int(audio.shape[0])],
            "dtype": "float32",
            "axis_semantics": ["samples"],
        },
        "times": {
            "path": times_path.name,
            "shape": list(times.shape),
            "dtype": "float32",
            "axis_semantics": ["frames"],
        },
        "frontend": {
            "sample_rate": int(HCQT_CONFIG["sample_rate"]),
            "hop_length": int(HCQT_CONFIG["hop_length"]),
            "fmin_hz": float(librosa.note_to_hz(HCQT_CONFIG["fmin_note"])),
            "harmonics": [float(value) for value in HCQT_CONFIG["harmonics"]],
            "n_bins": int(HCQT_CONFIG["n_bins"]),
            "bins_per_octave": int(HCQT_CONFIG["bins_per_octave"]),
            "top_db": 80.0,
        },
        "harmonics": harmonic_entries,
        "hcqt": {
            "path": hcqt_path.name,
            "shape": list(hcqt.shape),
            "dtype": "float32",
            "axis_semantics": ["harmonics", "freq_bins", "frames"],
        },
        "max_frames": int(num_frames),
        "summaries": {
            "resampled_audio": summarize(np.asarray(audio, dtype=np.float32)),
            "times": summarize(times),
            "harmonics": harmonic_summaries,
            "hcqt": summarize(hcqt),
        },
        "notes": [
            "Stage 1 stores the canonical Python resampled and RMS-normalized mono audio.",
            "Stage 2 stores per-harmonic VQT magnitudes and post-processed outputs before HCQT stacking.",
            "Stage 3 stores the final HCQT tensor used by the Rust feature-equivalence tests.",
            "The source of truth is the Python HCQT path from amt_tools/librosa, not the Rust implementation.",
        ],
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"[OK] Wrote stage fixture directory: {output_dir}")
    print(f"[OK] Metadata: {metadata_path}")
    print(f"[OK] Resampled audio shape: {audio.shape}")
    print(f"[OK] HCQT shape: {hcqt.shape}")
    print(f"[OK] Harmonics: {HCQT_CONFIG['harmonics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
