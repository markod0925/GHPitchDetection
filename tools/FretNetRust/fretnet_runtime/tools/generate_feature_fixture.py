#!/usr/bin/env python3
"""Generate a feature-only reference fixture from the canonical Python HCQT pipeline."""

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

from common import HCQT_CONFIG, compute_hcqt_feature_bundle, ensure_repo_on_path  # noqa: E402


DEFAULT_AUDIO_PATH = CRATE_ROOT.parent.parent.parent / "input" / "guitarset" / "audio" / "00_BN1-129-Eb_comp_mic.wav"
DEFAULT_OUTPUT_PREFIX = CRATE_ROOT / "assets" / "test_features_python" / "00_BN1-129-Eb_comp_mic_features_f130"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Python HCQT feature fixture for Rust equivalence tests.")
    parser.add_argument("--audio-path", default=str(DEFAULT_AUDIO_PATH), help="Path to a real input wave file.")
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT_PREFIX), help="Output path without suffix.")
    parser.add_argument("--max-frames", type=int, default=130, help="Optional frame limit applied after HCQT extraction.")
    return parser.parse_args()


def resolve_audio_path(audio_path: str) -> Path:
    candidate = Path(audio_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    for path in [Path.cwd() / candidate, CRATE_ROOT / candidate, PYTHON_REPO_ROOT / candidate]:
        if path.exists():
            return path.resolve()

    return (Path.cwd() / candidate).resolve()


def main() -> int:
    args = parse_args()
    ensure_repo_on_path()

    import amt_tools.tools as tools

    audio_path = resolve_audio_path(args.audio_path)
    output_prefix = Path(args.output_prefix)

    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        return 1

    audio, sample_rate = tools.load_normalize_audio(str(audio_path), HCQT_CONFIG["sample_rate"])
    feature_bundle, feature_meta = compute_hcqt_feature_bundle(audio, max_frames=args.max_frames)

    arrays = {
        "audio": audio.astype(np.float32),
        "hcqt": np.asarray(feature_bundle[tools.KEY_FEATS], dtype=np.float32),
        "times": np.asarray(feature_bundle[tools.KEY_TIMES], dtype=np.float32),
    }

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")
    np.savez_compressed(npz_path, **arrays)

    metadata = {
        "audio_path": str(audio_path.resolve()),
        "npz_path": str(npz_path.resolve()),
        "sample_rate": int(sample_rate),
        "normalization": "rms_norm",
        "load_resample_type": "kaiser_best",
        "max_frames": args.max_frames,
        "hcqt_config": HCQT_CONFIG,
        "axis_semantics": {
            "audio": ["samples"],
            "hcqt": ["harmonics", "freq_bins", "frames"],
            "times": ["frames"],
        },
        "audio_summary": {
            "shape": list(arrays["audio"].shape),
            "dtype": str(arrays["audio"].dtype),
            "min": float(arrays["audio"].min()),
            "max": float(arrays["audio"].max()),
            "mean": float(arrays["audio"].mean()),
        },
        "hcqt_summary": {
            "shape": list(arrays["hcqt"].shape),
            "dtype": str(arrays["hcqt"].dtype),
            "min": float(arrays["hcqt"].min()),
            "max": float(arrays["hcqt"].max()),
            "mean": float(arrays["hcqt"].mean()),
        },
        "times_summary": {
            "shape": list(arrays["times"].shape),
            "dtype": str(arrays["times"].dtype),
            "min": float(arrays["times"].min()) if arrays["times"].size else 0.0,
            "max": float(arrays["times"].max()) if arrays["times"].size else 0.0,
            "mean": float(arrays["times"].mean()) if arrays["times"].size else 0.0,
        },
        "feature_meta": feature_meta,
        "notes": [
            "Audio is mono, normalized, and resampled exactly by the canonical Python loader.",
            "HCQT comes from amt_tools.features.HCQT with repository configuration.",
            "The fixture stores raw HCQT, not the model input window tensor.",
        ],
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"[OK] Wrote feature fixture archive: {npz_path}")
    print(f"[OK] Wrote feature fixture metadata: {json_path}")
    print(f"[OK] Audio shape: {arrays['audio'].shape}")
    print(f"[OK] HCQT shape: {arrays['hcqt'].shape}")
    print(f"[OK] Times shape: {arrays['times'].shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
