#!/usr/bin/env python3
"""Generate a real-audio regression fixture for the Rust FretNet runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
CRATE_ROOT = SCRIPT_DIR.parent
PYTHON_REPO_ROOT = CRATE_ROOT.parent / "guitar-transcription-continuous"
PYTHON_TOOLS_DIR = PYTHON_REPO_ROOT / "tools"

if str(PYTHON_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_TOOLS_DIR))

from common import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    HCQT_CONFIG,
    apply_repo_device_workflow,
    compute_hcqt_feature_bundle,
    ensure_repo_on_path,
    extract_model_candidate,
    load_checkpoint,
    prepare_model_forward_input,
    resolve_model_path,
    summarize_profile,
)


DEFAULT_AUDIO_PATH = CRATE_ROOT.parent.parent.parent / "input" / "guitarset" / "audio" / "00_BN1-129-Eb_comp_mic.wav"
DEFAULT_OUTPUT_PREFIX = CRATE_ROOT / "assets" / "test_features" / "00_BN1-129-Eb_comp_mic_f130"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a regression fixture from a real GuitarSet wave.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the original PyTorch model.")
    parser.add_argument("--audio-path", default=str(DEFAULT_AUDIO_PATH), help="Path to the real input wave file.")
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT_PREFIX), help="Output path without suffix.")
    parser.add_argument("--max-frames", type=int, default=130, help="Maximum number of HCQT frames to keep.")
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

    model_path = resolve_model_path(args.model_path)
    audio_path = resolve_audio_path(args.audio_path)
    output_prefix = Path(args.output_prefix)

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return 1

    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        return 1

    loaded_obj = load_checkpoint(model_path, map_location="cpu")
    model, source_key = extract_model_candidate(loaded_obj)
    if model is None:
        print("[ERROR] Could not extract a model object from the checkpoint.")
        return 1

    apply_repo_device_workflow(model, device="cpu")
    model.eval()

    import amt_tools.tools as tools

    print(f"[Model] Source: {'top-level object' if source_key is None else f'dict[{source_key!r}]'}")
    print(f"[Model] Path: {model_path}")
    print(f"[Audio] Path: {audio_path}")

    audio, _ = tools.load_normalize_audio(str(audio_path), HCQT_CONFIG["sample_rate"])
    feature_bundle, feature_meta = compute_hcqt_feature_bundle(audio, max_frames=args.max_frames)
    example_input, _prepared = prepare_model_forward_input(model, feature_bundle)

    with torch.no_grad():
        raw_output = model(example_input)

    if not isinstance(raw_output, dict):
        print(f"[ERROR] Expected dict output from model.forward, got {type(raw_output).__name__}.")
        return 1

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")

    arrays: dict[str, np.ndarray] = {
        "features": example_input.detach().cpu().numpy().astype(np.float32),
        "times": np.asarray(feature_bundle[tools.KEY_TIMES]),
    }

    for name, tensor in raw_output.items():
        if not torch.is_tensor(tensor):
            print(f"[ERROR] Output '{name}' is not a tensor: {type(tensor).__name__}")
            return 1
        arrays[str(name)] = tensor.detach().cpu().numpy().astype(np.float32)

    np.savez_compressed(npz_path, **arrays)

    metadata = {
        "model_path": str(model_path.resolve()),
        "audio_path": str(audio_path.resolve()),
        "npz_path": str(npz_path.resolve()),
        "max_frames": args.max_frames,
        "hcqt_config": HCQT_CONFIG,
        "feature_summary": {
            "shape": list(arrays["features"].shape),
            "dtype": str(arrays["features"].dtype),
            "min": float(arrays["features"].min()),
            "max": float(arrays["features"].max()),
            "mean": float(arrays["features"].mean()),
        },
        "outputs": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(value.min()),
                "max": float(value.max()),
                "mean": float(value.mean()),
            }
            for name, value in arrays.items()
            if name not in {"features", "times"}
        },
        "profile": summarize_profile(getattr(model, "profile", None)),
        "python_output_type": f"{type(raw_output).__module__}.{type(raw_output).__name__}",
        "notes": [
            "Fixture generated from a real GuitarSet wave.",
            "Features are the ONNX-ready tensor returned by model.pre_proc(...), not raw HCQT stack.",
            "Reference outputs come from the original PyTorch model on CPU.",
        ],
        "feature_meta": feature_meta,
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"[OK] Wrote fixture archive: {npz_path}")
    print(f"[OK] Wrote fixture metadata: {json_path}")
    print(f"[OK] Feature tensor shape: {arrays['features'].shape}")
    for name, value in arrays.items():
        if name in {"features", "times"}:
            continue
        print(f"[OK] Output {name}: shape={value.shape} min={value.min():.6f} max={value.max():.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
