#!/usr/bin/env python3
"""Run the canonical Python FretNet path and dump comparable raw outputs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
RUST_CRATE_ROOT = THIS_DIR.parent
FRETNET_PY_ROOT = RUST_CRATE_ROOT.parent / "guitar-transcription-continuous"
FRETNET_PY_TOOLS = FRETNET_PY_ROOT / "tools"

if str(FRETNET_PY_TOOLS) not in sys.path:
    sys.path.insert(0, str(FRETNET_PY_TOOLS))

from common import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    HCQT_CONFIG,
    apply_repo_device_workflow,
    compute_hcqt_feature_bundle,
    ensure_repo_on_path,
    extract_model_candidate,
    load_checkpoint,
    parse_device,
    prepare_model_forward_input,
    resolve_model_path,
)


def load_model(model_path: str, device: str) -> torch.nn.Module:
    ensure_repo_on_path()

    resolved = resolve_model_path(model_path)
    requested_device = parse_device(device)
    loaded_obj = load_checkpoint(resolved, map_location=requested_device)
    model, source_key = extract_model_candidate(loaded_obj)
    if model is None:
        raise RuntimeError(
            "Loaded object is not a directly usable model and no nested model key was found."
        )

    apply_repo_device_workflow(model, device)
    model.eval()
    model._benchmark_model_path = str(resolved)  # type: ignore[attr-defined]
    model._benchmark_source = source_key  # type: ignore[attr-defined]
    return model


def tensor_dict_to_numpy(tensors: dict[str, Any]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key, value in tensors.items():
        if torch.is_tensor(value):
            output[key] = value.detach().cpu().numpy().astype(np.float32, copy=False)
    return output


def finalize_semantics(model: torch.nn.Module, raw_output: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    import amt_tools.tools as tools

    with torch.no_grad():
        semantic_output = model.post_proc({tools.KEY_OUTPUT: copy.deepcopy(raw_output)})
    return tensor_dict_to_numpy(semantic_output)


def run_reference(
    audio_path: str | Path,
    model: torch.nn.Module,
    max_frames: int | None = None,
    include_semantic: bool = True,
) -> dict[str, Any]:
    import amt_tools.tools as tools

    audio_path = str(Path(audio_path).resolve())
    audio, _ = tools.load_normalize_audio(audio_path, HCQT_CONFIG["sample_rate"])
    feature_bundle, feature_meta = compute_hcqt_feature_bundle(audio.astype(np.float32), max_frames=max_frames)
    feats, prepared = prepare_model_forward_input(model, feature_bundle)

    with torch.no_grad():
        raw_output = model(feats)

    raw_numpy = tensor_dict_to_numpy(raw_output)
    semantic_numpy = finalize_semantics(model, raw_output) if include_semantic else {}

    metadata = {
        "audio_path": audio_path,
        "model_path": getattr(model, "_benchmark_model_path", "<unknown>"),
        "sample_rate": HCQT_CONFIG["sample_rate"],
        "max_frames": max_frames,
        "prepared_input_shape": list(feats.shape),
        "frame_times_shape": list(feature_bundle[tools.KEY_TIMES].shape),
        "hcqt_summary": feature_meta["feature_summary"],
        "times_summary": feature_meta["times_summary"],
        "raw_tensors": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in raw_numpy.items()
        },
        "semantic_tensors": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in semantic_numpy.items()
        },
    }

    return {
        "raw_tensors": raw_numpy,
        "semantic_tensors": semantic_numpy,
        "frame_times": np.asarray(feature_bundle[tools.KEY_TIMES], dtype=np.float32),
        "metadata": metadata,
    }


def save_result(output_prefix: Path, result: dict[str, Any]) -> dict[str, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")

    npz_payload: dict[str, np.ndarray] = {}
    for name, array in result["raw_tensors"].items():
        npz_payload[name] = array
    for name, array in result["semantic_tensors"].items():
        npz_payload[f"semantic__{name}"] = array
    if "frame_times" in result:
        npz_payload["meta__frame_times"] = np.asarray(result["frame_times"], dtype=np.float32)
    np.savez_compressed(npz_path, **npz_payload)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result["metadata"], handle, indent=2, sort_keys=True)

    return {"npz_path": npz_path, "json_path": json_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the canonical Python reference path and dump raw outputs.")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip-semantic", action="store_true")
    args = parser.parse_args()

    model = load_model(args.model_path, args.device)
    result = run_reference(
        audio_path=args.audio_path,
        model=model,
        max_frames=args.max_frames,
        include_semantic=not args.skip_semantic,
    )
    paths = save_result(Path(args.output_prefix), result)

    print(f"Audio path: {args.audio_path}")
    print(f"Model path: {result['metadata']['model_path']}")
    print(f"NPZ path: {paths['npz_path']}")
    print(f"JSON path: {paths['json_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
