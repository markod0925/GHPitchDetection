#!/usr/bin/env python3
"""Dataset-scale end-to-end consistency benchmark between Python and Rust pipelines."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
RUST_CRATE_ROOT = THIS_DIR.parent
FRETNET_PY_ROOT = RUST_CRATE_ROOT.parent / "guitar-transcription-continuous"
FRETNET_PY_TOOLS = FRETNET_PY_ROOT / "tools"

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(FRETNET_PY_TOOLS) not in sys.path:
    sys.path.insert(0, str(FRETNET_PY_TOOLS))

from run_python_reference import load_model, run_reference, save_result  # noqa: E402


DEFAULT_OUTPUT_DIR = RUST_CRATE_ROOT / "output" / "python_vs_rust_consistency"
DEFAULT_RUST_BINARY = RUST_CRATE_ROOT / "target" / "release" / "examples" / (
    "run_end_to_end_dump.exe" if sys.platform.startswith("win") else "run_end_to_end_dump"
)
DEFAULT_ONNX_PATH = RUST_CRATE_ROOT / "../guitar-transcription-continuous/output/export/model.onnx"


@dataclass
class AudioInput:
    path: Path
    file_id: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Python reference and Rust end-to-end outputs across GuitarSet audio."
    )
    parser.add_argument("--audio-path", action="append", default=[], help="Explicit audio path. Repeatable.")
    parser.add_argument("--audio-dir", default=None, help="Directory to scan for audio files.")
    parser.add_argument("--glob", default="*.wav", help="Glob pattern used with --audio-dir.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to process.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N discovered files.")
    parser.add_argument("--full-dataset", action="store_true", help="Scan the full inferred/default GuitarSet directory.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle discovered files before applying offset/limit.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for --shuffle.")
    parser.add_argument("--python-model-path", default="model/model-2000.pt")
    parser.add_argument("--python-device", default="cpu")
    parser.add_argument("--rust-binary", default=None, help="Optional path to compiled Rust dump binary.")
    parser.add_argument("--rust-model-path", default=str(DEFAULT_ONNX_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future parallel execution.")
    parser.add_argument("--reuse-existing-results", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--worst-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir = output_dir / "per_file"
    per_file_dir.mkdir(parents=True, exist_ok=True)

    if args.workers != 1:
        print("[WARN] workers > 1 is not implemented yet for the shared Python reference model; running sequentially.")

    audio_inputs = discover_audio_inputs(args)
    if not audio_inputs:
        raise RuntimeError("no audio inputs discovered; provide --audio-path or --audio-dir, or enable --full-dataset")

    rust_binary = prepare_rust_binary(args.rust_binary)
    model = load_model(args.python_model_path, args.python_device)
    print(f"[Config] files={len(audio_inputs)} output_dir={output_dir}")
    print(f"[Config] python_model={getattr(model, '_benchmark_model_path', args.python_model_path)}")
    print(f"[Config] rust_binary={rust_binary}")
    print(f"[Config] rust_model={Path(args.rust_model_path).resolve()}")

    results: list[dict[str, Any]] = []
    results_jsonl = output_dir / "results.jsonl"
    with results_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        for index, audio_input in enumerate(audio_inputs, start=1):
            print(f"[{index}/{len(audio_inputs)}] {audio_input.file_id} :: {audio_input.path}")
            try:
                result = process_audio_file(
                    audio_input=audio_input,
                    model=model,
                    rust_binary=rust_binary,
                    rust_model_path=Path(args.rust_model_path).resolve(),
                    per_file_dir=per_file_dir,
                    skip_semantic=args.skip_semantic,
                    max_frames=args.max_frames,
                    reuse_existing_results=args.reuse_existing_results,
                )
            except Exception as exc:
                failure = {
                    "file_id": audio_input.file_id,
                    "audio_path": str(audio_input.path),
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "guitarset_metadata": audio_input.metadata,
                }
                results.append(failure)
                jsonl_handle.write(json.dumps(failure, sort_keys=True) + "\n")
                jsonl_handle.flush()
                print(f"  FAILED: {failure['failure_reason']}")
                if args.fail_fast:
                    raise
                continue

            results.append(result)
            jsonl_handle.write(json.dumps(result, sort_keys=True) + "\n")
            jsonl_handle.flush()
            if result["status"] == "ok":
                print(f"  OK raw_mean={result['overall']['raw_mean_abs_diff']:.6e}")
            else:
                print(f"  FAILED: {result['failure_reason']}")
                if args.fail_fast:
                    raise RuntimeError(result["failure_reason"])

    summary = build_summary(results, worst_k=args.worst_k)
    summary_path = output_dir / "summary.json"
    worst_cases_path = output_dir / "worst_cases.json"
    csv_path = output_dir / "results.csv"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with worst_cases_path.open("w", encoding="utf-8") as handle:
        json.dump(summary["worst_cases"], handle, indent=2, sort_keys=True)
    write_csv(results, csv_path)

    print()
    print("Summary:")
    print(f"  files_attempted: {summary['files_attempted']}")
    print(f"  files_succeeded_both: {summary['files_succeeded_both']}")
    print(f"  files_failed_python: {summary['files_failed_python']}")
    print(f"  files_failed_rust: {summary['files_failed_rust']}")
    for tensor_name, metrics in summary["raw_tensor_aggregates"].items():
        print(
            f"  raw[{tensor_name}] mean(mean_abs_diff)={metrics['mean_of_mean_abs_diff']:.6e} "
            f"median={metrics['median_of_mean_abs_diff']:.6e} p95={metrics['p95_of_mean_abs_diff']:.6e} "
            f"max(max_abs_diff)={metrics['max_observed_max_abs_diff']:.6e}"
        )
    if summary["semantic_tensor_aggregates"]:
        for tensor_name, metrics in summary["semantic_tensor_aggregates"].items():
            print(
                f"  semantic[{tensor_name}] mean(mean_abs_diff)={metrics['mean_of_mean_abs_diff']:.6e} "
                f"median={metrics['median_of_mean_abs_diff']:.6e} p95={metrics['p95_of_mean_abs_diff']:.6e} "
                f"max(max_abs_diff)={metrics['max_observed_max_abs_diff']:.6e}"
            )
    print(f"  summary_json: {summary_path}")
    print(f"  results_jsonl: {results_jsonl}")
    print(f"  results_csv: {csv_path}")
    print(f"  worst_cases_json: {worst_cases_path}")
    return 0


def discover_audio_inputs(args: argparse.Namespace) -> list[AudioInput]:
    paths: list[Path] = []

    if args.audio_path:
        paths.extend(Path(item).resolve() for item in args.audio_path)
    else:
        audio_dir = None
        if args.audio_dir:
            audio_dir = Path(args.audio_dir).resolve()
        elif args.full_dataset or (not args.audio_path and args.audio_dir is None):
            audio_dir = infer_default_guitarset_audio_dir()

        if audio_dir is not None:
            if not audio_dir.exists():
                raise RuntimeError(f"audio directory does not exist: {audio_dir}")
            paths.extend(sorted(audio_dir.glob(args.glob)))

    unique_paths = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    if args.shuffle:
        random.Random(args.seed).shuffle(unique_paths)

    if args.offset:
        unique_paths = unique_paths[args.offset :]
    if args.limit is not None:
        unique_paths = unique_paths[: args.limit]

    return [
        AudioInput(
            path=path,
            file_id=make_file_id(path),
            metadata=parse_guitarset_metadata(path),
        )
        for path in unique_paths
    ]


def infer_default_guitarset_audio_dir() -> Path:
    stage_root = RUST_CRATE_ROOT / "assets" / "test_features_python_stages"
    metadata_files = sorted(stage_root.glob("*/metadata.json"))
    for metadata_path in metadata_files:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_audio_path = Path(payload["source_audio_path"])
        if source_audio_path.exists():
            return source_audio_path.parent
    return (RUST_CRATE_ROOT.parent.parent.parent / "input" / "guitarset" / "audio").resolve()


def prepare_rust_binary(explicit_binary: str | None) -> Path:
    if explicit_binary:
        binary = Path(explicit_binary).resolve()
    else:
        binary = DEFAULT_RUST_BINARY.resolve()
        if not binary.exists():
            subprocess.run(
                ["cargo", "build", "--release", "--example", "run_end_to_end_dump"],
                cwd=RUST_CRATE_ROOT,
                check=True,
            )
    if not binary.exists():
        raise RuntimeError(f"Rust benchmark binary does not exist: {binary}")
    return binary


def process_audio_file(
    *,
    audio_input: AudioInput,
    model: torch.nn.Module,
    rust_binary: Path,
    rust_model_path: Path,
    per_file_dir: Path,
    skip_semantic: bool,
    max_frames: int | None,
    reuse_existing_results: bool,
) -> dict[str, Any]:
    file_dir = per_file_dir / audio_input.file_id
    result_path = file_dir / "result.json"
    if reuse_existing_results and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    file_dir.mkdir(parents=True, exist_ok=True)
    python_prefix = file_dir / "python_reference"
    rust_prefix = file_dir / "rust_reference"

    python_failure = None
    rust_failure = None
    python_result = None
    rust_raw = None

    try:
        python_result = run_reference(
            audio_path=audio_input.path,
            model=model,
            max_frames=max_frames,
            include_semantic=not skip_semantic,
        )
        save_result(python_prefix, python_result)
    except Exception as exc:  # noqa: BLE001
        python_failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=10),
        }

    try:
        run_rust_reference(
            rust_binary=rust_binary,
            audio_path=audio_input.path,
            rust_model_path=rust_model_path,
            output_prefix=rust_prefix,
            max_frames=max_frames,
        )
        rust_raw = load_npz_tensors(rust_prefix.with_suffix(".npz"))
    except Exception as exc:  # noqa: BLE001
        rust_failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=10),
        }

    result: dict[str, Any] = {
        "file_id": audio_input.file_id,
        "audio_path": str(audio_input.path),
        "guitarset_metadata": audio_input.metadata,
        "status": "ok",
        "python_success": python_failure is None,
        "rust_success": rust_failure is None,
        "python_failure": python_failure,
        "rust_failure": rust_failure,
    }

    if python_failure or rust_failure or python_result is None or rust_raw is None:
        result["status"] = "failed"
        result["failure_reason"] = failure_reason(python_failure, rust_failure)
        write_json(result_path, result)
        return result

    raw_metrics = compare_tensor_dicts(
        python_result["raw_tensors"],
        rust_raw,
        comparison_label="raw",
    )
    semantic_metrics = {}
    if not skip_semantic:
        rust_semantic = finalize_semantics_from_numpy(model, rust_raw)
        semantic_metrics = compare_tensor_dicts(
            python_result["semantic_tensors"],
            rust_semantic,
            comparison_label="semantic",
        )

    result.update(
        {
            "python_metadata_path": str(python_prefix.with_suffix(".json")),
            "python_npz_path": str(python_prefix.with_suffix(".npz")),
            "rust_metadata_path": str(rust_prefix.with_suffix(".json")),
            "rust_npz_path": str(rust_prefix.with_suffix(".npz")),
            "raw": raw_metrics,
            "semantic": semantic_metrics,
            "overall": {
                "raw_mean_abs_diff": aggregate_overall_score(raw_metrics),
                "semantic_mean_abs_diff": aggregate_overall_score(semantic_metrics) if semantic_metrics else None,
            },
        }
    )

    write_json(result_path, result)
    return result


def run_rust_reference(
    *,
    rust_binary: Path,
    audio_path: Path,
    rust_model_path: Path,
    output_prefix: Path,
    max_frames: int | None,
) -> None:
    command = [
        str(rust_binary),
        "--audio-path",
        str(audio_path),
        "--model-path",
        str(rust_model_path),
        "--output-prefix",
        str(output_prefix),
    ]
    if max_frames is not None:
        command.extend(["--max-frames", str(max_frames)])
    subprocess.run(command, cwd=RUST_CRATE_ROOT, check=True, capture_output=True, text=True)


def load_npz_tensors(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        tensors: dict[str, np.ndarray] = {}
        for key in archive.files:
            if key.startswith("semantic__"):
                continue
            normalized_key = key[:-4] if key.endswith(".npy") else key
            tensors[normalized_key] = np.asarray(archive[key], dtype=np.float32)
        return tensors


def finalize_semantics_from_numpy(model: torch.nn.Module, raw_tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import amt_tools.tools as tools

    tensor_dict = {
        key: torch.from_numpy(np.asarray(value, dtype=np.float32))
        for key, value in raw_tensors.items()
    }
    with torch.no_grad():
        semantic_output = model.post_proc({tools.KEY_OUTPUT: copy.deepcopy(tensor_dict)})
    return {
        key: value.detach().cpu().numpy().astype(np.float32, copy=False)
        for key, value in semantic_output.items()
        if torch.is_tensor(value)
    }


def compare_tensor_dicts(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    comparison_label: str,
) -> dict[str, Any]:
    reference_keys = sorted(reference.keys())
    candidate_keys = sorted(candidate.keys())
    shared_keys = sorted(set(reference_keys) & set(candidate_keys))

    result: dict[str, Any] = {
        "reference_keys": reference_keys,
        "candidate_keys": candidate_keys,
        "shared_keys": shared_keys,
        "missing_in_candidate": sorted(set(reference_keys) - set(candidate_keys)),
        "missing_in_reference": sorted(set(candidate_keys) - set(reference_keys)),
        "tensors": {},
    }

    if result["missing_in_candidate"] or result["missing_in_reference"]:
        result["comparison_error"] = f"{comparison_label} tensor name mismatch"

    for key in shared_keys:
        ref = np.asarray(reference[key], dtype=np.float32)
        cand = np.asarray(candidate[key], dtype=np.float32)
        tensor_result: dict[str, Any] = {
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
            "shape_match": list(ref.shape) == list(cand.shape),
        }
        if ref.shape != cand.shape:
            tensor_result["comparison_error"] = "shape mismatch"
            result["tensors"][key] = tensor_result
            continue

        abs_diff = np.abs(ref - cand)
        tensor_result.update(
            {
                "max_abs_diff": float(np.max(abs_diff)),
                "mean_abs_diff": float(np.mean(abs_diff)),
                "rmse": float(np.sqrt(np.mean(np.square(ref - cand)))),
                "p95_abs_diff": float(np.percentile(abs_diff, 95)),
            }
        )
        result["tensors"][key] = tensor_result

    return result


def aggregate_overall_score(section: dict[str, Any]) -> float | None:
    if not section:
        return None
    values = [
        tensor_metrics["mean_abs_diff"]
        for tensor_metrics in section.get("tensors", {}).values()
        if "mean_abs_diff" in tensor_metrics
    ]
    if not values:
        return None
    return float(np.mean(values))


def build_summary(results: list[dict[str, Any]], worst_k: int) -> dict[str, Any]:
    raw_aggregate = aggregate_tensor_metrics(results, "raw")
    semantic_aggregate = aggregate_tensor_metrics(results, "semantic")
    successful = [result for result in results if result.get("status") == "ok"]

    summary = {
        "files_attempted": len(results),
        "files_succeeded_both": len(successful),
        "files_failed_python": sum(
            1 for result in results if result.get("python_success") is False and result.get("rust_success") is True
        ),
        "files_failed_rust": sum(
            1 for result in results if result.get("rust_success") is False and result.get("python_success") is True
        ),
        "files_failed_both": sum(
            1 for result in results if result.get("python_success") is False and result.get("rust_success") is False
        ),
        "raw_tensor_aggregates": raw_aggregate,
        "semantic_tensor_aggregates": semantic_aggregate,
        "worst_cases": {
            "overall_raw": top_k_results(successful, lambda item: item["overall"]["raw_mean_abs_diff"], worst_k),
            "overall_semantic": top_k_results(
                [item for item in successful if item["overall"].get("semantic_mean_abs_diff") is not None],
                lambda item: item["overall"]["semantic_mean_abs_diff"],
                worst_k,
            ),
            "raw_per_tensor": top_k_per_tensor(successful, "raw", worst_k),
            "semantic_per_tensor": top_k_per_tensor(successful, "semantic", worst_k),
        },
    }
    return summary


def aggregate_tensor_metrics(results: list[dict[str, Any]], section_name: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, float]]] = {}
    for result in results:
        if result.get("status") != "ok":
            continue
        section = result.get(section_name, {})
        for tensor_name, tensor_metrics in section.get("tensors", {}).items():
            if "mean_abs_diff" not in tensor_metrics:
                continue
            buckets.setdefault(tensor_name, []).append(tensor_metrics)

    aggregate: dict[str, Any] = {}
    for tensor_name, items in buckets.items():
        means = np.array([item["mean_abs_diff"] for item in items], dtype=np.float64)
        maxes = np.array([item["max_abs_diff"] for item in items], dtype=np.float64)
        aggregate[tensor_name] = {
            "count": int(len(items)),
            "mean_of_mean_abs_diff": float(np.mean(means)),
            "median_of_mean_abs_diff": float(np.median(means)),
            "p95_of_mean_abs_diff": float(np.percentile(means, 95)),
            "max_observed_max_abs_diff": float(np.max(maxes)),
        }
    return aggregate


def top_k_results(results: list[dict[str, Any]], score_fn, k: int) -> list[dict[str, Any]]:
    ranked = sorted(
        [result for result in results if score_fn(result) is not None],
        key=score_fn,
        reverse=True,
    )
    return [
        {
            "file_id": result["file_id"],
            "audio_path": result["audio_path"],
            "score": score_fn(result),
            "guitarset_metadata": result.get("guitarset_metadata"),
        }
        for result in ranked[:k]
    ]


def top_k_per_tensor(results: list[dict[str, Any]], section_name: str, k: int) -> dict[str, Any]:
    per_tensor: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        section = result.get(section_name, {})
        for tensor_name, tensor_metrics in section.get("tensors", {}).items():
            if "mean_abs_diff" not in tensor_metrics:
                continue
            per_tensor.setdefault(tensor_name, []).append(
                {
                    "file_id": result["file_id"],
                    "audio_path": result["audio_path"],
                    "mean_abs_diff": tensor_metrics["mean_abs_diff"],
                    "max_abs_diff": tensor_metrics["max_abs_diff"],
                    "guitarset_metadata": result.get("guitarset_metadata"),
                }
            )

    return {
        tensor_name: sorted(items, key=lambda item: item["mean_abs_diff"], reverse=True)[:k]
        for tensor_name, items in per_tensor.items()
    }


def failure_reason(python_failure: dict[str, Any] | None, rust_failure: dict[str, Any] | None) -> str:
    if python_failure and rust_failure:
        return f"python={python_failure['type']}: {python_failure['message']} | rust={rust_failure['type']}: {rust_failure['message']}"
    if python_failure:
        return f"python={python_failure['type']}: {python_failure['message']}"
    if rust_failure:
        return f"rust={rust_failure['type']}: {rust_failure['message']}"
    return "unknown failure"


def make_file_id(path: Path) -> str:
    stem = path.stem.replace(" ", "_")
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def parse_guitarset_metadata(path: Path) -> dict[str, Any]:
    stem = path.stem
    tokens = stem.split("_")
    metadata: dict[str, Any] = {"stem": stem}
    if len(tokens) >= 4:
        metadata["performer_id"] = tokens[0]
        middle = tokens[1].split("-")
        if len(middle) >= 3:
            metadata["take_id"] = middle[0]
            metadata["tempo_bpm"] = middle[1]
            metadata["key"] = middle[2]
        metadata["segment"] = tokens[2]
        metadata["source"] = tokens[3]
    return metadata


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "file_id",
        "audio_path",
        "status",
        "python_success",
        "rust_success",
        "raw_mean_abs_diff",
        "semantic_mean_abs_diff",
        "failure_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "file_id": result.get("file_id"),
                    "audio_path": result.get("audio_path"),
                    "status": result.get("status"),
                    "python_success": result.get("python_success"),
                    "rust_success": result.get("rust_success"),
                    "raw_mean_abs_diff": (result.get("overall") or {}).get("raw_mean_abs_diff"),
                    "semantic_mean_abs_diff": (result.get("overall") or {}).get("semantic_mean_abs_diff"),
                    "failure_reason": result.get("failure_reason"),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
