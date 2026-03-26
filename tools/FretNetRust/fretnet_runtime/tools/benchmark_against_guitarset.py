#!/usr/bin/env python3
"""Compare Python and Rust FretNet predictions against GuitarSet annotations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import jams
import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent

from benchmark_python_vs_rust import (  # noqa: E402
    AudioInput,
    aggregate_tensor_metrics,
    compare_tensor_dicts,
    discover_audio_inputs,
    finalize_semantics_from_numpy,
    load_npz_tensors,
    top_k_per_tensor,
)
from run_python_reference import load_model, run_reference, save_result  # noqa: E402


DEFAULT_OUTPUT_DIR = THIS_DIR.parent / "output" / "annotation_eval"
DEFAULT_ONNX_PATH = THIS_DIR.parent / "../guitar-transcription-continuous/output/export/model.onnx"
DEFAULT_RUST_BATCH_BINARY = THIS_DIR.parent / "target" / "release" / "examples" / (
    "run_end_to_end_batch_dump.exe" if sys.platform.startswith("win") else "run_end_to_end_batch_dump"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Python and Rust FretNet pipelines against GuitarSet annotations."
    )
    parser.add_argument("--audio-path", action="append", default=[], help="Explicit audio path. Repeatable.")
    parser.add_argument("--audio-dir", default=None, help="Directory to scan for audio files.")
    parser.add_argument("--annotation-dir", default=None, help="Optional directory containing GuitarSet JAMS files.")
    parser.add_argument("--glob", default="*.wav", help="Glob pattern used with --audio-dir.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to process.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N discovered files.")
    parser.add_argument("--full-dataset", action="store_true", help="Scan the inferred/default GuitarSet audio directory.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle discovered files before applying offset/limit.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for --shuffle.")
    parser.add_argument("--python-model-path", default="model/model-2000.pt")
    parser.add_argument("--python-device", default="cpu")
    parser.add_argument("--rust-binary", default=None, help="Optional path to compiled Rust dump binary.")
    parser.add_argument("--rust-model-path", default=str(DEFAULT_ONNX_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-python-vs-rust", action="store_true", help="Skip raw Python-vs-Rust tensor comparison.")
    parser.add_argument("--skip-semantic", action="store_true", help="Skip semantic comparison output in reports.")
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
        print("[WARN] workers > 1 is not implemented yet; running sequentially.")

    audio_inputs = discover_audio_inputs(args)
    if not audio_inputs:
        raise RuntimeError("no audio inputs discovered; provide --audio-path or --audio-dir, or enable --full-dataset")

    annotation_dir = Path(args.annotation_dir).resolve() if args.annotation_dir else None
    rust_binary = prepare_rust_batch_binary(args.rust_binary)
    python_model_load_started = time.perf_counter()
    model = load_model(args.python_model_path, args.python_device)
    python_model_load_seconds = time.perf_counter() - python_model_load_started

    print(f"[Config] files={len(audio_inputs)} output_dir={output_dir}")
    print(f"[Config] python_model={getattr(model, '_benchmark_model_path', args.python_model_path)}")
    print(f"[Config] python_model_load_seconds={python_model_load_seconds:.6f}")
    print(f"[Config] rust_batch_binary={rust_binary}")
    print(f"[Config] rust_model={Path(args.rust_model_path).resolve()}")
    if annotation_dir is not None:
        print(f"[Config] annotation_dir={annotation_dir}")

    rust_batch_summary = run_or_reuse_rust_batch(
        audio_inputs=audio_inputs,
        rust_batch_binary=rust_binary,
        rust_model_path=Path(args.rust_model_path).resolve(),
        per_file_dir=per_file_dir,
        output_dir=output_dir,
        max_frames=args.max_frames,
        reuse_existing_results=args.reuse_existing_results,
    )
    rust_batch_results = {
        item["file_id"]: item
        for item in rust_batch_summary.get("jobs", [])
        if isinstance(item, dict) and "file_id" in item
    }

    results: list[dict[str, Any]] = []
    results_jsonl = output_dir / "results.jsonl"
    with results_jsonl.open("w", encoding="utf-8") as jsonl_handle:
        for index, audio_input in enumerate(audio_inputs, start=1):
            print(f"[{index}/{len(audio_inputs)}] {audio_input.file_id} :: {audio_input.path}")
            try:
                result = process_audio_file(
                    audio_input=audio_input,
                    model=model,
                    annotation_dir=annotation_dir,
                    per_file_dir=per_file_dir,
                    max_frames=args.max_frames,
                    skip_python_vs_rust=args.skip_python_vs_rust,
                    skip_semantic=args.skip_semantic,
                    reuse_existing_results=args.reuse_existing_results,
                    rust_batch_result=rust_batch_results.get(audio_input.file_id),
                )
            except Exception as exc:  # noqa: BLE001
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
                py_f1 = result["python_vs_gt"]["tablature_f1"]
                rust_f1 = result["rust_vs_gt"]["tablature_f1"]
                print(
                    "  OK "
                    f"py_tab_f1={py_f1:.4f} "
                    f"rust_tab_f1={rust_f1:.4f} "
                    f"delta={result['delta_vs_gt']['tablature_f1']:+.4f} "
                    f"py_time={result['timings']['python_pipeline_seconds']:.3f}s "
                    f"rust_time={result['timings']['rust_comparable_pipeline_seconds']:.3f}s"
                )
            else:
                print(f"  FAILED: {result['failure_reason']}")
                if args.fail_fast:
                    raise RuntimeError(result["failure_reason"])

    summary = build_summary(
        results,
        worst_k=args.worst_k,
        include_raw=not args.skip_python_vs_rust,
        python_model_load_seconds=python_model_load_seconds,
        rust_runtime_load_seconds=rust_batch_summary.get("runtime_load_seconds"),
        rust_extractor_create_seconds=rust_batch_summary.get("extractor_create_seconds"),
    )
    summary_path = output_dir / "summary.json"
    worst_cases_path = output_dir / "worst_cases.json"
    csv_path = output_dir / "results.csv"

    write_json(summary_path, summary)
    write_json(worst_cases_path, summary["worst_cases"])
    write_csv(results, csv_path)

    print_console_summary(summary, results_jsonl, csv_path, summary_path, worst_cases_path)
    return 0


def prepare_rust_batch_binary(explicit_binary: str | None) -> Path:
    if explicit_binary:
        binary = Path(explicit_binary).resolve()
    else:
        binary = DEFAULT_RUST_BATCH_BINARY.resolve()
        if not binary.exists():
            subprocess.run(
                ["cargo", "build", "--release", "--example", "run_end_to_end_batch_dump"],
                cwd=THIS_DIR.parent,
                check=True,
            )
    if not binary.exists():
        raise RuntimeError(f"Rust batch benchmark binary does not exist: {binary}")
    return binary


def run_or_reuse_rust_batch(
    *,
    audio_inputs: list[AudioInput],
    rust_batch_binary: Path,
    rust_model_path: Path,
    per_file_dir: Path,
    output_dir: Path,
    max_frames: int | None,
    reuse_existing_results: bool,
) -> dict[str, Any]:
    manifest_path = output_dir / "rust_batch_manifest.json"
    summary_path = output_dir / "rust_batch_summary.json"

    jobs = []
    for audio_input in audio_inputs:
        output_prefix = per_file_dir / audio_input.file_id / "rust_reference"
        jobs.append(
            {
                "file_id": audio_input.file_id,
                "audio_path": str(audio_input.path),
                "output_prefix": str(output_prefix),
                "max_frames": max_frames,
            }
        )

    if not (reuse_existing_results and summary_path.exists()):
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump({"jobs": jobs}, handle, indent=2, sort_keys=True)
        command = [
            str(rust_batch_binary),
            "--manifest-path",
            str(manifest_path),
            "--model-path",
            str(rust_model_path),
            "--summary-path",
            str(summary_path),
        ]
        subprocess.run(command, cwd=THIS_DIR.parent, check=True, capture_output=True, text=True)

    if not summary_path.exists():
        raise RuntimeError(f"Rust batch summary was not produced: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def process_audio_file(
    *,
    audio_input: AudioInput,
    model: torch.nn.Module,
    annotation_dir: Path | None,
    per_file_dir: Path,
    max_frames: int | None,
    skip_python_vs_rust: bool,
    skip_semantic: bool,
    reuse_existing_results: bool,
    rust_batch_result: dict[str, Any] | None,
) -> dict[str, Any]:
    file_dir = per_file_dir / audio_input.file_id
    result_path = file_dir / "result.json"
    if reuse_existing_results and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    file_dir.mkdir(parents=True, exist_ok=True)
    python_prefix = file_dir / "python_reference"
    rust_prefix = file_dir / "rust_reference"

    annotation_path = derive_annotation_path(audio_input.path, annotation_dir)

    python_failure = None
    rust_failure = None
    annotation_failure = None
    python_result = None
    rust_raw = None
    rust_semantic = None
    ground_truth = None
    python_pipeline_seconds = None
    rust_pipeline_seconds = None
    rust_semantic_decode_seconds = None
    annotation_ground_truth_seconds = None
    evaluation_seconds = None

    try:
        started = time.perf_counter()
        python_result = run_reference(
            audio_path=audio_input.path,
            model=model,
            max_frames=max_frames,
            include_semantic=True,
        )
        save_result(python_prefix, python_result)
        python_pipeline_seconds = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        python_failure = error_payload(exc)

    try:
        if rust_batch_result is None:
            raise RuntimeError(f"missing Rust batch result for {audio_input.file_id}")
        if rust_batch_result.get("status") != "ok":
            raise RuntimeError(rust_batch_result.get("error") or "Rust batch runner reported failure")
        rust_npz_path = Path(rust_batch_result["npz_path"])
        rust_raw = load_npz_tensors(rust_npz_path)
        rust_timings = rust_batch_result.get("timings") or {}
        rust_pipeline_seconds = rust_timings.get("total_seconds")
        semantic_started = time.perf_counter()
        rust_semantic = finalize_semantics_from_numpy(model, rust_raw)
        rust_semantic_decode_seconds = time.perf_counter() - semantic_started
    except Exception as exc:  # noqa: BLE001
        rust_failure = error_payload(exc)

    if python_failure is None and python_result is not None:
        try:
            started = time.perf_counter()
            frame_times = np.asarray(python_result["frame_times"], dtype=np.float32)
            ground_truth = load_guitarset_ground_truth(
                annotation_path=annotation_path,
                frame_times=frame_times,
                profile=model.profile,
            )
            annotation_ground_truth_seconds = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001
            annotation_failure = error_payload(exc)

    result: dict[str, Any] = {
        "file_id": audio_input.file_id,
        "audio_path": str(audio_input.path),
        "annotation_path": str(annotation_path),
        "guitarset_metadata": audio_input.metadata,
        "status": "ok",
        "python_success": python_failure is None,
        "rust_success": rust_failure is None,
        "annotation_success": annotation_failure is None,
        "python_failure": python_failure,
        "rust_failure": rust_failure,
        "annotation_failure": annotation_failure,
    }

    if python_failure or rust_failure or annotation_failure or python_result is None or rust_raw is None or rust_semantic is None or ground_truth is None:
        result["status"] = "failed"
        result["failure_reason"] = failure_reason(python_failure, rust_failure, annotation_failure)
        write_json(result_path, result)
        return result

    raw_metrics: dict[str, Any] = {}
    semantic_metrics: dict[str, Any] = {}
    if not skip_python_vs_rust:
        raw_metrics = compare_tensor_dicts(
            python_result["raw_tensors"],
            rust_raw,
            comparison_label="raw",
        )
        if not skip_semantic:
            semantic_metrics = compare_tensor_dicts(
                python_result["semantic_tensors"],
                rust_semantic,
                comparison_label="semantic",
            )

    evaluation_started = time.perf_counter()
    python_eval = evaluate_semantic_outputs_against_gt(
        semantic_tensors=python_result["semantic_tensors"],
        ground_truth=ground_truth,
        profile=model.profile,
        label="python",
    )
    rust_eval = evaluate_semantic_outputs_against_gt(
        semantic_tensors=rust_semantic,
        ground_truth=ground_truth,
        profile=model.profile,
        label="rust",
    )
    delta_eval = compute_metric_delta(rust_eval["metrics"], python_eval["metrics"])
    evaluation_seconds = time.perf_counter() - evaluation_started

    result.update(
        {
            "python_metadata_path": str(python_prefix.with_suffix(".json")),
            "python_npz_path": str(python_prefix.with_suffix(".npz")),
            "rust_metadata_path": str(rust_prefix.with_suffix(".json")),
            "rust_npz_path": str(rust_prefix.with_suffix(".npz")),
            "rust_batch_result": rust_batch_result,
            "raw": raw_metrics,
            "semantic": semantic_metrics,
            "ground_truth_summary": ground_truth["summary"],
            "python_vs_gt": python_eval["metrics"],
            "rust_vs_gt": rust_eval["metrics"],
            "delta_vs_gt": delta_eval,
            "python_eval_details": python_eval["details"],
            "rust_eval_details": rust_eval["details"],
            "timings": {
                "python_pipeline_seconds": python_pipeline_seconds,
                "rust_pipeline_seconds": rust_pipeline_seconds,
                "rust_semantic_decode_seconds": rust_semantic_decode_seconds,
                "rust_comparable_pipeline_seconds": (
                    None
                    if rust_pipeline_seconds is None or rust_semantic_decode_seconds is None
                    else float(rust_pipeline_seconds + rust_semantic_decode_seconds)
                ),
                "annotation_ground_truth_seconds": annotation_ground_truth_seconds,
                "evaluation_seconds": evaluation_seconds,
                "rust_minus_python_pipeline_seconds": (
                    None
                    if python_pipeline_seconds is None
                    or rust_pipeline_seconds is None
                    or rust_semantic_decode_seconds is None
                    else float((rust_pipeline_seconds + rust_semantic_decode_seconds) - python_pipeline_seconds)
                ),
            },
            "overall": {
                "raw_mean_abs_diff": aggregate_overall_score(raw_metrics),
                "semantic_mean_abs_diff": aggregate_overall_score(semantic_metrics) if semantic_metrics else None,
                "overall_degradation_score": overall_degradation_score(delta_eval),
            },
        }
    )

    write_json(result_path, result)
    return result


def derive_annotation_path(audio_path: Path, annotation_dir: Path | None) -> Path:
    if annotation_dir is None:
        annotation_dir = audio_path.parent.parent / "annotation"
    if not audio_path.exists():
        raise FileNotFoundError(f"audio path does not exist: {audio_path}")
    annotation_stem = audio_path.stem[:-4] if audio_path.stem.endswith("_mic") else audio_path.stem
    annotation_path = annotation_dir / f"{annotation_stem}.jams"
    if not annotation_path.exists():
        raise FileNotFoundError(f"annotation file does not exist: {annotation_path}")
    return annotation_path.resolve()


def load_guitarset_ground_truth(
    *,
    annotation_path: Path,
    frame_times: np.ndarray,
    profile: Any,
) -> dict[str, Any]:
    import amt_tools.tools as tools

    jams_data = jams.load(str(annotation_path), validate=False)
    stacked_notes = tools.extract_stacked_notes_jams(jams_data)
    stacked_onsets = tools.stacked_notes_to_stacked_onsets(stacked_notes, frame_times, profile)
    stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, frame_times, profile)
    tablature = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, profile)
    multi_pitch = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)
    onset_multi_pitch = tools.stacked_multi_pitch_to_multi_pitch(stacked_onsets)

    return {
        "frame_times": np.asarray(frame_times, dtype=np.float32),
        "annotation_path": str(annotation_path),
        "tablature": np.asarray(tablature, dtype=np.int64),
        "stacked_onsets": np.asarray(stacked_onsets, dtype=np.float32),
        "stacked_multi_pitch": np.asarray(stacked_multi_pitch, dtype=np.float32),
        "multi_pitch": np.asarray(multi_pitch, dtype=np.float32),
        "onset_multi_pitch": np.asarray(onset_multi_pitch, dtype=np.float32),
        "summary": {
            "annotation_path": str(annotation_path),
            "frame_times_shape": list(frame_times.shape),
            "tablature_shape": list(tablature.shape),
            "stacked_onsets_shape": list(stacked_onsets.shape),
            "stacked_multi_pitch_shape": list(stacked_multi_pitch.shape),
        },
    }


def evaluate_semantic_outputs_against_gt(
    *,
    semantic_tensors: dict[str, np.ndarray],
    ground_truth: dict[str, Any],
    profile: Any,
    label: str,
) -> dict[str, Any]:
    import amt_tools.tools as tools
    from amt_tools.evaluate import MultipitchEvaluator, TablatureEvaluator

    pred_tablature = expect_semantic_tablature(semantic_tensors, label)
    pred_onsets = expect_semantic_onsets(semantic_tensors, label)

    gt_tablature = np.asarray(ground_truth["tablature"], dtype=np.int64)
    gt_onsets = np.asarray(ground_truth["stacked_onsets"], dtype=np.float32)

    if pred_tablature.shape != gt_tablature.shape:
        raise ValueError(
            f"{label} tablature shape mismatch: predicted={list(pred_tablature.shape)} reference={list(gt_tablature.shape)}"
        )
    if pred_onsets.shape != gt_onsets.shape:
        raise ValueError(
            f"{label} onsets shape mismatch: predicted={list(pred_onsets.shape)} reference={list(gt_onsets.shape)}"
        )

    tablature_eval = TablatureEvaluator(profile=profile).evaluate(pred_tablature, gt_tablature)
    onset_multi_pitch_pred = tools.stacked_multi_pitch_to_multi_pitch(pred_onsets)
    onset_multi_pitch_ref = ground_truth["onset_multi_pitch"]
    onset_eval = MultipitchEvaluator().evaluate(onset_multi_pitch_pred, onset_multi_pitch_ref)

    pred_stacked_multi_pitch = tools.tablature_to_stacked_multi_pitch(pred_tablature, profile)
    pred_multi_pitch = tools.stacked_multi_pitch_to_multi_pitch(pred_stacked_multi_pitch)
    pitch_eval = MultipitchEvaluator().evaluate(pred_multi_pitch, ground_truth["multi_pitch"])

    exact_frame_accuracy = float(np.mean(np.all(pred_tablature == gt_tablature, axis=0)))

    metrics = {
        "onset_precision": float(onset_eval[tools.KEY_PRECISION]),
        "onset_recall": float(onset_eval[tools.KEY_RECALL]),
        "onset_f1": float(onset_eval[tools.KEY_F1]),
        "tablature_precision": float(tablature_eval[tools.KEY_PRECISION]),
        "tablature_recall": float(tablature_eval[tools.KEY_RECALL]),
        "tablature_f1": float(tablature_eval[tools.KEY_F1]),
        "tablature_tdr": float(tablature_eval[tools.KEY_TDR]),
        "pitch_precision": float(pitch_eval[tools.KEY_PRECISION]),
        "pitch_recall": float(pitch_eval[tools.KEY_RECALL]),
        "pitch_f1": float(pitch_eval[tools.KEY_F1]),
        "exact_tablature_frame_accuracy": exact_frame_accuracy,
    }

    details = {
        "predicted_tablature_shape": list(pred_tablature.shape),
        "ground_truth_tablature_shape": list(gt_tablature.shape),
        "predicted_onsets_shape": list(pred_onsets.shape),
        "ground_truth_onsets_shape": list(gt_onsets.shape),
        "frame_count": int(pred_tablature.shape[-1]),
    }

    return {"metrics": metrics, "details": details}


def expect_semantic_tablature(semantic_tensors: dict[str, np.ndarray], label: str) -> np.ndarray:
    if "tablature" not in semantic_tensors:
        raise KeyError(f"{label} semantic outputs do not contain 'tablature'")
    tablature = np.asarray(semantic_tensors["tablature"])
    if tablature.ndim == 3 and tablature.shape[0] == 1:
        tablature = tablature[0]
    if tablature.ndim != 2:
        raise ValueError(f"{label} semantic tablature must be [strings, frames], got {list(tablature.shape)}")
    rounded = np.rint(tablature).astype(np.int64)
    if not np.allclose(tablature, rounded, atol=1e-4):
        raise ValueError(f"{label} semantic tablature is not close to integer-valued classes")
    return rounded


def expect_semantic_onsets(semantic_tensors: dict[str, np.ndarray], label: str) -> np.ndarray:
    if "onsets" not in semantic_tensors:
        raise KeyError(f"{label} semantic outputs do not contain 'onsets'")
    onsets = np.asarray(semantic_tensors["onsets"], dtype=np.float32)
    if onsets.ndim == 4 and onsets.shape[0] == 1:
        onsets = onsets[0]
    if onsets.ndim != 3:
        raise ValueError(f"{label} semantic onsets must be [strings, pitches, frames], got {list(onsets.shape)}")
    return (onsets > 0.5).astype(np.float32)


def compute_metric_delta(rust_metrics: dict[str, float], python_metrics: dict[str, float]) -> dict[str, float]:
    shared = sorted(set(rust_metrics) & set(python_metrics))
    return {key: float(rust_metrics[key] - python_metrics[key]) for key in shared}


def overall_degradation_score(delta_metrics: dict[str, float]) -> float:
    tracked = [
        "onset_f1",
        "tablature_f1",
        "pitch_f1",
        "exact_tablature_frame_accuracy",
    ]
    losses = [max(0.0, -float(delta_metrics.get(name, 0.0))) for name in tracked]
    return float(np.mean(losses))


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


def build_summary(
    results: list[dict[str, Any]],
    *,
    worst_k: int,
    include_raw: bool,
    python_model_load_seconds: float | None,
    rust_runtime_load_seconds: float | None,
    rust_extractor_create_seconds: float | None,
) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "ok"]
    python_gt_agg = aggregate_flat_metrics(successful, "python_vs_gt")
    rust_gt_agg = aggregate_flat_metrics(successful, "rust_vs_gt")
    delta_agg = aggregate_flat_metrics(successful, "delta_vs_gt")
    timing_agg = aggregate_flat_metrics(successful, "timings")

    summary: dict[str, Any] = {
        "files_attempted": len(results),
        "files_fully_evaluated": len(successful),
        "files_failed_python": sum(1 for result in results if result.get("python_success") is False),
        "files_failed_rust": sum(1 for result in results if result.get("rust_success") is False),
        "files_failed_annotation": sum(1 for result in results if result.get("annotation_success") is False),
        "one_time_costs": {
            "python_model_load_seconds": python_model_load_seconds,
            "rust_runtime_load_seconds": rust_runtime_load_seconds,
            "rust_extractor_create_seconds": rust_extractor_create_seconds,
        },
        "python_vs_gt_aggregates": python_gt_agg,
        "rust_vs_gt_aggregates": rust_gt_agg,
        "delta_vs_gt_aggregates": delta_agg,
        "timing_aggregates": timing_agg,
        "worst_cases": build_worst_cases(successful, worst_k=worst_k, include_raw=include_raw),
        "interpretation": build_interpretation(successful, delta_agg),
    }

    if include_raw:
        summary["raw_tensor_aggregates"] = aggregate_tensor_metrics(results, "raw")
        summary["semantic_tensor_aggregates"] = aggregate_tensor_metrics(results, "semantic")

    return summary


def aggregate_flat_metrics(results: list[dict[str, Any]], section_name: str) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {}
    for result in results:
        for metric_name, value in result.get(section_name, {}).items():
            if value is None:
                continue
            buckets.setdefault(metric_name, []).append(float(value))

    aggregate: dict[str, Any] = {}
    for metric_name, values in sorted(buckets.items()):
        array = np.asarray(values, dtype=np.float64)
        aggregate[metric_name] = {
            "count": int(array.size),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }
    return aggregate


def build_worst_cases(results: list[dict[str, Any]], *, worst_k: int, include_raw: bool) -> dict[str, Any]:
    worst: dict[str, Any] = {
        "by_delta_tablature_f1": top_k_by_metric(results, "delta_vs_gt", "tablature_f1", worst_k, ascending=True),
        "by_delta_onset_f1": top_k_by_metric(results, "delta_vs_gt", "onset_f1", worst_k, ascending=True),
        "by_delta_exact_tablature_frame_accuracy": top_k_by_metric(
            results, "delta_vs_gt", "exact_tablature_frame_accuracy", worst_k, ascending=True
        ),
        "slowest_python_pipeline": top_k_by_metric(results, "timings", "python_pipeline_seconds", worst_k, ascending=False),
        "slowest_rust_pipeline": top_k_by_metric(
            results, "timings", "rust_comparable_pipeline_seconds", worst_k, ascending=False
        ),
        "by_overall_degradation_score": sorted(
            [
                {
                    "file_id": result["file_id"],
                    "audio_path": result["audio_path"],
                    "annotation_path": result["annotation_path"],
                    "score": result["overall"]["overall_degradation_score"],
                    "guitarset_metadata": result.get("guitarset_metadata"),
                }
                for result in results
            ],
            key=lambda item: item["score"],
            reverse=True,
        )[:worst_k],
    }
    if include_raw:
        worst["raw_per_tensor"] = top_k_per_tensor(results, "raw", worst_k)
        worst["semantic_per_tensor"] = top_k_per_tensor(results, "semantic", worst_k)
    return worst


def top_k_by_metric(
    results: list[dict[str, Any]],
    section_name: str,
    metric_name: str,
    k: int,
    *,
    ascending: bool,
) -> list[dict[str, Any]]:
    ranked = [
        {
            "file_id": result["file_id"],
            "audio_path": result["audio_path"],
            "annotation_path": result["annotation_path"],
            "value": result.get(section_name, {}).get(metric_name),
            "guitarset_metadata": result.get("guitarset_metadata"),
        }
        for result in results
        if result.get(section_name, {}).get(metric_name) is not None
    ]
    ranked.sort(key=lambda item: item["value"], reverse=not ascending)
    return ranked[:k]


def build_interpretation(results: list[dict[str, Any]], delta_agg: dict[str, Any]) -> dict[str, Any]:
    if not results or not delta_agg:
        return {"status": "insufficient_data"}

    candidate_metrics = [
        name
        for name in ("onset_f1", "tablature_f1", "pitch_f1", "exact_tablature_frame_accuracy")
        if name in delta_agg
    ]
    most_sensitive_metric = min(candidate_metrics, key=lambda name: delta_agg[name]["mean"]) if candidate_metrics else None
    files_with_visible_drop = sum(
        1
        for result in results
        if any(float(result["delta_vs_gt"].get(metric, 0.0)) < -0.02 for metric in candidate_metrics)
    )

    return {
        "most_sensitive_metric": most_sensitive_metric,
        "most_sensitive_mean_delta": (
            float(delta_agg[most_sensitive_metric]["mean"]) if most_sensitive_metric is not None else None
        ),
        "files_with_metric_drop_below_minus_0_02": int(files_with_visible_drop),
        "guidance": (
            "Use the mean/p95 Rust-minus-Python deltas and the worst-case files before moving to frontend performance work."
        ),
    }


def print_console_summary(
    summary: dict[str, Any],
    results_jsonl: Path,
    csv_path: Path,
    summary_path: Path,
    worst_cases_path: Path,
) -> None:
    print()
    print("Summary:")
    print(f"  files_attempted: {summary['files_attempted']}")
    print(f"  files_fully_evaluated: {summary['files_fully_evaluated']}")
    print(f"  files_failed_python: {summary['files_failed_python']}")
    print(f"  files_failed_rust: {summary['files_failed_rust']}")
    print(f"  files_failed_annotation: {summary['files_failed_annotation']}")
    print(f"  one_time_python_model_load_seconds: {summary['one_time_costs']['python_model_load_seconds']:.6f}")
    rust_runtime = summary["one_time_costs"].get("rust_runtime_load_seconds")
    rust_extractor = summary["one_time_costs"].get("rust_extractor_create_seconds")
    if rust_runtime is not None:
        print(f"  one_time_rust_runtime_load_seconds: {rust_runtime:.6f}")
    if rust_extractor is not None:
        print(f"  one_time_rust_extractor_create_seconds: {rust_extractor:.6f}")

    for section_name in ("python_vs_gt_aggregates", "rust_vs_gt_aggregates", "delta_vs_gt_aggregates"):
        print(f"  {section_name}:")
        for metric_name, aggregate in summary[section_name].items():
            print(
                f"    {metric_name}: mean={aggregate['mean']:.6f} "
                f"median={aggregate['median']:.6f} p95={aggregate['p95']:.6f} "
                f"min={aggregate['min']:.6f} max={aggregate['max']:.6f}"
            )

    print("  timing_aggregates:")
    for metric_name, aggregate in summary["timing_aggregates"].items():
        print(
            f"    {metric_name}: mean={aggregate['mean']:.6f}s "
            f"median={aggregate['median']:.6f}s p95={aggregate['p95']:.6f}s "
            f"min={aggregate['min']:.6f}s max={aggregate['max']:.6f}s"
        )

    interpretation = summary.get("interpretation", {})
    if interpretation:
        print("  interpretation:")
        print(f"    most_sensitive_metric: {interpretation.get('most_sensitive_metric')}")
        mean_delta = interpretation.get("most_sensitive_mean_delta")
        if mean_delta is not None:
            print(f"    most_sensitive_mean_delta: {mean_delta:.6f}")
        print(
            "    files_with_metric_drop_below_minus_0_02: "
            f"{interpretation.get('files_with_metric_drop_below_minus_0_02')}"
        )

    print(f"  summary_json: {summary_path}")
    print(f"  results_jsonl: {results_jsonl}")
    print(f"  results_csv: {csv_path}")
    print(f"  worst_cases_json: {worst_cases_path}")


def error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=12),
    }


def failure_reason(
    python_failure: dict[str, Any] | None,
    rust_failure: dict[str, Any] | None,
    annotation_failure: dict[str, Any] | None,
) -> str:
    reasons = []
    if python_failure:
        reasons.append(f"python={python_failure['type']}: {python_failure['message']}")
    if rust_failure:
        reasons.append(f"rust={rust_failure['type']}: {rust_failure['message']}")
    if annotation_failure:
        reasons.append(f"annotation={annotation_failure['type']}: {annotation_failure['message']}")
    return " | ".join(reasons) if reasons else "unknown failure"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "file_id",
        "audio_path",
        "annotation_path",
        "status",
        "python_success",
        "rust_success",
        "annotation_success",
        "python_onset_f1",
        "rust_onset_f1",
        "delta_onset_f1",
        "python_tablature_f1",
        "rust_tablature_f1",
        "delta_tablature_f1",
        "python_pipeline_seconds",
        "rust_pipeline_seconds",
        "rust_minus_python_pipeline_seconds",
        "rust_semantic_decode_seconds",
        "annotation_ground_truth_seconds",
        "evaluation_seconds",
        "python_pitch_f1",
        "rust_pitch_f1",
        "delta_pitch_f1",
        "python_exact_tablature_frame_accuracy",
        "rust_exact_tablature_frame_accuracy",
        "delta_exact_tablature_frame_accuracy",
        "overall_degradation_score",
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
                    "annotation_path": result.get("annotation_path"),
                    "status": result.get("status"),
                    "python_success": result.get("python_success"),
                    "rust_success": result.get("rust_success"),
                    "annotation_success": result.get("annotation_success"),
                    "python_onset_f1": (result.get("python_vs_gt") or {}).get("onset_f1"),
                    "rust_onset_f1": (result.get("rust_vs_gt") or {}).get("onset_f1"),
                    "delta_onset_f1": (result.get("delta_vs_gt") or {}).get("onset_f1"),
                    "python_tablature_f1": (result.get("python_vs_gt") or {}).get("tablature_f1"),
                    "rust_tablature_f1": (result.get("rust_vs_gt") or {}).get("tablature_f1"),
                    "delta_tablature_f1": (result.get("delta_vs_gt") or {}).get("tablature_f1"),
                    "python_pipeline_seconds": (result.get("timings") or {}).get("python_pipeline_seconds"),
                    "rust_pipeline_seconds": (result.get("timings") or {}).get("rust_pipeline_seconds"),
                    "rust_minus_python_pipeline_seconds": (
                        result.get("timings") or {}
                    ).get("rust_minus_python_pipeline_seconds"),
                    "rust_semantic_decode_seconds": (result.get("timings") or {}).get("rust_semantic_decode_seconds"),
                    "annotation_ground_truth_seconds": (
                        result.get("timings") or {}
                    ).get("annotation_ground_truth_seconds"),
                    "evaluation_seconds": (result.get("timings") or {}).get("evaluation_seconds"),
                    "python_pitch_f1": (result.get("python_vs_gt") or {}).get("pitch_f1"),
                    "rust_pitch_f1": (result.get("rust_vs_gt") or {}).get("pitch_f1"),
                    "delta_pitch_f1": (result.get("delta_vs_gt") or {}).get("pitch_f1"),
                    "python_exact_tablature_frame_accuracy": (
                        result.get("python_vs_gt") or {}
                    ).get("exact_tablature_frame_accuracy"),
                    "rust_exact_tablature_frame_accuracy": (
                        result.get("rust_vs_gt") or {}
                    ).get("exact_tablature_frame_accuracy"),
                    "delta_exact_tablature_frame_accuracy": (
                        result.get("delta_vs_gt") or {}
                    ).get("exact_tablature_frame_accuracy"),
                    "overall_degradation_score": (result.get("overall") or {}).get("overall_degradation_score"),
                    "failure_reason": result.get("failure_reason"),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
