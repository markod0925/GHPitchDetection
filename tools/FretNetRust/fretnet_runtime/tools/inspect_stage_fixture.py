#!/usr/bin/env python3
"""Inspect a stage-by-stage Python frontend fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a stage-by-stage Python frontend fixture.")
    parser.add_argument("fixture_dir", help="Path to the fixture directory.")
    return parser.parse_args()


def summarize(name: str, value: np.ndarray) -> dict[str, object]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()) if value.size else 0.0,
        "max": float(value.max()) if value.size else 0.0,
        "mean": float(value.mean()) if value.size else 0.0,
    }


def inspect_npz(path: Path) -> list[dict[str, object]]:
    archive = np.load(path)
    return [summarize(key, archive[key]) for key in archive.files]


def main() -> int:
    args = parse_args()
    fixture_dir = Path(args.fixture_dir)
    if not fixture_dir.exists() or not fixture_dir.is_dir():
        print(f"[ERROR] Fixture directory not found: {fixture_dir}")
        return 1

    metadata_path = fixture_dir / "metadata.json"
    if metadata_path.exists():
        print(f"Metadata: {metadata_path.resolve()}")
        print(metadata_path.read_text(encoding="utf-8"))
    else:
        print("Metadata file not found.")

    for npz_path in sorted(fixture_dir.glob("*.npz")):
        print(f"\nArchive: {npz_path.name}")
        for summary in inspect_npz(npz_path):
            print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
