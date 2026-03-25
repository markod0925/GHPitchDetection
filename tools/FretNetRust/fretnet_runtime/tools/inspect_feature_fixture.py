#!/usr/bin/env python3
"""Inspect a feature-only fixture generated from the Python HCQT pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a Python HCQT feature fixture.")
    parser.add_argument("fixture_path", help="Path to the .npz fixture archive.")
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


def main() -> int:
    args = parse_args()
    fixture_path = Path(args.fixture_path)
    if not fixture_path.exists():
        print(f"[ERROR] Fixture archive not found: {fixture_path}")
        return 1

    metadata_path = fixture_path.with_suffix(".json")
    archive = np.load(fixture_path)

    print(f"Fixture archive: {fixture_path.resolve()}")
    print("Arrays:")
    for key in archive.files:
        print(json.dumps(summarize(key, archive[key]), indent=2, sort_keys=True))

    if metadata_path.exists():
        print(f"\nMetadata: {metadata_path.resolve()}")
        print(metadata_path.read_text(encoding='utf-8'))
    else:
        print("\nMetadata file not found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
