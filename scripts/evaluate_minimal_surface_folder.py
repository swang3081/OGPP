#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_minimal_surface_folder.py

Evaluate all NPZ files in a folder and compute averaged minimal surface metrics.

Usage:
    python scripts/evaluate_minimal_surface_folder.py --folder outputs/minimal_surface_eval/ours_e4721/step_20/ours_e4721

Output:
    Creates <folder>/metrics.txt with averaged metrics (one per line):
        area_fraction=<value>
        area_fraction_error=<value>
        angle_smoothness=<value>
        mean_curvature=<value>
        curvature_std=<value>
        uniformity_cv=<value>
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import metric functions from eval_minimal_surface_metrics.py
from scripts.eval_minimal_surface_metrics import (
    evaluate_sample,
    evaluate_batch,
)


# Default parameters
DEFAULT_TARGET_AREA_FRACTION = 0.7


def evaluate_folder(
    folder: str,
    target_area_fraction: float = DEFAULT_TARGET_AREA_FRACTION,
) -> dict:
    """
    Evaluate the NPZ file in <folder>/npz/generated.npz and return metrics.

    Args:
        folder: Folder containing npz/ subdirectory with generated.npz
        target_area_fraction: Expected area fraction for minimal surface

    Returns:
        Dictionary with averaged metrics
    """
    folder_path = Path(folder)
    npz_dir = folder_path / "npz"
    npz_file = npz_dir / "generated.npz"

    if not npz_file.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")

    print(f"[INFO] Loading NPZ file: {npz_file}")

    # Load data
    data = np.load(npz_file)
    edge_points = data['edge_points']  # (B, N, 2)
    anchors = data['anchors']  # (B, 3, 2)

    B = edge_points.shape[0]
    print(f"[INFO] Found {B} samples in NPZ file")

    # Evaluate batch
    results, per_sample = evaluate_batch(
        edge_points, anchors,
        ground_truth=None,
        target_area_fraction=target_area_fraction
    )

    # Extract averaged metrics (results contains *_mean and *_std keys)
    avg_metrics = {
        "area_fraction": results['area_fraction_mean'],
        "area_fraction_error": results['area_fraction_error_mean'],
        "angle_smoothness": results['angle_smoothness_mean'],
        "mean_curvature": results['mean_curvature_mean'],
        "curvature_std": results['curvature_std_mean'],
        "uniformity_cv": results['uniformity_cv_mean'],
    }

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate minimal surface metrics for NPZ file in a folder")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing npz/ subdirectory with generated.npz")
    parser.add_argument("--target-area-fraction", type=float, default=DEFAULT_TARGET_AREA_FRACTION,
                        help=f"Target area fraction (default: {DEFAULT_TARGET_AREA_FRACTION})")

    args = parser.parse_args()

    print(f"[INFO] Evaluating folder: {args.folder}")

    avg_metrics = evaluate_folder(
        folder=args.folder,
        target_area_fraction=args.target_area_fraction,
    )

    # Write metrics to file
    out_path = Path(args.folder) / "metrics.txt"
    metric_keys = [
        "area_fraction", "area_fraction_error", "angle_smoothness",
        "mean_curvature", "curvature_std", "uniformity_cv"
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        for key in metric_keys:
            f.write(f"{key}={avg_metrics[key]}\n")

    # Print results
    npz_file = Path(args.folder) / "npz" / "generated.npz"
    data = np.load(npz_file)
    n_samples = data['edge_points'].shape[0]

    print(f"\n[RESULT] Averaged metrics over {n_samples} samples:")
    print(f"  area_fraction      = {avg_metrics['area_fraction']:.6f}")
    print(f"  area_fraction_error = {avg_metrics['area_fraction_error']:.6f}")
    print(f"  angle_smoothness   = {avg_metrics['angle_smoothness']:.6f}")
    print(f"  mean_curvature     = {avg_metrics['mean_curvature']:.6f}")
    print(f"  curvature_std      = {avg_metrics['curvature_std']:.6f}")
    print(f"  uniformity_cv      = {avg_metrics['uniformity_cv']:.6f}")
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
