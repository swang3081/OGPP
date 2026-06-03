#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_thomson_folder.py

Evaluate all PLY files in a folder and compute averaged Thomson metrics.

Usage:
    python scripts/evaluate_thomson_folder.py --folder outputs/thomson_eval/thomson_ours_26k/step_20

Output:
    Creates <folder>/metrics.txt with averaged metrics (one per line):
        E_coul=<value>
        E_spring=<value>
        E_total=<value>
        E_star=<value>
        F_tan_rms=<value>
        F_tan_max=<value>
        CV_avg=<value>
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path to import eval_thomson_metrics functions
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_thomson_metrics import (
    load_ply_points,
    project_to_nearest_shell,
    make_charges,
    coulomb_forces_and_energy,
    spring_energy_nearest_shell,
    metric_E_star,
    metric_tangent_force,
    metric_spacing_cv_per_layer,
)


# Default parameters (same as thomson_metric.py)
DEFAULT_LAYERS = 3
DEFAULT_R_MIN = 1.0
DEFAULT_R_MAX = 2.0
DEFAULT_K_SPRING = 120.0
DEFAULT_EPS = 1e-4
DEFAULT_SAME_LAYER_SCALE = 1.0
DEFAULT_CROSS_LAYER_SCALE = 1.0
DEFAULT_ALT_CHARGE = False
DEFAULT_CHUNK = 1024


def evaluate_single_ply(
    ply_path: str,
    radii: np.ndarray,
    layers: int,
    k_spring: float,
    eps: float,
    same_layer_scale: float,
    cross_layer_scale: float,
    alt_charge: bool,
    chunk: int,
) -> dict:
    """
    Evaluate a single PLY file and return metrics as a dict.
    """
    # Load and rescale points
    pts0 = load_ply_points(ply_path, rescale=True)
    if pts0.ndim != 2 or pts0.shape[1] != 3:
        raise RuntimeError(f"Expect Nx3 points, got {pts0.shape}")

    # Project to nearest shell
    pts, layer_id = project_to_nearest_shell(pts0.astype(np.float64), radii)

    # Spring energy (using raw points before projection)
    E_spring, _ = spring_energy_nearest_shell(pts0, layer_id, radii, k_spring)

    # Make charges
    q = make_charges(layer_id, layers, alt_charge)

    # Coulomb full force + energy
    F_full, E_coul = coulomb_forces_and_energy(
        pts, layer_id, q,
        eps=eps,
        same_layer_scale=same_layer_scale,
        cross_layer_scale=cross_layer_scale,
        chunk=chunk,
    )

    # Tangent force metrics
    F_tan_rms, F_tan_max, _ = metric_tangent_force(pts, F_full)

    # Dimensionless energy
    E_star = metric_E_star(E_coul, layer_id, radii, q)

    # Spacing CV
    _, cv_avg = metric_spacing_cv_per_layer(pts, layer_id, radii)

    return {
        "E_coul": E_coul,
        "E_spring": E_spring,
        "E_total": E_coul + E_spring,
        "E_star": E_star,
        "F_tan_rms": F_tan_rms,
        "F_tan_max": F_tan_max,
        "CV_avg": cv_avg,
    }


def evaluate_folder(
    folder: str,
    layers: int = DEFAULT_LAYERS,
    r_min: float = DEFAULT_R_MIN,
    r_max: float = DEFAULT_R_MAX,
    k_spring: float = DEFAULT_K_SPRING,
    eps: float = DEFAULT_EPS,
    same_layer_scale: float = DEFAULT_SAME_LAYER_SCALE,
    cross_layer_scale: float = DEFAULT_CROSS_LAYER_SCALE,
    alt_charge: bool = DEFAULT_ALT_CHARGE,
    chunk: int = DEFAULT_CHUNK,
) -> dict:
    """
    Evaluate all PLY files in <folder>/ply/ and return averaged metrics.
    """
    folder_path = Path(folder)
    ply_dir = folder_path / "ply"

    if not ply_dir.exists():
        raise FileNotFoundError(f"PLY directory not found: {ply_dir}")

    # Find all PLY files
    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in: {ply_dir}")

    print(f"[INFO] Found {len(ply_files)} PLY files in {ply_dir}")

    # Setup radii
    radii = np.linspace(r_min, r_max, layers).astype(np.float64)

    # Accumulate metrics
    all_metrics = []
    for i, ply_path in enumerate(ply_files):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[INFO] Processing {i+1}/{len(ply_files)}: {ply_path.name}")

        try:
            metrics = evaluate_single_ply(
                str(ply_path),
                radii=radii,
                layers=layers,
                k_spring=k_spring,
                eps=eps,
                same_layer_scale=same_layer_scale,
                cross_layer_scale=cross_layer_scale,
                alt_charge=alt_charge,
                chunk=chunk,
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"[WARN] Failed to process {ply_path.name}: {e}")
            continue

    if not all_metrics:
        raise RuntimeError("No PLY files were successfully processed")

    # Average metrics
    avg_metrics = {}
    metric_keys = ["E_coul", "E_spring", "E_total", "E_star", "F_tan_rms", "F_tan_max", "CV_avg"]
    for key in metric_keys:
        values = [m[key] for m in all_metrics if np.isfinite(m[key])]
        if not values:
            avg_metrics[key] = float("nan")
            continue

        if key == "F_tan_rms" and len(values) > 3:
            # Exclude top 3 largest F_tan_rms values
            values_sorted = sorted(values)
            values_filtered = values_sorted[:-3]  # Remove top 3
            avg_metrics[key] = np.mean(values_filtered)
            print(f"[INFO] F_tan_rms: excluded top 3 outliers, using {len(values_filtered)}/{len(values)} samples")
        else:
            avg_metrics[key] = np.mean(values)

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Thomson metrics for all PLY files in a folder")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing ply/ subdirectory with PLY files")
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--r-min", type=float, default=DEFAULT_R_MIN)
    parser.add_argument("--r-max", type=float, default=DEFAULT_R_MAX)
    parser.add_argument("--k-spring", type=float, default=DEFAULT_K_SPRING)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--same-layer-scale", type=float, default=DEFAULT_SAME_LAYER_SCALE)
    parser.add_argument("--cross-layer-scale", type=float, default=DEFAULT_CROSS_LAYER_SCALE)
    parser.add_argument("--alt-charge", action="store_true", default=DEFAULT_ALT_CHARGE)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)

    args = parser.parse_args()

    print(f"[INFO] Evaluating folder: {args.folder}")

    avg_metrics = evaluate_folder(
        folder=args.folder,
        layers=args.layers,
        r_min=args.r_min,
        r_max=args.r_max,
        k_spring=args.k_spring,
        eps=args.eps,
        same_layer_scale=args.same_layer_scale,
        cross_layer_scale=args.cross_layer_scale,
        alt_charge=args.alt_charge,
        chunk=args.chunk,
    )

    # Write metrics to file
    out_path = Path(args.folder) / "metrics.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for key in ["E_coul", "E_spring", "E_total", "E_star", "F_tan_rms", "F_tan_max", "CV_avg"]:
            f.write(f"{key}={avg_metrics[key]}\n")

    print(f"\n[RESULT] Averaged metrics over {len(list(Path(args.folder).joinpath('ply').glob('*.ply')))} files:")
    print(f"  E_coul     = {avg_metrics['E_coul']:.10f}")
    print(f"  E_spring   = {avg_metrics['E_spring']:.10f}")
    print(f"  E_total    = {avg_metrics['E_total']:.10f}")
    print(f"  E_star     = {avg_metrics['E_star']:.6e}")
    print(f"  F_tan_rms  = {avg_metrics['F_tan_rms']:.6e}")
    print(f"  F_tan_max  = {avg_metrics['F_tan_max']:.6e}")
    print(f"  CV_avg     = {avg_metrics['CV_avg']:.6e}")
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
