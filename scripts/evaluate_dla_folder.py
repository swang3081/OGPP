#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_dla_folder.py

Evaluate all PLY files in a folder and compute averaged DLA fractal metrics.
Optimized with multiprocessing and vectorized operations.

Usage:
    python scripts/evaluate_dla_folder.py --folder outputs/dla_eval/dla_ours_e2001/step_20/dla_ours_e2001

Output:
    Creates <folder>/metrics.txt with averaged metrics (one per line):
        D_f_gyration=<value>
        D_f_boxcount=<value>
        overall_score=<value>
        D_f_score=<value>
        fit_score=<value>
        isotropy_score=<value>
        angular_uniformity=<value>
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from scipy.stats import linregress

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


# Default parameters
DEFAULT_GRID_SIZE = 256  # Same as eval_dla_metrics.py for consistency
DEFAULT_EXPECTED_DF = 1.71
DEFAULT_NUM_WORKERS = 8


# ============== Fast Metric Functions (Vectorized) ==============

def fast_fractal_dimension_gyration(positions: np.ndarray, n_points: int = 15, min_particles: int = 50):
    """
    Fast gyration-based fractal dimension.

    For DLA, we compute the radius of gyration R_g for subsets of N particles
    and fit N ~ R_g^D_f to find the fractal dimension.

    IMPORTANT: Points must be in growth order (index 0 = seed point).
    """
    n_total = len(positions)

    # Not enough points for meaningful analysis
    if n_total < min_particles:
        return np.nan, 0.0

    # First point is the seed (center of DLA growth)
    center = positions[0]

    n_values = np.unique(np.logspace(
        np.log10(min_particles),
        np.log10(n_total),
        n_points
    ).astype(int))
    n_values = n_values[n_values <= n_total]

    # Need at least 2 points for linear regression
    if len(n_values) < 2:
        return np.nan, 0.0

    rg_values = []
    for n in n_values:
        # Compute radius of gyration for first n particles (in growth order)
        r_squared = np.sum((positions[:n] - center) ** 2, axis=1)
        rg = np.sqrt(np.mean(r_squared))
        # Avoid log(0) by skipping near-zero values
        if rg < 1e-10:
            rg = 1e-10
        rg_values.append(rg)

    log_n = np.log(n_values)
    log_rg = np.log(rg_values)

    # Check for valid values
    if not np.all(np.isfinite(log_n)) or not np.all(np.isfinite(log_rg)):
        return np.nan, 0.0

    slope, _, r_value, _, _ = linregress(log_n, log_rg)
    D_f = 1.0 / slope if slope > 0 else np.nan

    return D_f, r_value**2


def fast_fractal_dimension_boxcount(positions: np.ndarray, grid_size: int = 128):
    """Vectorized box-counting using reshape instead of loops."""
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    coords = np.clip(positions.astype(int), 0, grid_size - 1)
    grid[coords[:, 0], coords[:, 1]] = 1

    max_power = int(np.log2(grid_size))
    box_sizes = [2**i for i in range(1, max_power)]

    counts = []
    for box_size in box_sizes:
        n_boxes = grid_size // box_size
        if n_boxes == 0:
            continue
        # Reshape grid into boxes and check non-empty
        # grid shape: (grid_size, grid_size)
        # We want boxes of shape (n_boxes, n_boxes), each covering (box_size, box_size)
        trimmed = grid[:n_boxes * box_size, :n_boxes * box_size]
        # Reshape to (n_boxes, box_size, n_boxes, box_size)
        reshaped = trimmed.reshape(n_boxes, box_size, n_boxes, box_size)
        # Transpose to (n_boxes, n_boxes, box_size, box_size) so each box is contiguous
        reshaped = reshaped.transpose(0, 2, 1, 3)
        # Check if any point exists in each box
        box_has_point = reshaped.any(axis=(2, 3))
        counts.append(np.sum(box_has_point))

    if len(counts) < 2:
        return np.nan, 0.0

    log_eps = np.log(1.0 / np.array(box_sizes[:len(counts)]))
    log_n = np.log(counts)

    slope, _, r_value, _, _ = linregress(log_eps, log_n)
    return slope, r_value**2


def fast_angular_uniformity(positions: np.ndarray, n_bins: int = 36):
    """Fast angular uniformity calculation."""
    center = positions[0]
    dx = positions[:, 0] - center[0]
    dy = positions[:, 1] - center[1]
    angles = np.arctan2(dy, dx)

    counts, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(n_bins)

    return entropy / max_entropy


def fast_evaluate_fractal_quality(positions: np.ndarray, grid_size: int = 128, expected_D_f: float = 1.71):
    """Fast fractal quality evaluation with minimal metrics."""
    # Fractal dimension (gyration)
    D_f_gyration, r2_gyration = fast_fractal_dimension_gyration(positions)

    # Fractal dimension (box-counting) - vectorized
    D_f_boxcount, r2_boxcount = fast_fractal_dimension_boxcount(positions, grid_size)

    # Angular uniformity
    uniformity = fast_angular_uniformity(positions)

    # Quality scores
    D_f_avg = (D_f_gyration + D_f_boxcount) / 2 if np.isfinite(D_f_boxcount) else D_f_gyration
    D_f_error = abs(D_f_avg - expected_D_f) / expected_D_f
    D_f_score = max(0, 1 - D_f_error)
    fit_score = (r2_gyration + r2_boxcount) / 2 if np.isfinite(r2_boxcount) else r2_gyration
    isotropy_score = uniformity
    overall_score = D_f_score * 0.4 + fit_score * 0.3 + isotropy_score * 0.3

    return {
        "D_f_gyration": D_f_gyration,
        "D_f_boxcount": D_f_boxcount,
        "overall_score": overall_score,
        "D_f_score": D_f_score,
        "fit_score": fit_score,
        "isotropy_score": isotropy_score,
        "angular_uniformity": uniformity,
    }


def load_ply_points(ply_path: str) -> np.ndarray:
    """
    Load points from a PLY file.

    Args:
        ply_path: Path to PLY file

    Returns:
        points: (N, 3) numpy array of xyz coordinates
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        return points
    except ImportError:
        # Fallback to trimesh
        import trimesh
        mesh = trimesh.load(ply_path)
        if hasattr(mesh, 'vertices'):
            return np.asarray(mesh.vertices)
        else:
            return np.asarray(mesh.points) if hasattr(mesh, 'points') else None


def evaluate_single_ply(
    ply_path: str,
    grid_size: int = DEFAULT_GRID_SIZE,
    expected_D_f: float = DEFAULT_EXPECTED_DF,
) -> dict:
    """
    Evaluate a single PLY file and return metrics as a dict.

    Args:
        ply_path: Path to PLY file
        grid_size: Grid size for grid-based methods
        expected_D_f: Expected fractal dimension for DLA

    Returns:
        Dictionary with metrics
    """
    # Load points
    points = load_ply_points(ply_path)
    if points is None or len(points) == 0:
        raise RuntimeError(f"Failed to load points from {ply_path}")

    # Check if 2D or 3D data
    # Note: Open3D always returns 3D points, so we check z variance to detect 2D data
    if points.shape[1] == 2:
        # Pure 2D data: keep original order (points are already in growth order)
        positions = points
    elif points.shape[1] >= 3:
        # Check if z values have meaningful variation (i.e., actual 3D/time data)
        z_values = points[:, 2]
        z_range = z_values.max() - z_values.min()

        if z_range < 1e-6:
            # Z values are essentially constant (2D data stored as 3D)
            # Keep original order - points are already in growth order
            positions = points[:, :2]
        else:
            # 3D data: sort by z (time dimension) for correct gyration calculation
            sort_idx = np.argsort(z_values)
            points = points[sort_idx]
            positions = points[:, :2]  # Only use x, y for 2D fractal analysis
    else:
        raise RuntimeError(f"Invalid point dimension: {points.shape[1]}")

    # Normalize to [0, grid_size] range for proper fractal analysis
    pos_min = positions.min(axis=0, keepdims=True)
    pos_max = positions.max(axis=0, keepdims=True)
    pos_range = (pos_max - pos_min).max()
    if pos_range < 1e-8:
        pos_range = 1.0
    # Scale to [margin, grid_size - margin] to avoid edge issues
    margin = grid_size * 0.05
    positions = (positions - pos_min) / pos_range * (grid_size - 2 * margin) + margin

    # Evaluate fractal quality using fast vectorized functions
    return fast_evaluate_fractal_quality(positions, grid_size, expected_D_f)


def _evaluate_single_ply_wrapper(args):
    """Wrapper for multiprocessing - unpacks arguments."""
    ply_path, grid_size, expected_D_f = args
    try:
        return evaluate_single_ply(ply_path, grid_size, expected_D_f)
    except Exception as e:
        return None


def evaluate_folder(
    folder: str,
    grid_size: int = DEFAULT_GRID_SIZE,
    expected_D_f: float = DEFAULT_EXPECTED_DF,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict:
    """
    Evaluate all PLY files in <folder>/ply/ and return averaged metrics.
    Uses multiprocessing for parallel evaluation.

    Args:
        folder: Folder containing ply/ subdirectory
        grid_size: Grid size for grid-based methods
        expected_D_f: Expected fractal dimension for DLA
        num_workers: Number of parallel workers

    Returns:
        Dictionary with averaged metrics
    """
    folder_path = Path(folder)
    ply_dir = folder_path / "ply"

    if not ply_dir.exists():
        raise FileNotFoundError(f"PLY directory not found: {ply_dir}")

    # Find all PLY files
    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in: {ply_dir}")

    n_files = len(ply_files)
    print(f"[INFO] Found {n_files} PLY files in {ply_dir}")
    print(f"[INFO] Using {num_workers} workers for parallel processing")

    # Prepare arguments for multiprocessing
    args_list = [(str(p), grid_size, expected_D_f) for p in ply_files]

    # Parallel evaluation
    all_metrics = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_evaluate_single_ply_wrapper, args): i
                   for i, args in enumerate(args_list)}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0 or completed == n_files:
                print(f"[INFO] Progress: {completed}/{n_files}")

            result = future.result()
            if result is not None:
                all_metrics.append(result)

    if not all_metrics:
        raise RuntimeError("No PLY files were successfully processed")

    print(f"[INFO] Successfully processed {len(all_metrics)}/{n_files} files")

    # Average metrics
    avg_metrics = {}
    metric_keys = [
        "D_f_gyration", "D_f_boxcount", "overall_score",
        "D_f_score", "fit_score", "isotropy_score", "angular_uniformity"
    ]

    for key in metric_keys:
        values = [m[key] for m in all_metrics if np.isfinite(m[key])]
        if not values:
            avg_metrics[key] = float("nan")
        else:
            avg_metrics[key] = np.mean(values)

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DLA metrics for all PLY files in a folder")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing ply/ subdirectory with PLY files")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE,
                        help=f"Grid size for grid-based methods (default: {DEFAULT_GRID_SIZE})")
    parser.add_argument("--expected-df", type=float, default=DEFAULT_EXPECTED_DF,
                        help=f"Expected fractal dimension (default: {DEFAULT_EXPECTED_DF})")
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_NUM_WORKERS})")

    args = parser.parse_args()

    print(f"[INFO] Evaluating folder: {args.folder}")

    avg_metrics = evaluate_folder(
        folder=args.folder,
        grid_size=args.grid_size,
        expected_D_f=args.expected_df,
        num_workers=args.workers,
    )

    # Write metrics to file
    out_path = Path(args.folder) / "metrics.txt"
    metric_keys = [
        "D_f_gyration", "D_f_boxcount", "overall_score",
        "D_f_score", "fit_score", "isotropy_score", "angular_uniformity"
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        for key in metric_keys:
            f.write(f"{key}={avg_metrics[key]}\n")

    n_files = len(list(Path(args.folder).joinpath('ply').glob('*.ply')))
    print(f"\n[RESULT] Averaged metrics over {n_files} files:")
    print(f"  D_f_gyration      = {avg_metrics['D_f_gyration']:.6f}")
    print(f"  D_f_boxcount      = {avg_metrics['D_f_boxcount']:.6f}")
    print(f"  overall_score     = {avg_metrics['overall_score']:.6f}")
    print(f"  D_f_score         = {avg_metrics['D_f_score']:.6f}")
    print(f"  fit_score         = {avg_metrics['fit_score']:.6f}")
    print(f"  isotropy_score    = {avg_metrics['isotropy_score']:.6f}")
    print(f"  angular_uniformity = {avg_metrics['angular_uniformity']:.6f}")
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
