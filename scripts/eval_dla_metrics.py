"""
DLA Fractal Quality Metrics

Evaluates DLA fractal quality using established metrics:
1. Fractal Dimension (box-counting and radius of gyration)
2. Radius of Gyration scaling
3. Radial Density Profile
4. Lacunarity (texture/gappiness)
5. Angular Uniformity
6. Branch Statistics
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import ndimage
from scipy.stats import linregress
from collections import deque


def compute_center_of_mass(positions: np.ndarray) -> np.ndarray:
    """Compute center of mass of particle positions."""
    return np.mean(positions, axis=0)


def compute_radius_of_gyration(positions: np.ndarray, center: np.ndarray = None) -> float:
    """
    Compute radius of gyration: Rg = sqrt(mean(|r - r_cm|^2))

    For DLA, Rg ~ N^(1/D_f) where D_f is fractal dimension (~1.71 in 2D).
    """
    if center is None:
        center = compute_center_of_mass(positions)

    r_squared = np.sum((positions - center) ** 2, axis=1)
    return np.sqrt(np.mean(r_squared))


def compute_fractal_dimension_gyration(
    positions: np.ndarray,
    n_points: int = 20,
    min_particles: int = 50,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute fractal dimension from radius of gyration scaling.

    Method: Rg(N) ~ N^(1/D_f), so log(Rg) = (1/D_f) * log(N) + const

    Returns:
        D_f: Fractal dimension
        r_squared: R² of the linear fit
        log_n: Log of particle counts used
        log_rg: Log of radius of gyration values
    """
    n_total = len(positions)
    center = positions[0]  # Use seed as center (first particle)

    # Sample at different N values
    n_values = np.unique(np.logspace(
        np.log10(min_particles),
        np.log10(n_total),
        n_points
    ).astype(int))
    n_values = n_values[n_values <= n_total]

    rg_values = []
    for n in n_values:
        rg = compute_radius_of_gyration(positions[:n], center)
        rg_values.append(rg)

    log_n = np.log(n_values)
    log_rg = np.log(rg_values)

    # Linear fit: log(Rg) = (1/D_f) * log(N) + const
    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_rg)

    D_f = 1.0 / slope if slope > 0 else np.nan

    return D_f, r_value**2, log_n, log_rg


def compute_fractal_dimension_boxcount(
    positions: np.ndarray,
    grid_size: int = 256,
    n_scales: int = 10,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute fractal dimension using box-counting method.

    Method: N(ε) ~ ε^(-D_f), count boxes of size ε that contain particles.

    Returns:
        D_f: Fractal dimension
        r_squared: R² of the linear fit
        log_eps: Log of box sizes
        log_n: Log of box counts
    """
    # Create binary grid
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    coords = np.clip(positions.astype(int), 0, grid_size - 1)
    grid[coords[:, 0], coords[:, 1]] = 1

    # Box sizes (powers of 2)
    max_power = int(np.log2(grid_size))
    box_sizes = [2**i for i in range(1, max_power)]
    box_sizes = box_sizes[:n_scales]

    counts = []
    for box_size in box_sizes:
        # Count non-empty boxes
        n_boxes_per_dim = grid_size // box_size
        count = 0
        for i in range(n_boxes_per_dim):
            for j in range(n_boxes_per_dim):
                box = grid[
                    i * box_size:(i + 1) * box_size,
                    j * box_size:(j + 1) * box_size
                ]
                if np.any(box):
                    count += 1
        counts.append(count)

    log_eps = np.log(1.0 / np.array(box_sizes))
    log_n = np.log(counts)

    # Linear fit: log(N) = D_f * log(1/ε) + const
    slope, intercept, r_value, p_value, std_err = linregress(log_eps, log_n)

    return slope, r_value**2, log_eps, log_n


def compute_radial_density(
    positions: np.ndarray,
    n_bins: int = 50,
    center: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute radial density profile ρ(r).

    For DLA in 2D: ρ(r) ~ r^(D_f - 2) where D_f ≈ 1.71
    So we expect ρ(r) ~ r^(-0.29)

    Returns:
        r_centers: Bin centers (radial distances)
        density: Particle density at each radius
        exponent: Fitted power law exponent (should be ~D_f - 2 ≈ -0.29)
    """
    if center is None:
        center = positions[0]  # Seed position

    # Compute radial distances
    r = np.sqrt(np.sum((positions - center) ** 2, axis=1))
    r_max = np.max(r)

    # Create radial bins
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2

    # Count particles in annular bins
    counts, _ = np.histogram(r, bins=r_edges)

    # Compute area of each annulus
    areas = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)

    # Density = count / area
    density = counts / (areas + 1e-10)

    # Fit power law to non-zero density values (exclude inner region)
    valid = (density > 0) & (r_centers > r_max * 0.1) & (r_centers < r_max * 0.9)
    if np.sum(valid) > 5:
        log_r = np.log(r_centers[valid])
        log_rho = np.log(density[valid])
        slope, _, _, _, _ = linregress(log_r, log_rho)
        exponent = slope
    else:
        exponent = np.nan

    return r_centers, density, exponent


def compute_lacunarity(
    positions: np.ndarray,
    grid_size: int = 256,
    box_sizes: list = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute lacunarity (texture measure) using gliding-box method.

    Lacunarity Λ(r) = Var(mass) / Mean(mass)² + 1

    - High lacunarity: heterogeneous, gappy structure
    - Low lacunarity: uniform, space-filling structure
    - For fractals, Λ(r) ~ r^α where α relates to the fractal properties

    Returns:
        box_sizes: Box sizes used
        lacunarity: Lacunarity at each scale
    """
    # Create binary grid
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    coords = np.clip(positions.astype(int), 0, grid_size - 1)
    grid[coords[:, 0], coords[:, 1]] = 1

    if box_sizes is None:
        box_sizes = [2, 4, 8, 16, 32, 64]

    lacunarity = []
    for box_size in box_sizes:
        # Compute mass in each box using convolution
        kernel = np.ones((box_size, box_size))
        mass_map = ndimage.convolve(grid, kernel, mode='constant')

        # Sample masses at valid positions
        valid_range = grid_size - box_size
        if valid_range <= 0:
            lacunarity.append(np.nan)
            continue

        masses = mass_map[:valid_range, :valid_range].flatten()

        mean_mass = np.mean(masses)
        var_mass = np.var(masses)

        if mean_mass > 0:
            lac = var_mass / (mean_mass ** 2) + 1
        else:
            lac = np.nan
        lacunarity.append(lac)

    return np.array(box_sizes), np.array(lacunarity)


def compute_angular_uniformity(
    positions: np.ndarray,
    center: np.ndarray = None,
    n_bins: int = 36,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Measure angular uniformity of particle distribution.

    DLA should be roughly isotropic (uniform angular distribution).

    Returns:
        uniformity: 1 = perfectly uniform, 0 = all in one direction
        angles: Bin centers (in radians)
        counts: Particle counts per angular bin
    """
    if center is None:
        center = positions[0]

    # Compute angles
    dx = positions[:, 0] - center[0]
    dy = positions[:, 1] - center[1]
    angles = np.arctan2(dy, dx)

    # Histogram
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(angles, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Uniformity metric (normalized entropy)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(n_bins)
    uniformity = entropy / max_entropy

    return uniformity, bin_centers, counts


def compute_branch_statistics(
    positions: np.ndarray,
    grid_size: int = 256,
) -> Dict:
    """
    Compute branch statistics of the DLA structure.

    Returns:
        n_tips: Number of branch tips (endpoints)
        n_branches: Estimated number of branches
        mean_branch_length: Average branch length
        max_radius: Maximum extent from center
    """
    # Create binary grid
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    coords = np.clip(positions.astype(int), 0, grid_size - 1)
    for x, y in coords:
        grid[x, y] = 1

    # Find tips (particles with only 1 neighbor)
    tips = []
    for x, y in coords:
        neighbors = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    if grid[nx, ny] > 0:
                        neighbors += 1
        if neighbors == 1:
            tips.append((x, y))

    n_tips = len(tips)

    # Estimate branch count (tips / 2 roughly, since branches have 2 ends but one connects to main)
    n_branches = max(1, n_tips)

    # Compute max radius
    center = positions[0]
    distances = np.sqrt(np.sum((positions - center) ** 2, axis=1))
    max_radius = np.max(distances)

    # Mean branch length estimate (total particles / branches)
    mean_branch_length = len(positions) / n_branches if n_branches > 0 else len(positions)

    return {
        "n_tips": n_tips,
        "n_branches": n_branches,
        "mean_branch_length": mean_branch_length,
        "max_radius": max_radius,
    }


def evaluate_fractal_quality(
    positions: np.ndarray,
    grid_size: int = 256,
    expected_D_f: float = 1.71,
    verbose: bool = True,
) -> Dict:
    """
    Comprehensive fractal quality evaluation.

    Args:
        positions: Particle positions (N, 2)
        grid_size: Grid size for grid-based methods
        expected_D_f: Expected fractal dimension for DLA (1.71 in 2D)
        verbose: Print results

    Returns:
        Dictionary with all metrics and quality scores
    """
    results = {}

    # 1. Fractal dimension (gyration method)
    D_f_gyration, r2_gyration, log_n, log_rg = compute_fractal_dimension_gyration(positions)
    results["D_f_gyration"] = D_f_gyration
    results["D_f_gyration_r2"] = r2_gyration

    # 2. Fractal dimension (box-counting)
    D_f_boxcount, r2_boxcount, log_eps, log_n_box = compute_fractal_dimension_boxcount(
        positions, grid_size
    )
    results["D_f_boxcount"] = D_f_boxcount
    results["D_f_boxcount_r2"] = r2_boxcount

    # 3. Radial density profile
    r_centers, density, density_exponent = compute_radial_density(positions)
    results["density_exponent"] = density_exponent
    results["expected_density_exponent"] = expected_D_f - 2  # Should be ~-0.29

    # 4. Lacunarity
    box_sizes, lacunarity = compute_lacunarity(positions, grid_size)
    results["lacunarity_mean"] = np.nanmean(lacunarity)
    results["lacunarity_values"] = lacunarity
    results["lacunarity_scales"] = box_sizes

    # 5. Angular uniformity
    uniformity, angles, angle_counts = compute_angular_uniformity(positions)
    results["angular_uniformity"] = uniformity

    # 6. Branch statistics
    branch_stats = compute_branch_statistics(positions, grid_size)
    results.update(branch_stats)

    # 7. Basic statistics
    results["n_particles"] = len(positions)
    results["radius_of_gyration"] = compute_radius_of_gyration(positions)

    # 8. Quality scores (0-1, higher is better)
    # D_f score: how close to expected 1.71
    D_f_avg = (D_f_gyration + D_f_boxcount) / 2
    D_f_error = abs(D_f_avg - expected_D_f) / expected_D_f
    results["D_f_score"] = max(0, 1 - D_f_error)

    # Fit quality score
    results["fit_score"] = (r2_gyration + r2_boxcount) / 2

    # Angular uniformity score (already 0-1)
    results["isotropy_score"] = uniformity

    # Overall quality score
    results["overall_score"] = (
        results["D_f_score"] * 0.4 +
        results["fit_score"] * 0.3 +
        results["isotropy_score"] * 0.3
    )

    if verbose:
        print("\n" + "=" * 60)
        print("DLA Fractal Quality Metrics")
        print("=" * 60)
        print(f"\nParticles: {results['n_particles']}")
        print(f"Max Radius: {results['max_radius']:.1f}")
        print(f"Radius of Gyration: {results['radius_of_gyration']:.2f}")
        print(f"\n--- Fractal Dimension ---")
        print(f"D_f (gyration):   {D_f_gyration:.3f} (R²={r2_gyration:.4f})")
        print(f"D_f (box-count):  {D_f_boxcount:.3f} (R²={r2_boxcount:.4f})")
        print(f"Expected D_f:     {expected_D_f:.3f}")
        print(f"\n--- Radial Density ---")
        print(f"Density exponent: {density_exponent:.3f} (expected: {expected_D_f - 2:.3f})")
        print(f"\n--- Structure ---")
        print(f"Angular uniformity: {uniformity:.3f} (1.0 = perfect)")
        print(f"Branch tips: {branch_stats['n_tips']}")
        print(f"Mean lacunarity: {results['lacunarity_mean']:.3f}")
        print(f"\n--- Quality Scores (0-1) ---")
        print(f"D_f accuracy:     {results['D_f_score']:.3f}")
        print(f"Fit quality:      {results['fit_score']:.3f}")
        print(f"Isotropy:         {results['isotropy_score']:.3f}")
        print(f"OVERALL SCORE:    {results['overall_score']:.3f}")
        print("=" * 60)

    return results


def evaluate_batch(
    data: np.ndarray,
    grid_size: int = 256,
    sample_size: int = 100,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a batch of DLA simulations.

    Args:
        data: Batch data (n_runs, n_particles, 3)
        grid_size: Grid size
        sample_size: Number of runs to sample (for speed)
        verbose: Print summary

    Returns:
        Dictionary with aggregated metrics
    """
    n_runs = len(data)
    sample_indices = np.random.choice(n_runs, min(sample_size, n_runs), replace=False)

    all_metrics = []
    for idx in sample_indices:
        positions = data[idx, :, :2]  # x, y only
        metrics = evaluate_fractal_quality(positions, grid_size, verbose=False)
        all_metrics.append(metrics)

    # Aggregate - only mean values, consistent with evaluate_dla_folder.py
    agg = {
        "D_f_gyration": np.mean([m["D_f_gyration"] for m in all_metrics]),
        "D_f_boxcount": np.mean([m["D_f_boxcount"] for m in all_metrics]),
        "overall_score": np.mean([m["overall_score"] for m in all_metrics]),
        "D_f_score": np.mean([m["D_f_score"] for m in all_metrics]),
        "fit_score": np.mean([m["fit_score"] for m in all_metrics]),
        "isotropy_score": np.mean([m["isotropy_score"] for m in all_metrics]),
        "angular_uniformity": np.mean([m["angular_uniformity"] for m in all_metrics]),
    }

    if verbose:
        print(f"\n[RESULT] Averaged metrics over {len(sample_indices)} files:")
        print(f"  D_f_gyration      = {agg['D_f_gyration']:.6f}")
        print(f"  D_f_boxcount      = {agg['D_f_boxcount']:.6f}")
        print(f"  overall_score     = {agg['overall_score']:.6f}")
        print(f"  D_f_score         = {agg['D_f_score']:.6f}")
        print(f"  fit_score         = {agg['fit_score']:.6f}")
        print(f"  isotropy_score    = {agg['isotropy_score']:.6f}")
        print(f"  angular_uniformity = {agg['angular_uniformity']:.6f}")

    return agg


def evaluate_ply_folder(
    folder_path: str,
    grid_size: int = 256,
    verbose: bool = True,
    mode: str = "3d",
) -> Dict:
    """
    Evaluate all PLY files in a folder.

    Args:
        folder_path: Path to folder containing PLY files
        grid_size: Grid size for grid-based methods
        verbose: Print results
        mode: "3d" (sort by z, extract xy) or "2d" (data is already sorted xy)

    Returns:
        Dictionary with aggregated metrics
    """
    import glob
    import os

    try:
        import open3d as o3d
    except ImportError:
        print("请先安装 open3d: pip install open3d")
        return {}

    # Find all PLY files
    ply_files = sorted(glob.glob(os.path.join(folder_path, "*.ply")))

    if not ply_files:
        print(f"[ERROR] No PLY files found in {folder_path}")
        return {}

    print(f"Found {len(ply_files)} PLY files in {folder_path}, mode={mode}")

    all_metrics = []
    for ply_file in ply_files:
        # Load PLY and extract xy coordinates
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)

        if mode == "2d":
            # 2D mode: data is already sorted, just use xy (z might be 0)
            positions = points[:, :2]
        else:
            # 3D mode: Sort by z (time dimension) for correct gyration calculation
            sort_idx = np.argsort(points[:, 2])
            points = points[sort_idx]
            positions = points[:, :2]  # Only use x, y

        # Normalize to [0, grid_size] range for proper fractal analysis
        pos_min = positions.min(axis=0, keepdims=True)
        pos_max = positions.max(axis=0, keepdims=True)
        pos_range = (pos_max - pos_min).max()
        if pos_range < 1e-8:
            pos_range = 1.0
        # Scale to [margin, grid_size - margin] to avoid edge issues
        margin = grid_size * 0.05
        positions = (positions - pos_min) / pos_range * (grid_size - 2 * margin) + margin

        # Evaluate
        metrics = evaluate_fractal_quality(positions, grid_size, verbose=False)
        all_metrics.append(metrics)

    # Aggregate - only mean values, consistent with evaluate_dla_folder.py
    agg = {
        "D_f_gyration": np.mean([m["D_f_gyration"] for m in all_metrics]),
        "D_f_boxcount": np.mean([m["D_f_boxcount"] for m in all_metrics]),
        "overall_score": np.mean([m["overall_score"] for m in all_metrics]),
        "D_f_score": np.mean([m["D_f_score"] for m in all_metrics]),
        "fit_score": np.mean([m["fit_score"] for m in all_metrics]),
        "isotropy_score": np.mean([m["isotropy_score"] for m in all_metrics]),
        "angular_uniformity": np.mean([m["angular_uniformity"] for m in all_metrics]),
    }

    if verbose:
        print(f"\n[RESULT] Averaged metrics over {len(ply_files)} files:")
        print(f"  D_f_gyration      = {agg['D_f_gyration']:.6f}")
        print(f"  D_f_boxcount      = {agg['D_f_boxcount']:.6f}")
        print(f"  overall_score     = {agg['overall_score']:.6f}")
        print(f"  D_f_score         = {agg['D_f_score']:.6f}")
        print(f"  fit_score         = {agg['fit_score']:.6f}")
        print(f"  isotropy_score    = {agg['isotropy_score']:.6f}")
        print(f"  angular_uniformity = {agg['angular_uniformity']:.6f}")

    return agg


def evaluate_npz_dataset(
    npz_path: str,
    grid_size: int = 256,
    n_samples: int = 100,
    verbose: bool = True,
    mode: str = "3d",
) -> Dict:
    """
    Evaluate NPZ dataset (ground truth DLA data).

    NPZ format: data['points'] with shape (num_samples, num_points, 3) for 3D
                or (num_samples, num_points, 2) for 2D

    Args:
        npz_path: Path to NPZ file
        grid_size: Grid size for grid-based methods
        n_samples: Number of samples to evaluate
        verbose: Print results
        mode: "3d" (sort by z, extract xy) or "2d" (data is already sorted xy)

    Returns:
        Dictionary with aggregated metrics
    """
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)

    # Support both 'points' and 'data' keys
    if 'points' in data:
        points_all = data['points']  # (num_samples, num_points, D)
    elif 'data' in data:
        points_all = data['data']
    else:
        raise KeyError(f"NPZ file must contain 'points' or 'data' field. Found: {list(data.keys())}")

    n_total = points_all.shape[0]
    print(f"Loaded {n_total} samples, shape: {points_all.shape}, mode={mode}")

    # Sample randomly if needed
    if n_samples < n_total:
        indices = np.random.choice(n_total, n_samples, replace=False)
    else:
        indices = np.arange(n_total)
        n_samples = n_total

    print(f"Evaluating {n_samples} samples...")

    all_metrics = []
    for idx in indices:
        points = points_all[idx]  # (num_points, D)

        if mode == "2d":
            # 2D mode: data is already sorted xy
            positions = points[:, :2]
        else:
            # 3D mode: Sort by z (time dimension)
            sort_idx = np.argsort(points[:, 2])
            points = points[sort_idx]
            positions = points[:, :2]  # Only use x, y

        # Normalize to [0, grid_size] range for proper fractal analysis
        pos_min = positions.min(axis=0, keepdims=True)
        pos_max = positions.max(axis=0, keepdims=True)
        pos_range = (pos_max - pos_min).max()
        if pos_range < 1e-8:
            pos_range = 1.0
        margin = grid_size * 0.05
        positions = (positions - pos_min) / pos_range * (grid_size - 2 * margin) + margin

        # Evaluate
        metrics = evaluate_fractal_quality(positions, grid_size, verbose=False)
        all_metrics.append(metrics)

    # Aggregate - only mean values, consistent with evaluate_dla_folder.py
    agg = {
        "D_f_gyration": np.mean([m["D_f_gyration"] for m in all_metrics]),
        "D_f_boxcount": np.mean([m["D_f_boxcount"] for m in all_metrics]),
        "overall_score": np.mean([m["overall_score"] for m in all_metrics]),
        "D_f_score": np.mean([m["D_f_score"] for m in all_metrics]),
        "fit_score": np.mean([m["fit_score"] for m in all_metrics]),
        "isotropy_score": np.mean([m["isotropy_score"] for m in all_metrics]),
        "angular_uniformity": np.mean([m["angular_uniformity"] for m in all_metrics]),
    }

    if verbose:
        print(f"\n[RESULT] Averaged metrics over {len(indices)} files:")
        print(f"  D_f_gyration      = {agg['D_f_gyration']:.6f}")
        print(f"  D_f_boxcount      = {agg['D_f_boxcount']:.6f}")
        print(f"  overall_score     = {agg['overall_score']:.6f}")
        print(f"  D_f_score         = {agg['D_f_score']:.6f}")
        print(f"  fit_score         = {agg['fit_score']:.6f}")
        print(f"  isotropy_score    = {agg['isotropy_score']:.6f}")
        print(f"  angular_uniformity = {agg['angular_uniformity']:.6f}")

    return agg


def main():
    """CLI for fractal metrics evaluation."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Evaluate DLA fractal quality")
    parser.add_argument(
        "input",
        type=str,
        help="Input folder containing PLY files, or NPZ file (ground truth dataset)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="3d",
        choices=["3d", "2d"],
        help="Data mode: 3d (extract xy from xyz) or 2d (data is already xy)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Number of samples for evaluation (default: 100)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=256,
        help="Grid size (default: 256)",
    )

    args = parser.parse_args()

    # Check if input is a folder or npz file
    if os.path.isdir(args.input):
        # PLY folder mode
        evaluate_ply_folder(args.input, args.grid_size, verbose=True, mode=args.mode)
    elif args.input.endswith('.npz'):
        # NPZ dataset mode (ground truth)
        evaluate_npz_dataset(args.input, args.grid_size, args.sample, verbose=True, mode=args.mode)
    else:
        print(f"[ERROR] Input must be a folder or .npz file, got: {args.input}")


if __name__ == "__main__":
    main()
