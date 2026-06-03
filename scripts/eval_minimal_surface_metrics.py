#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_minimal_surface_metrics.py

Evaluation metrics for minimal surface generation:
1. Area Fraction: area enclosed by points / convex hull area (target ~0.7)
2. Smoothness: measures curve smoothness via angle variation
3. Uniformity: measures point distribution uniformity along the curve
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def order_points_by_angle(points):
    """
    Order 2D points by angle from centroid (for proper polygon area calculation).

    Args:
        points: (N, 2) array
    Returns:
        ordered_points: (N, 2) array ordered counter-clockwise
        order: indices of the ordering
    """
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    order = np.argsort(angles)
    return points[order], order


def compute_polygon_area(points):
    """
    Compute area of polygon using shoelace formula.
    Points should be ordered (either CW or CCW).

    Args:
        points: (N, 2) array of ordered points
    Returns:
        area: absolute area of the polygon
    """
    n = len(points)
    if n < 3:
        return 0.0

    # Shoelace formula
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


def compute_convex_hull_area(points):
    """
    Compute convex hull area of 2D points.

    Args:
        points: (N, 2) array
    Returns:
        hull_area: area of convex hull
    """
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except:
        return 0.0


def compute_triangle_area(anchors):
    """
    Compute area of triangle formed by 3 anchor points.

    Args:
        anchors: (3, 2) array of anchor points
    Returns:
        area: triangle area
    """
    if anchors.shape[0] != 3:
        return compute_convex_hull_area(anchors)

    # Shoelace formula for triangle
    x = anchors[:, 0]
    y = anchors[:, 1]
    area = 0.5 * np.abs(
        x[0] * (y[1] - y[2]) +
        x[1] * (y[2] - y[0]) +
        x[2] * (y[0] - y[1])
    )
    return area


def compute_area_fraction(points, anchors, target_fraction=0.7):
    """
    Compute area fraction metric.
    Area fraction = edge_points_polygon_area / anchors_triangle_area

    Args:
        points: (N, 2) array of edge points
        anchors: (3, 2) array of anchor points
        target_fraction: expected area fraction (default 0.7)
    Returns:
        fraction: edge_points_area / anchors_triangle_area
        error: |fraction - target_fraction|
        polygon_area: area of polygon formed by edge points
        anchor_area: area of triangle formed by anchors
    """
    # Order points for polygon area calculation
    ordered_points, _ = order_points_by_angle(points)

    polygon_area = compute_polygon_area(ordered_points)
    anchor_area = compute_triangle_area(anchors)

    if anchor_area < 1e-10:
        return 0.0, 1.0, polygon_area, anchor_area

    fraction = polygon_area / anchor_area
    error = np.abs(fraction - target_fraction)

    return fraction, error, polygon_area, anchor_area


def compute_smoothness(points):
    """
    Compute smoothness metric based on angle variation between consecutive segments.
    Lower values indicate smoother curves.

    Args:
        points: (N, 2) array of edge points (should be ordered)
    Returns:
        smoothness: mean absolute angle change (radians)
        angle_std: std of angle changes
    """
    # Order points by angle from centroid
    ordered_points, _ = order_points_by_angle(points)
    n = len(ordered_points)

    if n < 3:
        return 0.0, 0.0

    # Compute vectors between consecutive points (closed loop)
    vectors = np.diff(ordered_points, axis=0, append=ordered_points[:1])

    # Compute angles of each vector
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # Compute angle changes between consecutive segments
    angle_changes = np.diff(angles, append=angles[:1])

    # Normalize to [-pi, pi]
    angle_changes = np.arctan2(np.sin(angle_changes), np.cos(angle_changes))

    # Smoothness metrics
    mean_abs_change = np.mean(np.abs(angle_changes))
    std_change = np.std(angle_changes)

    return mean_abs_change, std_change


def compute_uniformity(points):
    """
    Compute uniformity metric based on spacing between consecutive points.
    Lower coefficient of variation indicates more uniform distribution.

    Args:
        points: (N, 2) array of edge points
    Returns:
        cv: coefficient of variation of point spacings (std/mean)
        spacing_std: std of spacings
        spacing_mean: mean spacing
    """
    # Order points
    ordered_points, _ = order_points_by_angle(points)
    n = len(ordered_points)

    if n < 2:
        return 0.0, 0.0, 0.0

    # Compute distances between consecutive points (closed loop)
    rolled = np.roll(ordered_points, -1, axis=0)
    spacings = np.linalg.norm(ordered_points - rolled, axis=1)

    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)

    if mean_spacing < 1e-10:
        return 0.0, 0.0, 0.0

    cv = std_spacing / mean_spacing  # Coefficient of variation

    return cv, std_spacing, mean_spacing


def compute_curvature_smoothness(points):
    """
    Compute smoothness based on discrete curvature estimation.
    Uses the Menger curvature for triplets of points.

    Args:
        points: (N, 2) array of edge points
    Returns:
        mean_curvature: mean absolute curvature
        curvature_std: std of curvatures
    """
    ordered_points, _ = order_points_by_angle(points)
    n = len(ordered_points)

    if n < 3:
        return 0.0, 0.0

    curvatures = []

    for i in range(n):
        p1 = ordered_points[i]
        p2 = ordered_points[(i + 1) % n]
        p3 = ordered_points[(i + 2) % n]

        # Menger curvature: 4 * area(triangle) / (|p1-p2| * |p2-p3| * |p3-p1|)
        # Area using cross product
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * np.abs(v1[0] * v2[1] - v1[1] * v2[0])

        d12 = np.linalg.norm(v1)
        d23 = np.linalg.norm(p3 - p2)
        d31 = np.linalg.norm(v2)

        denom = d12 * d23 * d31
        if denom > 1e-10:
            curvature = 4 * area / denom
        else:
            curvature = 0.0

        curvatures.append(curvature)

    curvatures = np.array(curvatures)
    return np.mean(curvatures), np.std(curvatures)


def evaluate_sample(edge_points, anchors, target_area_fraction=0.7):
    """
    Evaluate a single sample.

    Args:
        edge_points: (N, 2) array
        anchors: (3, 2) array - anchor points forming reference triangle
        target_area_fraction: expected area fraction
    Returns:
        metrics: dict of metrics
    """
    metrics = {}

    # Area fraction (edge_points area / anchors triangle area)
    area_frac, area_error, poly_area, anchor_area = compute_area_fraction(
        edge_points, anchors, target_area_fraction
    )
    metrics['area_fraction'] = area_frac
    metrics['area_fraction_error'] = area_error
    metrics['polygon_area'] = poly_area
    metrics['anchor_triangle_area'] = anchor_area

    # Smoothness (angle-based)
    angle_smoothness, angle_std = compute_smoothness(edge_points)
    metrics['angle_smoothness'] = angle_smoothness
    metrics['angle_std'] = angle_std

    # Smoothness (curvature-based)
    mean_curv, curv_std = compute_curvature_smoothness(edge_points)
    metrics['mean_curvature'] = mean_curv
    metrics['curvature_std'] = curv_std

    # Uniformity
    cv, spacing_std, spacing_mean = compute_uniformity(edge_points)
    metrics['uniformity_cv'] = cv
    metrics['spacing_std'] = spacing_std
    metrics['spacing_mean'] = spacing_mean

    return metrics


def evaluate_batch(edge_points, anchors, ground_truth=None, target_area_fraction=0.7):
    """
    Evaluate a batch of samples.

    Args:
        edge_points: (B, N, 2) array - generated points
        anchors: (B, 3, 2) array - anchor points
        ground_truth: (B, N, 2) array (optional) - for comparison
        target_area_fraction: expected area fraction
    Returns:
        results: dict with mean/std metrics
        per_sample: list of per-sample metrics
    """
    B = edge_points.shape[0]
    per_sample = []

    for i in range(B):
        metrics = evaluate_sample(edge_points[i], anchors[i], target_area_fraction)
        per_sample.append(metrics)

    # Aggregate metrics
    results = {}
    metric_keys = per_sample[0].keys()

    for key in metric_keys:
        values = [m[key] for m in per_sample]
        results[f'{key}_mean'] = np.mean(values)
        results[f'{key}_std'] = np.std(values)

    # If ground truth provided, compute comparison
    if ground_truth is not None:
        gt_per_sample = []
        for i in range(B):
            gt_metrics = evaluate_sample(ground_truth[i], anchors[i], target_area_fraction)
            gt_per_sample.append(gt_metrics)

        results['gt_metrics'] = {}
        for key in metric_keys:
            values = [m[key] for m in gt_per_sample]
            results['gt_metrics'][f'{key}_mean'] = np.mean(values)
            results['gt_metrics'][f'{key}_std'] = np.std(values)

    return results, per_sample


def visualize_metrics(edge_points, anchors, save_path, sample_idx=0):
    """
    Visualize a sample with metric annotations.
    """
    pts = edge_points[sample_idx]
    anch = anchors[sample_idx]

    metrics = evaluate_sample(pts, anch)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Order points for visualization
    ordered_pts, _ = order_points_by_angle(pts)

    # Plot anchor triangle (reference area)
    from matplotlib.patches import Polygon
    anchor_triangle = Polygon(anch, alpha=0.15, facecolor='red', edgecolor='red',
                              linewidth=2, linestyle='--', label='Anchor triangle')
    ax.add_patch(anchor_triangle)

    # Plot edge points polygon (filled)
    polygon = Polygon(ordered_pts, alpha=0.3, facecolor='blue', edgecolor='blue', linewidth=2)
    ax.add_patch(polygon)

    # Plot points
    ax.scatter(pts[:, 0], pts[:, 1], c='blue', s=10, alpha=0.7, label='Edge points')

    # Plot anchors
    ax.scatter(anch[:, 0], anch[:, 1], c='red', s=150, marker='^',
               edgecolors='black', linewidths=2, label='Anchors', zorder=10)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Add metrics text
    text = (f"Area Fraction: {metrics['area_fraction']:.4f}\n"
            f"  (edge_area / anchor_triangle_area)\n"
            f"Area Error (from 0.7): {metrics['area_fraction_error']:.4f}\n"
            f"Polygon Area: {metrics['polygon_area']:.4f}\n"
            f"Anchor Triangle Area: {metrics['anchor_triangle_area']:.4f}\n"
            f"Angle Smoothness: {metrics['angle_smoothness']:.4f}\n"
            f"Uniformity CV: {metrics['uniformity_cv']:.4f}\n"
            f"Mean Curvature: {metrics['mean_curvature']:.4f}")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(f'Sample {sample_idx} Metrics')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIS] Saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate minimal surface generation metrics")
    parser.add_argument("--npz_path", type=str, required=True,
                        help="Path to generated.npz file")
    parser.add_argument("--target_area_fraction", type=float, default=0.7,
                        help="Target area fraction (default: 0.7)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for visualizations")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    args = parser.parse_args()

    # Load data
    data = np.load(args.npz_path)
    edge_points = data['edge_points']
    anchors = data['anchors']  # Required
    ground_truth = data.get('ground_truth', None)

    # Check if ground_truth is valid (not empty or wrong shape)
    if ground_truth is not None:
        if ground_truth.size == 0 or len(ground_truth.shape) < 3:
            ground_truth = None

    print(f"[Data] Loaded {args.npz_path}")
    print(f"  - edge_points: {edge_points.shape}")
    print(f"  - anchors: {anchors.shape}")
    if ground_truth is not None:
        print(f"  - ground_truth: {ground_truth.shape}")
    else:
        print(f"  - ground_truth: None (not available)")

    # Evaluate
    print(f"\n[Evaluating] Target area fraction: {args.target_area_fraction}")
    results, per_sample = evaluate_batch(
        edge_points, anchors, ground_truth, args.target_area_fraction
    )

    # Print results
    print("\n" + "="*60)
    print("GENERATED SAMPLES METRICS")
    print("="*60)
    print(f"  Area Fraction (edge/anchor): {results['area_fraction_mean']:.4f} ± {results['area_fraction_std']:.4f}")
    print(f"  Area Error (|frac - {args.target_area_fraction}|): {results['area_fraction_error_mean']:.4f} ± {results['area_fraction_error_std']:.4f}")
    print(f"  Polygon Area:     {results['polygon_area_mean']:.4f} ± {results['polygon_area_std']:.4f}")
    print(f"  Anchor Tri Area:  {results['anchor_triangle_area_mean']:.4f} ± {results['anchor_triangle_area_std']:.4f}")
    print(f"  Angle Smoothness: {results['angle_smoothness_mean']:.4f} ± {results['angle_smoothness_std']:.4f}")
    print(f"  Curvature Mean:   {results['mean_curvature_mean']:.4f} ± {results['mean_curvature_std']:.4f}")
    print(f"  Curvature Std:    {results['curvature_std_mean']:.4f} ± {results['curvature_std_std']:.4f}")
    print(f"  Uniformity CV:    {results['uniformity_cv_mean']:.4f} ± {results['uniformity_cv_std']:.4f}")
    print(f"  Spacing Mean:     {results['spacing_mean_mean']:.4f} ± {results['spacing_mean_std']:.4f}")
    print(f"  Spacing Std:      {results['spacing_std_mean']:.4f} ± {results['spacing_std_std']:.4f}")

    if 'gt_metrics' in results:
        print("\n" + "="*60)
        print("GROUND TRUTH METRICS (for reference)")
        print("="*60)
        gt = results['gt_metrics']
        print(f"  Area Fraction:    {gt['area_fraction_mean']:.4f} ± {gt['area_fraction_std']:.4f}")
        print(f"  Angle Smoothness: {gt['angle_smoothness_mean']:.4f} ± {gt['angle_smoothness_std']:.4f}")
        print(f"  Curvature Mean:   {gt['mean_curvature_mean']:.4f} ± {gt['mean_curvature_std']:.4f}")
        print(f"  Uniformity CV:    {gt['uniformity_cv_mean']:.4f} ± {gt['uniformity_cv_std']:.4f}")

    # Per-sample metrics
    print("\n" + "="*60)
    print("PER-SAMPLE METRICS")
    print("="*60)
    print(f"{'Sample':<8} {'AreaFrac':<10} {'AreaErr':<10} {'Smooth':<10} {'Uniform':<10}")
    print("-"*48)
    for i, m in enumerate(per_sample):
        print(f"{i:<8} {m['area_fraction']:<10.4f} {m['area_fraction_error']:<10.4f} "
              f"{m['angle_smoothness']:<10.4f} {m['uniformity_cv']:<10.4f}")

    # Visualizations
    if args.visualize:
        if args.output_dir is None:
            args.output_dir = os.path.dirname(args.npz_path)
        os.makedirs(args.output_dir, exist_ok=True)

        for i in range(min(4, edge_points.shape[0])):
            vis_path = os.path.join(args.output_dir, f"metrics_sample_{i:02d}.png")
            visualize_metrics(edge_points, anchors, vis_path, sample_idx=i)

    print("\n[Done]")

    return results, per_sample


if __name__ == "__main__":
    main()
