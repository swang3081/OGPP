#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PLY -> Coulomb tangent forces + nearest-shell spring energy + quality metrics (uniformity)

Defaults (as requested):
  --layers 3 --per-layer 128 --r-min 1.0 --r-max 2.0 --k-spring 120

Key choice:
  - Coulomb forces are projected to the shell tangent space:
        F_tan = F - (F·rhat) rhat
    So both metric and (optional) optimization only act tangentially on shells.

Outputs:
  .npz with points, layer_id, charges, forces (tangent), energies, metrics.
"""

import argparse
import numpy as np


# ---------------- IO ----------------

def load_ply_points(path: str, rescale=True) -> np.ndarray:
    """
    Load Nx3 points from PLY via trimesh (preferred) or open3d (fallback),
    optionally rescale coordinates to [-2, 2] using GLOBAL min/max.
    """
    try:
        import trimesh
        m = trimesh.load(path, process=False)
        if hasattr(m, "vertices") and m.vertices is not None and len(m.vertices) > 0:
            pts = np.asarray(m.vertices, dtype=np.float64)
        elif hasattr(m, "points") and m.points is not None and len(m.points) > 0:
            pts = np.asarray(m.points, dtype=np.float64)
        else:
            raise RuntimeError("trimesh loaded file but found no vertices/points.")
    except Exception as e_trimesh:
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points, dtype=np.float64)
            if pts.size == 0:
                raise RuntimeError("open3d loaded file but points are empty.")
        except Exception as e_o3d:
            raise RuntimeError(
                f"Failed to load PLY.\n"
                f"trimesh error: {e_trimesh}\n"
                f"open3d error: {e_o3d}\n"
                f"Please install trimesh or open3d."
            )

    if not rescale:
        return pts

    # ===== rescale to [-2, 2] using GLOBAL min/max =====
    vmin = pts.min()
    vmax = pts.max()

    if vmax - vmin < 1e-12:
        raise ValueError("Degenerate point cloud: max == min, cannot rescale.")

    pts = 4.0 * (pts - vmin) / (vmax - vmin) - 2.0
    return pts


# ---------------- Shell assignment ----------------

def assign_layers_by_r(points: np.ndarray, radii: np.ndarray):
    r = np.linalg.norm(points, axis=1)
    d = np.abs(r[:, None] - radii[None, :])
    layer_id = np.argmin(d, axis=1).astype(np.int32)
    r_near = radii[layer_id]
    return r, layer_id, r_near


def project_to_nearest_shell(points: np.ndarray, radii: np.ndarray):
    """Project each point to its nearest shell radius (by current |x|)."""
    r = np.linalg.norm(points, axis=1)
    d = np.abs(r[:, None] - radii[None, :])
    lid = np.argmin(d, axis=1).astype(np.int32)
    r_target = radii[lid]

    x = points.copy().astype(np.float64)
    rn = np.linalg.norm(x, axis=1)
    rn = np.maximum(rn, 1e-12)
    x *= (r_target / rn)[:, None]
    return x, lid


def make_charges(layer_id: np.ndarray, layers: int, alt_charge: bool):
    if not alt_charge:
        return np.ones_like(layer_id, dtype=np.float64)
    sign = np.where((np.arange(layers) % 2) == 0, 1.0, -1.0).astype(np.float64)
    return sign[layer_id]


# ---------------- Coulomb force + energy ----------------

def coulomb_forces_and_energy(points: np.ndarray,
                             layer_id: np.ndarray,
                             q: np.ndarray,
                             eps: float,
                             same_layer_scale: float,
                             cross_layer_scale: float,
                             chunk: int = 1024):
    """
    O(N^2) Coulomb; chunked to limit memory.
    Returns:
      forces (N,3) [FULL 3D Coulomb force], E_coul (float)
    """
    x = points.astype(np.float64)
    N = x.shape[0]
    F = np.zeros((N, 3), dtype=np.float64)
    E = 0.0

    lid = layer_id.astype(np.int32)
    q = q.astype(np.float64)

    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        xi = x[i0:i1]
        li = lid[i0:i1]
        qi = q[i0:i1]

        d = xi[:, None, :] - x[None, :, :]          # (B,N,3)
        r = np.linalg.norm(d, axis=2) + eps         # (B,N)

        same = (li[:, None] == lid[None, :])
        s = np.where(same, same_layer_scale, cross_layer_scale).astype(np.float64)
        qq = (qi[:, None] * q[None, :])

        rows = np.arange(i0, i1)[:, None]
        cols = np.arange(N)[None, :]
        self_mask = (rows == cols)

        invr3 = 1.0 / (r * r * r)
        invr = 1.0 / r

        coefF = s * qq * invr3
        coefF[self_mask] = 0.0
        F[i0:i1] += np.einsum("bn,bnq->bq", coefF, d)

        gt = (cols > rows)
        coefE = s * qq * invr
        coefE[~gt] = 0.0
        E += np.sum(coefE)

    return F, float(E)


def project_force_to_tangent(points: np.ndarray, forces: np.ndarray):
    """
    Project force to shell tangent space at each point:
      F_tan = F - (F·rhat) rhat
    """
    x = points.astype(np.float64)
    f = forces.astype(np.float64)
    r = np.linalg.norm(x, axis=1, keepdims=True)
    r = np.maximum(r, 1e-12)
    rhat = x / r
    fr = np.sum(f * rhat, axis=1, keepdims=True)
    f_tan = f - fr * rhat
    return f_tan


# ---------------- Energies ----------------

def spring_energy_nearest_shell(points: np.ndarray, layer_id: np.ndarray, radii: np.ndarray, k_spring: float):
    r = np.linalg.norm(points, axis=1)
    r_near = radii[layer_id]
    dr = r - r_near
    # print(r)
    print("dr max abs:", np.max(np.abs(dr)))
    e_per = 0.5 * float(k_spring) * (dr * dr)
    return float(np.sum(np.abs(e_per))), e_per


# ---------------- Metrics (goodness) ----------------

def metric_E_star(E_coul: float, layer_id: np.ndarray, radii: np.ndarray, q: np.ndarray):
    """
    Dimensionless Coulomb energy:
      E* = E_coul / sum_{i<j} |qi qj| / R_ij
      R_ij = 0.5*(r_layer(i)+r_layer(j))
    Lower is "better" for comparable configs.
    """
    lid = layer_id.astype(np.int32)
    q = q.astype(np.float64)
    r_layer = radii[lid].astype(np.float64)
    N = lid.shape[0]

    denom = 0.0
    for i in range(N):
        Ri = r_layer[i]
        qi = abs(q[i])
        for j in range(i + 1, N):
            Rj = r_layer[j]
            denom += (qi * abs(q[j])) / (0.5 * (Ri + Rj) + 1e-12)

    return float(E_coul / max(denom, 1e-30))


def metric_tangent_force(points: np.ndarray, forces_full: np.ndarray):
    """Return RMS / max of tangent force."""
    f_tan = project_force_to_tangent(points, forces_full)
    n = np.linalg.norm(f_tan, axis=1)
    return float(np.sqrt(np.mean(n * n))), float(np.max(n)), f_tan


def metric_spacing_cv_per_layer(points: np.ndarray, layer_id: np.ndarray, radii: np.ndarray):
    """
    Per layer: CV of nearest-neighbor spherical distance (geodesic).
      d = R * arccos(u_i·u_j)
    Lower CV => more uniform.
    """
    x = points.astype(np.float64)
    lid = layer_id.astype(np.int32)
    layers = int(np.max(lid)) + 1
    cvs = []

    for l in range(layers):
        idx = np.where(lid == l)[0]
        if idx.size < 2:
            cvs.append(np.nan)
            continue

        xl = x[idx]
        R = float(radii[l])
        u = xl / np.maximum(np.linalg.norm(xl, axis=1, keepdims=True), 1e-12)

        nn = np.full((idx.size,), np.inf, dtype=np.float64)
        for i in range(idx.size):
            dots = u @ u[i]
            dots = np.clip(dots, -1.0, 1.0)
            ang = np.arccos(dots)
            ang[i] = np.inf
            nn[i] = np.min(ang) * R

        m = np.mean(nn)
        s = np.std(nn)
        cv = float(s / max(m, 1e-12))
        cvs.append(cv)

    valid = [c for c in cvs if np.isfinite(c)]
    cv_avg = float(np.mean(valid)) if valid else float("nan")
    return cvs, cv_avg


def metric_score(E_star: float, F_tan_rms: float, cv_avg: float,
                 wE=0.5, wF=0.3, wCV=0.2):
    """Lower is better."""
    return float(wE * E_star + wF * np.log1p(max(F_tan_rms, 0.0)) + wCV * cv_avg)


# ---------------- Optional: tangent-only relaxation ----------------

def tangent_relax(points: np.ndarray,
                  radii: np.ndarray,
                  layers: int,
                  same_layer_scale: float,
                  cross_layer_scale: float,
                  eps: float,
                  k_spring: float,
                  alt_charge: bool,
                  iters: int,
                  lr: float,
                  lr_min: float,
                  chunk: int,
                  seed: int,
                  jitter: float):
    """
    Optional: minimize / relax on shells using tangent-only Coulomb forces.
    Steps:
      - small direction jitter
      - project to nearest shell
      - iterate: compute FULL Coulomb force -> project to tangent -> x += lr*F_tan -> project to shell
    """
    rng = np.random.default_rng(seed)
    x = points.astype(np.float64).copy()

    # jitter directions slightly (keep radius)
    if jitter > 0:
        rad = np.linalg.norm(x, axis=1, keepdims=True)
        rad = np.maximum(rad, 1e-12)
        dirs = x / rad
        dirs = dirs + float(jitter) * rng.normal(size=dirs.shape)
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        x = dirs * rad

    x, lid = project_to_nearest_shell(x, radii)
    q = make_charges(lid, layers, alt_charge)

    def lr_t(t):
        a = t / max(1, iters)
        return lr_min + (lr - lr_min) * (1.0 - a) ** 0.9

    for it in range(1, iters + 1):
        F_full, _ = coulomb_forces_and_energy(
            x, lid, q, eps, same_layer_scale, cross_layer_scale, chunk=chunk
        )
        F_tan = project_force_to_tangent(x, F_full)
        x = x + lr_t(it) * F_tan
        x, lid = project_to_nearest_shell(x, radii)
        q = make_charges(lid, layers, alt_charge)

    return x, lid


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="PLY Coulomb tangent metric on multi-shells")

    ap.add_argument("--ply", type=str, required=True, help="input .ply point cloud")

    # defaults you requested
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--per-layer", type=int, default=128)  # just for reporting/sanity
    ap.add_argument("--r-min", type=float, default=1.0)
    ap.add_argument("--r-max", type=float, default=2.0)
    ap.add_argument("--k-spring", type=float, default=120.0)

    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--same-layer-scale", type=float, default=1.0)
    ap.add_argument("--cross-layer-scale", type=float, default=1.0)
    ap.add_argument("--alt-charge", action="store_true")

    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out", type=str, default="ply_coulomb_tangent_metrics.npz")

    # optional relaxation
    ap.add_argument("--relax", action="store_true", help="tangent-only relaxation on shells before measuring")
    ap.add_argument("--relax-iters", type=int, default=800)
    ap.add_argument("--relax-lr", type=float, default=1e-2)
    ap.add_argument("--relax-lr-min", type=float, default=1e-4)
    ap.add_argument("--relax-seed", type=int, default=0)
    ap.add_argument("--relax-jitter", type=float, default=0.03)

    args = ap.parse_args()

    pts0 = load_ply_points(args.ply)
    if pts0.ndim != 2 or pts0.shape[1] != 3:
        raise RuntimeError(f"Expect Nx3 points, got {pts0.shape}")

    radii = np.linspace(float(args.r_min), float(args.r_max), int(args.layers)).astype(np.float64)
    
    # (optional) relax on shells using tangent-only forces
    if args.relax:
        pts, layer_id = tangent_relax(
            pts0, radii, int(args.layers),
            float(args.same_layer_scale), float(args.cross_layer_scale),
            float(args.eps), float(args.k_spring),
            bool(args.alt_charge),
            iters=int(args.relax_iters),
            lr=float(args.relax_lr),
            lr_min=float(args.relax_lr_min),
            chunk=int(args.chunk),
            seed=int(args.relax_seed),
            jitter=float(args.relax_jitter),
        )
    else:
        # no relax: just project to nearest shell once for consistent definition
        pts, layer_id = project_to_nearest_shell(pts0.astype(np.float64), radii)
    E_spring, e_spring_per = spring_energy_nearest_shell(
        pts0, layer_id, radii, float(args.k_spring)
    )
    q = make_charges(layer_id, int(args.layers), bool(args.alt_charge))

    # Coulomb full force + energy
    F_full, E_coul = coulomb_forces_and_energy(
        pts, layer_id, q,
        eps=float(args.eps),
        same_layer_scale=float(args.same_layer_scale),
        cross_layer_scale=float(args.cross_layer_scale),
        chunk=int(args.chunk),
    )

    # Tangent-only force (what you "use")
    F_tan_rms, F_tan_max, F_tan = metric_tangent_force(pts, F_full)

    # spring energy (nearest shell)
    

    # metrics
    E_star = metric_E_star(E_coul, layer_id, radii, q)
    cv_layers, cv_avg = metric_spacing_cv_per_layer(pts, layer_id, radii)
    Score = metric_score(E_star, F_tan_rms, cv_avg)

    # prints
    N = pts.shape[0]
    counts = np.bincount(layer_id, minlength=int(args.layers))
    print(f"[info] N={N}, layers={args.layers}, expected per-layer={args.per_layer}")
    print(f"[info] radii={np.array2string(radii, precision=6)}")
    print(f"[info] counts per layer={counts.tolist()}")
    print(f"[info] alt_charge={bool(args.alt_charge)} relax={bool(args.relax)}")

    print(f"[energy] E_coul   = {E_coul:.10f}")
    print(f"[energy] E_spring = {E_spring:.10f}")
    print(f"[energy] E_total  = {E_coul + E_spring:.10f}")

    print(f"[metric] E_star (dimensionless) = {E_star:.6e}   (lower better)")
    print(f"[metric] F_tan_rms              = {F_tan_rms:.6e} (lower better)")
    print(f"[metric] F_tan_max              = {F_tan_max:.6e} (lower better)")
    print(f"[metric] spacing CV per layer   = {cv_layers}  (lower better)")
    print(f"[metric] spacing CV avg         = {cv_avg:.6e} (lower better)")
    print(f"[metric] Score                  = {Score:.6e}   (lower better)")

    np.savez_compressed(
        args.out,
        points=pts.astype(np.float32),
        points_raw=pts0.astype(np.float32),
        radii=radii.astype(np.float32),
        layer_id=layer_id.astype(np.int32),
        charge=q.astype(np.float32),

        # forces: full and tangent
        forces_full=F_full.astype(np.float32),
        forces_tangent=F_tan.astype(np.float32),

        # energies
        eps=float(args.eps),
        same_layer_scale=float(args.same_layer_scale),
        cross_layer_scale=float(args.cross_layer_scale),
        k_spring=float(args.k_spring),
        E_coul=float(E_coul),
        E_spring=float(E_spring),
        E_total=float(E_coul + E_spring),
        e_spring_per=e_spring_per.astype(np.float32),

        # metrics
        E_star=float(E_star),
        F_tan_rms=float(F_tan_rms),
        F_tan_max=float(F_tan_max),
        cv_layers=np.array(cv_layers, dtype=np.float32),
        cv_avg=float(cv_avg),
        Score=float(Score),

        # config
        layers=int(args.layers),
        per_layer_expected=int(args.per_layer),
        r_min=float(args.r_min),
        r_max=float(args.r_max),
        alt_charge=bool(args.alt_charge),
        relax=bool(args.relax),
        relax_iters=int(args.relax_iters),
        relax_lr=float(args.relax_lr),
        relax_lr_min=float(args.relax_lr_min),
        relax_seed=int(args.relax_seed),
        relax_jitter=float(args.relax_jitter),
    )
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
