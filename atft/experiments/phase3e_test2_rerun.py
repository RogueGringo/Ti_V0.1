#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""Phase 3e Test 2+3 RERUN — Fixed GUE unfolding.

Only runs:
  - Test 2: 10 proper D-E GUE realizations with spacing-preserving unfolding
  - Test 3: Edge count + S for all sources including proper GUE
"""
from __future__ import annotations

import gc
import json
import sys
import time

import numpy as np
import torch
from scipy.linalg import eigvalsh_tridiagonal
from scipy.spatial.distance import pdist

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian

K = 200
N = 1000
EPSILON = 3.0
K_EIG = 20


def gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_proper_gue(n, z_min, z_max, seed):
    """D-E GUE with spacing-preserving rescale."""
    rng = np.random.default_rng(seed)
    diag = rng.standard_normal(n)
    dof = 2.0 * np.arange(n - 1, 0, -1, dtype=np.float64)
    sub = np.sqrt(rng.chisquare(dof)) / np.sqrt(2.0)
    eigs = eigvalsh_tridiagonal(diag, sub)
    eigs /= np.sqrt(2.0 * n)

    sorted_eigs = np.sort(eigs)
    spacings = np.diff(sorted_eigs)
    target_mean = (z_max - z_min) / (n - 1)
    scaled = spacings * (target_mean / spacings.mean())

    pts = np.zeros(n)
    pts[0] = z_min
    pts[1:] = z_min + np.cumsum(scaled)
    return pts


def run_point(zeros, sigma, label):
    t0 = time.time()
    try:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        lap = TorchSheafLaplacian(builder, zeros, transport_mode="superposition")
        eigs = lap.smallest_eigenvalues(EPSILON, k=K_EIG)
        s = float(np.sum(eigs))
        elapsed = time.time() - t0
        print(f"  [{label}] S={s:.6f} ({elapsed:.1f}s)")
        sys.stdout.flush()
        return {"spectral_sum": s, "eigs_top5": eigs[:5].tolist(), "time_s": elapsed}
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        return None
    finally:
        gpu_cleanup()


def main():
    print("=" * 70)
    print("  PHASE 3e RERUN — Test 2 + Test 3 (fixed GUE unfolding)")
    print(f"  K={K}, N={N}, eps={EPSILON}")
    print("=" * 70)

    # Load zeta zeros
    source = ZetaZerosSource("data/odlyzko_zeros.txt")
    cloud = source.generate(N)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    z_min, z_max = float(zeta_zeros.min()), float(zeta_zeros.max())
    mean_sp = float(np.mean(np.diff(np.sort(zeta_zeros))))
    print(f"  Zeta: N={N}, range=[{z_min:.2f}, {z_max:.2f}], mean_sp={mean_sp:.4f}")

    # Verify GUE unfolding produces different points per seed
    test1 = generate_proper_gue(N, z_min, z_max, 1000)
    test2 = generate_proper_gue(N, z_min, z_max, 1001)
    print(f"  GUE seed 1000 spacing std: {np.std(np.diff(test1)):.4f}")
    print(f"  GUE seed 1001 spacing std: {np.std(np.diff(test2)):.4f}")
    print(f"  Points differ: {not np.allclose(test1, test2)}")

    results = {}

    # ── TEST 2: 10 proper GUE realizations at sigma=0.5 ──
    print(f"\n{'='*70}")
    print("  TEST 2: PROPER GUE ENSEMBLE (D-E, spacing-preserving, 10 seeds)")
    print(f"{'='*70}")

    gue_sums = []
    gue_edges = []

    for i in range(10):
        seed = 2000 + i
        pts = generate_proper_gue(N, z_min, z_max, seed)
        n_e = int(np.sum(pdist(pts.reshape(-1, 1)) <= EPSILON))

        r = run_point(pts, 0.5, f"GUE-{seed}")
        if r:
            gue_sums.append(r["spectral_sum"])
            gue_edges.append(n_e)
            print(f"    |E|={n_e}, S/|E|={r['spectral_sum']/n_e:.6f}")

        if (i + 1) % 5 == 0:
            print(f"  --- {i+1}/10: mean S={np.mean(gue_sums):.6f} "
                  f"+/- {np.std(gue_sums):.6f}, mean |E|={np.mean(gue_edges):.0f} ---")

    results["GUE_Ensemble"] = {
        "n": len(gue_sums),
        "spectral_sums": gue_sums,
        "edge_counts": gue_edges,
        "mean_S": float(np.mean(gue_sums)),
        "std_S": float(np.std(gue_sums)),
        "mean_edges": float(np.mean(gue_edges)),
        "CI_95": [float(np.percentile(gue_sums, 2.5)),
                  float(np.percentile(gue_sums, 97.5))],
    }

    # ── TEST 3: Edge-count comparison (all sources) ──
    print(f"\n{'='*70}")
    print("  TEST 3: EDGE COUNT + S/|E| (all sources)")
    print(f"{'='*70}")

    # Prepare all sources
    even_pts = np.linspace(z_min, z_max, N)
    rng = np.random.default_rng(42)
    rand_pts = np.sort(rng.uniform(z_min, z_max, N))
    from atft.experiments.phase3d_torch_k200 import generate_gue_points
    wigner_pts = generate_gue_points(N, mean_sp, z_min, rng)
    proper_gue_pts = generate_proper_gue(N, z_min, z_max, 2000)

    sources = {
        "Zeta": zeta_zeros,
        "EvenSpaced": even_pts,
        "ProperGUE": proper_gue_pts,
        "WignerGUE": wigner_pts,
        "Random": rand_pts,
    }

    edge_data = {}
    for name, pts in sources.items():
        n_e = int(np.sum(pdist(pts.reshape(-1, 1)) <= EPSILON))
        r = run_point(pts, 0.5, name)
        s = r["spectral_sum"] if r else 0
        ratio = s / n_e if n_e > 0 else 0
        edge_data[name] = {"edges": n_e, "S": s, "S_per_E": ratio}
        print(f"    {name}: |E|={n_e}, S={s:.6f}, S/|E|={ratio:.6f}")

    results["EdgeAnalysis"] = edge_data

    # ── ANALYSIS ──
    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")

    # Test 2
    s_zeta = 11.784063
    gue_mean = results["GUE_Ensemble"]["mean_S"]
    gue_std = results["GUE_Ensemble"]["std_S"]
    gue_ci = results["GUE_Ensemble"]["CI_95"]

    print(f"\n  TEST 2 — PROPER GUE ENSEMBLE:")
    print(f"    S(Zeta) = {s_zeta:.6f}")
    print(f"    S(GUE)  = {gue_mean:.6f} +/- {gue_std:.6f}")
    print(f"    95% CI: [{gue_ci[0]:.6f}, {gue_ci[1]:.6f}]")
    if gue_std > 0:
        z = (s_zeta - gue_mean) / gue_std
        print(f"    Z-score: {z:.2f}")
    if s_zeta < gue_ci[0]:
        print(f"    >>> ZETA BELOW GUE 95% CI. Arithmetic premium CONFIRMED.")
        results["test2_verdict"] = "SIGNIFICANT"
    elif s_zeta > gue_ci[1]:
        print(f"    >>> ZETA ABOVE GUE 95% CI. Surprising.")
        results["test2_verdict"] = "REVERSED"
    else:
        print(f"    >>> ZETA WITHIN GUE RANGE. Premium NOT significant.")
        results["test2_verdict"] = "NOT_SIGNIFICANT"

    # Test 3 — edge-normalized
    print(f"\n  TEST 3 — EDGE-NORMALIZED:")
    print(f"    {'Source':<12} {'|E|':>8} {'S':>12} {'S/|E|':>10}")
    ratios = []
    for name in ["Zeta", "EvenSpaced", "ProperGUE", "WignerGUE", "Random"]:
        d = edge_data[name]
        print(f"    {name:<12} {d['edges']:>8} {d['S']:>12.6f} {d['S_per_E']:>10.6f}")
        ratios.append(d["S_per_E"])

    cv = np.std(ratios) / np.mean(ratios) * 100
    print(f"\n    S/|E| CV: {cv:.1f}%")

    # Zeta vs ProperGUE edge-normalized
    z_spe = edge_data["Zeta"]["S_per_E"]
    g_spe = edge_data["ProperGUE"]["S_per_E"]
    print(f"\n    Zeta S/|E| = {z_spe:.6f}")
    print(f"    ProperGUE S/|E| = {g_spe:.6f}")
    if z_spe < g_spe:
        prem = (1 - z_spe / g_spe) * 100
        print(f"    >>> Zeta per-edge premium: {prem:.1f}% (transport IS tighter)")
    else:
        deficit = (z_spe / g_spe - 1) * 100
        print(f"    >>> Zeta per-edge DEFICIT: +{deficit:.1f}% (transport is LOOSER)")

    # Save
    save_path = "output/phase3e_test2_rerun_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
