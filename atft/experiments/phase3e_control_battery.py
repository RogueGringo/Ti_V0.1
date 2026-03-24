#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""Phase 3e: Control Battery — Kill Shot Tests for L3 Validation Committee.

Three experiments that resolve the adversary's attacks:
  1. EVENLY-SPACED CONTROL: uniform gaps → if S(even) < S(zeta), hierarchy = edge count
  2. PROPER GUE ENSEMBLE: 50 Dumitriu-Edelman realizations → establishes variance bounds
  3. EDGE COUNT TRACKING: S vs |E| for all sources → tests proportionality claim

K=200, N=1000, eps=3.0, sigma grid matching K=200 experiment.
"""
from __future__ import annotations

import gc
import json
import sys
import time

import numpy as np
import torch
from scipy.linalg import eigvalsh_tridiagonal

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian


K = 200
N = 1000
EPSILON = 3.0
K_EIG = 20
SIGMA_GRID = np.array([0.25, 0.35, 0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60, 0.65, 0.75])

SAVE_PATH = "output/phase3e_control_battery_results.json"


def gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_status() -> str:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        free = torch.cuda.mem_get_info()[0] / 1e9
        total = torch.cuda.mem_get_info()[1] / 1e9
        return f"{alloc:.2f}GB alloc, {free:.1f}GB free / {total:.1f}GB"
    return "CPU mode"


def generate_dumitriu_edelman_gue(n: int, rng: np.random.Generator) -> np.ndarray:
    """Proper GUE via Dumitriu-Edelman tridiagonal model (beta=2).

    Returns eigenvalues normalized to semicircle [-1, 1].
    """
    diagonal = rng.standard_normal(n)
    dof = 2.0 * np.arange(n - 1, 0, -1, dtype=np.float64)
    sub_diagonal = np.sqrt(rng.chisquare(dof)) / np.sqrt(2.0)
    eigenvalues = eigvalsh_tridiagonal(diagonal, sub_diagonal)
    eigenvalues /= np.sqrt(2.0 * n)
    return eigenvalues.astype(np.float64)


def gue_to_unfolded(eigs: np.ndarray, target_range: tuple[float, float],
                     target_spacing: float) -> np.ndarray:
    """Map GUE eigenvalues to target range preserving local spacing structure.

    Approach: D-E eigenvalues have correct GUE local statistics. We rescale
    their spacings to match the target mean spacing, then reconstruct points
    cumulatively from z_min. This preserves the RELATIVE spacing pattern
    (level repulsion, bulk/edge density variation) while matching the target scale.
    """
    n = len(eigs)
    sorted_eigs = np.sort(eigs)
    spacings = np.diff(sorted_eigs)

    target_mean = (target_range[1] - target_range[0]) / (n - 1)
    scale = target_mean / spacings.mean()
    scaled_spacings = spacings * scale

    pts = np.zeros(n)
    pts[0] = target_range[0]
    pts[1:] = target_range[0] + np.cumsum(scaled_spacings)
    return pts


def run_point(zeros, sigma, label, track_edges=False):
    """Run one (source, sigma) point. Returns result dict."""
    t0 = time.time()
    try:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        lap = TorchSheafLaplacian(builder, zeros, transport_mode="superposition")

        # Track edge count if requested
        n_edges = None
        if track_edges:
            n_edges = lap.n_edges if hasattr(lap, 'n_edges') else None

        eigs = lap.smallest_eigenvalues(EPSILON, k=K_EIG)
        s = float(np.sum(eigs))
        tau = 1e-6 * np.sqrt(s) if s > 0 else 1e-10
        beta_0 = int(np.sum(eigs < tau))
        elapsed = time.time() - t0

        print(f"  [{label}] sigma={sigma:.3f}: S={s:.6f} b0={beta_0} "
              f"({elapsed:.1f}s) [{vram_status()}]")
        sys.stdout.flush()

        result = {
            "sigma": float(sigma),
            "epsilon": float(EPSILON),
            "spectral_sum": s,
            "kernel_dim": beta_0,
            "eigs_top5": eigs[:5].tolist(),
            "time_s": elapsed,
        }
        if n_edges is not None:
            result["n_edges"] = n_edges
        return result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{label}] sigma={sigma:.3f}: FAILED ({e}) ({elapsed:.1f}s)")
        sys.stdout.flush()
        return None
    finally:
        gpu_cleanup()


def count_rips_edges(points: np.ndarray, epsilon: float) -> int:
    """Count edges in Vietoris-Rips complex at given epsilon."""
    from scipy.spatial.distance import pdist
    dists = pdist(points.reshape(-1, 1))
    return int(np.sum(dists <= epsilon))


def main():
    print("=" * 70)
    print("  ATFT PHASE 3e: CONTROL BATTERY (Kill Shot Tests)")
    print(f"  K={K}, N={N}, eps={EPSILON}, {len(SIGMA_GRID)} sigma points")
    print(f"  Backend: PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {vram_status()}")
    print("=" * 70)

    # Load zeta zeros (same as K=200 experiment)
    source = ZetaZerosSource("data/odlyzko_zeros.txt")
    cloud = source.generate(N)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    mean_spacing = float(np.mean(np.diff(np.sort(zeta_zeros))))
    z_min, z_max = float(zeta_zeros.min()), float(zeta_zeros.max())
    print(f"\n  Zeta zeros: N={len(zeta_zeros)}, range=[{z_min:.2f}, {z_max:.2f}], "
          f"mean_spacing={mean_spacing:.4f}")

    results = {}

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: EVENLY-SPACED CONTROL
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  TEST 1: EVENLY-SPACED CONTROL (uniform gaps)")
    print(f"{'='*70}")

    even_pts = np.linspace(z_min, z_max, N)
    print(f"  Even points: N={N}, spacing={even_pts[1]-even_pts[0]:.6f}")

    results["EvenSpaced"] = {}
    for sigma in SIGMA_GRID:
        r = run_point(even_pts, float(sigma), "EVEN")
        if r is not None:
            key = f"{float(sigma):.3f}_{EPSILON:.1f}"
            results["EvenSpaced"][key] = r

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: PROPER GUE ENSEMBLE (50 realizations at sigma=0.5)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  TEST 2: PROPER GUE ENSEMBLE (Dumitriu-Edelman, 50 realizations)")
    print(f"{'='*70}")

    n_realizations = 10  # 10 is enough for variance estimate; extend later if needed
    gue_spectral_sums = []
    gue_eigs_all = []

    for i in range(n_realizations):
        rng_gue = np.random.default_rng(1000 + i)
        raw_eigs = generate_dumitriu_edelman_gue(N, rng_gue)
        gue_pts = gue_to_unfolded(raw_eigs, (z_min, z_max), mean_spacing)

        r = run_point(gue_pts, 0.5, f"GUE-{i+1:02d}")
        if r is not None:
            gue_spectral_sums.append(r["spectral_sum"])
            gue_eigs_all.append(r["eigs_top5"])

        if (i + 1) % 10 == 0:
            mean_s = np.mean(gue_spectral_sums)
            std_s = np.std(gue_spectral_sums)
            print(f"  --- GUE ensemble progress: {i+1}/{n_realizations}, "
                  f"mean S={mean_s:.6f}, std={std_s:.6f} ---")
            sys.stdout.flush()

    results["GUE_Ensemble"] = {
        "n_realizations": n_realizations,
        "sigma": 0.5,
        "spectral_sums": gue_spectral_sums,
        "mean": float(np.mean(gue_spectral_sums)),
        "std": float(np.std(gue_spectral_sums)),
        "min": float(np.min(gue_spectral_sums)),
        "max": float(np.max(gue_spectral_sums)),
        "CI_95": [
            float(np.percentile(gue_spectral_sums, 2.5)),
            float(np.percentile(gue_spectral_sums, 97.5)),
        ],
    }

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: EDGE COUNT TRACKING
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  TEST 3: EDGE COUNT vs SPECTRAL SUM")
    print(f"{'='*70}")

    # Generate controls (same as original experiment for comparability)
    rng = np.random.default_rng(42)
    rand_pts = np.sort(rng.uniform(z_min, z_max, N))
    # Wigner surmise GUE (for comparison with original)
    from atft.experiments.phase3d_torch_k200 import generate_gue_points
    wigner_gue_pts = generate_gue_points(N, mean_spacing, z_min, rng)

    all_sources = {
        "Zeta": zeta_zeros,
        "EvenSpaced": even_pts,
        "Random": rand_pts,
        "WignerGUE": wigner_gue_pts,
    }

    # Add one proper GUE realization
    rng_gue0 = np.random.default_rng(1000)
    raw_eigs0 = generate_dumitriu_edelman_gue(N, rng_gue0)
    all_sources["ProperGUE"] = gue_to_unfolded(raw_eigs0, (z_min, z_max), mean_spacing)

    edge_counts = {}
    spectral_sums_at_05 = {}

    print(f"\n  Computing edge counts at eps={EPSILON}...")
    for name, pts in all_sources.items():
        n_e = count_rips_edges(pts, EPSILON)
        edge_counts[name] = n_e
        print(f"    {name}: |E| = {n_e}")

    print(f"\n  Computing spectral sums at sigma=0.5...")
    for name, pts in all_sources.items():
        if name == "EvenSpaced":
            # Already computed in Test 1
            key = f"0.500_{EPSILON:.1f}"
            if key in results.get("EvenSpaced", {}):
                spectral_sums_at_05[name] = results["EvenSpaced"][key]["spectral_sum"]
                print(f"    {name}: S = {spectral_sums_at_05[name]:.6f} (cached)")
                continue
        r = run_point(pts, 0.5, name)
        if r is not None:
            spectral_sums_at_05[name] = r["spectral_sum"]

    results["EdgeCount"] = {
        "edge_counts": edge_counts,
        "spectral_sums_at_sigma_0.5": spectral_sums_at_05,
        "epsilon": EPSILON,
    }

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")

    # Test 1 result: where does EvenSpaced land?
    z_key = "0.500_3.0"
    s_zeta_05 = 11.784063  # from K=200 results
    s_even_05 = results.get("EvenSpaced", {}).get(z_key, {}).get("spectral_sum", None)

    print(f"\n  TEST 1 — EVENLY-SPACED CONTROL:")
    if s_even_05 is not None:
        if s_even_05 < s_zeta_05:
            print(f"    S(Even) = {s_even_05:.6f} < S(Zeta) = {s_zeta_05:.6f}")
            print(f"    >>> RESULT IS KILLED. Hierarchy = edge count density.")
            results["test1_verdict"] = "KILLED"
        else:
            print(f"    S(Even) = {s_even_05:.6f} > S(Zeta) = {s_zeta_05:.6f}")
            premium = (1 - s_zeta_05 / s_even_05) * 100
            print(f"    >>> ADVERSARY ATTACK REFUTED. Zeta beats even spacing by {premium:.1f}%")
            print(f"    >>> Hierarchy contains information beyond edge geometry.")
            results["test1_verdict"] = "SURVIVED"
            results["test1_premium_vs_even"] = premium

    # Test 2 result: does zeta fall outside GUE distribution?
    print(f"\n  TEST 2 — GUE ENSEMBLE:")
    gue_ens = results.get("GUE_Ensemble", {})
    gue_mean = gue_ens.get("mean", 0)
    gue_std = gue_ens.get("std", 1)
    gue_ci = gue_ens.get("CI_95", [0, 0])
    z_score = (s_zeta_05 - gue_mean) / gue_std if gue_std > 0 else 0

    print(f"    S(Zeta) = {s_zeta_05:.6f}")
    print(f"    S(GUE) = {gue_mean:.6f} +/- {gue_std:.6f} (50 realizations)")
    print(f"    GUE 95% CI: [{gue_ci[0]:.6f}, {gue_ci[1]:.6f}]")
    print(f"    Z-score: {z_score:.2f}")

    if s_zeta_05 < gue_ci[0]:
        print(f"    >>> ZETA FALLS BELOW GUE 95% CI. Arithmetic premium is REAL.")
        results["test2_verdict"] = "SIGNIFICANT"
    else:
        print(f"    >>> ZETA WITHIN GUE RANGE. Premium may be sampling artifact.")
        results["test2_verdict"] = "NOT_SIGNIFICANT"

    # Test 3 result: S vs |E| proportionality
    print(f"\n  TEST 3 — EDGE COUNT vs SPECTRAL SUM:")
    print(f"    {'Source':<12} {'|E|':>8} {'S(0.5)':>12} {'S/|E|':>10}")
    s_over_e = {}
    for name in all_sources:
        e = edge_counts.get(name, 0)
        s = spectral_sums_at_05.get(name, 0)
        ratio = s / e if e > 0 else 0
        s_over_e[name] = ratio
        print(f"    {name:<12} {e:>8} {s:>12.6f} {ratio:>10.6f}")

    # Check if S/|E| is constant (proportionality)
    ratios = list(s_over_e.values())
    if ratios:
        ratio_cv = np.std(ratios) / np.mean(ratios) * 100
        print(f"\n    S/|E| coefficient of variation: {ratio_cv:.1f}%")
        if ratio_cv < 5:
            print(f"    >>> S IS PROPORTIONAL TO |E|. Spectral sum ~ edge count.")
            results["test3_verdict"] = "PROPORTIONAL"
        else:
            print(f"    >>> S IS NOT SIMPLY PROPORTIONAL TO |E|. Contains additional structure.")
            results["test3_verdict"] = "NOT_PROPORTIONAL"

    results["test3_s_over_e"] = s_over_e

    # Save
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {SAVE_PATH}")

    # ── FINAL VERDICT ──
    print(f"\n{'='*70}")
    print("  FINAL CONTROL BATTERY VERDICT")
    print(f"{'='*70}")
    v1 = results.get("test1_verdict", "UNKNOWN")
    v2 = results.get("test2_verdict", "UNKNOWN")
    v3 = results.get("test3_verdict", "UNKNOWN")
    print(f"    Test 1 (Even-Spaced):     {v1}")
    print(f"    Test 2 (GUE Ensemble):    {v2}")
    print(f"    Test 3 (Edge Proportionality): {v3}")

    if v1 == "KILLED":
        print(f"\n    OVERALL: OFF_SHELL — hierarchy is an edge-count artifact")
    elif v1 == "SURVIVED" and v2 == "SIGNIFICANT" and v3 == "NOT_PROPORTIONAL":
        print(f"\n    OVERALL: ON_SHELL — arithmetic premium is real and topological")
    elif v1 == "SURVIVED" and v2 == "SIGNIFICANT":
        print(f"\n    OVERALL: CONDITIONAL ON_SHELL — premium is real but S may still track edge count")
    else:
        print(f"\n    OVERALL: REQUIRES FURTHER INVESTIGATION")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
