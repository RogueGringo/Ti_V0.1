#!/usr/bin/env python
"""Phase 3c: K=100 GPU Sweep using PyTorch backend (NVIDIA + AMD).

25 primes up to 97 -- the critical test for Fourier sharpening.
N=2000 with batched edge assembly for VRAM-safe operation on 12 GB GPUs.

VRAM budget (K=100, N=2000, batched assembly):
  eps=3.0: ~3-4 GB peak (Lanczos phase) -- comfortable on 12 GB
  eps=5.0: ~5-7 GB peak (Lanczos phase) -- fits with margin

Usage:
  python -u -m atft.experiments.phase3c_torch_k100 2>&1 | tee output/phase3c_torch_k100.log
"""
from __future__ import annotations

import gc
import json
import sys
import time

import numpy as np
import torch

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian


def gpu_cleanup():
    """Force PyTorch to release cached VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_status() -> str:
    """Return current VRAM usage string."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        free = torch.cuda.mem_get_info()[0] / 1e9
        total = torch.cuda.mem_get_info()[1] / 1e9
        return f"{alloc:.2f}GB alloc, {free:.1f}GB free / {total:.1f}GB"
    return "CPU mode"


def run_point(zeros, K, sigma, epsilon, k_eig, label):
    """Run one (sigma, epsilon) grid point with VRAM cleanup."""
    t0 = time.time()
    try:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        lap = TorchSheafLaplacian(builder, zeros, transport_mode="superposition")
        eigs = lap.smallest_eigenvalues(epsilon, k=k_eig)
        s = float(np.sum(eigs))
        tau = 1e-6 * np.sqrt(s) if s > 0 else 1e-10
        beta_0 = int(np.sum(eigs < tau))
        elapsed = time.time() - t0
        print(f"  [{label}] sigma={sigma:.3f} eps={epsilon:.1f}: "
              f"S={s:.6f} b0={beta_0} ({elapsed:.1f}s) "
              f"[{vram_status()}]")
        sys.stdout.flush()
        result = {"sigma": float(sigma), "epsilon": float(epsilon),
                  "spectral_sum": s, "kernel_dim": beta_0,
                  "eigs_top5": eigs[:5].tolist(), "time_s": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{label}] sigma={sigma:.3f} eps={epsilon:.1f}: "
              f"FAILED ({e}) ({elapsed:.1f}s)")
        sys.stdout.flush()
        result = None
    finally:
        gpu_cleanup()
    return result


def generate_gue_points(n, mean_spacing, start, rng):
    """GUE (beta=2) Wigner surmise via rejection sampling."""
    spacings = []
    c_reject = 2.0
    for _ in range(n - 1):
        while True:
            s = rng.rayleigh(scale=np.sqrt(np.pi / 8))
            target = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
            sigma2 = np.pi / 8
            proposal = (s / sigma2) * np.exp(-s**2 / (2 * sigma2))
            if proposal > 0 and rng.random() < target / (c_reject * proposal):
                spacings.append(s * mean_spacing)
                break
    return np.cumsum(np.array([start] + spacings))


def main():
    K = 100
    N = 2000
    k_eig = 20

    # Finer grid near 0.50 to resolve the peak
    sigma_grid = np.array([0.25, 0.35, 0.40, 0.44, 0.46, 0.48, 0.49,
                           0.50, 0.51, 0.52, 0.54, 0.56, 0.60, 0.65, 0.75])
    # eps=3.0 first (safe), then eps=5.0
    epsilon_grid = np.array([3.0, 5.0])

    primes = [p for p in range(2, K + 1)
              if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]

    print("=" * 70)
    print(f"  ATFT PHASE 3c: K=100 TORCH GPU SWEEP (25 primes)")
    print(f"  K={K}, N={N}, dim={N*K}, k_eig={k_eig}")
    print(f"  Primes: {primes}")
    print(f"  sigma grid: {len(sigma_grid)} points, "
          f"[{sigma_grid[0]:.2f} ... {sigma_grid[-1]:.2f}]")
    print(f"  epsilon grid: {list(epsilon_grid)}")
    n_pts = len(sigma_grid) * len(epsilon_grid)
    print(f"  Total points: {n_pts} zeta + {n_pts} random + {n_pts} GUE = {3*n_pts}")
    print(f"  Backend: PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print("=" * 70)

    # GPU check
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {vram_status()}")
    else:
        print("  WARNING: No GPU detected, falling back to CPU")

    # Load zeros
    source = ZetaZerosSource("data/odlyzko_zeros.txt")
    cloud = source.generate(N)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    mean_spacing = float(np.mean(np.diff(np.sort(zeta_zeros))))
    print(f"  Zeros loaded: N={len(zeta_zeros)}, mean_spacing={mean_spacing:.4f}")
    sys.stdout.flush()

    all_results = {}

    # Generate controls upfront (CPU, no VRAM)
    rng = np.random.default_rng(42)
    rand_pts = np.sort(rng.uniform(zeta_zeros.min(), zeta_zeros.max(), len(zeta_zeros)))
    gue_pts = generate_gue_points(len(zeta_zeros), mean_spacing, zeta_zeros.min(), rng)

    sources = [
        ("Zeta", zeta_zeros),
        ("Random", rand_pts),
        ("GUE", gue_pts),
    ]

    sweep_start = time.time()

    for source_label, pts in sources:
        print(f"\n  [{source_label.upper()} -- K={K}, N={N}]")
        all_results[source_label] = {}
        for eps in epsilon_grid:
            for sigma in sigma_grid:
                r = run_point(pts, K, float(sigma), float(eps), k_eig, source_label)
                if r is not None:
                    all_results[source_label][(float(sigma), float(eps))] = r

    sweep_elapsed = time.time() - sweep_start

    # ---- ANALYSIS ----
    print(f"\n{'='*70}")
    print(f"  RESULTS -- K={K}, N={N} (sweep took {sweep_elapsed/3600:.1f}h)")
    print(f"{'='*70}")

    zeta_results = all_results.get("Zeta", {})
    rand_results = all_results.get("Random", {})
    gue_results = all_results.get("GUE", {})

    for eps in epsilon_grid:
        print(f"\n  eps={eps:.1f}:")
        print(f"  {'sigma':>7} {'S(zeta)':>12} {'S(rand)':>12} {'S(GUE)':>12} {'zeta/rand':>10}")
        print(f"  {'-------':>7} {'------------':>12} {'------------':>12} {'------------':>12} {'----------':>10}")
        for sigma in sigma_grid:
            key = (float(sigma), float(eps))
            sz = zeta_results.get(key, {}).get("spectral_sum", float('nan'))
            sr = rand_results.get(key, {}).get("spectral_sum", float('nan'))
            sg = gue_results.get(key, {}).get("spectral_sum", float('nan'))
            ratio = sz / sr if sr > 1e-10 else float('inf')
            marker = " <--" if abs(sigma - 0.50) < 0.005 else ""
            print(f"  {sigma:7.3f} {sz:12.6f} {sr:12.6f} {sg:12.6f} {ratio:10.1f}{marker}")

    # Contrast and peak detection
    for eps in epsilon_grid:
        zeta_vals = {}
        for sigma in sigma_grid:
            key = (float(sigma), float(eps))
            if key in zeta_results:
                zeta_vals[float(sigma)] = zeta_results[key]["spectral_sum"]

        if not zeta_vals:
            continue

        sigmas = sorted(zeta_vals.keys())
        zv = [zeta_vals[s] for s in sigmas]

        peak_idx = int(np.argmax(zv))
        peak_sigma = sigmas[peak_idx]
        peak_val = zv[peak_idx]

        # Parabolic interpolation
        if 0 < peak_idx < len(sigmas) - 1:
            sl, sc, sr = zv[peak_idx-1], zv[peak_idx], zv[peak_idx+1]
            denom = 2 * (sl - 2*sc + sr)
            if abs(denom) > 1e-15:
                delta = sigmas[peak_idx] - sigmas[peak_idx-1]
                offset = delta * (sl - sr) / denom
                peak_sigma = sigmas[peak_idx] + offset

        s_min = min(zv)
        contrast = (peak_val - s_min) / peak_val if peak_val > 0 else 0

        print(f"\n  eps={eps:.1f}: peak_sigma={peak_sigma:.4f}, S(peak)={peak_val:.6f}, "
              f"contrast={contrast:.4f}")

        if 0.50 in zeta_vals:
            s50 = zeta_vals[0.50]
            if peak_val > 0:
                print(f"         S(0.50)={s50:.6f} vs S(peak)={peak_val:.6f} "
                      f"(ratio={s50/peak_val:.4f})")

    # ---- FALSIFICATION CHECK ----
    print(f"\n{'='*70}")
    print(f"  FALSIFICATION CRITERIA CHECK")
    print(f"{'='*70}")

    for eps in epsilon_grid:
        zeta_vals = {}
        rand_vals = {}
        gue_vals = {}
        for sigma in sigma_grid:
            key = (float(sigma), float(eps))
            if key in zeta_results:
                zeta_vals[float(sigma)] = zeta_results[key]["spectral_sum"]
            if key in rand_results:
                rand_vals[float(sigma)] = rand_results[key]["spectral_sum"]
            if key in gue_results:
                gue_vals[float(sigma)] = gue_results[key]["spectral_sum"]

        if not zeta_vals:
            continue

        sigmas = sorted(zeta_vals.keys())
        zv = [zeta_vals[s] for s in sigmas]

        peak_idx = int(np.argmax(zv))
        peak_sigma = sigmas[peak_idx]
        peak_val = zv[peak_idx]
        s_min = min(zv)
        contrast = (peak_val - s_min) / peak_val if peak_val > 0 else 0

        # Signal ratio: zeta / max(random, GUE)
        r_vals = [rand_vals.get(s, 0) for s in sigmas]
        g_vals = [gue_vals.get(s, 0) for s in sigmas]
        ctrl_max = max(max(r_vals) if r_vals else 0, max(g_vals) if g_vals else 0)
        signal_ratio = peak_val / ctrl_max if ctrl_max > 1e-10 else float('inf')

        print(f"\n  eps={eps:.1f}:")
        print(f"    Peak sigma:    {peak_sigma:.4f}  (P1: 0.45-0.52 for strong evidence)")
        print(f"    Contrast:      {contrast:.4f}  (P2: growing with K)")
        print(f"    Signal ratio:  {signal_ratio:.1f}  (P3: > R(K=50))")
        print(f"    F1 check:      peak in [{peak_sigma:.3f}]  ({'PASS' if 0.40 <= peak_sigma <= 0.60 else 'FAIL: outside [0.40, 0.60]'})")

    # ---- SAVE RESULTS ----
    save_path = "output/phase3c_torch_k100_results.json"
    serializable = {}
    for src_label, src_data in all_results.items():
        serializable[src_label] = {}
        for (sigma, eps), result in src_data.items():
            serializable[src_label][f"{sigma:.3f}_{eps:.1f}"] = result

    try:
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results saved to {save_path}")
    except Exception as e:
        print(f"\n  WARNING: Could not save results: {e}")

    print(f"\n  DONE ({sweep_elapsed/3600:.1f}h total)")


if __name__ == "__main__":
    main()
