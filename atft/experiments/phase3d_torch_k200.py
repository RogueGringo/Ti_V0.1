#!/usr/bin/env python
"""Phase 3d: K=200 GPU Sweep using PyTorch backend — Phased Tranches.

46 primes up to 199 — the decisive test for Fourier sharpening.
N=1000 for VRAM feasibility on 12 GB GPUs (dim = 200,000, same as K=100 N=2000).

Tranches:
  T1 (critical zone): sigma=[0.44, 0.48, 0.50, 0.52, 0.56], Zeta + GUE
  T2 (full profile):  sigma=[0.25, 0.35, 0.40, 0.60, 0.65, 0.75], Zeta + GUE
  T3 (random ctrl):   all 15 sigma, Random only

Usage:
  # T1 only (critical zone, ~2-3h):
  python -u -m atft.experiments.phase3d_torch_k200 --tranche T1 2>&1 | tee output/phase3d_k200_T1.log

  # T2 (full profile, ~3-4h):
  python -u -m atft.experiments.phase3d_torch_k200 --tranche T2 2>&1 | tee output/phase3d_k200_T2.log

  # T3 (random control, ~4-5h):
  python -u -m atft.experiments.phase3d_torch_k200 --tranche T3 2>&1 | tee output/phase3d_k200_T3.log

  # All tranches sequentially:
  python -u -m atft.experiments.phase3d_torch_k200 --tranche ALL 2>&1 | tee output/phase3d_k200_ALL.log
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian


# Tranche definitions
SIGMA_T1 = np.array([0.44, 0.48, 0.50, 0.52, 0.56])  # critical zone
SIGMA_T2 = np.array([0.25, 0.35, 0.40, 0.60, 0.65, 0.75])  # full profile
SIGMA_ALL = np.array([0.25, 0.35, 0.40, 0.44, 0.46, 0.48, 0.49,
                       0.50, 0.51, 0.52, 0.54, 0.56, 0.60, 0.65, 0.75])
EPSILON_GRID = np.array([3.0])  # eps=5.0 OOMs at K=200 N=1000 (8.5GB CSR+Lanczos)


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


def run_point(zeros, K, sigma, epsilon, k_eig, label):
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


def load_results(path):
    """Load existing results file if it exists (for incremental tranches)."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(results, path):
    try:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {path}")
    except Exception as e:
        print(f"  WARNING: Could not save results: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tranche", default="T1",
                        choices=["T1", "T2", "T3", "ALL"],
                        help="Which tranche to run")
    args = parser.parse_args()

    K = 200
    N = 1000
    k_eig = 20

    primes = [p for p in range(2, K + 1)
              if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]

    # Determine sigma grid and sources for this tranche
    if args.tranche == "T1":
        sigma_grid = SIGMA_T1
        source_names = ["Zeta", "GUE"]
    elif args.tranche == "T2":
        sigma_grid = SIGMA_T2
        source_names = ["Zeta", "GUE"]
    elif args.tranche == "T3":
        sigma_grid = SIGMA_ALL
        source_names = ["Random"]
    else:  # ALL
        sigma_grid = SIGMA_ALL
        source_names = ["Zeta", "GUE", "Random"]

    print("=" * 70)
    print(f"  ATFT PHASE 3d: K=200 TORCH GPU SWEEP ({len(primes)} primes)")
    print(f"  Tranche: {args.tranche}")
    print(f"  K={K}, N={N}, dim={N*K}, k_eig={k_eig}")
    print(f"  Primes: {primes[:10]}...{primes[-3:]} ({len(primes)} total)")
    print(f"  sigma grid: {len(sigma_grid)} points, "
          f"[{sigma_grid[0]:.2f} ... {sigma_grid[-1]:.2f}]")
    print(f"  epsilon grid: {list(EPSILON_GRID)}")
    print(f"  Sources: {source_names}")
    n_pts = len(sigma_grid) * len(EPSILON_GRID) * len(source_names)
    print(f"  Total points this tranche: {n_pts}")
    print(f"  Backend: PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print("=" * 70)

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

    # Generate controls
    rng = np.random.default_rng(42)
    rand_pts = np.sort(rng.uniform(zeta_zeros.min(), zeta_zeros.max(), len(zeta_zeros)))
    gue_pts = generate_gue_points(len(zeta_zeros), mean_spacing, zeta_zeros.min(), rng)

    source_data = {
        "Zeta": zeta_zeros,
        "Random": rand_pts,
        "GUE": gue_pts,
    }

    # Load existing results for incremental tranche support
    save_path = "output/phase3d_torch_k200_results.json"
    all_results = load_results(save_path)

    sweep_start = time.time()

    for source_label in source_names:
        pts = source_data[source_label]
        print(f"\n  [{source_label.upper()} -- K={K}, N={N}]")
        if source_label not in all_results:
            all_results[source_label] = {}
        for eps in EPSILON_GRID:
            for sigma in sigma_grid:
                key = f"{float(sigma):.3f}_{float(eps):.1f}"
                if key in all_results[source_label]:
                    print(f"  [{source_label}] sigma={sigma:.3f} eps={eps:.1f}: "
                          f"CACHED (S={all_results[source_label][key]['spectral_sum']:.6f})")
                    continue
                r = run_point(pts, K, float(sigma), float(eps), k_eig, source_label)
                if r is not None:
                    all_results[source_label][key] = r
                    # Save after each point for crash resilience
                    save_results(all_results, save_path)

    sweep_elapsed = time.time() - sweep_start

    # ---- ANALYSIS (if we have Zeta data) ----
    if "Zeta" in all_results and len(all_results["Zeta"]) > 0:
        print(f"\n{'='*70}")
        print(f"  RESULTS — K={K}, N={N}, Tranche={args.tranche} "
              f"({sweep_elapsed/3600:.1f}h)")
        print(f"{'='*70}")

        for eps in EPSILON_GRID:
            print(f"\n  eps={eps:.1f}:")
            header_parts = [f"{'sigma':>7}", f"{'S(zeta)':>12}"]
            if "Random" in all_results:
                header_parts.append(f"{'S(rand)':>12}")
            if "GUE" in all_results:
                header_parts.append(f"{'S(GUE)':>12}")
            if "GUE" in all_results:
                header_parts.append(f"{'zeta/GUE':>10}")
            print(f"  {'  '.join(header_parts)}")

            for sigma in sorted(set(SIGMA_ALL)):
                key = f"{float(sigma):.3f}_{float(eps):.1f}"
                sz = all_results.get("Zeta", {}).get(key, {}).get(
                    "spectral_sum", None)
                if sz is None:
                    continue
                parts = [f"{sigma:7.3f}", f"{sz:12.6f}"]
                sr = all_results.get("Random", {}).get(key, {}).get(
                    "spectral_sum", None)
                if "Random" in all_results:
                    parts.append(f"{sr:12.6f}" if sr else f"{'--':>12}")
                sg = all_results.get("GUE", {}).get(key, {}).get(
                    "spectral_sum", None)
                if "GUE" in all_results:
                    parts.append(f"{sg:12.6f}" if sg else f"{'--':>12}")
                if sg and sg > 0:
                    ratio = sz / sg
                    marker = " <--" if abs(sigma - 0.50) < 0.005 else ""
                    parts.append(f"{ratio:10.5f}{marker}")
                print(f"  {'  '.join(parts)}")

        # Arithmetic premium analysis (zeta/GUE)
        if "GUE" in all_results:
            print(f"\n  ARITHMETIC PREMIUM (zeta/GUE ratio):")
            for eps in EPSILON_GRID:
                ratios = {}
                for sigma in sorted(set(SIGMA_ALL)):
                    key = f"{float(sigma):.3f}_{float(eps):.1f}"
                    sz = all_results.get("Zeta", {}).get(key, {}).get(
                        "spectral_sum", None)
                    sg = all_results.get("GUE", {}).get(key, {}).get(
                        "spectral_sum", None)
                    if sz and sg and sg > 0:
                        ratios[sigma] = sz / sg

                if ratios:
                    min_s = min(ratios, key=ratios.get)
                    max_s = max(ratios, key=ratios.get)
                    variation = (max(ratios.values()) - min(ratios.values())) / \
                                np.mean(list(ratios.values())) * 100
                    print(f"    eps={eps}: min(zeta/GUE) at sigma={min_s:.2f} "
                          f"(C={ratios[min_s]:.5f}), "
                          f"max at sigma={max_s:.2f} "
                          f"(C={ratios[max_s]:.5f}), "
                          f"variation={variation:.3f}%")

    print(f"\n  DONE ({sweep_elapsed/3600:.1f}h total)")


if __name__ == "__main__":
    main()
