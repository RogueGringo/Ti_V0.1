#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""Phase 3f: K=400 GPU Sweep — 78 primes, the scaling test.

If the signal is real:
  - Premium should grow to ~25-28%
  - Peak should lock at σ=0.500 ± 0.005
  - Eigenvalue ratio should drop below 0.79

If the signal is artifact:
  - Premium saturates or reverses
  - Peak wanders away from 0.500

N=1000, eps=3.0. Sigma grid = [0.44, 0.48, 0.50, 0.52, 0.56] (critical zone only).
Full profile after critical zone confirms.
"""
from __future__ import annotations

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


K = 400
N = 1000
K_EIG = 20
EPSILON = 3.0
SIGMA_GRID = np.array([0.44, 0.48, 0.50, 0.52, 0.56])
SAVE_PATH = "output/phase3f_torch_k400_results.json"


def gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_status():
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
        print(f"  [{label}] sigma={sigma:.3f}: S={s:.6f} b0={beta_0} "
              f"({elapsed:.1f}s) [{vram_status()}]")
        sys.stdout.flush()
        return {"sigma": float(sigma), "epsilon": float(epsilon),
                "spectral_sum": s, "kernel_dim": beta_0,
                "eigs_top5": eigs[:5].tolist(), "time_s": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{label}] sigma={sigma:.3f}: FAILED ({e}) ({elapsed:.1f}s)")
        sys.stdout.flush()
        return None
    finally:
        gpu_cleanup()


def generate_gue_points(n, mean_spacing, start, rng):
    """Wigner surmise GUE (same as phase3d for comparability)."""
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
    primes = [p for p in range(2, K + 1)
              if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]

    print("=" * 70)
    print(f"  ATFT PHASE 3f: K=400 TORCH GPU SWEEP ({len(primes)} primes)")
    print(f"  K={K}, N={N}, dim={N*K}, k_eig={K_EIG}")
    print(f"  Primes: {primes[:5]}...{primes[-3:]} ({len(primes)} total)")
    print(f"  sigma grid: {list(SIGMA_GRID)}")
    print(f"  eps: {EPSILON}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {vram_status()}")
    print("=" * 70)

    # Load zeros
    source = ZetaZerosSource("data/odlyzko_zeros.txt")
    cloud = source.generate(N)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    mean_spacing = float(np.mean(np.diff(np.sort(zeta_zeros))))
    print(f"  Zeros: N={len(zeta_zeros)}, mean_spacing={mean_spacing:.4f}")

    rng = np.random.default_rng(42)
    rand_pts = np.sort(rng.uniform(zeta_zeros.min(), zeta_zeros.max(), N))
    gue_pts = generate_gue_points(N, mean_spacing, float(zeta_zeros.min()), rng)

    sources = {"Zeta": zeta_zeros, "GUE": gue_pts, "Random": rand_pts}

    all_results = {}
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH) as f:
            all_results = json.load(f)

    sweep_start = time.time()

    for src_name in ["Zeta", "GUE", "Random"]:
        pts = sources[src_name]
        print(f"\n  [{src_name.upper()} — K={K}, N={N}]")
        if src_name not in all_results:
            all_results[src_name] = {}
        for sigma in SIGMA_GRID:
            key = f"{float(sigma):.3f}_{EPSILON:.1f}"
            if key in all_results[src_name]:
                cached = all_results[src_name][key]["spectral_sum"]
                print(f"  [{src_name}] sigma={sigma:.3f}: CACHED (S={cached:.6f})")
                continue
            r = run_point(pts, K, float(sigma), EPSILON, K_EIG, src_name)
            if r is not None:
                all_results[src_name][key] = r
                with open(SAVE_PATH, "w") as f:
                    json.dump(all_results, f, indent=2)

    elapsed = time.time() - sweep_start

    # Quick analysis
    print(f"\n{'='*70}")
    print(f"  K=400 RESULTS ({elapsed/3600:.1f}h)")
    print(f"{'='*70}")

    for sigma in SIGMA_GRID:
        key = f"{float(sigma):.3f}_{EPSILON:.1f}"
        sz = all_results.get("Zeta", {}).get(key, {}).get("spectral_sum")
        sg = all_results.get("GUE", {}).get(key, {}).get("spectral_sum")
        sr = all_results.get("Random", {}).get(key, {}).get("spectral_sum")
        if sz and sg:
            prem = (1 - sz/sg) * 100
            marker = " <--" if abs(sigma - 0.5) < 0.005 else ""
            print(f"  sigma={sigma:.3f}: S(Z)={sz:.3f} S(G)={sg:.3f} "
                  f"S(R)={sr:.3f if sr else 0:.3f} premium={prem:.1f}%{marker}")

    print(f"\n  DONE ({elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
