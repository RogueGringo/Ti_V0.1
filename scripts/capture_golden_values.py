"""Capture golden reference values for regression tests.

Run this ONCE before any refactoring to freeze numerical baselines.
Output is printed as Python literals ready to paste into test_regression.py.
"""
import numpy as np
from pathlib import Path
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.sources.poisson import PoissonSource
from atft.sources.gue import GUESource
from atft.feature_maps.spectral_unfolding import SpectralUnfolding

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "odlyzko_zeros.txt"

# ── Test 1: Golden eigenvalues at K=6, N=10, fixed synthetic zeros ──────
print("=" * 60)
print("Test 1: Small-scale golden eigenvalues (K=6, N=10)")
print("=" * 60)

zeros_small = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
builder_small = TransportMapBuilder(K=6, sigma=0.5)
lap_small = SparseSheafLaplacian(builder_small, zeros_small, normalize=True)

for eps in [1.0, 2.0, 3.0]:
    eigs = lap_small.smallest_eigenvalues(eps, k=20)
    s = float(np.sum(eigs))
    print(f"  eps={eps}: spectral_sum={s!r}")
    print(f"    eigs[:10] = {eigs[:10].tolist()}")

# ── Test 2: Golden spectral sum at K=20, N=200 on real zeta zeros ──────
print()
print("=" * 60)
print("Test 2: K=20 zeta zeros (N=200)")
print("=" * 60)

zeta_src = ZetaZerosSource(data_path=DATA_PATH)
pc = zeta_src.generate(n_points=200)
unfold = SpectralUnfolding(method="zeta")
pc_unfolded = unfold.transform(pc)
zeta_zeros = pc_unfolded.points.ravel()

print(f"  zeta_zeros[:5] = {zeta_zeros[:5].tolist()}")
print(f"  zeta_zeros[-5:] = {zeta_zeros[-5:].tolist()}")

for sigma in [0.25, 0.50, 0.75]:
    builder = TransportMapBuilder(K=20, sigma=sigma)
    lap = SparseSheafLaplacian(builder, zeta_zeros, normalize=True)
    for eps in [3.0, 5.0]:
        s = lap.spectral_sum(eps, k=20)
        print(f"  sigma={sigma}, eps={eps}: spectral_sum={s!r}")

# ── Test 3: Discrimination ratio (zeta vs random) ──────────────────────
print()
print("=" * 60)
print("Test 3: Discrimination ratio at K=20, N=200")
print("=" * 60)

poisson_src = PoissonSource(seed=42)
pc_rand = poisson_src.generate(n_points=200)
# Poisson is already unfolded (mean gap = 1), just use the raw points
rand_zeros = pc_rand.points.ravel()

builder_disc = TransportMapBuilder(K=20, sigma=0.5)
lap_zeta_disc = SparseSheafLaplacian(builder_disc, zeta_zeros, normalize=True)
lap_rand_disc = SparseSheafLaplacian(builder_disc, rand_zeros, normalize=True)

for eps in [3.0, 5.0]:
    s_zeta = lap_zeta_disc.spectral_sum(eps, k=20)
    s_rand = lap_rand_disc.spectral_sum(eps, k=20)
    ratio = s_zeta / s_rand if s_rand > 0 else float('inf')
    print(f"  eps={eps}: S_zeta={s_zeta!r}, S_rand={s_rand!r}, ratio={ratio!r}")

# ── Test 4: FE vs superposition transport (must differ) ────────────────
print()
print("=" * 60)
print("Test 4: FE vs superposition transport at sigma=0.3")
print("=" * 60)

builder_fe = TransportMapBuilder(K=6, sigma=0.3)
lap_sup = SparseSheafLaplacian(builder_fe, zeros_small, transport_mode="superposition", normalize=True)
lap_fe = SparseSheafLaplacian(builder_fe, zeros_small, transport_mode="fe")

s_sup = lap_sup.spectral_sum(2.0, k=10)
s_fe = lap_fe.spectral_sum(2.0, k=10)
print(f"  superposition: {s_sup!r}")
print(f"  fe:            {s_fe!r}")
print(f"  difference:    {abs(s_sup - s_fe)!r}")

# ── Test 5: Monotonicity check at K=20 ─────────────────────────────────
print()
print("=" * 60)
print("Test 5: Monotonicity at K=20, eps=5.0")
print("=" * 60)

sigmas = [0.25, 0.50, 0.75]
sums = []
for sigma in sigmas:
    builder = TransportMapBuilder(K=20, sigma=sigma)
    lap = SparseSheafLaplacian(builder, zeta_zeros, normalize=True)
    s = lap.spectral_sum(5.0, k=20)
    sums.append(s)
    print(f"  sigma={sigma}: S={s!r}")

print(f"  Monotonic increasing: {sums[0] < sums[1] < sums[2]}")

# ── Test 6: GUE control ──────────────────────────────────────────────
print()
print("=" * 60)
print("Test 6: GUE control at K=20, N=200")
print("=" * 60)

gue_src = GUESource(seed=42)
pc_gue = gue_src.generate(n_points=200)
unfold_sc = SpectralUnfolding(method="semicircle")
pc_gue_unf = unfold_sc.transform(pc_gue)
gue_zeros = pc_gue_unf.points.ravel()

builder_gue = TransportMapBuilder(K=20, sigma=0.5)
lap_gue = SparseSheafLaplacian(builder_gue, gue_zeros, normalize=True)

for eps in [3.0, 5.0]:
    s_gue = lap_gue.spectral_sum(eps, k=20)
    s_zeta = lap_zeta_disc.spectral_sum(eps, k=20)
    print(f"  eps={eps}: S_gue={s_gue!r}, S_zeta={s_zeta!r}")

print()
print("=" * 60)
print("DONE — paste these values into tests/test_regression.py")
print("=" * 60)
