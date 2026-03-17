"""End-to-end regression tests with frozen golden reference values.

These tests validate the full pipeline:
  zeros -> transport -> Laplacian -> eigenvalues -> spectral sum

Golden values are computed once and frozen. Any change to the numerical
pipeline (edge discovery, transport, assembly, eigensolver) that alters
these values is a regression.

Golden values captured on 2026-03-17 using:
  - Python 3.14.0, numpy 2.4.3, scipy 1.17.1
  - SparseSheafLaplacian (CPU BSR backend, trusted reference)
  - scripts/capture_golden_values.py
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path

from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.sources.poisson import PoissonSource
from atft.feature_maps.spectral_unfolding import SpectralUnfolding

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "odlyzko_zeros.txt"


class TestGoldenReference:
    """Frozen golden eigenvalue values -- regression detection."""

    def test_spectral_sum_frozen_small(self):
        """K=6, N=10, eps=3.0, sigma=0.5 -- value frozen at commit time."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros, normalize=True)
        s = lap.spectral_sum(3.0, k=20)

        # FROZEN VALUE
        expected_s = 82.7088791191382
        assert s == pytest.approx(expected_s, abs=1e-8), (
            f"Golden spectral sum changed! Got {s}. "
            "If this is intentional, update the frozen value."
        )

    def test_spectral_sum_frozen_small_eps1(self):
        """K=6, N=10, eps=1.0, sigma=0.5 -- frozen spectral sum."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros, normalize=True)
        s = lap.spectral_sum(1.0, k=20)

        expected_s = 19.54886627844357
        assert s == pytest.approx(expected_s, abs=1e-8)

    def test_first_10_eigenvalues_frozen(self):
        """K=6, N=10, eps=1.0 -- frozen first 10 eigenvalues."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros, normalize=True)
        eigs = lap.smallest_eigenvalues(1.0, k=20)

        # FROZEN EIGENVALUES
        expected = np.array([
            0.0026995282459244666, 0.002911758561234296,
            0.019555334878115112, 0.029544365520144104,
            0.05418393699412956, 0.11453301533165135,
            0.4420121772852158, 0.45623982407175767,
            0.5067794171602239, 0.6057277794921198,
        ])
        npt.assert_allclose(eigs[:10], expected, atol=1e-8,
                            err_msg="Eigenvalue regression at eps=1.0")

    def test_spectral_sum_frozen_zeta(self):
        """K=20, N=200, eps=3.0, sigma=0.5 -- value frozen at commit time."""
        if not DATA_PATH.exists():
            pytest.skip("Odlyzko data not found")

        zeta_src = ZetaZerosSource(data_path=DATA_PATH)
        pc = zeta_src.generate(n_points=200)
        unfold = SpectralUnfolding(method="zeta")
        zeta_zeros = unfold.transform(pc).points.ravel()

        builder = TransportMapBuilder(K=20, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeta_zeros, normalize=True)
        s = lap.spectral_sum(3.0, k=20)

        # FROZEN VALUE
        expected_s = 0.15455824194814874
        assert s == pytest.approx(expected_s, abs=1e-8)

    @pytest.mark.parametrize("sigma,eps,expected", [
        (0.25, 3.0, 0.111058269234253),
        (0.25, 5.0, 0.47542736932030855),
        (0.50, 5.0, 0.6805539594815005),
        (0.75, 3.0, 0.12874646199245332),
        (0.75, 5.0, 0.5976520481575343),
    ])
    def test_spectral_sum_zeta_grid(self, sigma, eps, expected):
        """Frozen spectral sums across the (sigma, eps) grid on zeta zeros."""
        if not DATA_PATH.exists():
            pytest.skip("Odlyzko data not found")

        zeta_src = ZetaZerosSource(data_path=DATA_PATH)
        pc = zeta_src.generate(n_points=200)
        unfold = SpectralUnfolding(method="zeta")
        zeta_zeros = unfold.transform(pc).points.ravel()

        builder = TransportMapBuilder(K=20, sigma=sigma)
        lap = SparseSheafLaplacian(builder, zeta_zeros, normalize=True)
        s = lap.spectral_sum(eps, k=20)
        assert s == pytest.approx(expected, abs=1e-8)


class TestDiscriminationRatio:
    """Verify zeta-like data has higher spectral sum than random."""

    def test_zeta_vs_random(self):
        """Structured zeros should give much higher spectral sum than random."""
        if not DATA_PATH.exists():
            pytest.skip("Odlyzko data not found")

        zeta_src = ZetaZerosSource(data_path=DATA_PATH)
        pc_zeta = zeta_src.generate(n_points=200)
        unfold = SpectralUnfolding(method="zeta")
        zeta_zeros = unfold.transform(pc_zeta).points.ravel()

        poisson_src = PoissonSource(seed=42)
        rand_zeros = poisson_src.generate(n_points=200).points.ravel()

        builder = TransportMapBuilder(K=20, sigma=0.5)
        lap_zeta = SparseSheafLaplacian(builder, zeta_zeros, normalize=True)
        lap_rand = SparseSheafLaplacian(builder, rand_zeros, normalize=True)

        s_zeta = lap_zeta.spectral_sum(3.0, k=20)
        s_rand = lap_rand.spectral_sum(3.0, k=20)

        # Frozen expected values:
        # S_zeta=0.15455824194814874, S_rand=0.001993986943814548
        assert s_zeta > 0.15
        assert s_rand < 0.01
        assert (s_zeta / s_rand) > 50.0  # Ratio is ~77x

    def test_discrimination_values_frozen(self):
        """Frozen S_zeta and S_rand at eps=3.0."""
        if not DATA_PATH.exists():
            pytest.skip("Odlyzko data not found")

        zeta_src = ZetaZerosSource(data_path=DATA_PATH)
        pc_zeta = zeta_src.generate(n_points=200)
        unfold = SpectralUnfolding(method="zeta")
        zeta_zeros = unfold.transform(pc_zeta).points.ravel()

        poisson_src = PoissonSource(seed=42)
        rand_zeros = poisson_src.generate(n_points=200).points.ravel()

        builder = TransportMapBuilder(K=20, sigma=0.5)
        lap_zeta = SparseSheafLaplacian(builder, zeta_zeros, normalize=True)
        lap_rand = SparseSheafLaplacian(builder, rand_zeros, normalize=True)

        s_zeta = lap_zeta.spectral_sum(3.0, k=20)
        s_rand = lap_rand.spectral_sum(3.0, k=20)

        assert s_zeta == pytest.approx(0.15455824194814874, abs=1e-8)
        assert s_rand == pytest.approx(0.001993986943814548, abs=1e-8)


class TestModeDiscrimination:
    """Different transport modes should produce different spectral sums."""

    def test_superposition_vs_fe(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        builder = TransportMapBuilder(K=6, sigma=0.3)
        lap_sup = SparseSheafLaplacian(builder, zeros, transport_mode="superposition", normalize=True)
        lap_fe = SparseSheafLaplacian(builder, zeros, transport_mode="fe")
        s_sup = lap_sup.spectral_sum(2.0, k=10)
        s_fe = lap_fe.spectral_sum(2.0, k=10)

        # Frozen values: sup=9.976521254177285, fe=10.734685169022335
        assert s_sup == pytest.approx(9.976521254177285, abs=1e-8)
        assert s_fe == pytest.approx(10.734685169022335, abs=1e-8)
        assert abs(s_sup - s_fe) > 0.01


class TestSigmaProfile:
    """The sigma-dependence of S at K=20 on zeta zeros."""

    def test_signal_increases_toward_half(self):
        """S(0.25) < S(0.50) at eps=5.0 -- signal grows toward critical line."""
        if not DATA_PATH.exists():
            pytest.skip("Odlyzko data not found")

        zeta_src = ZetaZerosSource(data_path=DATA_PATH)
        pc = zeta_src.generate(n_points=200)
        unfold = SpectralUnfolding(method="zeta")
        zeta_zeros = unfold.transform(pc).points.ravel()

        sums = {}
        for sigma in [0.25, 0.50, 0.75]:
            builder = TransportMapBuilder(K=20, sigma=sigma)
            lap = SparseSheafLaplacian(builder, zeta_zeros, normalize=True)
            sums[sigma] = lap.spectral_sum(5.0, k=20)

        # At N=200 K=20, the profile peaks near sigma=0.50
        assert sums[0.25] < sums[0.50], "S should increase from 0.25 to 0.50"
        assert sums[0.75] < sums[0.50], "S should decrease from 0.50 to 0.75"

        # Frozen values
        assert sums[0.25] == pytest.approx(0.47542736932030855, abs=1e-8)
        assert sums[0.50] == pytest.approx(0.6805539594815005, abs=1e-8)
        assert sums[0.75] == pytest.approx(0.5976520481575343, abs=1e-8)
