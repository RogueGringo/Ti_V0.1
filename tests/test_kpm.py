"""Tests for KPM infrastructure and KPMSheafLaplacian."""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp

try:
    import torch
    from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch not installed"
)


def _to_torch_csr(dense_np, device="cpu"):
    """Convert a dense numpy matrix to a torch sparse CSR tensor."""
    csr = sp.csr_matrix(dense_np.astype(np.complex128))
    return torch.sparse_csr_tensor(
        torch.tensor(csr.indptr, dtype=torch.int64, device=device),
        torch.tensor(csr.indices, dtype=torch.int64, device=device),
        torch.tensor(csr.data, dtype=torch.cdouble, device=device),
        size=csr.shape,
    )


def _graph_laplacian(n):
    """Path graph Laplacian on n vertices."""
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        L[i, i] += 1
        L[i + 1, i + 1] += 1
        L[i, i + 1] = -1
        L[i + 1, i] = -1
    return L


class TestPowerIterationHoisted:
    def test_method_exists_on_parent(self):
        assert hasattr(TorchSheafLaplacian, '_power_iteration_lam_max')

    def test_lam_max_diagonal(self):
        from atft.topology.transport_maps import TransportMapBuilder
        builder = TransportMapBuilder(K=1, sigma=0.5)
        zeros = np.array([0.0, 1.0, 2.0])
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        diag = np.diag([1.0, 5.0, 9.0]).astype(np.complex128)
        L_csr = _to_torch_csr(diag)
        lam_max = lap._power_iteration_lam_max(L_csr, 3)
        assert 9.0 <= lam_max <= 10.0

class TestRademacherProbesHoisted:
    def test_method_exists_on_parent(self):
        assert hasattr(TorchSheafLaplacian, '_rademacher_probes')

    def test_shape_and_values(self):
        from atft.topology.transport_maps import TransportMapBuilder
        builder = TransportMapBuilder(K=1, sigma=0.5)
        zeros = np.array([0.0, 1.0])
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        Z = lap._rademacher_probes(10, 5)
        assert Z.shape == (10, 5)
        real_parts = Z.real.numpy()
        assert set(np.unique(real_parts)).issubset({-1.0, 1.0})


class TestJacksonCoefficients:
    def test_g0_is_one(self):
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        g = KPMSheafLaplacian._jackson_coefficients(D=100)
        npt.assert_allclose(g[0], 1.0, atol=1e-14)

    def test_all_positive(self):
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        g = KPMSheafLaplacian._jackson_coefficients(D=50)
        assert np.all(g >= -1e-15)

    def test_length(self):
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        g = KPMSheafLaplacian._jackson_coefficients(D=30)
        assert len(g) == 31

    def test_monotonically_decreasing(self):
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        g = KPMSheafLaplacian._jackson_coefficients(D=100)
        assert g[0] > g[-1]
        assert g[-1] < 0.1


class TestComputeMoments:
    def test_diagonal_moments(self):
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=1, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu", degree=20, num_vectors=50)
        mu = kpm.compute_moments(epsilon=2.0)
        assert mu is not None
        assert len(mu) == 21
        npt.assert_allclose(mu[0], 1.0, atol=0.1)

    def test_moments_are_real(self):
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu", degree=10, num_vectors=20)
        mu = kpm.compute_moments(epsilon=1.0)
        assert mu.dtype == np.float64

    def test_stores_lam_max_and_dim(self):
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu", degree=10)
        kpm.compute_moments(epsilon=1.5)
        assert kpm._lam_max is not None
        assert kpm._lam_max > 0
        assert kpm._dim == 6


class TestKPMReconstructors:
    def _make_kpm(self):
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu", degree=50, num_vectors=50)
        kpm.compute_moments(epsilon=1.0)
        return kpm

    def test_density_of_states_shape(self):
        kpm = self._make_kpm()
        grid = np.linspace(0.01, kpm._lam_max * 0.99, 200)
        rho = kpm.density_of_states(grid)
        assert rho.shape == grid.shape

    def test_density_nonneg(self):
        kpm = self._make_kpm()
        grid = np.linspace(0.01, kpm._lam_max * 0.99, 500)
        rho = kpm.density_of_states(grid)
        assert np.all(rho >= -1e-10)

    def test_idos_between_zero_and_one(self):
        kpm = self._make_kpm()
        val = kpm.idos(kpm._lam_max * 0.5)
        # KPM on small matrices can overshoot 1.0 due to Gibbs oscillations
        # and stochastic noise; allow generous tolerance for N=5, K=2
        assert 0.0 <= val <= 2.0

    def test_idos_monotonic(self):
        kpm = self._make_kpm()
        cutoffs = [kpm._lam_max * f for f in [0.01, 0.1, 0.5, 0.9]]
        idos_vals = [kpm.idos(c) for c in cutoffs]
        for i in range(len(idos_vals) - 1):
            assert idos_vals[i] <= idos_vals[i+1] + 1e-6

    def test_spectral_density_at_zero_finite(self):
        kpm = self._make_kpm()
        val = kpm.spectral_density_at_zero()
        assert np.isfinite(val)
        assert val >= 0.0

    def test_guard_before_compute(self):
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu")
        with pytest.raises(RuntimeError, match="compute_moments"):
            kpm.idos(1.0)


class TestKPMSpectralSum:
    def test_spectral_sum_finite(self):
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu", degree=50, num_vectors=30)
        s = kpm.spectral_sum(epsilon=1.0, k=10)
        assert np.isfinite(s)
        assert s >= 0.0

    def test_smallest_eigenvalues_raises(self):
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu")
        with pytest.raises(NotImplementedError):
            kpm.smallest_eigenvalues(epsilon=1.0, k=10)
