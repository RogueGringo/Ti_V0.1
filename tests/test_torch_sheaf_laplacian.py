"""Tests for TorchSheafLaplacian (PyTorch GPU/CPU backend).

Validates the PyTorch-based sheaf Laplacian against:
  1. A brute-force dense reference implementation
  2. The trusted SparseSheafLaplacian (CPU BSR backend)
  3. The TransportMapBuilder's batch methods

All tests run on CPU (device="cpu") so they work in CI without a GPU.
GPU-specific tests are marked with @pytest.mark.gpu and skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

try:
    import torch
    from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch not installed"
)


# ---------------------------------------------------------------------------
# Reference implementation (same as test_sparse_sheaf_laplacian.py)
# ---------------------------------------------------------------------------

def _build_dense_vector_laplacian(
    zeros, builder, epsilon, transport_mode="superposition", normalize=True,
):
    """Build explicit dense (N*K, N*K) vector-fiber sheaf Laplacian.

    Reference implementation for validating sparse/torch versions.
    """
    sorted_zeros = np.sort(zeros.ravel())
    N = len(sorted_zeros)
    K = builder.K
    L = np.zeros((N * K, N * K), dtype=np.complex128)
    I_K = np.eye(K, dtype=np.complex128)

    for i in range(N):
        for j in range(i + 1, N):
            gap = sorted_zeros[j] - sorted_zeros[i]
            if gap > epsilon:
                break
            if transport_mode == "superposition":
                U = builder.batch_transport_superposition(
                    np.array([gap]), normalize=normalize
                )[0]
            elif transport_mode == "fe":
                U = builder.batch_transport_fe(np.array([gap]))[0]
            else:
                U = builder.batch_transport_resonant(np.array([gap]))[0]

            Uh = U.conj().T
            L[i*K:(i+1)*K, j*K:(j+1)*K] = -Uh
            L[j*K:(j+1)*K, i*K:(i+1)*K] = -U
            L[i*K:(i+1)*K, i*K:(i+1)*K] += Uh @ U
            L[j*K:(j+1)*K, j*K:(j+1)*K] += I_K

    return L


# ---------------------------------------------------------------------------
# Edge discovery
# ---------------------------------------------------------------------------

class TestEdgeDiscovery:
    """Tests for TorchSheafLaplacian.build_edge_list()."""

    def test_shape(self):
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        i_idx, j_idx, gaps = lap.build_edge_list(1.5)
        assert i_idx.shape == j_idx.shape == gaps.shape
        assert i_idx.ndim == 1

    def test_edges_within_epsilon(self):
        zeros = np.array([0.0, 0.8, 1.5, 3.0, 3.2, 5.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        i_idx, j_idx, gaps = lap.build_edge_list(1.0)
        assert np.all(gaps <= 1.0 + 1e-12)
        assert np.all(gaps > 0)

    def test_edges_complete(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        i_idx, j_idx, gaps = lap.build_edge_list(0.5)
        assert len(i_idx) == 4

    def test_oriented_i_less_j(self):
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        i_idx, j_idx, _ = lap.build_edge_list(2.5)
        assert np.all(i_idx < j_idx)

    def test_eps_zero_no_edges(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        i_idx, j_idx, gaps = lap.build_edge_list(0.0)
        assert len(i_idx) == 0

    def test_large_eps_complete_graph(self):
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        i_idx, j_idx, _ = lap.build_edge_list(100.0)
        assert len(i_idx) == 6

    def test_matches_sparse_backend_edges(self):
        """Torch and sparse backends should discover the same edges."""
        zeros = np.array([0.0, 0.8, 1.5, 2.1, 3.0, 3.7, 4.5])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap_torch = TorchSheafLaplacian(builder, zeros, device="cpu")
        lap_sparse = SparseSheafLaplacian(builder, zeros)

        for eps in [0.5, 1.0, 1.5, 2.0, 5.0]:
            ti, tj, tg = lap_torch.build_edge_list(eps)
            si, sj, sg = lap_sparse.build_edge_list(eps)
            assert len(ti) == len(si), f"Edge count mismatch at eps={eps}"
            npt.assert_array_equal(ti, si)
            npt.assert_array_equal(tj, sj)
            npt.assert_allclose(tg, sg, atol=1e-14)


# ---------------------------------------------------------------------------
# GPU transport
# ---------------------------------------------------------------------------

class TestGPUTransport:
    """Tests for TorchSheafLaplacian.gpu_transport()."""

    def test_matches_builder_superposition(self):
        """gpu_transport should match builder.batch_transport_superposition."""
        zeros = np.array([0.0, 0.8, 1.5, 2.1, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")

        gaps = np.array([0.8, 0.7, 0.6, 0.9, 1.3])
        U_torch = lap.gpu_transport(gaps).numpy()
        U_ref = builder.batch_transport_superposition(gaps, normalize=True)

        npt.assert_allclose(U_torch, U_ref, atol=1e-10)

    def test_matches_builder_K10(self):
        """Cross-validate at K=10 (more primes in the connection)."""
        builder = TransportMapBuilder(K=10, sigma=0.5)
        zeros = np.linspace(0, 5, 8)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")

        gaps = np.array([0.5, 1.0, 1.5, 2.0])
        U_torch = lap.gpu_transport(gaps).numpy()
        U_ref = builder.batch_transport_superposition(gaps, normalize=True)

        npt.assert_allclose(U_torch, U_ref, atol=1e-10)

    def test_empty_gaps(self):
        builder = TransportMapBuilder(K=3, sigma=0.5)
        zeros = np.array([0.0, 1.0])
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")

        result = lap.gpu_transport(np.array([]))
        assert result.shape == (0, 3, 3)

    def test_no_primes_returns_identity(self):
        """K=1 has no primes; transport should be identity."""
        builder = TransportMapBuilder(K=1, sigma=0.5)
        zeros = np.array([0.0, 1.0])
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")

        result = lap.gpu_transport(np.array([0.5, 1.0])).numpy()
        I = np.eye(1, dtype=np.complex128)
        for m in range(2):
            npt.assert_allclose(result[m], I, atol=1e-14)


# ---------------------------------------------------------------------------
# CPU transport (fe/resonant fallback)
# ---------------------------------------------------------------------------

class TestCPUTransport:
    """Tests for _compute_transport (inherited from BaseSheafLaplacian)."""

    def test_fe_matches_builder(self):
        builder = TransportMapBuilder(K=6, sigma=0.5)
        zeros = np.array([0.0, 1.0, 2.0])
        lap = TorchSheafLaplacian(builder, zeros, transport_mode="fe", device="cpu")

        gaps = np.array([1.0, 2.0])
        U = lap._compute_transport(gaps)
        U_ref = builder.batch_transport_fe(gaps)
        npt.assert_allclose(U, U_ref, atol=1e-14)

    def test_resonant_matches_builder(self):
        builder = TransportMapBuilder(K=6, sigma=0.5)
        zeros = np.array([0.0, 1.0, 2.0])
        lap = TorchSheafLaplacian(builder, zeros, transport_mode="resonant", device="cpu")

        gaps = np.array([1.0, 2.0])
        U = lap._compute_transport(gaps)
        U_ref = builder.batch_transport_resonant(gaps)
        npt.assert_allclose(U, U_ref, atol=1e-14)

    def test_unknown_mode_raises(self):
        builder = TransportMapBuilder(K=3, sigma=0.5)
        zeros = np.array([0.0, 1.0])
        lap = TorchSheafLaplacian(builder, zeros, transport_mode="bogus", device="cpu")
        with pytest.raises(ValueError, match="Unknown transport_mode"):
            lap._compute_transport(np.array([1.0]))


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------

class TestBuildMatrix:
    """Tests for build_matrix() sparse CSR assembly."""

    def test_dense_equivalence_K3_N5(self):
        """Torch Laplacian matches dense reference at small scale."""
        zeros = np.array([0.0, 0.8, 1.5, 2.1, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        L_torch = lap.build_matrix(1.0).to_dense().numpy()
        L_dense = _build_dense_vector_laplacian(
            zeros, builder, 1.0, "superposition", True
        )
        npt.assert_allclose(L_torch, L_dense, atol=1e-10)

    def test_dense_equivalence_K6_N4(self):
        """Different K and sigma, still matches reference."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.4)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        L_torch = lap.build_matrix(1.5).to_dense().numpy()
        L_dense = _build_dense_vector_laplacian(
            zeros, builder, 1.5, "superposition", True
        )
        npt.assert_allclose(L_torch, L_dense, atol=1e-10)

    def test_hermitian(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        L = lap.build_matrix(1.5).to_dense().numpy()
        npt.assert_allclose(L, L.conj().T, atol=1e-10)

    def test_positive_semi_definite(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        L = lap.build_matrix(1.0).to_dense().numpy()
        eigenvals = np.linalg.eigvalsh(L)
        assert np.all(eigenvals > -1e-8)

    def test_shape(self):
        N, K = 5, 4
        zeros = np.arange(N, dtype=np.float64)
        builder = TransportMapBuilder(K=K, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        L = lap.build_matrix(1.5)
        assert L.shape == (N * K, N * K)

    def test_eps_zero_is_zero_matrix(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        L = lap.build_matrix(0.0)
        assert L._nnz() == 0

    def test_fe_transport_mode(self):
        """FE transport mode matches dense reference."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, transport_mode="fe", device="cpu")
        L_torch = lap.build_matrix(1.5).to_dense().numpy()
        L_dense = _build_dense_vector_laplacian(
            zeros, builder, 1.5, "fe", True
        )
        npt.assert_allclose(L_torch, L_dense, atol=1e-12)

    def test_resonant_transport_mode(self):
        """Resonant transport mode matches dense reference."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, transport_mode="resonant", device="cpu")
        L_torch = lap.build_matrix(1.5).to_dense().numpy()
        L_dense = _build_dense_vector_laplacian(
            zeros, builder, 1.5, "resonant", True
        )
        npt.assert_allclose(L_torch, L_dense, atol=1e-12)

    def test_matches_sparse_backend_matrix(self):
        """Torch and sparse backends produce the same Laplacian matrix."""
        zeros = np.array([0.0, 0.8, 1.5, 2.1, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap_torch = TorchSheafLaplacian(builder, zeros, device="cpu")
        lap_sparse = SparseSheafLaplacian(builder, zeros, normalize=True)

        L_torch = lap_torch.build_matrix(1.0).to_dense().numpy()
        L_sparse = lap_sparse.build_matrix(1.0).toarray()
        npt.assert_allclose(L_torch, L_sparse, atol=1e-10)


# ---------------------------------------------------------------------------
# Cross-validation: eigenvalues
# ---------------------------------------------------------------------------

class TestCrossValidation:
    """Cross-validate Torch eigenvalues against the trusted Sparse backend."""

    def test_eigenvalues_match_sparse_K3(self):
        """Eigenvalues from torch and sparse backends should match."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap_torch = TorchSheafLaplacian(builder, zeros, device="cpu")
        lap_sparse = SparseSheafLaplacian(builder, zeros, normalize=True)

        eigs_torch = lap_torch.smallest_eigenvalues(1.5, k=10)
        eigs_sparse = lap_sparse.smallest_eigenvalues(1.5, k=10)
        npt.assert_allclose(eigs_torch, eigs_sparse, atol=1e-8)

    def test_eigenvalues_match_sparse_K6(self):
        """Cross-validate at K=6 with more edges."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap_torch = TorchSheafLaplacian(builder, zeros, device="cpu")
        lap_sparse = SparseSheafLaplacian(builder, zeros, normalize=True)

        eigs_torch = lap_torch.smallest_eigenvalues(1.5, k=20)
        eigs_sparse = lap_sparse.smallest_eigenvalues(1.5, k=20)
        npt.assert_allclose(eigs_torch, eigs_sparse, atol=1e-8)

    def test_spectral_sum_match(self):
        """Spectral sums from both backends should be close."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        builder = TransportMapBuilder(K=4, sigma=0.5)
        lap_torch = TorchSheafLaplacian(builder, zeros, device="cpu")
        lap_sparse = SparseSheafLaplacian(builder, zeros, normalize=True)

        s_torch = lap_torch.spectral_sum(1.0, k=10)
        s_sparse = lap_sparse.spectral_sum(1.0, k=10)
        npt.assert_allclose(s_torch, s_sparse, atol=1e-6)

    def test_fe_eigenvalues_match(self):
        """FE mode eigenvalues match between backends."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap_torch = TorchSheafLaplacian(builder, zeros, transport_mode="fe", device="cpu")
        lap_sparse = SparseSheafLaplacian(builder, zeros, transport_mode="fe")

        eigs_torch = lap_torch.smallest_eigenvalues(1.5, k=10)
        eigs_sparse = lap_sparse.smallest_eigenvalues(1.5, k=10)
        npt.assert_allclose(eigs_torch, eigs_sparse, atol=1e-10)


# ---------------------------------------------------------------------------
# Eigensolver
# ---------------------------------------------------------------------------

class TestEigensolver:
    """Tests for smallest_eigenvalues() and spectral_sum()."""

    def test_eigenvalues_sorted_nonneg(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        eigs = lap.smallest_eigenvalues(1.5, k=10)
        assert len(eigs) == 10
        assert np.all(eigs[:-1] <= eigs[1:] + 1e-10)  # sorted
        assert np.all(eigs > -1e-10)  # nonneg

    def test_matches_dense_eigenvalues(self):
        """Should match dense eigh for small problem."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        eigs_torch = lap.smallest_eigenvalues(1.5, k=10)

        L_dense = _build_dense_vector_laplacian(zeros, builder, 1.5)
        eigs_dense = np.sort(np.linalg.eigvalsh(L_dense).real)[:10]
        eigs_dense = np.maximum(eigs_dense, 0.0)
        npt.assert_allclose(eigs_torch, eigs_dense, atol=1e-8)

    def test_eps_zero_returns_zeros(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        eigs = lap.smallest_eigenvalues(0.0, k=5)
        npt.assert_allclose(eigs, np.zeros(5), atol=1e-14)

    def test_pads_with_zeros(self):
        """Should pad with zeros when k > dim."""
        zeros = np.array([0.0, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        eigs = lap.smallest_eigenvalues(1.5, k=10)
        assert len(eigs) == 10

    def test_spectral_sum(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        eigs = lap.smallest_eigenvalues(1.0, k=10)
        s = lap.spectral_sum(1.0, k=10)
        npt.assert_allclose(s, float(np.sum(eigs)), atol=1e-10)

    def test_spectral_sum_eps_zero(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        s = lap.spectral_sum(0.0, k=5)
        assert s == 0.0


# ---------------------------------------------------------------------------
# Device / initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    """Tests for constructor and device handling."""

    def test_cpu_device(self):
        builder = TransportMapBuilder(K=3, sigma=0.5)
        zeros = np.array([0.0, 1.0, 2.0])
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        assert lap.device == torch.device("cpu")
        assert lap.N == 3
        assert lap.K == 3

    def test_default_device(self):
        builder = TransportMapBuilder(K=3, sigma=0.5)
        zeros = np.array([0.0, 1.0, 2.0])
        lap = TorchSheafLaplacian(builder, zeros)
        assert lap.device is not None

    def test_stores_transport_mode(self):
        builder = TransportMapBuilder(K=3, sigma=0.5)
        zeros = np.array([0.0, 1.0])
        lap = TorchSheafLaplacian(builder, zeros, transport_mode="fe", device="cpu")
        assert lap._transport_mode == "fe"

    def test_different_sigma_different_result(self):
        """Different sigma should produce different spectral sums."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder_a = TransportMapBuilder(K=6, sigma=0.3)
        builder_b = TransportMapBuilder(K=6, sigma=0.7)
        lap_a = TorchSheafLaplacian(builder_a, zeros, device="cpu")
        lap_b = TorchSheafLaplacian(builder_b, zeros, device="cpu")
        s_a = lap_a.spectral_sum(1.0, k=10)
        s_b = lap_b.spectral_sum(1.0, k=10)
        assert s_a != s_b


# ---------------------------------------------------------------------------
# Transport mode dispatch in build_matrix
# ---------------------------------------------------------------------------

class TestTransportModeDispatch:
    """Verify all three transport modes produce valid Laplacians."""

    @pytest.mark.parametrize("mode", ["superposition", "fe", "resonant"])
    def test_all_modes_produce_hermitian_psd(self, mode):
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, transport_mode=mode, device="cpu")
        L = lap.build_matrix(1.5).to_dense().numpy()

        # Hermitian
        npt.assert_allclose(L, L.conj().T, atol=1e-10)
        # PSD
        eigenvals = np.linalg.eigvalsh(L)
        assert np.all(eigenvals > -1e-8)

    def test_superposition_differs_from_fe(self):
        """Different transport modes should produce different Laplacians."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.3)
        lap_sup = TorchSheafLaplacian(builder, zeros, transport_mode="superposition", device="cpu")
        lap_fe = TorchSheafLaplacian(builder, zeros, transport_mode="fe", device="cpu")
        s_sup = lap_sup.spectral_sum(1.5, k=10)
        s_fe = lap_fe.spectral_sum(1.5, k=10)
        assert abs(s_sup - s_fe) > 1e-6
