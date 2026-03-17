"""Tests for the custom Lanczos eigensolver in torch_sheaf_laplacian.

Validates _lanczos_largest and lanczos_smallest against dense numpy
eigensolvers on known matrices (diagonal, Hermitian, graph Laplacian).

All tests run on CPU.
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp

try:
    import torch
    from atft.topology.torch_sheaf_laplacian import _lanczos_largest, lanczos_smallest
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_torch_csr(dense_np, device="cpu"):
    """Convert a dense numpy matrix to a torch sparse CSR tensor."""
    csr = sp.csr_matrix(dense_np.astype(np.complex128))
    return torch.sparse_csr_tensor(
        torch.tensor(csr.indptr, dtype=torch.int64, device=device),
        torch.tensor(csr.indices, dtype=torch.int64, device=device),
        torch.tensor(csr.data, dtype=torch.cdouble, device=device),
        size=csr.shape,
    )


def _make_matvec(M_np, device="cpu"):
    """Return a matvec closure for a dense numpy matrix."""
    M_t = torch.tensor(M_np.astype(np.complex128), dtype=torch.cdouble, device=device)
    def matvec(v):
        return M_t @ v
    return matvec


def _graph_laplacian(n):
    """Build the graph Laplacian of a path graph on n vertices.

    Eigenvalues: 2 - 2*cos(k*pi/n) for k=0,...,n-1.
    Smallest eigenvalue is always 0 (connected graph).
    """
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        L[i, i] += 1
        L[i + 1, i + 1] += 1
        L[i, i + 1] = -1
        L[i + 1, i] = -1
    return L


# ---------------------------------------------------------------------------
# _lanczos_largest
# ---------------------------------------------------------------------------

class TestLanczosLargest:
    """Tests for _lanczos_largest (find k largest eigenvalues)."""

    def test_diagonal_matrix(self):
        """Diagonal matrix: eigenvalues are the diagonal entries."""
        diag_vals = np.array([1.0, 5.0, 3.0, 7.0, 2.0, 9.0, 4.0, 6.0])
        M = np.diag(diag_vals)
        dim = len(diag_vals)
        k = 3

        matvec = _make_matvec(M)
        eigs = _lanczos_largest(
            matvec, dim, k=k, device="cpu", dtype=torch.cdouble
        )
        expected = np.sort(diag_vals)[-k:][::-1]  # [9, 7, 6]
        npt.assert_allclose(eigs, expected, atol=1e-8)

    def test_hermitian_matrix(self):
        """Random Hermitian matrix: compare against numpy eigvalsh."""
        rng = np.random.default_rng(42)
        n = 30
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = (A + A.conj().T) / 2
        k = 5

        ref_eigs = np.sort(np.linalg.eigvalsh(H))[-k:][::-1]

        matvec = _make_matvec(H)
        eigs = _lanczos_largest(
            matvec, n, k=k, device="cpu", dtype=torch.cdouble, max_iter=200
        )
        npt.assert_allclose(eigs, ref_eigs, atol=1e-8)

    def test_psd_graph_laplacian(self):
        """Graph Laplacian largest eigenvalues match analytic formula."""
        n = 20
        L = _graph_laplacian(n)
        k = 4

        ref_eigs = np.sort(np.linalg.eigvalsh(L))[-k:][::-1]

        matvec = _make_matvec(L)
        eigs = _lanczos_largest(
            matvec, n, k=k, device="cpu", dtype=torch.cdouble
        )
        npt.assert_allclose(eigs, ref_eigs, atol=1e-8)

    def test_single_eigenvalue(self):
        """Finding just the largest eigenvalue."""
        diag_vals = np.array([2.0, 8.0, 5.0, 1.0])
        M = np.diag(diag_vals)

        matvec = _make_matvec(M)
        eigs = _lanczos_largest(
            matvec, 4, k=1, device="cpu", dtype=torch.cdouble
        )
        npt.assert_allclose(eigs, [8.0], atol=1e-8)

    def test_identity_matrix(self):
        """Identity: Lanczos finds the eigenvalue 1.0.

        Since all eigenvalues are identical, the Krylov subspace is 1D
        and Lanczos hits invariant subspace breakdown after 1 iteration.
        This is correct behavior — verify the single returned value.
        """
        n = 10
        I = np.eye(n)

        matvec = _make_matvec(I)
        eigs = _lanczos_largest(
            matvec, n, k=1, device="cpu", dtype=torch.cdouble
        )
        npt.assert_allclose(eigs, [1.0], atol=1e-8)

    def test_returns_descending_order(self):
        """Output should be in descending order."""
        rng = np.random.default_rng(99)
        n = 15
        A = rng.standard_normal((n, n))
        H = (A + A.T) / 2
        k = 5

        matvec = _make_matvec(H)
        eigs = _lanczos_largest(
            matvec, n, k=k, device="cpu", dtype=torch.cdouble
        )
        assert len(eigs) == k
        assert np.all(eigs[:-1] >= eigs[1:] - 1e-10)


# ---------------------------------------------------------------------------
# lanczos_smallest (spectral flip)
# ---------------------------------------------------------------------------

class TestLanczosSmallest:
    """Tests for lanczos_smallest (find k smallest eigenvalues via spectral flip)."""

    def test_matches_dense_eigvalsh(self):
        """Smallest eigenvalues should match dense eigvalsh."""
        rng = np.random.default_rng(42)
        n = 30
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = (A + A.conj().T) / 2
        # Make PSD by adding lambda_max * I
        evals = np.linalg.eigvalsh(H)
        H_psd = H + (abs(evals.min()) + 1.0) * np.eye(n)
        k = 5

        ref_eigs = np.sort(np.linalg.eigvalsh(H_psd).real)[:k]
        ref_eigs = np.maximum(ref_eigs, 0.0)

        L_csr = _to_torch_csr(H_psd)
        eigs = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")
        npt.assert_allclose(eigs, ref_eigs, atol=1e-4)

    def test_graph_laplacian_smallest(self):
        """Path graph Laplacian: smallest eigenvalue is 0."""
        n = 20
        L = _graph_laplacian(n)
        k = 5

        ref_eigs = np.sort(np.linalg.eigvalsh(L))[:k]
        ref_eigs = np.maximum(ref_eigs, 0.0)

        L_csr = _to_torch_csr(L)
        eigs = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")
        npt.assert_allclose(eigs, ref_eigs, atol=1e-4)

    def test_psd_eigenvalues_nonneg(self):
        """All returned eigenvalues should be >= 0 for a PSD matrix."""
        n = 15
        L = _graph_laplacian(n)
        k = 5

        L_csr = _to_torch_csr(L)
        eigs = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")
        assert np.all(eigs >= -1e-10)

    def test_sorted_ascending(self):
        """Output should be sorted in ascending order."""
        n = 20
        L = _graph_laplacian(n)
        k = 8

        L_csr = _to_torch_csr(L)
        eigs = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")
        assert np.all(eigs[:-1] <= eigs[1:] + 1e-10)

    def test_zero_matrix_returns_zeros(self):
        """Zero matrix has all-zero eigenvalues; should return zeros via lam_max guard."""
        n = 5
        Z = np.zeros((n, n))
        L_csr = _to_torch_csr(Z)
        eigs = lanczos_smallest(L_csr, k=3, dim=n, device="cpu")
        npt.assert_allclose(eigs, np.zeros(3), atol=1e-14)

    def test_diagonal_psd(self):
        """Diagonal PSD matrix: known eigenvalues."""
        diag_vals = np.array([0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
        M = np.diag(diag_vals)
        n = len(diag_vals)
        k = 4

        L_csr = _to_torch_csr(M)
        eigs = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")
        expected = np.array([0.0, 1.0, 3.0, 5.0])
        npt.assert_allclose(eigs, expected, atol=1e-4)

    def test_spectral_flip_correctness(self):
        """Verify the spectral flip: lam_max - largest_of_M = smallest_of_L."""
        n = 15
        L = _graph_laplacian(n)
        k = 3

        # Dense reference
        all_eigs = np.sort(np.linalg.eigvalsh(L))
        ref_smallest = np.maximum(all_eigs[:k], 0.0)

        # Lanczos with spectral flip
        L_csr = _to_torch_csr(L)
        eigs = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")

        npt.assert_allclose(eigs, ref_smallest, atol=1e-4)


# ---------------------------------------------------------------------------
# Integration: Lanczos inside TorchSheafLaplacian
# ---------------------------------------------------------------------------

class TestLanczosIntegration:
    """Verify Lanczos eigensolver gives correct results when called
    through TorchSheafLaplacian.smallest_eigenvalues on problems
    small enough to cross-validate with dense eigensolvers."""

    def test_small_problem_uses_dense_path(self):
        """dim <= 500 should use dense eigvalsh, matching reference."""
        from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder

        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # N=5, K=3 => dim=15
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")

        eigs = lap.smallest_eigenvalues(1.0, k=10)
        L = lap.build_matrix(1.0).to_dense().numpy()
        ref = np.sort(np.linalg.eigvalsh(L).real)[:10]
        ref = np.maximum(ref, 0.0)

        npt.assert_allclose(eigs, ref, atol=1e-8)
