"""Tests for phase2a_abelian.py — eigenbasis diagnostic.

Key validation: diagonal blocks must reproduce the standard graph Laplacian
(and therefore Phase 1's scalar Betti curve).
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from atft.experiments.phase2a_abelian import Phase2aAbelian
from atft.topology.transport_maps import TransportMapBuilder


class TestTwistedLaplacian:
    """Test the scalar twisted graph Laplacian L_{kl}."""

    def test_zero_frequency_is_graph_laplacian(self):
        """At omega=0, twisted Laplacian must equal standard graph Laplacian."""
        zeros = np.array([0.0, 1.0, 2.5, 4.0])
        epsilon = 2.0
        diag = Phase2aAbelian._build_twisted_laplacian(zeros, 0.0, epsilon)

        # Standard graph Laplacian
        N = len(zeros)
        L_graph = np.zeros((N, N), dtype=np.complex128)
        for i in range(N):
            for j in range(i + 1, N):
                if abs(zeros[j] - zeros[i]) <= epsilon:
                    L_graph[i, i] += 1
                    L_graph[j, j] += 1
                    L_graph[i, j] -= 1
                    L_graph[j, i] -= 1

        assert_allclose(diag, L_graph, atol=1e-14)

    def test_twisted_laplacian_is_hermitian(self):
        """Twisted Laplacian must be Hermitian for any omega."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5])
        L = Phase2aAbelian._build_twisted_laplacian(zeros, omega=1.23, epsilon=1.0)
        assert_allclose(L, L.conj().T, atol=1e-14)

    def test_twisted_laplacian_psd(self):
        """Twisted Laplacian must be positive semidefinite."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        L = Phase2aAbelian._build_twisted_laplacian(zeros, omega=0.7, epsilon=1.5)
        eigs = np.linalg.eigvalsh(L)
        assert np.all(eigs >= -1e-12)


    def test_fast_matches_static(self):
        """Vectorized _build_twisted_laplacian_fast must match static version."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        diag = Phase2aAbelian(builder, zeros)

        for omega in [0.0, 1.23, -0.7, 3.14]:
            L_static = Phase2aAbelian._build_twisted_laplacian(zeros, omega, epsilon=1.5)
            L_fast = diag._build_twisted_laplacian_fast(omega, epsilon=1.5)
            assert_allclose(L_fast, L_static, atol=1e-14,
                            err_msg=f"Mismatch at omega={omega}")


class TestResonanceMatrix:
    """Test the full resonance matrix computation."""

    def test_resonance_matrix_shape(self):
        """Resonance matrix must be K×K."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        diag = Phase2aAbelian(builder, zeros)
        eps_grid = np.linspace(0, 4.0, 10)
        R = diag.compute_resonance_matrix(eps_grid)
        assert R.shape == (3, 3)

    def test_diagonal_entries_match_scalar_betti(self):
        """Diagonal R_{kk} must reproduce scalar Betti (connected components).

        At omega=0, the twisted Laplacian IS the standard graph Laplacian.
        The max kernel dim over epsilon is N (at eps=0 where there are no edges).
        All K diagonal entries must agree (they all use the same omega=0).
        """
        zeros = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        diag = Phase2aAbelian(builder, zeros)
        eps_grid = np.linspace(0, 7.0, 20)
        R = diag.compute_resonance_matrix(eps_grid)

        # All K diagonal entries must be identical (all use omega=0)
        for k in range(3):
            assert R[k, k] == R[0, 0]

        # Additionally verify: at eps=0, kernel = N (trivial full kernel)
        assert R[0, 0] == len(zeros)

        # Verify the zero-frequency Laplacian at a non-trivial epsilon
        # gives the correct connected component count.
        # zeros=[0,1,2.5,4,6], eps=1.5:
        #   edges with gap <= 1.5: (0,1) gap=1.0, (1,2) gap=1.5, (2,3) gap=1.5
        #   nodes 0,1,2,3 form one component; node 4 is isolated → 2 components
        L = Phase2aAbelian._build_twisted_laplacian(zeros, 0.0, epsilon=1.5)
        eigs = np.linalg.eigvalsh(L)
        n_kernel = int(np.sum(np.abs(eigs) < 1e-10))
        assert n_kernel == 2  # two connected components

    def test_resonance_matrix_symmetric(self):
        """R_{kl} = R_{lk} because omega_{kl} = -omega_{lk} and kernel dim is the same."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5])
        builder = TransportMapBuilder(K=4, sigma=0.5)
        diag = Phase2aAbelian(builder, zeros)
        eps_grid = np.linspace(0, 2.0, 10)
        R = diag.compute_resonance_matrix(eps_grid)
        assert_allclose(R, R.T)


class TestPhase2aRun:
    """Test the full experiment runner."""

    def test_run_returns_results(self):
        """Full run should return resonance matrix and distinct frequency count."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        diag = Phase2aAbelian(builder, zeros)
        eps_grid = np.linspace(0, 4.0, 10)
        results = diag.run(eps_grid)
        assert "resonance_matrix" in results
        assert "n_distinct_frequencies" in results
        assert "eigenvalues_A" in results
        assert results["resonance_matrix"].shape == (3, 3)
