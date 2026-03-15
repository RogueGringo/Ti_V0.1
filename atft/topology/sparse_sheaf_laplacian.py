"""Sparse sheaf Laplacian with C^K vector fibers.

Implements the Phase 3 vector-valued sheaf Laplacian using BSR sparse
matrices and shift-invert eigsh. Designed for N=10,000 scale with
K=50 fiber dimension.

Key differences from SheafLaplacian (Phase 2):
  - Vector fibers C^K instead of matrix fibers C^{K x K}
  - Explicit sparse matrix (BSR) instead of matrix-free LinearOperator
  - Coboundary: (delta s)_e = U_e s_i - s_j (left multiply, not conjugation)
  - Supports "superposition" transport mode (multi-prime phase interference)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh

from atft.topology.transport_maps import TransportMapBuilder


class SparseSheafLaplacian:
    """BSR sparse sheaf Laplacian with C^K vector fibers.

    The sheaf Laplacian L = delta_0^dagger delta_0 where
    (delta_0 s)_e = U_e s_i - s_j for oriented edge e = (i -> j).

    Block structure for each edge (i -> j) with transport U:
      L[i,i] += U^dagger U    (diagonal)
      L[j,j] += I_K           (diagonal)
      L[i,j] = -U^dagger      (off-diagonal)
      L[j,i] = -U             (off-diagonal)

    Args:
        builder: TransportMapBuilder providing K, sigma, transport methods.
        zeros: 1D array of (possibly unsorted) unfolded zeta zeros.
        transport_mode: "superposition" (default), "fe", or "resonant".
        normalize: Frobenius-normalize superposition generators (only for
            transport_mode="superposition").
    """

    def __init__(
        self,
        builder: TransportMapBuilder,
        zeros: NDArray[np.float64],
        transport_mode: str = "superposition",
        normalize: bool = True,
    ) -> None:
        self._builder = builder
        self._zeros = np.sort(zeros.ravel())
        self._N = len(self._zeros)
        self._K = builder.K
        self._transport_mode = transport_mode
        self._normalize = normalize

    @property
    def N(self) -> int:
        return self._N

    @property
    def K(self) -> int:
        return self._K

    def build_edge_list(
        self, epsilon: float
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
        """Discover all edges in the 1D Rips complex at scale epsilon.

        Uses binary search on sorted zeros for O(N log N + |E|) complexity.

        Returns:
            (i_idx, j_idx, gaps) where each is a 1D array of length |E|.
            All edges satisfy i < j and gaps[e] = zeros[j] - zeros[i] <= epsilon.
        """
        zeros = self._zeros
        N = self._N

        if epsilon <= 0 or N < 2:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )

        i_parts: list[NDArray[np.int64]] = []
        j_parts: list[NDArray[np.int64]] = []

        for i in range(N - 1):
            max_val = zeros[i] + epsilon
            # Binary search for rightmost j where zeros[j] <= max_val
            j_right = int(np.searchsorted(zeros, max_val, side='right'))
            j_right = min(j_right, N)
            if i + 1 < j_right:
                js = np.arange(i + 1, j_right, dtype=np.int64)
                i_parts.append(np.full(len(js), i, dtype=np.int64))
                j_parts.append(js)

        if i_parts:
            i_idx = np.concatenate(i_parts)
            j_idx = np.concatenate(j_parts)
        else:
            i_idx = np.array([], dtype=np.int64)
            j_idx = np.array([], dtype=np.int64)

        gaps = (
            self._zeros[j_idx] - self._zeros[i_idx]
            if len(i_idx) > 0
            else np.array([], dtype=np.float64)
        )
        return i_idx, j_idx, gaps
