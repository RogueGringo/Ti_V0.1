"""Abstract base class for Phase 3 vector-fiber sheaf Laplacians.

Consolidates shared logic across the three Phase 3 backends:
  - SparseSheafLaplacian (SciPy BSR, CPU)
  - GPUSheafLaplacian (CuPy CSR, NVIDIA CUDA)
  - TorchSheafLaplacian (PyTorch CSR, NVIDIA CUDA + AMD ROCm)

Shared logic (lives here):
  - Edge discovery: 1D Vietoris-Rips via binary search on sorted zeros
  - Transport dispatch: mode -> builder batch method mapping
  - Eigenvalue post-processing: sort, clamp, pad
  - Spectral sum: sum of k smallest eigenvalues

Backend-specific (abstract, implemented by subclasses):
  - build_matrix: sparse matrix assembly in backend-specific format
  - smallest_eigenvalues: eigensolver strategy (shift-invert / spectral flip / Lanczos)

The Phase 2 dense backend (SheafLaplacian) is NOT part of this hierarchy —
it uses C^{KxK} matrix fibers rather than C^K vector fibers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from atft.topology.transport_maps import TransportMapBuilder


class BaseSheafLaplacian(ABC):
    """Abstract base class for Phase 3 C^K vector-fiber sheaf Laplacians.

    Subclasses must implement:
      - build_matrix(epsilon) -> backend-specific sparse matrix
      - smallest_eigenvalues(epsilon, k) -> numpy array of k eigenvalues

    Args:
        builder: TransportMapBuilder providing K, sigma, and batch
            transport methods for all supported modes.
        zeros: 1D array of (possibly unsorted) unfolded zeta zeros.
            Sorted internally — subclasses always see sorted zeros.
        transport_mode: "superposition" (default), "fe", or "resonant".
    """

    def __init__(
        self,
        builder: TransportMapBuilder,
        zeros: NDArray[np.float64] | np.ndarray,
        transport_mode: str = "superposition",
    ) -> None:
        self._builder = builder
        self._zeros = np.sort(np.asarray(zeros, dtype=np.float64).ravel())
        self._N = len(self._zeros)
        self._K = builder.K
        self._transport_mode = transport_mode

    @property
    def N(self) -> int:
        """Number of points (vertices in the Rips complex)."""
        return self._N

    @property
    def K(self) -> int:
        """Fiber dimension."""
        return self._K

    # ------------------------------------------------------------------
    # Edge discovery
    # ------------------------------------------------------------------

    def build_edge_list(
        self, epsilon: float,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
        """Discover all edges in the 1D Vietoris-Rips complex at scale epsilon.

        For sorted 1D points, an edge (i, j) exists when
        zeros[j] - zeros[i] <= epsilon (with i < j, gap > 0).

        Uses pairwise comparison for N <= 5000 (vectorized, fast for
        small N) and binary search for N > 5000 (O(N log N + |E|),
        memory-safe for large N).

        Returns:
            (i_idx, j_idx, gaps) — each a 1D array of length |E|.
            All edges satisfy i < j and 0 < gaps[e] <= epsilon.
        """
        zeros = self._zeros
        N = self._N

        if epsilon <= 0 or N < 2:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )

        if N <= 5000:
            # Pairwise: vectorized, O(N^2) memory — fine for moderate N
            diff = zeros[None, :] - zeros[:, None]
            mask = (diff > 0) & (diff <= epsilon)
            i_idx, j_idx = np.where(mask)
            i_idx = i_idx.astype(np.int64)
            j_idx = j_idx.astype(np.int64)
        else:
            # Binary search: O(N log N + |E|), memory-safe for large N
            i_parts: list[NDArray[np.int64]] = []
            j_parts: list[NDArray[np.int64]] = []
            for i in range(N - 1):
                j_right = int(np.searchsorted(
                    zeros, zeros[i] + epsilon, side="right",
                ))
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
            zeros[j_idx] - zeros[i_idx]
            if len(i_idx) > 0
            else np.array([], dtype=np.float64)
        )
        return i_idx, j_idx, gaps

    # ------------------------------------------------------------------
    # Transport dispatch
    # ------------------------------------------------------------------

    def _compute_transport(
        self,
        gaps: NDArray[np.float64],
        normalize: bool = True,
    ) -> NDArray[np.complex128]:
        """Compute transport matrices for all edges via the builder.

        Dispatches to the appropriate builder batch method based on
        the transport_mode set at construction time.

        Args:
            gaps: 1D array of edge gap values (gamma_j - gamma_i).
            normalize: Frobenius-normalize superposition generators.
                Only applies to transport_mode="superposition".

        Returns:
            (M, K, K) complex128 array of transport matrices.

        Raises:
            ValueError: If transport_mode is not recognized.
        """
        if self._transport_mode == "superposition":
            return self._builder.batch_transport_superposition(
                gaps, normalize=normalize,
            )
        elif self._transport_mode == "fe":
            return self._builder.batch_transport_fe(gaps)
        elif self._transport_mode == "resonant":
            return self._builder.batch_transport_resonant(gaps)
        else:
            raise ValueError(
                f"Unknown transport_mode {self._transport_mode!r}. "
                "Must be 'superposition', 'fe', or 'resonant'."
            )

    # ------------------------------------------------------------------
    # Eigenvalue post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess_eigenvalues(
        eigs: np.ndarray, k: int,
    ) -> NDArray[np.float64]:
        """Sort, clamp negative values, and pad/truncate to length k.

        All backends apply this same post-processing to raw eigenvalues.

        Args:
            eigs: Raw eigenvalue array (may be complex, unsorted).
            k: Desired output length.

        Returns:
            Sorted 1D float64 array of length k. Tiny negatives from
            numerical noise are clamped to 0.
        """
        eigs = np.sort(np.asarray(eigs).real)
        eigs = np.maximum(eigs, 0.0)
        if len(eigs) >= k:
            return eigs[:k]
        return np.concatenate([eigs, np.zeros(k - len(eigs))])

    # ------------------------------------------------------------------
    # Abstract methods (backend-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def build_matrix(self, epsilon: float):
        """Assemble the N*K x N*K sheaf Laplacian in backend-specific format.

        Block structure per edge (i -> j) with transport U:
          L[i,i] += U^dagger U    (diagonal K x K block)
          L[j,j] += I_K           (diagonal K x K block)
          L[i,j]  = -U^dagger     (off-diagonal K x K block)
          L[j,i]  = -U            (off-diagonal K x K block)
        """
        ...

    @abstractmethod
    def smallest_eigenvalues(
        self, epsilon: float, k: int = 100,
    ) -> NDArray[np.float64]:
        """Compute the k smallest eigenvalues of the sheaf Laplacian.

        Returns:
            Sorted 1D float64 array of length k.
        """
        ...

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def spectral_sum(self, epsilon: float, k: int = 100) -> float:
        """Sum of the k smallest eigenvalues (primary metric).

        This is the total "constraint energy" in the near-kernel of the
        sheaf Laplacian. Lower values indicate more globally consistent
        sections (stronger topological signal).
        """
        return float(np.sum(self.smallest_eigenvalues(epsilon, k)))
