"""Gaussian Unitary Ensemble source via Dumitriu-Edelman tridiagonal model.

The beta=2 Hermite ensemble (GUE) eigenvalue distribution is exactly
reproduced by a real symmetric tridiagonal matrix:
  - Diagonal:     a_i ~ N(0, 1)
  - Sub-diagonal: b_i = (1/sqrt(2)) * chi_{2(N-i)},  i = 1, ..., N-1

This reduces GUE generation from O(N^3) complex Hermitian eigendecomposition
to O(N^2) real tridiagonal eigendecomposition, with O(N) memory for
construction instead of O(N^2).

Reference: Dumitriu & Edelman, "Matrix Models for Beta Ensembles" (2002).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

from atft.core.types import PointCloud, PointCloudBatch


class GUESource:
    """Generates eigenvalues from the GUE (beta=2 Hermite ensemble).

    Uses the Dumitriu-Edelman tridiagonal model:
      - O(N) memory to construct (two vectors of length N)
      - O(N^2) eigendecomposition via scipy.linalg.eigvalsh_tridiagonal
      - No complex arithmetic required

    Eigenvalues are normalized to semicircle support [-1, 1] via
    division by sqrt(2N), matching H = (A + A†) / (2*sqrt(N)).
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = np.random.default_rng(np.random.SeedSequence(seed))

    def _generate_single(self, n: int) -> np.ndarray:
        # Diagonal: N(0, 1)
        diagonal = self._rng.standard_normal(n)

        # Sub-diagonal: (1/sqrt(2)) * chi_{2k} for k = N-1, N-2, ..., 1
        # chi_{2k} = sqrt(chi^2_{2k}), and chi^2_{2k} ~ Gamma(k, 2)
        # so chi_{2k} = sqrt(Gamma(k, 2))
        dof = 2.0 * np.arange(n - 1, 0, -1, dtype=np.float64)
        sub_diagonal = np.sqrt(self._rng.chisquare(dof)) / np.sqrt(2.0)

        # Eigenvalues of the tridiagonal matrix
        eigenvalues = eigvalsh_tridiagonal(diagonal, sub_diagonal)

        # Normalize to semicircle [-1, 1]
        eigenvalues /= np.sqrt(2.0 * n)

        return eigenvalues.astype(np.float64)

    def generate(self, n_points: int, **kwargs) -> PointCloud:
        eigenvalues = self._generate_single(n_points)
        return PointCloud(
            points=eigenvalues.reshape(-1, 1),
            metadata={"source": "gue", "n_points": n_points, "seed": self._seed},
        )

    def generate_batch(self, n_points: int, batch_size: int, **kwargs) -> PointCloudBatch:
        clouds = [self.generate(n_points) for _ in range(batch_size)]
        return PointCloudBatch(clouds=clouds)
