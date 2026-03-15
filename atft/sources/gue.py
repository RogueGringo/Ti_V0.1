"""Gaussian Unitary Ensemble source."""
from __future__ import annotations

import numpy as np

from atft.core.types import PointCloud, PointCloudBatch


class GUESource:
    """Generates eigenvalues of GUE random Hermitian matrices.

    Entry distribution: off-diagonal A_{ij} = (X + iY) / sqrt(2),
    X, Y ~ N(0,1). Diagonal A_{ii} ~ N(0,1) (real).
    H = (A + A^dagger) / (2 * sqrt(N)), semicircle support [-1, 1].
    """

    def __init__(self, seed: int = 42, use_torch: bool = False, device: str = "cpu"):
        self._seed = seed
        self._rng = np.random.default_rng(np.random.SeedSequence(seed))
        self._use_torch = use_torch
        self._device = device

    def _generate_single(self, n: int) -> np.ndarray:
        real_part = self._rng.standard_normal((n, n))
        imag_part = self._rng.standard_normal((n, n))
        A = (real_part + 1j * imag_part) / np.sqrt(2)
        np.fill_diagonal(A, self._rng.standard_normal(n))
        H = (A + A.conj().T) / (2 * np.sqrt(n))
        eigenvalues = np.linalg.eigvalsh(H)
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
