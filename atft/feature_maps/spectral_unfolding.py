"""Spectral unfolding feature map."""
from __future__ import annotations

import numpy as np

from atft.core.types import PointCloud, PointCloudBatch


class SpectralUnfolding:
    """Normalizes spectra to mean gap = 1.

    Methods:
      - "rank": Rank-based unfolding via empirical CDF (for GUE).
      - "zeta": Analytic smooth staircase for Riemann zeta zeros.
    """

    def __init__(self, method: str = "semicircle"):
        if method not in ("semicircle", "rank", "zeta"):
            raise ValueError(f"Unknown method: {method!r}. Use 'semicircle', 'rank', or 'zeta'.")
        self._method = method

    def transform(self, cloud: PointCloud) -> PointCloud:
        pts = cloud.points[:, 0].copy()

        if self._method == "semicircle":
            unfolded = self._semicircle_unfold(pts)
        elif self._method == "rank":
            unfolded = self._rank_unfold(pts)
        elif self._method == "zeta":
            unfolded = self._zeta_unfold(pts)

        return PointCloud(
            points=unfolded.reshape(-1, 1),
            metadata={**cloud.metadata, "unfolding": self._method},
        )

    def transform_batch(self, batch: PointCloudBatch) -> PointCloudBatch:
        return PointCloudBatch(
            clouds=[self.transform(c) for c in batch.clouds]
        )

    @staticmethod
    def _semicircle_unfold(pts: np.ndarray) -> np.ndarray:
        """Unfold GUE eigenvalues via the Wigner semicircle CDF.

        For semicircle on [-a, a]:
          F(x) = 1/2 + (x/a * sqrt(1-(x/a)^2) + arcsin(x/a)) / pi
          N_smooth(lambda_i) = N * F(lambda_i)

        Uses the empirical spectral edge (plus small pad) as the support
        boundary to handle finite-N Tracy-Widom fluctuations.
        """
        n = len(pts)
        x = np.sort(pts)
        # Rescale support to empirical edge to avoid clipping
        x_max = np.max(np.abs(x))
        support = max(x_max * 1.001, 1.0)
        x_scaled = x / support
        cdf = 0.5 + (x_scaled * np.sqrt(1.0 - x_scaled**2) + np.arcsin(x_scaled)) / np.pi
        return n * cdf

    @staticmethod
    def _rank_unfold(pts: np.ndarray) -> np.ndarray:
        """Rank-based unfolding: positions become ranks 0 to N-1 (mean gap = 1)."""
        n = len(pts)
        sorted_idx = np.argsort(pts)
        ranks = np.empty(n, dtype=np.float64)
        ranks[sorted_idx] = np.arange(n, dtype=np.float64)
        return ranks

    @staticmethod
    def _zeta_unfold(gamma: np.ndarray) -> np.ndarray:
        """Unfold zeta zeros via the smooth staircase function.

        N_smooth(T) = (T/(2*pi)) * ln(T/(2*pi*e)) + 7/8
        """
        two_pi = 2.0 * np.pi
        n_smooth = (gamma / two_pi) * np.log(gamma / (two_pi * np.e)) + 7.0 / 8.0
        return n_smooth
