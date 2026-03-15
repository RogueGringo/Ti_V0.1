"""Analytical H_0 persistent homology for 1D point clouds."""
from __future__ import annotations

import numpy as np

from atft.core.types import PersistenceDiagram, PointCloud, PointCloudBatch


class AnalyticalH0:
    """Exact H_0 persistence for 1D point clouds via sorted gaps.

    Exploits the fact that for points on a line, H_0 persistence
    is completely determined by the sorted gap sequence.
    Uses the diameter convention: death at g_i (not g_i/2).
    Complexity: O(N log N).
    """

    def compute(self, cloud: PointCloud, max_degree: int = 0, epsilon_max: float | None = None) -> PersistenceDiagram:
        if cloud.dimension != 1:
            raise ValueError(f"AnalyticalH0 only supports 1D point clouds, got {cloud.dimension}D")
        if max_degree > 0:
            raise ValueError("AnalyticalH0 only computes H_0. Use RipserPH for higher degrees.")

        sorted_pts = np.sort(cloud.points[:, 0])
        gaps = np.diff(sorted_pts)

        births = np.zeros(len(gaps), dtype=np.float64)
        deaths = gaps.astype(np.float64)

        if epsilon_max is not None:
            mask = deaths <= epsilon_max
            births = births[mask]
            deaths = deaths[mask]

        # Add immortal feature
        births = np.append(births, 0.0)
        deaths = np.append(deaths, np.inf)

        diagram = np.column_stack([births, deaths])
        return PersistenceDiagram(
            diagrams={0: diagram},
            metadata={"method": "analytical_h0", "n_points": cloud.n_points},
        )

    def compute_batch(self, batch: PointCloudBatch, max_degree: int = 0, epsilon_max: float | None = None) -> list[PersistenceDiagram]:
        return [self.compute(c, max_degree, epsilon_max) for c in batch.clouds]
