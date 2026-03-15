"""Protocol interfaces for the ATFT pipeline.

All pipeline stages are defined as Protocols (structural subtyping).
Concrete implementations need only implement the methods -- no
inheritance required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from atft.core.types import (
    PersistenceDiagram,
    PointCloud,
    PointCloudBatch,
)


@runtime_checkable
class ConfigurationSource(Protocol):
    """Produces point clouds from a physical or mathematical source."""

    def generate(self, n_points: int, **kwargs) -> PointCloud: ...

    def generate_batch(
        self, n_points: int, batch_size: int, **kwargs
    ) -> PointCloudBatch: ...


@runtime_checkable
class FeatureMap(Protocol):
    """Transforms a point cloud into the metric space for PH computation."""

    def transform(self, cloud: PointCloud) -> PointCloud: ...

    def transform_batch(self, batch: PointCloudBatch) -> PointCloudBatch: ...


@runtime_checkable
class TopologicalOperator(Protocol):
    """The Adaptive Topological Operator."""

    def compute(
        self,
        cloud: PointCloud,
        max_degree: int = 0,
        epsilon_max: float | None = None,
    ) -> PersistenceDiagram: ...

    def compute_batch(
        self,
        batch: PointCloudBatch,
        max_degree: int = 0,
        epsilon_max: float | None = None,
    ) -> list[PersistenceDiagram]: ...


@runtime_checkable
class Cacheable(Protocol):
    """Serialization protocol for intermediate results."""

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> Cacheable: ...
