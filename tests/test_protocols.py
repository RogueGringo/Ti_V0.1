"""Tests for ATFT protocol conformance."""
import numpy as np

from atft.core.protocols import (
    Cacheable,
    ConfigurationSource,
    FeatureMap,
    TopologicalOperator,
)
from atft.core.types import (
    PersistenceDiagram,
    PointCloud,
    PointCloudBatch,
)


class TestProtocolConformance:
    """Verify that concrete implementations satisfy protocols at runtime."""

    def test_dummy_source_satisfies_protocol(self):
        class DummySource:
            def generate(self, n_points, **kwargs):
                pts = np.zeros((n_points, 1), dtype=np.float64)
                return PointCloud(points=pts)

            def generate_batch(self, n_points, batch_size, **kwargs):
                clouds = [self.generate(n_points) for _ in range(batch_size)]
                return PointCloudBatch(clouds=clouds)

        assert isinstance(DummySource(), ConfigurationSource)

    def test_dummy_feature_map_satisfies_protocol(self):
        class DummyMap:
            def transform(self, cloud):
                return cloud

            def transform_batch(self, batch):
                return batch

        assert isinstance(DummyMap(), FeatureMap)

    def test_dummy_operator_satisfies_protocol(self):
        class DummyOp:
            def compute(self, cloud, max_degree=0, epsilon_max=None):
                return PersistenceDiagram(diagrams={})

            def compute_batch(self, batch, max_degree=0, epsilon_max=None):
                return [self.compute(c) for c in batch.clouds]

        assert isinstance(DummyOp(), TopologicalOperator)

    def test_non_conforming_class_fails(self):
        class NotASource:
            pass

        assert not isinstance(NotASource(), ConfigurationSource)
