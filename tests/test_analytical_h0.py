"""Tests for analytical H_0 persistent homology."""
import numpy as np
import pytest

from atft.core.protocols import TopologicalOperator
from atft.core.types import PointCloud, PointCloudBatch
from atft.topology.analytical_h0 import AnalyticalH0


class TestAnalyticalH0:
    def test_satisfies_protocol(self):
        assert isinstance(AnalyticalH0(), TopologicalOperator)

    def test_known_gaps(self, simple_1d_points):
        """Points [1,2,4,7,11] have gaps [1,2,3,4]."""
        cloud = PointCloud(points=simple_1d_points)
        pd = AnalyticalH0().compute(cloud)
        pairs = pd.degree(0)
        assert pairs.shape[0] == 5  # 4 finite + 1 immortal
        np.testing.assert_array_equal(pairs[:, 0], 0.0)
        finite_deaths = pairs[pairs[:, 1] != np.inf, 1]
        np.testing.assert_array_almost_equal(np.sort(finite_deaths), [1.0, 2.0, 3.0, 4.0])
        assert np.sum(pairs[:, 1] == np.inf) == 1

    def test_uniform_gaps(self, uniform_1d_points):
        cloud = PointCloud(points=uniform_1d_points)
        pd = AnalyticalH0().compute(cloud)
        finite = pd.degree(0)
        finite = finite[finite[:, 1] != np.inf]
        assert finite.shape[0] == 9
        np.testing.assert_array_almost_equal(finite[:, 1], 1.0)

    def test_single_point(self):
        cloud = PointCloud(points=np.array([[5.0]]))
        pd = AnalyticalH0().compute(cloud)
        pairs = pd.degree(0)
        assert pairs.shape[0] == 1
        assert pairs[0, 1] == np.inf

    def test_two_points(self):
        cloud = PointCloud(points=np.array([[1.0], [4.0]]))
        pd = AnalyticalH0().compute(cloud)
        pairs = pd.degree(0)
        assert pairs.shape[0] == 2
        finite = pairs[pairs[:, 1] != np.inf]
        assert finite[0, 1] == 3.0

    def test_rejects_2d_input(self):
        cloud = PointCloud(points=np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError, match="1D"):
            AnalyticalH0().compute(cloud)

    def test_rejects_higher_degree(self):
        cloud = PointCloud(points=np.array([[1.0], [2.0]]))
        with pytest.raises(ValueError, match="H_0"):
            AnalyticalH0().compute(cloud, max_degree=1)

    def test_epsilon_max(self, simple_1d_points):
        cloud = PointCloud(points=simple_1d_points)
        pd = AnalyticalH0().compute(cloud, epsilon_max=2.5)
        finite = pd.degree(0)
        finite = finite[finite[:, 1] != np.inf]
        assert np.all(finite[:, 1] <= 2.5)
        assert finite.shape[0] == 2  # gaps 1.0 and 2.0

    def test_compute_batch(self):
        c1 = PointCloud(points=np.array([[1.0], [3.0], [6.0]]))
        c2 = PointCloud(points=np.array([[0.0], [5.0]]))
        batch = PointCloudBatch(clouds=[c1, c2])
        results = AnalyticalH0().compute_batch(batch)
        assert len(results) == 2
        assert results[0].degree(0).shape[0] == 3
        assert results[1].degree(0).shape[0] == 2

    def test_unsorted_input_handled(self):
        cloud = PointCloud(points=np.array([[7.0], [1.0], [4.0], [2.0]]))
        pd = AnalyticalH0().compute(cloud)
        finite = pd.degree(0)
        finite = finite[finite[:, 1] != np.inf]
        gaps = np.sort(finite[:, 1])
        np.testing.assert_array_almost_equal(gaps, [1.0, 2.0, 3.0])

    def test_metadata(self, simple_1d_points):
        cloud = PointCloud(points=simple_1d_points)
        pd = AnalyticalH0().compute(cloud)
        assert pd.metadata["method"] == "analytical_h0"
