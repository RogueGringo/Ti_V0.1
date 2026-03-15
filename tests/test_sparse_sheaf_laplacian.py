"""Tests for SparseSheafLaplacian (vector fibers, BSR sparse)."""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


class TestEdgeDiscovery:
    """Tests for build_edge_list()."""

    def test_shape(self):
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(1.5)
        assert i_idx.shape == j_idx.shape == gaps.shape
        assert i_idx.ndim == 1

    def test_edges_within_epsilon(self):
        """All discovered gaps should be <= epsilon."""
        zeros = np.array([0.0, 0.8, 1.5, 3.0, 3.2, 5.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(1.0)
        assert np.all(gaps <= 1.0 + 1e-12)
        assert np.all(gaps > 0)

    def test_edges_complete(self):
        """Should find ALL pairs within epsilon."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(0.5)
        # Pairs within 0.5: (0,1), (1,2), (2,3), (3,4)
        assert len(i_idx) == 4

    def test_oriented_i_less_j(self):
        """All edges should be oriented i < j."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, _ = lap.build_edge_list(2.5)
        assert np.all(i_idx < j_idx)

    def test_eps_zero_no_edges(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(0.0)
        assert len(i_idx) == 0

    def test_large_eps_complete_graph(self):
        """Large epsilon should connect all pairs."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, _ = lap.build_edge_list(100.0)
        # Complete graph on 4 vertices: 4*3/2 = 6 edges
        assert len(i_idx) == 6

    def test_unsorted_input_gets_sorted(self):
        """Input zeros in any order should produce correct edges."""
        zeros_unsorted = np.array([3.0, 1.0, 0.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros_unsorted)
        i_idx, j_idx, gaps = lap.build_edge_list(1.5)
        # After sorting: [0, 1, 2, 3]. Edges within 1.5: (0,1), (1,2), (2,3)
        assert len(i_idx) == 3
        assert np.all(gaps <= 1.5)
