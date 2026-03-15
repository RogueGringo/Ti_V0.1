"""Tests for Phase 2 sheaf types."""
from __future__ import annotations

import numpy as np
import pytest

from atft.core.types import SheafBettiCurve, SheafValidationResult


class TestSheafBettiCurve:
    def test_creation(self):
        curve = SheafBettiCurve(
            epsilon_grid=np.linspace(0, 3.0, 10),
            kernel_dimensions=np.array([400, 350, 200, 100, 50, 20, 10, 5, 2, 1]),
            smallest_eigenvalues=np.random.rand(10, 5),
            sigma=0.5,
            K=20,
        )
        assert curve.K == 20
        assert curve.sigma == 0.5
        assert len(curve.epsilon_grid) == 10
        assert len(curve.kernel_dimensions) == 10

    def test_frozen(self):
        curve = SheafBettiCurve(
            epsilon_grid=np.array([0.0, 1.0]),
            kernel_dimensions=np.array([10, 5]),
            smallest_eigenvalues=np.zeros((2, 3)),
            sigma=0.5,
            K=5,
        )
        with pytest.raises(AttributeError):
            curve.sigma = 0.7


class TestSheafValidationResult:
    def test_creation(self):
        result = SheafValidationResult(
            sigma_grid=np.array([0.3, 0.5, 0.7]),
            epsilon_grid=np.linspace(0, 3.0, 10),
            betti_heatmap=np.zeros((3, 10), dtype=np.int64),
            peak_sigma=0.5,
            peak_kernel_dim=42,
            is_unique_peak=True,
        )
        assert result.peak_sigma == 0.5
        assert result.is_unique_peak is True
        assert result.betti_heatmap.shape == (3, 10)

    def test_metadata_default(self):
        result = SheafValidationResult(
            sigma_grid=np.array([0.5]),
            epsilon_grid=np.array([0.0]),
            betti_heatmap=np.zeros((1, 1), dtype=np.int64),
            peak_sigma=0.5,
            peak_kernel_dim=0,
            is_unique_peak=True,
        )
        assert result.metadata == {}

    def test_frozen(self):
        result = SheafValidationResult(
            sigma_grid=np.array([0.5]),
            epsilon_grid=np.array([0.0]),
            betti_heatmap=np.zeros((1, 1), dtype=np.int64),
            peak_sigma=0.5,
            peak_kernel_dim=0,
            is_unique_peak=True,
        )
        with pytest.raises(AttributeError):
            result.peak_sigma = 0.7
