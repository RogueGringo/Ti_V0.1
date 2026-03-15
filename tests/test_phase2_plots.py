"""Smoke tests for Phase 2 visualization functions."""
from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atft.core.types import SheafBettiCurve, SheafValidationResult
from atft.visualization.plots import (
    plot_sheaf_betti_curves,
    plot_sigma_peak,
    plot_resonance_matrix,
)


def _make_dummy_curve(sigma: float = 0.5) -> SheafBettiCurve:
    n = 20
    return SheafBettiCurve(
        epsilon_grid=np.linspace(0, 3.0, n),
        kernel_dimensions=np.maximum(0, np.arange(n, 0, -1)),
        smallest_eigenvalues=np.random.rand(n, 5),
        sigma=sigma,
        K=5,
    )


def _make_dummy_result() -> SheafValidationResult:
    sigma_grid = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    eps_grid = np.linspace(0, 3.0, 10)
    return SheafValidationResult(
        sigma_grid=sigma_grid,
        epsilon_grid=eps_grid,
        betti_heatmap=np.random.randint(0, 50, size=(5, 10)),
        peak_sigma=0.5,
        peak_kernel_dim=42,
        is_unique_peak=True,
    )


class TestPlotSheafBettiCurves:
    def test_returns_figure(self):
        curves = [_make_dummy_curve(s) for s in [0.3, 0.5, 0.7]]
        fig = plot_sheaf_betti_curves(curves, highlight_sigma=0.5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        curves = [_make_dummy_curve(0.5)]
        path = tmp_path / "test_sheaf.png"
        fig = plot_sheaf_betti_curves(curves, save_path=path)
        assert path.exists()
        plt.close(fig)


class TestPlotSigmaPeak:
    def test_returns_figure(self):
        result = _make_dummy_result()
        fig = plot_sigma_peak(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotResonanceMatrix:
    def test_returns_figure(self):
        R = np.random.randint(0, 10, size=(5, 5))
        eigenvalues = np.linspace(-1, 1, 5)
        fig = plot_resonance_matrix(R, eigenvalues)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
