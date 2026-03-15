"""Tests for statistical validation (Mahalanobis + envelope)."""
import numpy as np
import pytest

from atft.analysis.statistical_tests import StatisticalValidator
from atft.core.types import (
    CurveType,
    EvolutionCurve,
    EvolutionCurveSet,
    WaypointSignature,
)


def _make_signature(onset, w1, w2, d1, d2, g, dg):
    return WaypointSignature(
        onset_scale=onset,
        waypoint_scales=np.array([w1, w2]),
        topo_derivatives=np.array([d1, d2]),
        gini_at_onset=g,
        gini_derivative_at_onset=dg,
    )


def _make_curves(eps, betti_vals, gini_vals):
    eps_arr = np.array(eps, dtype=np.float64)
    return EvolutionCurveSet(
        betti={0: EvolutionCurve(eps_arr, np.array(betti_vals, dtype=np.float64), CurveType.BETTI, 0)},
        gini={0: EvolutionCurve(eps_arr, np.array(gini_vals, dtype=np.float64), CurveType.GINI, 0)},
        persistence={0: EvolutionCurve(eps_arr, np.zeros(len(eps)), CurveType.PERSISTENCE, 0)},
    )


class TestStatisticalValidator:
    def test_identical_distribution_small_distance(self):
        rng = np.random.default_rng(42)
        sigs = [_make_signature(
            rng.normal(1, 0.1), rng.normal(3, 0.2), rng.normal(5, 0.3),
            rng.normal(-3, 0.1), rng.normal(-5, 0.1), rng.normal(0.4, 0.05),
            rng.normal(0.01, 0.005),
        ) for _ in range(100)]

        eps = np.linspace(0, 6, 20)
        curves = [_make_curves(eps, np.linspace(10, 1, 20), np.linspace(0, 0.5, 20))
                  for _ in range(100)]

        validator = StatisticalValidator()
        validator.fit_ensemble(sigs, curves)

        target_sig = _make_signature(1.0, 3.0, 5.0, -3.0, -5.0, 0.4, 0.01)
        target_curves = _make_curves(eps, np.linspace(10, 1, 20), np.linspace(0, 0.5, 20))
        result = validator.validate(target_sig, target_curves)
        assert result.p_value > 0.01

    def test_outlier_large_distance(self):
        rng = np.random.default_rng(42)
        sigs = [_make_signature(
            rng.normal(1, 0.1), rng.normal(3, 0.1), rng.normal(5, 0.1),
            rng.normal(-3, 0.1), rng.normal(-5, 0.1), rng.normal(0.4, 0.05),
            rng.normal(0.01, 0.005),
        ) for _ in range(100)]

        eps = np.linspace(0, 6, 20)
        curves = [_make_curves(eps, np.linspace(10, 1, 20), np.linspace(0, 0.5, 20))
                  for _ in range(100)]

        validator = StatisticalValidator()
        validator.fit_ensemble(sigs, curves)

        target_sig = _make_signature(5.0, 10.0, 20.0, -10.0, -15.0, 0.9, 0.5)
        target_curves = _make_curves(eps, np.linspace(10, 1, 20), np.linspace(0, 0.5, 20))
        result = validator.validate(target_sig, target_curves)
        assert result.p_value < 0.001
        assert result.mahalanobis_distance > 5.0

    def test_within_band_true(self):
        eps = np.linspace(0, 5, 10)
        rng = np.random.default_rng(42)
        mean_betti = np.linspace(10, 1, 10)
        mean_gini = np.linspace(0, 0.5, 10)

        sigs = [_make_signature(1, 3, 5, -3, -5, 0.4, 0.01) for _ in range(50)]
        curves = [
            _make_curves(eps, mean_betti + rng.normal(0, 0.5, 10),
                         np.clip(mean_gini + rng.normal(0, 0.05, 10), 0, 1))
            for _ in range(50)
        ]

        validator = StatisticalValidator(confidence_level=0.99)
        validator.fit_ensemble(sigs, curves)

        # Validate ensemble mean curve (should be inside band)
        target_curves = _make_curves(eps, mean_betti, mean_gini)
        result = validator.validate(sigs[0], target_curves)
        assert result.within_confidence_band is True

    def test_ensemble_size_recorded(self):
        sigs = [_make_signature(1, 3, 5, -3, -5, 0.4, 0.01) for _ in range(25)]
        eps = np.linspace(0, 5, 10)
        curves = [_make_curves(eps, np.ones(10), np.zeros(10)) for _ in range(25)]
        validator = StatisticalValidator()
        validator.fit_ensemble(sigs, curves)
        result = validator.validate(sigs[0], curves[0])
        assert result.ensemble_size == 25
