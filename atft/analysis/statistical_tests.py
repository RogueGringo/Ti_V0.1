"""Statistical validation: Mahalanobis distance + functional envelope."""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import chi2

from atft.core.types import (
    CurveType,
    EvolutionCurveSet,
    ValidationResult,
    WaypointSignature,
)

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """Two-pronged validation: functional envelope + Mahalanobis matching."""

    def __init__(self, confidence_level: float = 0.99):
        self.confidence_level = confidence_level
        self._ensemble_signatures: list[WaypointSignature] | None = None
        self._ensemble_curves: list[EvolutionCurveSet] | None = None
        self._covariance_inv: np.ndarray | None = None
        self._mean_vector: np.ndarray | None = None

    def fit_ensemble(self, signatures: list[WaypointSignature], curves: list[EvolutionCurveSet]) -> None:
        self._ensemble_signatures = signatures
        self._ensemble_curves = curves

        vectors = np.array([s.as_vector() for s in signatures])
        self._mean_vector = np.mean(vectors, axis=0)

        cov = np.cov(vectors, rowvar=False)
        dim = cov.shape[0]
        reg = 1e-6 * np.trace(cov) / dim
        cov_reg = cov + reg * np.eye(dim)

        cond = np.linalg.cond(cov_reg)
        if cond > 1e10:
            logger.warning("Covariance condition number %.2e exceeds 1e10", cond)

        self._covariance_inv = np.linalg.inv(cov_reg)

    def validate(self, target_signature: WaypointSignature, target_curves: EvolutionCurveSet, degree: int = 0) -> ValidationResult:
        l2_betti, within_betti = self._check_envelope(target_curves, CurveType.BETTI, degree)
        l2_gini, within_gini = self._check_envelope(target_curves, CurveType.GINI, degree)
        within_band = within_betti and within_gini

        target_vec = target_signature.as_vector()
        delta = target_vec - self._mean_vector
        d_mahal = float(np.sqrt(delta @ self._covariance_inv @ delta))

        dof = len(target_vec)
        p_value = float(1.0 - chi2.cdf(d_mahal**2, df=dof))

        return ValidationResult(
            mahalanobis_distance=d_mahal,
            p_value=p_value,
            l2_distance_betti=l2_betti,
            l2_distance_gini=l2_gini,
            within_confidence_band=within_band,
            ensemble_size=len(self._ensemble_signatures),
            metadata={
                "confidence_level": self.confidence_level,
                "dof": dof,
                "within_betti_band": within_betti,
                "within_gini_band": within_gini,
            },
        )

    def _check_envelope(self, target_curves: EvolutionCurveSet, curve_type: CurveType, degree: int) -> tuple[float, bool]:
        lookup = {CurveType.BETTI: "betti", CurveType.GINI: "gini", CurveType.PERSISTENCE: "persistence"}
        attr = lookup[curve_type]
        target_curve = getattr(target_curves, attr)[degree]

        ensemble_vals = np.array([
            np.interp(target_curve.epsilon_grid, getattr(c, attr)[degree].epsilon_grid, getattr(c, attr)[degree].values)
            for c in self._ensemble_curves
        ])

        mean_curve = np.mean(ensemble_vals, axis=0)

        alpha = 1.0 - self.confidence_level
        lower = np.percentile(ensemble_vals, 100 * alpha / 2, axis=0)
        upper = np.percentile(ensemble_vals, 100 * (1 - alpha / 2), axis=0)

        d_eps = target_curve.epsilon_grid[1] - target_curve.epsilon_grid[0]
        l2 = float(np.sqrt(np.sum((target_curve.values - mean_curve) ** 2) * d_eps))

        within = bool(np.all((target_curve.values >= lower) & (target_curve.values <= upper)))

        return l2, within
