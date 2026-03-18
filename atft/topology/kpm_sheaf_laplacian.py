"""KPM-based Sheaf Laplacian — density of states via Chebyshev moments.

Implements the Kernel Polynomial Method (KPM) with Jackson damping to
reconstruct the continuous density of states rho(lambda) from raw
Chebyshev moments computed on GPU via Stochastic Lanczos Quadrature.

Reference: Weisse et al., Rev. Mod. Phys. 78 (2006), Eq. 71.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian

logger = logging.getLogger(__name__)


class KPMSheafLaplacian(TorchSheafLaplacian):
    """KPM-based sheaf Laplacian — density of states via Chebyshev moments."""

    def __init__(self, builder, zeros, transport_mode="superposition",
                 device=None, num_vectors=30, degree=300):
        super().__init__(builder, zeros, transport_mode, device)
        self._kpm_num_vectors = num_vectors
        self._kpm_degree = degree
        self._moments = None
        self._lam_max = None
        self._dim = None

    @staticmethod
    def _jackson_coefficients(D: int) -> NDArray[np.float64]:
        """Jackson kernel damping factors g_n for n=0..D.
        Convention: D = polynomial degree, N_Jackson = D+1.
        Follows Weisse et al., Rev. Mod. Phys. 78 (2006), Eq. 71.
        g[0] = 1.0 exactly.
        """
        n = np.arange(D + 1, dtype=np.float64)
        Dp1 = D + 1
        g = ((Dp1 - n) * np.cos(np.pi * n / Dp1)
             + np.sin(np.pi * n / Dp1) / np.tan(np.pi / Dp1)) / Dp1
        return g

    def compute_moments(self, epsilon: float) -> NDArray[np.float64]:
        """Compute raw Chebyshev moments on GPU via Hutchinson trace estimation."""
        L_csr = self.build_matrix(epsilon)
        dim = L_csr.shape[0]
        device = self.device
        D = self._kpm_degree
        num_vectors = self._kpm_num_vectors

        if dim == 0 or L_csr._nnz() == 0:
            self._moments = np.zeros(D + 1, dtype=np.float64)
            self._lam_max = 0.0
            self._dim = dim
            return self._moments

        lam_max = self._power_iteration_lam_max(L_csr, dim)
        if lam_max < 1e-10:
            self._moments = np.zeros(D + 1, dtype=np.float64)
            self._moments[0] = 1.0
            self._lam_max = 0.0
            self._dim = dim
            return self._moments

        Z = self._rademacher_probes(dim, num_vectors)
        scale = 2.0 / lam_max

        def L_norm_mm(V):
            return scale * (L_csr @ V) - V

        mu = np.empty(D + 1, dtype=np.float64)
        T_prev = Z.clone()
        T_curr = L_norm_mm(Z)

        def hutchinson_trace(Z_mat, T_mat):
            per_vec = torch.real(torch.sum(Z_mat.conj() * T_mat, dim=0))
            return float(per_vec.mean().cpu()) / dim

        mu[0] = hutchinson_trace(Z, T_prev)
        mu[1] = hutchinson_trace(Z, T_curr)

        for k in range(2, D + 1):
            T_next = 2.0 * L_norm_mm(T_curr) - T_prev
            mu[k] = hutchinson_trace(Z, T_next)
            T_prev = T_curr
            T_curr = T_next

        # Noise floor check
        noise_floor = 2.0 / np.sqrt(num_vectors * dim)
        tail_moments = np.abs(mu[max(0, D - 10):])
        if np.mean(tail_moments) < noise_floor:
            logger.warning(
                "KPM moments mu[%d:%d] (mean=%.2e) are below the Hutchinson "
                "noise floor (%.2e). Increase num_vectors for reliable "
                "high-frequency reconstruction.", D - 10, D,
                float(np.mean(tail_moments)), noise_floor,
            )

        self._moments = mu
        self._lam_max = lam_max
        self._dim = dim

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return mu

    def _check_moments(self):
        if self._moments is None:
            raise RuntimeError(
                "Call compute_moments(epsilon) first before using "
                "reconstruction methods (density_of_states, idos, etc.)."
            )

    def density_of_states(self, lambda_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct rho(lambda) using Jackson-damped KPM expansion."""
        self._check_moments()
        mu = self._moments
        g = self._jackson_coefficients(self._kpm_degree)
        lam_max = self._lam_max

        if lam_max < 1e-10:
            return np.zeros_like(lambda_grid)

        x = 2.0 * lambda_grid / lam_max - 1.0
        x = np.clip(x, -1.0 + 1e-10, 1.0 - 1e-10)

        T_prev = np.ones_like(x)
        T_curr = x.copy()

        rho = g[0] * mu[0] * T_prev
        if len(mu) > 1:
            rho += 2.0 * g[1] * mu[1] * T_curr

        for n in range(2, len(mu)):
            T_next = 2.0 * x * T_curr - T_prev
            rho += 2.0 * g[n] * mu[n] * T_next
            T_prev = T_curr
            T_curr = T_next

        weight = 1.0 / (np.pi * np.sqrt(1.0 - x**2))
        rho *= weight
        rho *= 2.0 / lam_max

        return rho

    def idos(self, cutoff: float) -> float:
        """Integrated Density of States: fraction of eigenvalues below cutoff."""
        self._check_moments()
        if self._lam_max < 1e-10:
            return 0.0
        lambda_grid = np.linspace(1e-12, cutoff, 1000)
        rho = self.density_of_states(lambda_grid)
        rho = np.maximum(rho, 0.0)
        return float(np.trapezoid(rho, lambda_grid))

    def spectral_density_at_zero(self) -> float:
        """Spectral weight at lambda=0: IDOS up to KPM resolution limit."""
        self._check_moments()
        if self._lam_max < 1e-10:
            return 0.0
        # Note: _lam_max includes a 5% safety margin from power iteration.
        # This makes the resolution limit ~5% larger than the true spectral
        # resolution, which is conservative (integrates over a slightly wider
        # band). This is acceptable for falsification.
        resolution_limit = np.pi * self._lam_max / self._kpm_degree
        return self.idos(cutoff=resolution_limit)

    def spectral_sum(self, epsilon: float, k: int = 100) -> float:
        """Spectral sum proxy via KPM eigenvalue-weighted density.
        The k parameter is accepted for API compatibility but ignored.
        """
        self.compute_moments(epsilon)
        if self._lam_max < 1e-10:
            return 0.0
        cutoff = self._lam_max * 0.01
        lambda_grid = np.linspace(1e-12, cutoff, 1000)
        rho = self.density_of_states(lambda_grid)
        rho = np.maximum(rho, 0.0)
        return float(self._dim * np.trapezoid(lambda_grid * rho, lambda_grid))

    def smallest_eigenvalues(self, epsilon: float, k: int = 100) -> NDArray[np.float64]:
        """Not available — use compute_moments() + idos() instead."""
        raise NotImplementedError(
            "KPMSheafLaplacian does not compute individual eigenvalues. "
            "Use compute_moments(epsilon) + idos(cutoff) for the integrated "
            "density of states, or spectral_sum(epsilon) for a compatible "
            "proxy that works with sigma sweep scripts."
        )
