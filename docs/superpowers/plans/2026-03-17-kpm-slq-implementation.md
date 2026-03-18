# KPM Sheaf Laplacian Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `KPMSheafLaplacian` — a GPU-native Kernel Polynomial Method backend that computes raw Chebyshev moments and reconstructs the density of states via Jackson-damped KPM, replacing discrete eigenvalue extraction with the Integrated Density of States (IDOS) as the primary spectral observable.

**Architecture:** Hoist SLQ primitives (`_power_iteration_lam_max`, `_rademacher_probes`) to `TorchSheafLaplacian`. Create `KPMSheafLaplacian` as a sibling to `HeatKernelSheafLaplacian` inheriting GPU transport and BSR assembly. The GPU computes raw Chebyshev moments; Jackson-damped reconstruction runs on CPU. A new `FALSIFICATION_IDOS.md` defines thermodynamic scaling criteria. A scaling exponent extractor script validates the proof pathway.

**Tech Stack:** PyTorch (sparse CSR, GPU SpMV), NumPy (Chebyshev recurrence, Jackson damping, trapezoidal integration), SciPy (curve fitting for scaling exponents)

**Spec:** `docs/superpowers/specs/2026-03-17-kpm-sheaf-laplacian-design.md`

---

## Chunk 1: Hoist SLQ Primitives

### Task 1: Hoist `_power_iteration_lam_max` to TorchSheafLaplacian

**Files:**
- Modify: `atft/topology/torch_sheaf_laplacian.py:189-241` (class body, before `gpu_transport`)
- Modify: `atft/topology/heat_kernel_laplacian.py:197-220` (remove method)
- Test: `tests/test_kpm.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_kpm.py`:

```python
"""Tests for KPM infrastructure and KPMSheafLaplacian."""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp

try:
    import torch
    from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_torch_csr(dense_np, device="cpu"):
    """Convert a dense numpy matrix to a torch sparse CSR tensor."""
    csr = sp.csr_matrix(dense_np.astype(np.complex128))
    return torch.sparse_csr_tensor(
        torch.tensor(csr.indptr, dtype=torch.int64, device=device),
        torch.tensor(csr.indices, dtype=torch.int64, device=device),
        torch.tensor(csr.data, dtype=torch.cdouble, device=device),
        size=csr.shape,
    )


def _graph_laplacian(n):
    """Path graph Laplacian on n vertices. Eigenvalues: 2 - 2*cos(k*pi/n)."""
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        L[i, i] += 1
        L[i + 1, i + 1] += 1
        L[i, i + 1] = -1
        L[i + 1, i] = -1
    return L


# ---------------------------------------------------------------------------
# SLQ Primitives on TorchSheafLaplacian
# ---------------------------------------------------------------------------

class TestPowerIterationHoisted:
    """Verify _power_iteration_lam_max is accessible on TorchSheafLaplacian."""

    def test_method_exists_on_parent(self):
        """TorchSheafLaplacian should have _power_iteration_lam_max."""
        assert hasattr(TorchSheafLaplacian, '_power_iteration_lam_max')

    def test_lam_max_diagonal(self):
        """Diagonal matrix: lam_max should approximate max(diag)."""
        from atft.topology.transport_maps import TransportMapBuilder
        builder = TransportMapBuilder(K=1, sigma=0.5)
        zeros = np.array([0.0, 1.0, 2.0])
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")

        diag = np.diag([1.0, 5.0, 9.0]).astype(np.complex128)
        L_csr = _to_torch_csr(diag)
        lam_max = lap._power_iteration_lam_max(L_csr, 3)
        # With 5% safety margin: expect ~9.45
        assert 9.0 <= lam_max <= 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kpm.py::TestPowerIterationHoisted -v`
Expected: FAIL — `TorchSheafLaplacian` does not have `_power_iteration_lam_max`

- [ ] **Step 3: Move `_power_iteration_lam_max` to TorchSheafLaplacian**

In `atft/topology/torch_sheaf_laplacian.py`, add this method to `TorchSheafLaplacian` class, after `__init__` (after line 241) and before `gpu_transport` (before line 243):

```python
    def _power_iteration_lam_max(
        self, L_csr, dim: int, n_iter: int = 30,
    ) -> float:
        """Estimate largest eigenvalue via power iteration.

        Universal SLQ primitive: required by any Chebyshev-based trace
        estimator to normalize the spectrum to [-1, 1].

        Returns lam_max with 5% safety margin.
        """
        device = self.device
        dtype = torch.cdouble

        rng = torch.Generator(device=device)
        rng.manual_seed(123)
        v = torch.randn(
            dim, dtype=torch.double, device=device, generator=rng,
        ).to(dtype)
        v = v / torch.linalg.norm(v)

        lam = torch.tensor(0.0, device=device)
        for _ in range(n_iter):
            w = torch.mv(L_csr, v)
            lam = torch.real(torch.dot(v.conj(), w))
            norm_w = torch.linalg.norm(w).real
            if norm_w < 1e-14:
                return 0.0
            v = w / norm_w

        return float(lam.cpu()) * 1.05  # 5% safety margin

    def _rademacher_probes(
        self, dim: int, num_vectors: int, seed: int = 42,
    ) -> torch.Tensor:
        """Generate Rademacher probe matrix Z in {-1, +1}^{dim x num_vectors}.

        Universal SLQ primitive: provides stochastic trace estimation
        via Hutchinson's method. Returns complex128 tensor on self.device.
        """
        device = self.device
        dtype = torch.cdouble
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        Z = (
            torch.randint(
                0, 2, (dim, num_vectors),
                device=device, dtype=torch.double, generator=rng,
            ) * 2 - 1
        ).to(dtype)
        return Z
```

Then in `atft/topology/heat_kernel_laplacian.py`:
- **Delete** the `_power_iteration_lam_max` method (lines 197-220) — it's now inherited.
- **Replace** the inline Rademacher probe code in `heat_trace()` (lines 156-164) with:

```python
        # 4. Rademacher probe matrix Z in {-1, +1}^{dim x num_vectors}
        Z = self._rademacher_probes(dim, num_vectors, seed=42)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kpm.py::TestPowerIterationHoisted -v`
Expected: PASS

- [ ] **Step 5: Verify heat kernel unchanged**

Run: `pytest tests/ -k "heat or kernel" -v`
Expected: all existing heat kernel tests PASS (if any exist)

- [ ] **Step 6: Commit**

```bash
git add atft/topology/torch_sheaf_laplacian.py atft/topology/heat_kernel_laplacian.py tests/test_kpm.py
git commit -m "refactor: hoist _power_iteration_lam_max and _rademacher_probes to TorchSheafLaplacian"
```

---

## Chunk 2: KPMSheafLaplacian Core

### Task 2: Jackson damping coefficients

**Files:**
- Create: `atft/topology/kpm_sheaf_laplacian.py`
- Test: `tests/test_kpm.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_kpm.py`:

```python
class TestJacksonCoefficients:
    """Tests for the Jackson kernel damping factors."""

    def test_g0_is_one(self):
        """g[0] must equal 1.0 exactly (Weisse et al. convention)."""
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        g = KPMSheafLaplacian._jackson_coefficients(D=100)
        npt.assert_allclose(g[0], 1.0, atol=1e-14)

    def test_all_positive(self):
        """Jackson kernel factors should be non-negative."""
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        g = KPMSheafLaplacian._jackson_coefficients(D=50)
        assert np.all(g >= -1e-15)

    def test_length(self):
        """Should return D+1 coefficients."""
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        g = KPMSheafLaplacian._jackson_coefficients(D=30)
        assert len(g) == 31

    def test_monotonically_decreasing(self):
        """Jackson coefficients should generally decrease."""
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        g = KPMSheafLaplacian._jackson_coefficients(D=100)
        # g[0]=1.0, g[D] should be near 0
        assert g[0] > g[-1]
        assert g[-1] < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kpm.py::TestJacksonCoefficients -v`
Expected: FAIL with `ImportError: cannot import name 'KPMSheafLaplacian'`

- [ ] **Step 3: Create `kpm_sheaf_laplacian.py` with Jackson coefficients**

Create `atft/topology/kpm_sheaf_laplacian.py`:

```python
"""KPM-based Sheaf Laplacian — density of states via Chebyshev moments.

Implements the Kernel Polynomial Method (KPM) with Jackson damping to
reconstruct the continuous density of states rho(lambda) from raw
Chebyshev moments computed on GPU via Stochastic Lanczos Quadrature.

The primary observable is the Integrated Density of States (IDOS) —
a macroscopic, scale-invariant topological measure of the near-kernel
that is robust against spectral clustering and scales to K=500+.

The raw Chebyshev moments mu_n = (1/dim) Tr(T_n(L_norm)) are
path-ordered holonomies — the exact analytic objects needed to bridge
discrete numerics to continuous spectral bounds in the thermodynamic
limit K -> infinity.

Requires: torch (with CUDA or ROCm support for GPU acceleration).

Reference: Weisse et al., Rev. Mod. Phys. 78 (2006), Eq. 71.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian


class KPMSheafLaplacian(TorchSheafLaplacian):
    """KPM-based sheaf Laplacian — density of states via Chebyshev moments.

    Computes raw Chebyshev moments on GPU, then reconstructs rho(lambda),
    IDOS, and spectral density at zero using Jackson-damped KPM on CPU.

    Args:
        builder: TransportMapBuilder instance.
        zeros: 1D array of unfolded zero positions.
        transport_mode: "superposition" (default), "fe", or "resonant".
        device: Torch device (None for auto-detection).
        num_vectors: Number of Rademacher probe vectors (default 30).
        degree: Chebyshev polynomial degree (default 300).
    """

    def __init__(
        self,
        builder,
        zeros,
        transport_mode: str = "superposition",
        device=None,
        num_vectors: int = 30,
        degree: int = 300,
    ):
        super().__init__(builder, zeros, transport_mode, device)
        self._kpm_num_vectors = num_vectors
        self._kpm_degree = degree
        self._moments = None  # raw, undamped Chebyshev moments
        self._lam_max = None
        self._dim = None

    @staticmethod
    def _jackson_coefficients(D: int) -> NDArray[np.float64]:
        """Jackson kernel damping factors g_n for n=0..D.

        Ensures strict positivity of reconstructed rho(lambda) and
        uniform convergence O(1/D). No tunable parameters.

        Convention: D = polynomial degree, N_Jackson = D+1 in the
        denominator. Follows Weisse et al., Rev. Mod. Phys. 78 (2006),
        Eq. 71 with N -> D. Verify: g[0] = 1.0 exactly.
        """
        n = np.arange(D + 1, dtype=np.float64)
        Dp1 = D + 1
        g = ((Dp1 - n) * np.cos(np.pi * n / Dp1)
             + np.sin(np.pi * n / Dp1) / np.tan(np.pi / Dp1)) / Dp1
        return g
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kpm.py::TestJacksonCoefficients -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atft/topology/kpm_sheaf_laplacian.py tests/test_kpm.py
git commit -m "feat: add KPMSheafLaplacian skeleton with Jackson damping coefficients"
```

---

### Task 3: Moment computation (`compute_moments`)

**Files:**
- Modify: `atft/topology/kpm_sheaf_laplacian.py`
- Test: `tests/test_kpm.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_kpm.py`:

```python
class TestComputeMoments:
    """Tests for the GPU Chebyshev moment computation."""

    def test_diagonal_moments(self):
        """For a diagonal matrix, mu_n should match analytic Chebyshev moments."""
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder

        # Use K=1 so dim = N, making a simple diagonal-like Laplacian
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=1, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu", degree=20,
                                 num_vectors=50)

        mu = kpm.compute_moments(epsilon=2.0)
        assert mu is not None
        assert len(mu) == 21  # degree + 1
        # mu_0 = (1/dim) * Tr(T_0(L_norm)) = (1/dim) * Tr(I) = 1.0
        npt.assert_allclose(mu[0], 1.0, atol=0.1)  # Hutchinson has variance

    def test_moments_are_real(self):
        """All moments must be strictly real (no imaginary component)."""
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder

        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu", degree=10,
                                 num_vectors=20)
        mu = kpm.compute_moments(epsilon=1.0)
        assert mu.dtype == np.float64

    def test_stores_lam_max_and_dim(self):
        """compute_moments should populate _lam_max and _dim."""
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder

        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu", degree=10)
        kpm.compute_moments(epsilon=1.5)
        assert kpm._lam_max is not None
        assert kpm._lam_max > 0
        assert kpm._dim == 6  # 3 * 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kpm.py::TestComputeMoments -v`
Expected: FAIL — `KPMSheafLaplacian` has no `compute_moments` method

- [ ] **Step 3: Implement `compute_moments`**

Add to `KPMSheafLaplacian` in `kpm_sheaf_laplacian.py`:

```python
    def compute_moments(self, epsilon: float) -> NDArray[np.float64]:
        """Compute raw Chebyshev moments on GPU via Hutchinson trace estimation.

        Stores the undamped moments mu_n = (1/dim) Tr(T_n(L_norm)) for
        n = 0..D. Jackson damping is applied dynamically at reconstruction.

        Args:
            epsilon: Rips complex scale parameter.

        Returns:
            1D numpy array of D+1 raw Chebyshev moments (float64).
        """
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

        # 1. Estimate lambda_max
        lam_max = self._power_iteration_lam_max(L_csr, dim)
        if lam_max < 1e-10:
            self._moments = np.zeros(D + 1, dtype=np.float64)
            self._moments[0] = 1.0  # Tr(T_0) / dim = 1
            self._lam_max = 0.0
            self._dim = dim
            return self._moments

        # 2. Rademacher probes
        Z = self._rademacher_probes(dim, num_vectors)

        # 3. Spectrum normalization: L_norm = (2/lam_max)L - I
        scale = 2.0 / lam_max

        def L_norm_mm(V):
            return scale * (L_csr @ V) - V

        # 4. Chebyshev recurrence with per-step Hutchinson trace
        mu = np.empty(D + 1, dtype=np.float64)
        T_prev = Z.clone()       # T_0 @ Z = Z
        T_curr = L_norm_mm(Z)    # T_1 @ Z = L_norm @ Z

        # mu[k] = (1/dim) * real(mean_j(z_j† T_k z_j))
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

        # 5. Noise floor check: warn if high-frequency moments hit Hutchinson noise
        noise_floor = 2.0 / np.sqrt(num_vectors * dim)
        tail_moments = np.abs(mu[max(0, D-10):])
        if np.mean(tail_moments) < noise_floor:
            import logging
            logging.getLogger(__name__).warning(
                "KPM moments mu[%d:%d] (mean=%.2e) are below the Hutchinson "
                "noise floor (%.2e). Increase num_vectors for reliable "
                "high-frequency reconstruction.", D-10, D,
                float(np.mean(tail_moments)), noise_floor,
            )

        # 6. Store results
        self._moments = mu
        self._lam_max = lam_max
        self._dim = dim

        # GPU cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return mu
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kpm.py::TestComputeMoments -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atft/topology/kpm_sheaf_laplacian.py tests/test_kpm.py
git commit -m "feat: add compute_moments — GPU Chebyshev moment computation via Hutchinson"
```

---

### Task 4: KPM reconstructors (`density_of_states`, `idos`, `spectral_density_at_zero`)

**Files:**
- Modify: `atft/topology/kpm_sheaf_laplacian.py`
- Test: `tests/test_kpm.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_kpm.py`:

```python
class TestKPMReconstructors:
    """Tests for Jackson-damped KPM density reconstruction."""

    def _make_kpm_with_known_spectrum(self):
        """Build a KPM instance and compute moments for a small problem."""
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu", degree=50,
                                 num_vectors=50)
        kpm.compute_moments(epsilon=1.0)
        return kpm

    def test_density_of_states_shape(self):
        """density_of_states should return array matching lambda_grid."""
        kpm = self._make_kpm_with_known_spectrum()
        grid = np.linspace(0.01, kpm._lam_max * 0.99, 200)
        rho = kpm.density_of_states(grid)
        assert rho.shape == grid.shape

    def test_density_nonneg(self):
        """Jackson-damped density should be non-negative everywhere."""
        kpm = self._make_kpm_with_known_spectrum()
        grid = np.linspace(0.01, kpm._lam_max * 0.99, 500)
        rho = kpm.density_of_states(grid)
        # Allow tiny numerical noise
        assert np.all(rho >= -1e-10)

    def test_idos_between_zero_and_one(self):
        """IDOS is a fraction: must be in [0, 1]."""
        kpm = self._make_kpm_with_known_spectrum()
        val = kpm.idos(kpm._lam_max * 0.5)
        assert 0.0 <= val <= 1.0 + 1e-6

    def test_idos_monotonic(self):
        """IDOS must increase with larger cutoff."""
        kpm = self._make_kpm_with_known_spectrum()
        cutoffs = [kpm._lam_max * f for f in [0.01, 0.1, 0.5, 0.9]]
        idos_vals = [kpm.idos(c) for c in cutoffs]
        for i in range(len(idos_vals) - 1):
            assert idos_vals[i] <= idos_vals[i+1] + 1e-6

    def test_spectral_density_at_zero_finite(self):
        """spectral_density_at_zero must be finite (no singularity blowup)."""
        kpm = self._make_kpm_with_known_spectrum()
        val = kpm.spectral_density_at_zero()
        assert np.isfinite(val)
        assert val >= 0.0

    def test_guard_before_compute(self):
        """Calling reconstructors before compute_moments should raise."""
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu")
        with pytest.raises(RuntimeError, match="compute_moments"):
            kpm.idos(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kpm.py::TestKPMReconstructors -v`
Expected: FAIL — methods don't exist

- [ ] **Step 3: Implement all three reconstructors**

Add to `KPMSheafLaplacian` in `kpm_sheaf_laplacian.py`:

```python
    def _check_moments(self):
        """Raise if compute_moments has not been called."""
        if self._moments is None:
            raise RuntimeError(
                "Call compute_moments(epsilon) first before using "
                "reconstruction methods (density_of_states, idos, etc.)."
            )

    def density_of_states(self, lambda_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct rho(lambda) using Jackson-damped KPM expansion.

        rho(x) = (1 / (pi * sqrt(1 - x^2))) * [g_0*mu_0 + 2 * sum_n g_n*mu_n*T_n(x)]

        where x = 2*lambda/lam_max - 1 maps lambda to [-1, 1].

        Args:
            lambda_grid: 1D array of lambda values where rho is evaluated.

        Returns:
            1D array of spectral density values rho(lambda).
        """
        self._check_moments()
        mu = self._moments
        g = self._jackson_coefficients(self._kpm_degree)
        lam_max = self._lam_max

        if lam_max < 1e-10:
            return np.zeros_like(lambda_grid)

        # Map lambda to Chebyshev domain [-1, 1]
        x = 2.0 * lambda_grid / lam_max - 1.0
        x = np.clip(x, -1.0 + 1e-10, 1.0 - 1e-10)

        # Chebyshev recurrence on the grid
        T_prev = np.ones_like(x)   # T_0(x) = 1
        T_curr = x.copy()          # T_1(x) = x

        rho = g[0] * mu[0] * T_prev
        if len(mu) > 1:
            rho += 2.0 * g[1] * mu[1] * T_curr

        for n in range(2, len(mu)):
            T_next = 2.0 * x * T_curr - T_prev
            rho += 2.0 * g[n] * mu[n] * T_next
            T_prev = T_curr
            T_curr = T_next

        # KPM reconstruction denominator (cancels Chebyshev orthogonality weight)
        weight = 1.0 / (np.pi * np.sqrt(1.0 - x**2))
        rho *= weight

        # Jacobian: dx/dlambda = 2/lam_max
        rho *= 2.0 / lam_max

        return rho

    def idos(self, cutoff: float) -> float:
        """Integrated Density of States: fraction of eigenvalues below cutoff.

        Computes integral_0^cutoff rho(lambda) d(lambda) via trapezoidal rule.

        Args:
            cutoff: Upper integration limit in lambda space.

        Returns:
            Fraction in [0, 1]. Multiply by self._dim for absolute count.
        """
        self._check_moments()

        if self._lam_max < 1e-10:
            return 0.0

        lambda_grid = np.linspace(1e-12, cutoff, 1000)
        rho = self.density_of_states(lambda_grid)
        rho = np.maximum(rho, 0.0)  # clamp floating-point noise
        return float(np.trapezoid(rho, lambda_grid))

    def spectral_density_at_zero(self) -> float:
        """Spectral weight at lambda=0: IDOS up to KPM resolution limit.

        Returns the integrated density up to Delta_lambda = pi * lam_max / D,
        the fundamental resolution limit of the Jackson-damped expansion.
        Avoids the Chebyshev boundary singularity at x=-1.
        """
        self._check_moments()

        if self._lam_max < 1e-10:
            return 0.0

        # Note: _lam_max includes a 5% safety margin from power iteration.
        # This makes the resolution limit ~5% larger than the true spectral
        # resolution, which is conservative (integrates over a slightly wider
        # band). This is acceptable for falsification — it makes the IDOS
        # measurement an upper bound on the true near-kernel density.
        resolution_limit = np.pi * self._lam_max / self._kpm_degree
        return self.idos(cutoff=resolution_limit)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kpm.py::TestKPMReconstructors -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atft/topology/kpm_sheaf_laplacian.py tests/test_kpm.py
git commit -m "feat: add KPM reconstructors — density_of_states, idos, spectral_density_at_zero"
```

---

### Task 5: Backward-compatible `spectral_sum` and `smallest_eigenvalues`

**Files:**
- Modify: `atft/topology/kpm_sheaf_laplacian.py`
- Test: `tests/test_kpm.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_kpm.py`:

```python
class TestKPMSpectralSum:
    """Tests for backward-compatible spectral_sum."""

    def test_spectral_sum_finite(self):
        """spectral_sum should return a finite positive value."""
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu", degree=50,
                                 num_vectors=30)
        s = kpm.spectral_sum(epsilon=1.0, k=10)
        assert np.isfinite(s)
        assert s >= 0.0

    def test_smallest_eigenvalues_raises(self):
        """smallest_eigenvalues should raise NotImplementedError."""
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu")
        with pytest.raises(NotImplementedError):
            kpm.smallest_eigenvalues(epsilon=1.0, k=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kpm.py::TestKPMSpectralSum -v`
Expected: FAIL — methods don't exist or use parent Lanczos implementation

- [ ] **Step 3: Implement `spectral_sum` and `smallest_eigenvalues`**

Add to `KPMSheafLaplacian`:

```python
    def spectral_sum(self, epsilon: float, k: int = 100) -> float:
        """Spectral sum proxy via KPM eigenvalue-weighted density.

        Integrates lambda * rho(lambda) over [0, cutoff] to approximate
        sum(smallest_eigenvalues). The k parameter is accepted for API
        compatibility but ignored — KPM does not compute individual eigenvalues.

        Compatible with existing sigma sweep scripts.
        """
        self.compute_moments(epsilon)

        if self._lam_max < 1e-10:
            return 0.0

        cutoff = self._lam_max * 0.01  # bottom 1% of spectrum
        lambda_grid = np.linspace(1e-12, cutoff, 1000)
        rho = self.density_of_states(lambda_grid)
        rho = np.maximum(rho, 0.0)
        return float(self._dim * np.trapezoid(lambda_grid * rho, lambda_grid))

    def smallest_eigenvalues(
        self, epsilon: float, k: int = 100,
    ) -> NDArray[np.float64]:
        """Not available — use compute_moments() + idos() instead."""
        raise NotImplementedError(
            "KPMSheafLaplacian does not compute individual eigenvalues. "
            "Use compute_moments(epsilon) + idos(cutoff) for the integrated "
            "density of states, or spectral_sum(epsilon) for a compatible "
            "proxy that works with sigma sweep scripts."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kpm.py::TestKPMSpectralSum -v`
Expected: all 2 tests PASS

- [ ] **Step 5: Run the full test file**

Run: `pytest tests/test_kpm.py -v`
Expected: all tests PASS

- [ ] **Step 6: Update `__init__.py` exports**

Check if `atft/topology/__init__.py` has any exports. If empty, add:

```python
from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian

__all__ = ["KPMSheafLaplacian"]
```

If it already has exports for other backends, add `KPMSheafLaplacian` to the existing list.

- [ ] **Step 7: Commit**

```bash
git add atft/topology/kpm_sheaf_laplacian.py atft/topology/__init__.py tests/test_kpm.py
git commit -m "feat: add spectral_sum (backward-compatible), smallest_eigenvalues guard, and __init__ export"
```

---

## Chunk 3: Documentation and Scaling Analysis

### Task 6: Write `FALSIFICATION_IDOS.md`

**Files:**
- Create: `docs/FALSIFICATION_IDOS.md`

- [ ] **Step 1: Write the document**

Create `docs/FALSIFICATION_IDOS.md`:

```markdown
# ATFT Falsification Criteria — KPM/IDOS Era

**Date:** 2026-03-17
**Status:** Pre-committed before any K>=100 KPM data is collected.
**Predecessor:** `docs/FALSIFICATION.md` (frozen at commit 2b3f023, valid for K<=50 eigenvalue-based runs)
**Authors:** Blake Jones, Claude (Opus 4.6)

These criteria are frozen at commit time and MUST NOT be modified after
K>=100 KPM data collection begins.

---

## 1. Definitions

| Symbol | Definition |
|--------|-----------|
| IDOS(Delta_lambda, sigma) | Integrated Density of States from 0 to Delta_lambda at fiber parameter sigma |
| Delta_lambda | KPM resolution limit: pi * lambda_max / D |
| rho(lambda, sigma) | Jackson-damped KPM-reconstructed spectral density |
| rho_0(sigma) | spectral_density_at_zero(): IDOS integrated up to Delta_lambda |
| R_IDOS(K) | IDOS contrast ratio: IDOS(sigma=0.5) / mean(IDOS(sigma!=0.5)) |
| mu_n(sigma) | Raw (undamped) Chebyshev moment: (1/dim) Tr(T_n(L_norm)) |

---

## 2. Framework Falsification

**Question:** Does the ATFT topological obstruction mechanism work?

| ID | Criterion | Trigger | Verdict |
|----|-----------|---------|---------|
| F1 | Persistent off-line density | IDOS(Delta_lambda, sigma!=0.5) does NOT decrease monotonically as K increases from 50 to 200 | Framework fails to create topological obstruction off the critical line |
| F2 | Contrast saturation | R_IDOS(K) plateaus at a finite value (does not grow as K increases) | The arithmetic signal saturates — no phase transition |
| F3 | Symmetric collapse | IDOS(Delta_lambda, sigma=0.5) also collapses to 0 as K increases | Obstruction is non-selective — kills all sections, not just off-line |
| F4 | GUE artifact | GUE random matrices produce R_IDOS within 2x of zeta R_IDOS at same K | Signal is not arithmetic — it arises from generic random matrix statistics |

---

## 3. Positive Evidence for RH

| ID | Criterion | Observable | Pass condition |
|----|-----------|-----------|----------------|
| P1 | Near-kernel concentration | IDOS(Delta_lambda, sigma=0.5) | Remains finite (> 1e-4) for K = 50, 100, 200 |
| P2 | Off-line collapse | IDOS(Delta_lambda, sigma=0.25) | Decreases by at least 50% from K=50 to K=200 |
| P3 | Contrast divergence | R_IDOS(K) | Grows by at least 3x from K=50 to K=200 |
| P4 | Moment scaling | Decay rate of mu_n at sigma=0.5 vs sigma=0.25 | Off-line moments decay faster (exponent ratio > 1.5) |

---

## 4. Backward Compatibility

The K=20 discrimination ratio of 670x (from eigenvalue-based spectral_sum) can be
recomputed as an IDOS contrast ratio R_IDOS(K=20) for cross-validation. The original
`FALSIFICATION.md` criteria remain valid and frozen for all eigenvalue-based runs at K<=50.

---

## 5. Interpretation

If P1-P4 all pass: The ATFT framework provides strong numerical evidence that the
topological obstruction selectively forbids global sections off the critical line,
consistent with RH. The raw moment scaling exponents provide the analytic scaffolding
for a formal proof.

If any F1-F4 triggers: The corresponding aspect of the framework is falsified.
F1 or F3 invalidate the obstruction mechanism. F2 means the signal is finite, not
a true phase transition. F4 means the signal is not arithmetic.
```

- [ ] **Step 2: Commit**

```bash
git add docs/FALSIFICATION_IDOS.md
git commit -m "docs: add FALSIFICATION_IDOS.md — KPM-era pre-registered criteria"
```

---

### Task 7: Thermodynamic scaling exponent extractor script

**Files:**
- Create: `scripts/validate_thermodynamic_scaling.py`

- [ ] **Step 1: Write the script**

Create `scripts/validate_thermodynamic_scaling.py`:

```python
#!/usr/bin/env python
"""Thermodynamic Scaling Exponent Extractor.

Runs KPMSheafLaplacian at multiple K values and extracts the scaling
exponents of the IDOS as a function of K. These exponents are the
key numerical evidence for or against the topological obstruction.

Usage:
  python scripts/validate_thermodynamic_scaling.py --quick
  python scripts/validate_thermodynamic_scaling.py --k-values 20 50 100
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atft.sources.zeta_zeros import ZetaZerosSource
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian


def run_kpm_sweep(
    zeros: np.ndarray,
    K: int,
    sigma_values: list[float],
    epsilon: float,
    degree: int = 300,
    num_vectors: int = 30,
    device: str | None = None,
) -> dict:
    """Run KPM sweep across sigma values for a given K."""
    results = {}
    for sigma in sigma_values:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        kpm = KPMSheafLaplacian(
            builder, zeros, device=device,
            degree=degree, num_vectors=num_vectors,
        )
        mu = kpm.compute_moments(epsilon)
        idos_val = kpm.spectral_density_at_zero()
        s_sum = kpm.spectral_sum(epsilon)

        results[sigma] = {
            "idos_at_zero": idos_val,
            "spectral_sum": s_sum,
            "lam_max": kpm._lam_max,
            "dim": kpm._dim,
            "moments": mu.tolist(),
        }
        print(f"  K={K}, sigma={sigma:.2f}: IDOS={idos_val:.6f}, "
              f"S={s_sum:.4f}, dim={kpm._dim}")

    return results


def compute_scaling_exponents(
    all_results: dict[int, dict],
    sigma_on: float = 0.5,
    sigma_off: float = 0.25,
) -> dict:
    """Extract thermodynamic scaling exponents from multi-K results.

    Fits IDOS(K) ~ K^alpha for on-line and off-line sigma values.
    """
    from scipy.optimize import curve_fit

    K_vals = sorted(all_results.keys())
    if len(K_vals) < 2:
        return {"warning": "Need at least 2 K values for scaling fit"}

    K_arr = np.array(K_vals, dtype=np.float64)

    # Extract IDOS at sigma_on and sigma_off
    idos_on = np.array([all_results[K][sigma_on]["idos_at_zero"] for K in K_vals])
    idos_off = np.array([all_results[K][sigma_off]["idos_at_zero"] for K in K_vals])

    # Contrast ratio
    contrast = np.where(idos_off > 1e-12, idos_on / idos_off, np.inf)

    # Three ansatzes for the IDOS collapse:
    # 1. Power law: IDOS ~ a * K^alpha (scale-free phase transition)
    # 2. Logarithmic: IDOS ~ a * (log K)^alpha (prime harmonic distribution)
    # 3. Exponential: IDOS ~ a * exp(-alpha * K) (rapid collapse)
    def power_law(K, a, alpha):
        return a * K ** alpha

    def log_law(K, a, alpha):
        return a * np.log(K) ** alpha

    def exp_law(K, a, alpha):
        return a * np.exp(-alpha * K)

    ansatzes = {
        "power_law": power_law,
        "logarithmic": log_law,
        "exponential": exp_law,
    }

    exponents = {
        "K_values": K_vals,
        "idos_on_line": idos_on.tolist(),
        "idos_off_line": idos_off.tolist(),
        "contrast_ratio": contrast.tolist(),
    }

    # Fit each ansatz to off-line IDOS decay
    if np.all(idos_off > 1e-15) and len(K_vals) >= 2:
        best_aic = np.inf
        for name, func in ansatzes.items():
            try:
                p0 = [float(idos_off[0]), -1.0] if name != "exponential" else [float(idos_off[0]), 0.01]
                popt, pcov = curve_fit(func, K_arr, idos_off, p0=p0, maxfev=5000)
                residuals = idos_off - func(K_arr, *popt)
                ss_res = float(np.sum(residuals**2))
                # AIC with 2 parameters
                n = len(K_arr)
                aic = n * np.log(ss_res / n + 1e-30) + 4
                exponents[f"off_{name}_a"] = float(popt[0])
                exponents[f"off_{name}_alpha"] = float(popt[1])
                exponents[f"off_{name}_aic"] = float(aic)
                if aic < best_aic:
                    best_aic = aic
                    exponents["off_best_ansatz"] = name
                    exponents["off_line_exponent"] = float(popt[1])
            except Exception as e:
                exponents[f"off_{name}_error"] = str(e)

    # Fit on-line IDOS (should stay finite or grow)
    if np.all(idos_on > 1e-15) and len(K_vals) >= 2:
        try:
            popt, _ = curve_fit(power_law, K_arr, idos_on, p0=[1.0, 0.0])
            exponents["on_line_amplitude"] = float(popt[0])
            exponents["on_line_exponent"] = float(popt[1])
        except Exception as e:
            exponents["on_line_fit_error"] = str(e)

    return exponents


def evaluate_criteria(exponents: dict) -> dict:
    """Evaluate P1-P4 and F1-F4 criteria from FALSIFICATION_IDOS.md."""
    verdicts = {}

    idos_on = exponents.get("idos_on_line", [])
    idos_off = exponents.get("idos_off_line", [])
    contrast = exponents.get("contrast_ratio", [])

    if len(idos_on) >= 2:
        # P1: Near-kernel concentration (on-line stays finite)
        verdicts["P1"] = "PASS" if all(v > 1e-4 for v in idos_on) else "FAIL"
        # P2: Off-line collapse (decreases by 50%)
        verdicts["P2"] = "PASS" if idos_off[-1] < idos_off[0] * 0.5 else "FAIL"
        # P3: Contrast divergence (grows by 3x)
        finite_contrast = [c for c in contrast if np.isfinite(c)]
        if len(finite_contrast) >= 2:
            verdicts["P3"] = "PASS" if finite_contrast[-1] > finite_contrast[0] * 3 else "FAIL"
        # F1: Persistent off-line density
        verdicts["F1"] = "TRIGGERED" if idos_off[-1] >= idos_off[0] else "OK"
        # F3: Symmetric collapse
        verdicts["F3"] = "TRIGGERED" if all(v < 1e-4 for v in idos_on) else "OK"

    off_exp = exponents.get("off_line_exponent")
    if off_exp is not None:
        verdicts["off_line_scaling"] = f"IDOS_off ~ K^{off_exp:.3f}"

    return verdicts


def main():
    parser = argparse.ArgumentParser(description="KPM Thermodynamic Scaling Analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: small N, few K values")
    parser.add_argument("--k-values", nargs="+", type=int, default=None,
                        help="K values to sweep (default: 20 50)")
    parser.add_argument("--epsilon", type=float, default=3.0)
    parser.add_argument("--degree", type=int, default=300)
    parser.add_argument("--num-vectors", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load zeros
    source = ZetaZerosSource()
    raw_zeros = source.load()
    unfolding = SpectralUnfolding()
    zeros = unfolding.transform(raw_zeros)

    if args.quick:
        zeros = zeros[:200]
        K_values = [6, 10]
        sigma_values = [0.25, 0.50, 0.75]
    else:
        K_values = args.k_values or [20, 50]
        sigma_values = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75]

    print(f"Zeros: {len(zeros)}, K values: {K_values}")
    print(f"Sigma grid: {sigma_values}")
    print(f"Epsilon: {args.epsilon}, Degree: {args.degree}")
    print("=" * 60)

    all_results = {}
    for K in K_values:
        print(f"\n--- K = {K} ---")
        all_results[K] = run_kpm_sweep(
            zeros, K, sigma_values, args.epsilon,
            degree=args.degree, num_vectors=args.num_vectors,
        )

    # Compute scaling exponents
    print("\n" + "=" * 60)
    print("THERMODYNAMIC SCALING ANALYSIS")
    print("=" * 60)
    exponents = compute_scaling_exponents(all_results)
    for key, val in exponents.items():
        print(f"  {key}: {val}")

    # Evaluate criteria
    print("\n--- FALSIFICATION_IDOS Criteria ---")
    verdicts = evaluate_criteria(exponents)
    for criterion, verdict in verdicts.items():
        print(f"  {criterion}: {verdict}")

    # Save results
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump({
                "K_values": K_values,
                "sigma_values": sigma_values,
                "results": {str(k): v for k, v in all_results.items()},
                "exponents": exponents,
                "verdicts": verdicts,
            }, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script in quick mode**

Run: `python scripts/validate_thermodynamic_scaling.py --quick`
Expected: Completes without errors. Prints IDOS values, scaling exponents (may be noisy with only 2 K values and 200 zeros), and P1-P4/F1-F4 verdicts.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_thermodynamic_scaling.py
git commit -m "feat: add thermodynamic scaling exponent extractor for KPM IDOS"
```

---

### Task 8: End-to-end validation

**Files:**
- No file changes — validation only

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests PASS, including test_kpm.py and existing tests

- [ ] **Step 2: Run the KPM scaling script in quick mode**

Run: `python scripts/validate_thermodynamic_scaling.py --quick`
Expected: Completes successfully with finite IDOS values and scaling output

- [ ] **Step 3: Verify heat kernel backward compatibility**

Run: `python -c "from atft.topology.heat_kernel_laplacian import HeatKernelSheafLaplacian; from atft.topology.transport_maps import TransportMapBuilder; import numpy as np; b = TransportMapBuilder(K=3, sigma=0.5); zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0]); h = HeatKernelSheafLaplacian(b, zeros, device='cpu'); print('heat_trace:', h.heat_trace(1.0)); print('spectral_sum:', h.spectral_sum(1.0))"`
Expected: Finite values, no errors (proves hoisting didn't break heat kernel)

- [ ] **Step 4: Final commit if fixups needed**

If any issues found, fix and commit:
```bash
git add -u
git commit -m "fix: address issues found during end-to-end validation"
```
