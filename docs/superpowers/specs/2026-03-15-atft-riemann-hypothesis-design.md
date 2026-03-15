# ATFT Riemann Hypothesis: Phase 1 Design Specification

**Date:** 2026-03-15
**Status:** Approved
**Authors:** Blake Jones, Claude (Opus 4.6)

---

## 1. Overview

### 1.1 Goal

Apply the Adaptive Topological Field Theory (ATFT) framework to the Riemann Hypothesis by treating the non-trivial zeros of the Riemann zeta function as a spectral point cloud and comparing its topological evolution signature against the Gaussian Unitary Ensemble (GUE).

### 1.2 The Core Insight

The ATFT paper (Jones, Feb 2026) demonstrates that field equations emerge as "topological waypoints" — algebraic constraints on how persistent homology evolves across filtration scales. The Hilbert-Polya conjecture posits that zeta zeros are eigenvalues of an unknown self-adjoint operator. If true, their spacing statistics must match GUE random matrix theory (the Montgomery-Odlyzko connection).

This experiment translates that conjecture into the native language of ATFT: **the zeta zeros and GUE eigenvalues must produce statistically indistinguishable waypoint signatures**.

### 1.3 Project Trajectory

- **Phase 1 (this spec):** Computational benchmark — Zeta vs GUE vs Poisson topological comparison. Validates the ATFT pipeline and produces publishable evidence.
- **Phase 2 (future):** Sheaf-valued upgrade — attach u(N) Lie algebra fibers to the Rips complex. Attempt to identify the Hilbert-Polya operator as a sheaf Laplacian.
- **Phase 3 (future):** Rigorous reformulation — prove that RH is equivalent to a specific topological waypoint constraint.

### 1.4 Success Criteria

The experiment succeeds if:

| Test | Success | Failure |
|------|---------|---------|
| Zeta beta_0 within GUE 99% band | Curves stay inside envelope | Curves deviate from band |
| Zeta G_0 within GUE 99% band | Curves stay inside envelope | Curves deviate from band |
| Zeta Mahalanobis p-value | p > 0.05 | p < 0.05 |
| Poisson beta_0 within GUE band | Must FAIL (negative control) | If passes, pipeline is broken |
| Poisson Mahalanobis p-value | Must be p near 0 (negative control) | If p > 0.05, pipeline is broken |

---

## 2. Mathematical Foundation

### 2.1 Configuration Space: Zeros as Spectrum

The imaginary parts of the non-trivial zeta zeros, gamma_n, form a 1D point cloud in R^1. This mirrors the ATFT paper's treatment of the quantum harmonic oscillator spectrum (Section 6.3), where energy levels are treated as a finite metric space.

### 2.2 The GUE Connection

Freeman Dyson and Hugh Montgomery discovered that the pair correlation of zeta zeros matches the GUE eigenvalue spacing distribution. GUE exhibits "level repulsion" — eigenvalues actively avoid clustering. This produces a rigid, highly specific gap distribution that differs qualitatively from uncorrelated (Poisson) spacings.

### 2.3 Spectral Unfolding

Raw spectra have non-uniform average density. Unfolding normalizes to mean gap = 1, isolating the fluctuations (which encode universality class) from the trend (which is system-specific).

- **Zeta zeros:** N_smooth(T) = (T / 2*pi) * ln(T / (2*pi*e)) + 7/8. Unfolded zeros: x_i = N_smooth(gamma_i).
- **GUE eigenvalues:** Unfold via Wigner semicircle law or empirical CDF.
- **Poisson:** Cumulative sum of Exp(1) gaps; trivially unfolded.

### 2.4 The H_0 Analytical Shortcut

For a 1D point cloud, H_0 persistent homology is analytically computable in O(N log N):

1. Sort the unfolded spectrum: x_1 < x_2 < ... < x_N
2. Compute gaps: g_i = x_{i+1} - x_i
3. H_0 persistence diagram = {(0, g_i) : i = 1,...,N-1} union {(0, infinity)}

No Rips complex construction or Ripser computation needed. Each feature's death scale is exactly the gap at which two adjacent components merge.

### 2.5 Evolution Curves

From the persistence diagram, three evolution functions are computed on a discrete epsilon grid:

- **Betti curve** beta_0(eps) = count of features alive at scale eps. For H_0, starts at N and decreases monotonically.
- **Gini trajectory** G_0(eps) = Gini coefficient of the lifetime distribution of features alive at eps. Measures hierarchical organization: G=0 (uniform/disordered) to G->1 (single dominant feature).
- **Persistence curve** P_0(eps) = sum of lifetimes of features alive at eps.

### 2.6 Waypoint Signature (Adapted for H_0)

W_0(C) = (eps*, {eps_{w,i}}_{i=1}^K, {delta_0(eps_{w,i})}_{i=1}^K, G_0(eps*), dG_0/deps|_{eps*}) in R^(2K+3)

Where:
- eps* = onset scale (first eps where topology changes)
- eps_{w,i} = top-K waypoint scales (extrema of |d(beta_0)/d(eps)| sorted by magnitude)
- delta_0(eps_{w,i}) = topological derivative values at waypoints
- G_0(eps*) = Gini coefficient at onset
- dG_0/deps|_{eps*} = Gini trajectory slope at onset

K is fixed at 2 for Phase 1, yielding vectors in R^7.

### 2.7 Statistical Validation: Two-Pronged Criterion

**Prong 1 (Macroscopic — functional envelope):** Generate M=1,000 GUE matrices. Compute Betti and Gini curves for each. Build pointwise 99% confidence bands using empirical percentiles (not Gaussian assumption). Compute L^2 functional distance between zeta curves and ensemble mean. Check that zeta curves stay entirely within the 99% band across the full adaptive filtration.

**Prong 2 (Formal — Mahalanobis waypoint matching):** Extract R^7 waypoint signature for all 1,000 GUE matrices. Compute the multivariate mean and regularized covariance matrix (Tikhonov regularization: reg = 1e-6 * trace(cov) / dim). Extract the same R^7 signature for the zeta zeros. Compute Mahalanobis distance D_M. Report p-value under chi-square distribution with 7 degrees of freedom.

The Mahalanobis distance was chosen over Euclidean distance because it accounts for covariance between waypoint components (e.g., onset scale and first waypoint are likely correlated). Bottleneck distance was rejected because its L-infinity nature makes it hypersensitive to single outlier gaps in finite samples, penalizing the entire comparison based on microscopic fluctuations rather than the global level-repulsion signature.

---

## 3. Software Architecture

### 3.1 Design Philosophy

The code mirrors the ATFT paper's Dimensional Pathway. Each mathematical stage maps to a software module. Modules communicate through shared immutable data types and depend only on the contract layer, never on each other directly.

### 3.2 Pipeline

```
ConfigurationSource -> FeatureMap -> TopologicalOperator -> EvolutionCurves
                                                                  |
                                                                  v
                                    StatisticalValidator <- WaypointExtractor
```

### 3.3 Project Layout

```
atft/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── types.py              # PointCloud, PersistenceDiagram, EvolutionCurve,
│   │                         #   WaypointSignature, ValidationResult
│   └── protocols.py          # ConfigurationSource, FeatureMap,
│                              #   TopologicalOperator, Cacheable
├── sources/
│   ├── __init__.py
│   ├── zeta_zeros.py          # Load/parse Odlyzko dataset
│   ├── gue.py                 # Generate GUE random matrices (GPU via PyTorch)
│   └── poisson.py             # Generate Poisson point processes
├── feature_maps/
│   ├── __init__.py
│   ├── identity.py            # Pass-through for pre-processed data
│   └── spectral_unfolding.py  # Normalize spacings to mean gap = 1
├── topology/
│   ├── __init__.py
│   ├── analytical_h0.py       # Exact H_0 for 1D point clouds (O(N log N))
│   ├── ripser_ph.py           # GPU-accelerated PH via giotto-tda (Phase 2)
│   └── sheaf_ph.py            # Sheaf-valued PH stub (Phase 2)
├── analysis/
│   ├── __init__.py
│   ├── evolution_curves.py    # Betti, Gini, Persistence curve computation
│   ├── waypoint_extractor.py  # Top-K waypoint extraction
│   └── statistical_tests.py   # Mahalanobis, L2 envelope, confidence bands
├── visualization/
│   ├── __init__.py
│   └── plots.py               # Three-panel publication figure
├── io/
│   ├── __init__.py
│   └── cache.py               # HDF5 serialization for intermediate results
└── experiments/
    ├── __init__.py
    └── phase1_benchmark.py     # Orchestrator: Zeta vs GUE vs Poisson
```

### 3.4 Data Flow

```
                                float32              float64
                               (GPU-friendly)       (precision)
                                    |                    |
ZetaZerosSource ──► PointCloud ──► SpectralUnfolding ──► AnalyticalH0
GUESource ─────────► (N, 1)                              ──► PersistenceDiagram
PoissonSource ─────►                                          ──► EvolutionCurveSet
                                                                   ──► WaypointSignature
                                                                        ──► ValidationResult
```

---

## 4. Core Types

### 4.1 PointCloud

```python
@dataclass(frozen=True)
class PointCloud:
    points: NDArray[np.float32]          # shape (N, d)
    metadata: dict = field(default_factory=dict)
```

Immutable. N points in R^d. float32 for VRAM efficiency.

### 4.2 PointCloudBatch

```python
@dataclass(frozen=True)
class PointCloudBatch:
    clouds: list[PointCloud]
```

Supports ragged batches (different N per cloud, same d). Enables GPU batch processing for the GUE ensemble without Python-loop overhead.

### 4.3 PersistenceDiagram

```python
@dataclass(frozen=True)
class PersistenceDiagram:
    diagrams: dict[int, NDArray[np.float64]]   # degree -> (n_features, 2)
    metadata: dict = field(default_factory=dict)
```

float64 for birth/death precision. Implements Cacheable protocol (HDF5 serialization).

### 4.4 EvolutionCurve and EvolutionCurveSet

```python
@dataclass(frozen=True)
class EvolutionCurve:
    epsilon_grid: NDArray[np.float64]    # shape (n_steps,)
    values: NDArray[np.float64]          # shape (n_steps,)
    curve_type: CurveType                # BETTI, GINI, or PERSISTENCE
    degree: int                          # homological degree k

@dataclass(frozen=True)
class EvolutionCurveSet:
    betti: dict[int, EvolutionCurve]
    gini: dict[int, EvolutionCurve]
    persistence: dict[int, EvolutionCurve]
```

### 4.5 WaypointSignature

```python
@dataclass(frozen=True)
class WaypointSignature:
    onset_scale: float
    waypoint_scales: NDArray[np.float64]         # exactly K values
    topo_derivatives: NDArray[np.float64]         # exactly K values
    gini_at_onset: float
    gini_derivative_at_onset: float

    def as_vector(self) -> NDArray[np.float64]:
        """Flatten to R^(2K+3) for Mahalanobis computation."""
```

**Critical constraint:** K is fixed across the ensemble (Top-K extraction). The WaypointExtractor always returns exactly K waypoints (the top K extrema of |d(beta_k)/d(eps)| by magnitude, zero-padded if fewer than K exist). This guarantees all signature vectors live in the same R^(2K+3) for covariance computation.

### 4.6 ValidationResult

```python
@dataclass(frozen=True)
class ValidationResult:
    mahalanobis_distance: float
    p_value: float
    l2_distance_betti: float
    l2_distance_gini: float
    within_confidence_band: bool
    ensemble_size: int
    metadata: dict = field(default_factory=dict)
```

---

## 5. Protocols

All pipeline stages are defined by Python Protocols (structural subtyping). Phase 2 modules need only implement the same interface — no inheritance required.

```python
class ConfigurationSource(Protocol):
    def generate(self, n_points: int, **kwargs) -> PointCloud: ...
    def generate_batch(self, n_points: int, batch_size: int, **kwargs) -> PointCloudBatch: ...

class FeatureMap(Protocol):
    def transform(self, cloud: PointCloud) -> PointCloud: ...
    def transform_batch(self, batch: PointCloudBatch) -> PointCloudBatch: ...

class TopologicalOperator(Protocol):
    def compute(self, cloud: PointCloud, max_degree: int = 0,
                epsilon_max: float | None = None) -> PersistenceDiagram: ...
    def compute_batch(self, batch: PointCloudBatch, max_degree: int = 0,
                      epsilon_max: float | None = None) -> list[PersistenceDiagram]: ...

class Cacheable(Protocol):
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Cacheable": ...
```

---

## 6. Phase 1 Implementations

### 6.1 Configuration Sources

**ZetaZerosSource:** Parses Odlyzko's plain-text dataset of zeta zero imaginary parts. Loads first N values, reshapes to (N, 1) PointCloud.

**GUESource:** Generates N x N random Hermitian matrices via H = (A + A^dagger) / sqrt(2N) where A has i.i.d. complex normal entries. Eigendecomposition via torch.linalg.eigh on CUDA for batch processing. Returns eigenvalues as (N, 1) PointCloud.

**PoissonSource:** Generates N i.i.d. Exp(1) gaps, cumulative sums to positions. No correlations, no level repulsion. Serves as negative control.

### 6.2 Spectral Unfolding (Feature Map)

Normalizes any raw spectrum to mean gap = 1:

- Zeta: Uses analytic smooth staircase N_smooth(T) = (T/2pi) * ln(T/(2pi*e)) + 7/8
- GUE: Uses Wigner semicircle law or empirical CDF
- Poisson: Already uniform (identity transform)

### 6.3 Analytical H_0 (Topological Operator)

For 1D point clouds, H_0 persistence is exactly the sorted gap sequence:

1. Sort points: x_1 < x_2 < ... < x_N — O(N log N)
2. Gaps: g_i = x_{i+1} - x_i — O(N)
3. Persistence diagram: {(0, g_i)} — no Rips complex needed

This bypass reduces computation from Ripser's sparse matrix reduction to a simple array operation. The 1,000-matrix GUE ensemble processes in seconds on CPU.

### 6.4 Evolution Curve Computer

Samples Betti, Gini, and Persistence curves on a uniform epsilon grid (n_steps = 1,000):

- beta_0(eps) = count of (birth, death) pairs where birth <= eps < death
- G_0(eps) = Gini coefficient of lifetimes of alive features (sorted formula)
- P_0(eps) = sum of lifetimes of alive features

Gini coefficient uses the efficient sorted formula: G = (2 * sum(i * v_i)) / (n * sum(v_i)) - (n+1)/n

Handles edge cases: empty feature sets return G = 0.0.

### 6.5 Waypoint Extractor (Top-K)

1. Compute topological derivative: delta_0(eps) = d(beta_0)/d(eps) via np.gradient
2. Find local extrema of |delta_0| via scipy.signal.argrelextrema
3. Take top K by magnitude (K=2 for Phase 1)
4. If fewer than K extrema exist, take the K largest |delta_0| grid points
5. Sort selected waypoints by epsilon position

Onset scale logic branches on homological degree:
- H_0 (degree=0): first eps where beta_0 drops below initial value N
- H_k (degree>0): first eps where beta_k rises above 0

### 6.6 Statistical Validator

**Prong 1 — Functional Envelope:**
- Interpolate all ensemble curves to common epsilon grid
- Compute pointwise percentile bands (empirical, not Gaussian)
- L^2 distance = sqrt(sum((target - mean)^2 * d_eps))
- within_band = all(lower <= target <= upper) at every grid point

**Prong 2 — Mahalanobis Distance:**
- Stack ensemble signature vectors: (M, 2K+3) matrix
- Mean vector: mu = mean over M rows
- Regularized covariance: C_reg = C + (1e-6 * trace(C) / dim) * I
- D_M = sqrt((w - mu)^T * C_reg^{-1} * (w - mu))
- p-value = 1 - chi2.cdf(D_M^2, df=2K+3)

### 6.7 Caching

Intermediate results (PersistenceDiagram, EvolutionCurveSet) are serialized to HDF5 via h5py:

```
cache/
├── zeta_zeros/
│   ├── persistence_N10000.h5
│   └── curves_N10000.h5
├── gue_ensemble/
│   ├── persistence_M1000_N10000.h5
│   └── curves_M1000_N10000.h5
└── poisson/
    ├── persistence_M1000_N10000.h5
    └── curves_M1000_N10000.h5
```

---

## 7. Experiment Orchestration

### 7.1 Configuration

```python
Phase1Config:
    n_points: 10,000
    ensemble_size: 1,000
    k_waypoints: 2
    n_epsilon_steps: 1,000
    confidence_level: 0.99
    device: "cuda"
```

### 7.2 Execution Flow

1. Generate/load data: zeta zeros (Odlyzko), 1,000 GUE matrices, 1,000 Poisson samples
2. Unfold all spectra to mean gap = 1
3. Compute H_0 persistence diagrams (analytical shortcut)
4. Cache persistence diagrams to HDF5
5. Compute evolution curves (Betti, Gini, Persistence)
6. Extract Top-2 waypoint signatures for all configurations
7. Fit StatisticalValidator on GUE ensemble
8. Validate zeta zeros against GUE (should PASS)
9. Validate Poisson against GUE (should FAIL — negative control)
10. Generate three-panel publication figure

### 7.3 Output: Three-Panel Figure

- **Panel A:** Betti curves beta_0(eps) — GUE 99% band (grey), Zeta (blue), Poisson (red)
- **Panel B:** Gini trajectories G_0(eps) — same color scheme
- **Panel C:** PCA projection of R^7 waypoint space — GUE cloud (grey), Zeta star (blue), Poisson cross (red)
- **Footer:** Mahalanobis distances and p-values for both zeta and Poisson

---

## 8. Dependencies

```
numpy >= 1.24
scipy >= 1.11
h5py >= 3.9
matplotlib >= 3.7
torch >= 2.1          # GUE batch eigendecomposition on CUDA
giotto-tda >= 0.6     # Ripser backend (Phase 2 / H_1)
```

Phase 1 core computation has no hard dependency on GPU or Ripser — the analytical H_0 shortcut runs on CPU. GPU is used for batch GUE eigendecomposition (torch.linalg.eigh).

---

## 9. Phase 2 Extension Points

The architecture is designed so that Phase 2 (sheaf-valued persistent homology) requires only:

1. A new `SheafSpecification` type in `core/types.py` defining stalks (u(N) fibers) and restriction maps
2. A new `SheafPH` class in `topology/sheaf_ph.py` implementing the `TopologicalOperator` protocol
3. A new feature map in `feature_maps/` if needed for the sheaf construction

No changes to the analysis, validation, or visualization layers. The waypoint extraction and statistical validation work identically on sheaf-valued outputs once the evolution curves are computed.

---

## 10. Open Questions

1. **Optimal K for waypoint extraction.** K=2 is a reasonable starting point, but the true number of significant topological waypoints may differ. Consider running a sensitivity analysis on K = 1, 2, 3, 4.

2. **H_1 analysis.** The analytical H_0 shortcut is specific to 1D point clouds. When moving to higher-dimensional feature maps or H_1 computation, we need Ripser. Does H_1 carry additional discriminative information for the zeta/GUE comparison?

3. **Scaling behavior.** How do the waypoint signatures evolve as N increases from 1,000 to 100,000? Convergence of the signatures with N would strengthen the claim.

4. **The sheaf construction.** What are the natural restriction maps for u(N) fibers on the Rips complex of zeta zeros? The Phase 1 empirical data should inform this choice.

5. **Connection to known results.** How does the Betti curve onset scale relate to the known pair correlation function R_2(r)? Can we derive a closed-form prediction for the GUE Betti curve from random matrix theory?
