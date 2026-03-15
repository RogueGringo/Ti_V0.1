# ATFT Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular Python framework that compares topological evolution signatures of Riemann zeta zeros against GUE random matrix eigenvalues and Poisson baselines, using the Adaptive Topological Field Theory pipeline.

**Architecture:** Data flows through a six-stage pipeline (Source -> FeatureMap -> TopologicalOperator -> EvolutionCurves -> WaypointExtractor -> StatisticalValidator), with each stage defined by a Python Protocol and communicating through immutable frozen dataclasses. Phase 1 exploits an analytical H_0 shortcut for 1D point clouds (O(N log N) via sorted gaps), avoiding Ripser entirely.

**Tech Stack:** Python 3.11+, NumPy, SciPy, h5py, matplotlib, PyTorch (GPU eigendecomposition only), pytest

**Spec:** `docs/superpowers/specs/2026-03-15-atft-riemann-hypothesis-design.md`

---

## File Structure

### Files to Create

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, pytest config |
| `atft/__init__.py` | Package root |
| `atft/core/__init__.py` | Core subpackage |
| `atft/core/types.py` | All immutable data types (PointCloud, PersistenceDiagram, EvolutionCurve, WaypointSignature, ValidationResult) |
| `atft/core/protocols.py` | Protocol interfaces (ConfigurationSource, FeatureMap, TopologicalOperator, Cacheable) |
| `atft/sources/__init__.py` | Sources subpackage |
| `atft/sources/poisson.py` | PoissonSource: i.i.d. Exp(1) gaps |
| `atft/sources/gue.py` | GUESource: random Hermitian matrices via PyTorch |
| `atft/sources/zeta_zeros.py` | ZetaZerosSource: parse Odlyzko dataset |
| `atft/feature_maps/__init__.py` | Feature maps subpackage |
| `atft/feature_maps/identity.py` | IdentityMap: pass-through |
| `atft/feature_maps/spectral_unfolding.py` | SpectralUnfolding: normalize to mean gap=1 |
| `atft/topology/__init__.py` | Topology subpackage |
| `atft/topology/analytical_h0.py` | AnalyticalH0: exact H_0 PH for 1D via sorted gaps |
| `atft/analysis/__init__.py` | Analysis subpackage |
| `atft/analysis/evolution_curves.py` | EvolutionCurveComputer: Betti, Gini, Persistence curves |
| `atft/analysis/waypoint_extractor.py` | WaypointExtractor: top-K gap-based extraction |
| `atft/analysis/statistical_tests.py` | StatisticalValidator: Mahalanobis + envelope |
| `atft/visualization/__init__.py` | Visualization subpackage |
| `atft/visualization/plots.py` | Three-panel publication figure |
| `atft/io/__init__.py` | IO subpackage |
| `atft/io/cache.py` | HDF5 serialization for PersistenceDiagram and EvolutionCurveSet |
| `atft/experiments/__init__.py` | Experiments subpackage |
| `atft/experiments/phase1_benchmark.py` | Phase1Experiment orchestrator |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/test_types.py` | Tests for core data types |
| `tests/test_sources.py` | Tests for all three sources |
| `tests/test_feature_maps.py` | Tests for identity and spectral unfolding |
| `tests/test_analytical_h0.py` | Tests for analytical H_0 computation |
| `tests/test_evolution_curves.py` | Tests for Betti, Gini, Persistence curves |
| `tests/test_waypoint_extractor.py` | Tests for top-K waypoint extraction |
| `tests/test_statistical_tests.py` | Tests for Mahalanobis and envelope validation |
| `tests/test_cache.py` | Tests for HDF5 serialization round-trip |
| `tests/test_phase1_integration.py` | End-to-end integration test with small N |

---

## Chunk 1: Foundation

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: All `__init__.py` files (14 files)
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```python
# File: pyproject.toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "atft"
version = "0.1.0"
description = "Adaptive Topological Field Theory framework"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.11",
    "h5py>=3.9",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
gpu = ["torch>=2.1"]
dev = ["pytest>=7.4", "pytest-cov>=4.1", "scikit-learn>=1.3"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Create all directory structure and __init__.py files**

```bash
mkdir -p atft/core atft/sources atft/feature_maps atft/topology atft/analysis atft/visualization atft/io atft/experiments tests
touch atft/__init__.py atft/core/__init__.py atft/sources/__init__.py atft/feature_maps/__init__.py atft/topology/__init__.py atft/analysis/__init__.py atft/visualization/__init__.py atft/io/__init__.py atft/experiments/__init__.py
```

- [ ] **Step 3: Create tests/conftest.py with shared fixtures**

```python
# File: tests/conftest.py
"""Shared fixtures for ATFT test suite."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic random number generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def simple_1d_points():
    """5 sorted points on a line with known gaps.

    Points: [1.0, 2.0, 4.0, 7.0, 11.0]
    Gaps:   [1.0, 2.0, 3.0, 4.0]
    Sorted gaps (desc): [4.0, 3.0, 2.0, 1.0]
    """
    return np.array([[1.0], [2.0], [4.0], [7.0], [11.0]], dtype=np.float64)


@pytest.fixture
def uniform_1d_points():
    """10 uniformly spaced points (all gaps = 1.0).

    Gini coefficient of gaps should be 0.0 (perfect equality).
    """
    return np.array([[float(i)] for i in range(10)], dtype=np.float64)
```

- [ ] **Step 4: Verify pytest discovers conftest**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/ --collect-only`
Expected: "no tests ran" (no test files yet), but no import errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml atft/ tests/conftest.py
git commit -m "feat: scaffold project structure with pyproject.toml and test fixtures"
```

---

### Task 2: Core Types

**Files:**
- Create: `atft/core/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write failing tests for all core types**

```python
# File: tests/test_types.py
"""Tests for ATFT core data types."""
import numpy as np
import pytest

from atft.core.types import (
    CurveType,
    EvolutionCurve,
    EvolutionCurveSet,
    PersistenceDiagram,
    PointCloud,
    PointCloudBatch,
    ValidationResult,
    WaypointSignature,
)


class TestPointCloud:
    def test_creation(self, simple_1d_points):
        pc = PointCloud(points=simple_1d_points)
        assert pc.n_points == 5
        assert pc.dimension == 1

    def test_immutable(self, simple_1d_points):
        pc = PointCloud(points=simple_1d_points)
        with pytest.raises(AttributeError):
            pc.points = np.zeros((3, 1))

    def test_metadata(self, simple_1d_points):
        pc = PointCloud(points=simple_1d_points, metadata={"source": "test"})
        assert pc.metadata["source"] == "test"

    def test_2d_points(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        pc = PointCloud(points=pts)
        assert pc.n_points == 2
        assert pc.dimension == 2


class TestPointCloudBatch:
    def test_creation(self, simple_1d_points, uniform_1d_points):
        c1 = PointCloud(points=simple_1d_points)
        c2 = PointCloud(points=uniform_1d_points)
        batch = PointCloudBatch(clouds=[c1, c2])
        assert batch.batch_size == 2

    def test_uniform_size_same(self):
        c1 = PointCloud(points=np.zeros((5, 1)))
        c2 = PointCloud(points=np.zeros((5, 1)))
        batch = PointCloudBatch(clouds=[c1, c2])
        assert batch.uniform_size() == 5

    def test_uniform_size_different(self):
        c1 = PointCloud(points=np.zeros((5, 1)))
        c2 = PointCloud(points=np.zeros((3, 1)))
        batch = PointCloudBatch(clouds=[c1, c2])
        assert batch.uniform_size() is None


class TestPersistenceDiagram:
    def test_creation(self):
        diag = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float64)
        pd = PersistenceDiagram(diagrams={0: diag})
        assert pd.max_degree == 0

    def test_degree_access(self):
        diag = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float64)
        pd = PersistenceDiagram(diagrams={0: diag})
        assert pd.degree(0).shape == (2, 2)

    def test_missing_degree_returns_empty(self):
        pd = PersistenceDiagram(diagrams={0: np.array([[0.0, 1.0]])})
        result = pd.degree(1)
        assert result.shape == (0, 2)
        assert result.dtype == np.float64

    def test_lifetimes(self):
        diag = np.array([[0.0, 1.0], [0.0, 3.0]], dtype=np.float64)
        pd = PersistenceDiagram(diagrams={0: diag})
        lts = pd.lifetimes(0)
        np.testing.assert_array_equal(lts, [1.0, 3.0])

    def test_lifetimes_empty(self):
        pd = PersistenceDiagram(diagrams={})
        lts = pd.lifetimes(0)
        assert len(lts) == 0
        assert lts.dtype == np.float64

    def test_max_degree_empty(self):
        pd = PersistenceDiagram(diagrams={})
        assert pd.max_degree == -1


class TestEvolutionCurve:
    def test_creation(self):
        eps = np.linspace(0, 1, 10)
        vals = np.arange(10, dtype=np.float64)
        ec = EvolutionCurve(
            epsilon_grid=eps, values=vals,
            curve_type=CurveType.BETTI, degree=0
        )
        assert ec.n_steps == 10
        assert ec.curve_type == CurveType.BETTI


class TestEvolutionCurveSet:
    def test_curve_lookup(self):
        eps = np.linspace(0, 1, 5)
        betti = EvolutionCurve(eps, np.ones(5), CurveType.BETTI, 0)
        gini = EvolutionCurve(eps, np.zeros(5), CurveType.GINI, 0)
        pers = EvolutionCurve(eps, np.ones(5) * 2, CurveType.PERSISTENCE, 0)
        cs = EvolutionCurveSet(
            betti={0: betti}, gini={0: gini}, persistence={0: pers}
        )
        assert cs.curve(CurveType.BETTI, 0) is betti
        assert cs.curve(CurveType.GINI, 0) is gini


class TestWaypointSignature:
    def test_as_vector_shape(self):
        ws = WaypointSignature(
            onset_scale=0.5,
            waypoint_scales=np.array([1.0, 2.0]),
            topo_derivatives=np.array([-3.0, -2.0]),
            gini_at_onset=0.3,
            gini_derivative_at_onset=0.01,
        )
        vec = ws.as_vector()
        assert vec.shape == (7,)  # 2*2 + 3 = 7
        assert ws.n_waypoints == 2
        assert ws.vector_dimension == 7

    def test_as_vector_values(self):
        ws = WaypointSignature(
            onset_scale=0.5,
            waypoint_scales=np.array([1.0, 2.0]),
            topo_derivatives=np.array([-3.0, -2.0]),
            gini_at_onset=0.3,
            gini_derivative_at_onset=0.01,
        )
        vec = ws.as_vector()
        expected = np.array([0.5, 1.0, 2.0, -3.0, -2.0, 0.3, 0.01])
        np.testing.assert_array_almost_equal(vec, expected)


class TestValidationResult:
    def test_creation(self):
        vr = ValidationResult(
            mahalanobis_distance=1.5,
            p_value=0.23,
            l2_distance_betti=0.05,
            l2_distance_gini=0.02,
            within_confidence_band=True,
            ensemble_size=1000,
        )
        assert vr.p_value == 0.23
        assert vr.within_confidence_band is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atft.core.types'`

- [ ] **Step 3: Implement all core types**

```python
# File: atft/core/types.py
"""Core data types for the ATFT framework.

All types are immutable (frozen dataclasses). Modules communicate
through these types and never depend on each other directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class CurveType(Enum):
    BETTI = "betti"
    GINI = "gini"
    PERSISTENCE = "persistence"


@dataclass(frozen=True)
class PointCloud:
    """A finite metric space: N points in R^d."""

    points: NDArray[np.float64]
    metadata: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def dimension(self) -> int:
        return self.points.shape[1]


@dataclass(frozen=True)
class PointCloudBatch:
    """A collection of point clouds for batch processing."""

    clouds: list[PointCloud]

    @property
    def batch_size(self) -> int:
        return len(self.clouds)

    def uniform_size(self) -> int | None:
        """Returns N if all clouds have the same size, else None."""
        sizes = {c.n_points for c in self.clouds}
        return sizes.pop() if len(sizes) == 1 else None


@dataclass(frozen=True)
class PersistenceDiagram:
    """Birth-death pairs for each homological degree.

    diagrams[k] is an (n_features, 2) array of (birth, death) pairs.
    The immortal H_0 feature is stored with death = np.inf.
    """

    diagrams: dict[int, NDArray[np.float64]]
    metadata: dict = field(default_factory=dict)

    def degree(self, k: int) -> NDArray[np.float64]:
        """Get persistence pairs for degree k. Returns empty (0,2) if missing."""
        return self.diagrams.get(k, np.empty((0, 2), dtype=np.float64))

    def lifetimes(self, k: int) -> NDArray[np.float64]:
        """Persistence = death - birth for degree k."""
        d = self.degree(k)
        if len(d) > 0:
            return d[:, 1] - d[:, 0]
        return np.array([], dtype=np.float64)

    @property
    def max_degree(self) -> int:
        return max(self.diagrams.keys()) if self.diagrams else -1


@dataclass(frozen=True)
class EvolutionCurve:
    """A topological evolution function sampled on an epsilon grid."""

    epsilon_grid: NDArray[np.float64]
    values: NDArray[np.float64]
    curve_type: CurveType
    degree: int

    @property
    def n_steps(self) -> int:
        return len(self.epsilon_grid)


@dataclass(frozen=True)
class EvolutionCurveSet:
    """All evolution curves for a single configuration."""

    betti: dict[int, EvolutionCurve]
    gini: dict[int, EvolutionCurve]
    persistence: dict[int, EvolutionCurve]

    def curve(self, curve_type: CurveType, degree: int) -> EvolutionCurve:
        lookup = {
            CurveType.BETTI: self.betti,
            CurveType.GINI: self.gini,
            CurveType.PERSISTENCE: self.persistence,
        }
        return lookup[curve_type][degree]


@dataclass(frozen=True)
class WaypointSignature:
    """The finite-dimensional topological fingerprint W(C).

    W_0(C) = (eps*, {eps_w,i}, {delta_0(eps_w,i)}, G_0(eps*), dG_0/deps|_eps*)
    Lives in R^(2K+3) where K = len(waypoint_scales).
    """

    onset_scale: float
    waypoint_scales: NDArray[np.float64]
    topo_derivatives: NDArray[np.float64]
    gini_at_onset: float
    gini_derivative_at_onset: float

    def as_vector(self) -> NDArray[np.float64]:
        """Flatten to R^(2K+3) for statistical comparison."""
        return np.concatenate([
            [self.onset_scale],
            self.waypoint_scales,
            self.topo_derivatives,
            [self.gini_at_onset],
            [self.gini_derivative_at_onset],
        ])

    @property
    def n_waypoints(self) -> int:
        return len(self.waypoint_scales)

    @property
    def vector_dimension(self) -> int:
        return 2 * self.n_waypoints + 3


@dataclass(frozen=True)
class ValidationResult:
    """Output of the statistical comparison."""

    mahalanobis_distance: float
    p_value: float
    l2_distance_betti: float
    l2_distance_gini: float
    within_confidence_band: bool
    ensemble_size: int
    metadata: dict = field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_types.py -v`
Expected: All 18 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atft/core/types.py tests/test_types.py
git commit -m "feat: implement core data types with full test coverage"
```

---

### Task 3: Core Protocols

**Files:**
- Create: `atft/core/protocols.py`
- Create: `tests/test_protocols.py`

- [ ] **Step 1: Write failing tests for protocol conformance**

```python
# File: tests/test_protocols.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_protocols.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement protocols**

```python
# File: atft/core/protocols.py
"""Protocol interfaces for the ATFT pipeline.

All pipeline stages are defined as Protocols (structural subtyping).
Concrete implementations need only implement the methods — no
inheritance required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from atft.core.types import (
    PersistenceDiagram,
    PointCloud,
    PointCloudBatch,
)


@runtime_checkable
class ConfigurationSource(Protocol):
    """Produces point clouds from a physical or mathematical source."""

    def generate(self, n_points: int, **kwargs) -> PointCloud: ...

    def generate_batch(
        self, n_points: int, batch_size: int, **kwargs
    ) -> PointCloudBatch: ...


@runtime_checkable
class FeatureMap(Protocol):
    """Transforms a point cloud into the metric space for PH computation."""

    def transform(self, cloud: PointCloud) -> PointCloud: ...

    def transform_batch(self, batch: PointCloudBatch) -> PointCloudBatch: ...


@runtime_checkable
class TopologicalOperator(Protocol):
    """The Adaptive Topological Operator."""

    def compute(
        self,
        cloud: PointCloud,
        max_degree: int = 0,
        epsilon_max: float | None = None,
    ) -> PersistenceDiagram: ...

    def compute_batch(
        self,
        batch: PointCloudBatch,
        max_degree: int = 0,
        epsilon_max: float | None = None,
    ) -> list[PersistenceDiagram]: ...


@runtime_checkable
class Cacheable(Protocol):
    """Serialization protocol for intermediate results."""

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> Cacheable: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_protocols.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atft/core/protocols.py tests/test_protocols.py
git commit -m "feat: implement protocol interfaces for pipeline stages"
```

---

## Chunk 2: Data Sources & Feature Maps

### Task 4: PoissonSource

**Files:**
- Create: `atft/sources/poisson.py`
- Create: `tests/test_sources.py`

- [ ] **Step 1: Write failing tests**

```python
# File: tests/test_sources.py
"""Tests for configuration sources."""
import numpy as np
import pytest

from atft.core.protocols import ConfigurationSource
from atft.sources.poisson import PoissonSource


class TestPoissonSource:
    def test_satisfies_protocol(self):
        assert isinstance(PoissonSource(seed=42), ConfigurationSource)

    def test_generate_shape(self):
        src = PoissonSource(seed=42)
        cloud = src.generate(100)
        assert cloud.points.shape == (100, 1)
        assert cloud.points.dtype == np.float64

    def test_generate_sorted_and_positive(self):
        src = PoissonSource(seed=42)
        cloud = src.generate(100)
        pts = cloud.points[:, 0]
        assert np.all(pts > 0)
        assert np.all(np.diff(pts) > 0)  # monotonically increasing

    def test_gaps_are_exponential(self):
        """Gaps should be approximately Exp(1) distributed (mean ~1)."""
        src = PoissonSource(seed=42)
        cloud = src.generate(10_000)
        gaps = np.diff(cloud.points[:, 0])
        assert 0.95 < np.mean(gaps) < 1.05

    def test_reproducibility(self):
        c1 = PoissonSource(seed=42).generate(100)
        c2 = PoissonSource(seed=42).generate(100)
        np.testing.assert_array_equal(c1.points, c2.points)

    def test_generate_batch(self):
        src = PoissonSource(seed=42)
        batch = src.generate_batch(100, batch_size=5)
        assert batch.batch_size == 5
        assert all(c.n_points == 100 for c in batch.clouds)

    def test_batch_members_differ(self):
        src = PoissonSource(seed=42)
        batch = src.generate_batch(100, batch_size=3)
        # Different batch members should have different points
        assert not np.array_equal(
            batch.clouds[0].points, batch.clouds[1].points
        )

    def test_metadata(self):
        src = PoissonSource(seed=42)
        cloud = src.generate(100)
        assert cloud.metadata["source"] == "poisson"
        assert cloud.metadata["n_points"] == 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_sources.py::TestPoissonSource -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement PoissonSource**

```python
# File: atft/sources/poisson.py
"""Poisson point process source (negative control baseline)."""
from __future__ import annotations

import numpy as np

from atft.core.types import PointCloud, PointCloudBatch


class PoissonSource:
    """Generates 1D point clouds from a Poisson process.

    Points are cumulative sums of i.i.d. Exp(1) gaps.
    Already unfolded by construction (mean gap = 1).
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = np.random.default_rng(np.random.SeedSequence(seed))

    def generate(self, n_points: int, **kwargs) -> PointCloud:
        gaps = self._rng.exponential(scale=1.0, size=n_points - 1)
        positions = np.concatenate([[0.0], np.cumsum(gaps)])
        # Shift so first point is positive
        positions += self._rng.exponential(scale=1.0)
        return PointCloud(
            points=positions.reshape(-1, 1).astype(np.float64),
            metadata={"source": "poisson", "n_points": n_points, "seed": self._seed},
        )

    def generate_batch(
        self, n_points: int, batch_size: int, **kwargs
    ) -> PointCloudBatch:
        clouds = [self.generate(n_points) for _ in range(batch_size)]
        return PointCloudBatch(clouds=clouds)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_sources.py::TestPoissonSource -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atft/sources/poisson.py tests/test_sources.py
git commit -m "feat: implement PoissonSource with Exp(1) gap generation"
```

---

### Task 5: GUESource

**Files:**
- Create: `atft/sources/gue.py`
- Modify: `tests/test_sources.py` (append new test class)

- [ ] **Step 1: Write failing tests (append to tests/test_sources.py)**

```python
# Append to: tests/test_sources.py
from atft.sources.gue import GUESource


class TestGUESource:
    def test_satisfies_protocol(self):
        assert isinstance(GUESource(seed=42), ConfigurationSource)

    def test_generate_shape(self):
        src = GUESource(seed=42)
        cloud = src.generate(50)
        assert cloud.points.shape == (50, 1)
        assert cloud.points.dtype == np.float64

    def test_eigenvalues_are_real_and_sorted(self):
        src = GUESource(seed=42)
        cloud = src.generate(50)
        pts = cloud.points[:, 0]
        assert np.all(np.isreal(pts))
        assert np.all(np.diff(pts) >= 0)  # sorted ascending

    def test_eigenvalues_bounded_by_semicircle(self):
        """GUE eigenvalues should be within [-1, 1] support (approximately)."""
        src = GUESource(seed=42)
        cloud = src.generate(200)
        pts = cloud.points[:, 0]
        # Allow some tail leakage for finite N
        assert np.all(pts > -1.5)
        assert np.all(pts < 1.5)

    def test_reproducibility(self):
        c1 = GUESource(seed=42).generate(50)
        c2 = GUESource(seed=42).generate(50)
        np.testing.assert_array_almost_equal(c1.points, c2.points)

    def test_generate_batch(self):
        src = GUESource(seed=42)
        batch = src.generate_batch(50, batch_size=5)
        assert batch.batch_size == 5
        assert all(c.n_points == 50 for c in batch.clouds)

    def test_metadata(self):
        src = GUESource(seed=42)
        cloud = src.generate(50)
        assert cloud.metadata["source"] == "gue"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_sources.py::TestGUESource -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement GUESource**

```python
# File: atft/sources/gue.py
"""Gaussian Unitary Ensemble source."""
from __future__ import annotations

import numpy as np

from atft.core.types import PointCloud, PointCloudBatch


class GUESource:
    """Generates eigenvalues of GUE random Hermitian matrices.

    Entry distribution: off-diagonal A_{ij} = (X + iY) / sqrt(2),
    X, Y ~ N(0,1). Diagonal A_{ii} ~ N(0,1) (real).
    H = (A + A^dagger) / (2 * sqrt(N)), semicircle support [-1, 1].

    Uses NumPy by default. Set use_torch=True for GPU batch processing.
    """

    def __init__(self, seed: int = 42, use_torch: bool = False, device: str = "cpu"):
        self._seed = seed
        self._rng = np.random.default_rng(np.random.SeedSequence(seed))
        self._use_torch = use_torch
        self._device = device

    def _generate_single(self, n: int) -> np.ndarray:
        """Generate eigenvalues of a single N x N GUE matrix."""
        # Complex normal entries
        real_part = self._rng.standard_normal((n, n))
        imag_part = self._rng.standard_normal((n, n))
        A = (real_part + 1j * imag_part) / np.sqrt(2)
        # Make diagonal real
        np.fill_diagonal(A, self._rng.standard_normal(n))
        # Hermitian matrix
        H = (A + A.conj().T) / (2 * np.sqrt(n))
        # Eigenvalues (real for Hermitian)
        eigenvalues = np.linalg.eigvalsh(H)
        return eigenvalues.astype(np.float64)

    def generate(self, n_points: int, **kwargs) -> PointCloud:
        eigenvalues = self._generate_single(n_points)
        return PointCloud(
            points=eigenvalues.reshape(-1, 1),
            metadata={"source": "gue", "n_points": n_points, "seed": self._seed},
        )

    def generate_batch(
        self, n_points: int, batch_size: int, **kwargs
    ) -> PointCloudBatch:
        clouds = [self.generate(n_points) for _ in range(batch_size)]
        return PointCloudBatch(clouds=clouds)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_sources.py::TestGUESource -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atft/sources/gue.py tests/test_sources.py
git commit -m "feat: implement GUESource with Hermitian matrix eigendecomposition"
```

---

### Task 6: ZetaZerosSource

**Files:**
- Create: `atft/sources/zeta_zeros.py`
- Create: `tests/data/test_zeros.txt` (small test fixture)
- Modify: `tests/test_sources.py` (append new test class)

- [ ] **Step 1: Create test fixture with known zeta zeros**

```
# File: tests/data/test_zeros.txt
# First 10 non-trivial zeta zeros (imaginary parts)
14.134725141734693
21.022039638771555
25.010857580145688
30.424876125859513
32.935061587739189
37.586178158825671
40.918719012147495
43.327073280914999
48.005150881167159
49.773832477672302
```

- [ ] **Step 2: Write failing tests (append to tests/test_sources.py)**

```python
# Append to: tests/test_sources.py
from pathlib import Path

from atft.sources.zeta_zeros import ZetaZerosSource

TEST_ZEROS_PATH = Path(__file__).parent / "data" / "test_zeros.txt"


class TestZetaZerosSource:
    def test_satisfies_protocol(self):
        assert isinstance(
            ZetaZerosSource(data_path=TEST_ZEROS_PATH), ConfigurationSource
        )

    def test_generate_shape(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        cloud = src.generate(5)
        assert cloud.points.shape == (5, 1)
        assert cloud.points.dtype == np.float64

    def test_generate_loads_first_n(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        cloud = src.generate(3)
        expected = np.array([[14.134725141734693],
                             [21.022039638771555],
                             [25.010857580145688]])
        np.testing.assert_array_almost_equal(cloud.points, expected)

    def test_generate_all(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        cloud = src.generate(10)
        assert cloud.points.shape == (10, 1)

    def test_sorted_ascending(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        cloud = src.generate(10)
        assert np.all(np.diff(cloud.points[:, 0]) > 0)

    def test_metadata(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        cloud = src.generate(5)
        assert cloud.metadata["source"] == "zeta_zeros"

    def test_n_exceeds_available_raises(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        with pytest.raises(ValueError, match="Requested 20.*only 10"):
            src.generate(20)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_sources.py::TestZetaZerosSource -v`
Expected: FAIL

- [ ] **Step 4: Implement ZetaZerosSource**

```python
# File: atft/sources/zeta_zeros.py
"""Riemann zeta zeros source (Odlyzko dataset)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from atft.core.types import PointCloud, PointCloudBatch


class ZetaZerosSource:
    """Loads non-trivial zeta zero imaginary parts from a text file.

    Expects one zero per line. Comments (lines starting with #) are skipped.
    """

    def __init__(self, data_path: Path | str):
        self._data_path = Path(data_path)
        self._zeros: np.ndarray | None = None

    def _load(self) -> np.ndarray:
        """Lazy-load and cache the zeros from disk."""
        if self._zeros is None:
            lines = []
            with open(self._data_path) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        lines.append(float(stripped))
            self._zeros = np.array(lines, dtype=np.float64)
        return self._zeros

    def generate(self, n_points: int, **kwargs) -> PointCloud:
        all_zeros = self._load()
        if n_points > len(all_zeros):
            raise ValueError(
                f"Requested {n_points} zeros but only {len(all_zeros)} available "
                f"in {self._data_path}"
            )
        selected = all_zeros[:n_points]
        return PointCloud(
            points=selected.reshape(-1, 1),
            metadata={
                "source": "zeta_zeros",
                "n_points": n_points,
                "data_path": str(self._data_path),
            },
        )

    def generate_batch(
        self, n_points: int, batch_size: int, **kwargs
    ) -> PointCloudBatch:
        # Zeta zeros are a single dataset — batch returns the same cloud
        cloud = self.generate(n_points)
        return PointCloudBatch(clouds=[cloud] * batch_size)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_sources.py::TestZetaZerosSource -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
mkdir -p tests/data
git add atft/sources/zeta_zeros.py tests/data/test_zeros.txt tests/test_sources.py
git commit -m "feat: implement ZetaZerosSource with Odlyzko dataset parser"
```

---

### Task 7: Identity Feature Map

**Files:**
- Create: `atft/feature_maps/identity.py`
- Create: `tests/test_feature_maps.py`

- [ ] **Step 1: Write failing tests**

```python
# File: tests/test_feature_maps.py
"""Tests for feature maps."""
import numpy as np

from atft.core.protocols import FeatureMap
from atft.core.types import PointCloud, PointCloudBatch
from atft.feature_maps.identity import IdentityMap


class TestIdentityMap:
    def test_satisfies_protocol(self):
        assert isinstance(IdentityMap(), FeatureMap)

    def test_transform_passthrough(self, simple_1d_points):
        fm = IdentityMap()
        cloud = PointCloud(points=simple_1d_points)
        result = fm.transform(cloud)
        np.testing.assert_array_equal(result.points, cloud.points)

    def test_transform_batch(self, simple_1d_points, uniform_1d_points):
        fm = IdentityMap()
        batch = PointCloudBatch(clouds=[
            PointCloud(points=simple_1d_points),
            PointCloud(points=uniform_1d_points),
        ])
        result = fm.transform_batch(batch)
        assert result.batch_size == 2
```

- [ ] **Step 2: Run to verify failure, then implement**

```python
# File: atft/feature_maps/identity.py
"""Identity feature map (pass-through)."""
from __future__ import annotations

from atft.core.types import PointCloud, PointCloudBatch


class IdentityMap:
    """Pass-through feature map for pre-processed data."""

    def transform(self, cloud: PointCloud) -> PointCloud:
        return cloud

    def transform_batch(self, batch: PointCloudBatch) -> PointCloudBatch:
        return batch
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_feature_maps.py::TestIdentityMap -v`
Expected: All 3 tests PASS

- [ ] **Step 4: Commit**

```bash
git add atft/feature_maps/identity.py tests/test_feature_maps.py
git commit -m "feat: implement IdentityMap feature map"
```

---

### Task 8: Spectral Unfolding Feature Map

**Files:**
- Create: `atft/feature_maps/spectral_unfolding.py`
- Modify: `tests/test_feature_maps.py` (append)

- [ ] **Step 1: Write failing tests (append to tests/test_feature_maps.py)**

```python
# Append to: tests/test_feature_maps.py
from atft.feature_maps.spectral_unfolding import SpectralUnfolding


class TestSpectralUnfolding:
    def test_satisfies_protocol(self):
        assert isinstance(SpectralUnfolding(method="rank"), FeatureMap)

    def test_rank_unfolding_mean_gap(self):
        """After rank-based unfolding, mean gap should be ~1."""
        rng = np.random.default_rng(42)
        pts = np.sort(rng.standard_normal(200)).reshape(-1, 1)
        cloud = PointCloud(points=pts.astype(np.float64))
        fm = SpectralUnfolding(method="rank")
        result = fm.transform(cloud)
        gaps = np.diff(result.points[:, 0])
        assert 0.9 < np.mean(gaps) < 1.1

    def test_rank_unfolding_sorted(self):
        """Unfolded spectrum should remain sorted."""
        rng = np.random.default_rng(42)
        pts = np.sort(rng.standard_normal(100)).reshape(-1, 1)
        cloud = PointCloud(points=pts.astype(np.float64))
        fm = SpectralUnfolding(method="rank")
        result = fm.transform(cloud)
        assert np.all(np.diff(result.points[:, 0]) > 0)

    def test_zeta_unfolding_mean_gap(self):
        """Zeta unfolding with known zeros should give mean gap ~1."""
        # Use the first 10 known zeros
        zeros = np.array([
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
        ]).reshape(-1, 1)
        cloud = PointCloud(points=zeros.astype(np.float64))
        fm = SpectralUnfolding(method="zeta")
        result = fm.transform(cloud)
        gaps = np.diff(result.points[:, 0])
        # With only 10 zeros, gap ~ 1 is approximate
        assert 0.5 < np.mean(gaps) < 2.0

    def test_transform_batch(self):
        rng = np.random.default_rng(42)
        clouds = [
            PointCloud(points=np.sort(rng.standard_normal(50)).reshape(-1, 1))
            for _ in range(3)
        ]
        batch = PointCloudBatch(clouds=clouds)
        fm = SpectralUnfolding(method="rank")
        result = fm.transform_batch(batch)
        assert result.batch_size == 3
```

- [ ] **Step 2: Run to verify failure, then implement**

```python
# File: atft/feature_maps/spectral_unfolding.py
"""Spectral unfolding feature map."""
from __future__ import annotations

import numpy as np

from atft.core.types import PointCloud, PointCloudBatch


class SpectralUnfolding:
    """Normalizes spectra to mean gap = 1.

    Methods:
      - "rank": Rank-based unfolding via empirical CDF (for GUE).
      - "zeta": Analytic smooth staircase for Riemann zeta zeros.
    """

    def __init__(self, method: str = "rank"):
        if method not in ("rank", "zeta"):
            raise ValueError(f"Unknown method: {method!r}. Use 'rank' or 'zeta'.")
        self._method = method

    def transform(self, cloud: PointCloud) -> PointCloud:
        pts = cloud.points[:, 0].copy()

        if self._method == "rank":
            unfolded = self._rank_unfold(pts)
        elif self._method == "zeta":
            unfolded = self._zeta_unfold(pts)

        return PointCloud(
            points=unfolded.reshape(-1, 1),
            metadata={**cloud.metadata, "unfolding": self._method},
        )

    def transform_batch(self, batch: PointCloudBatch) -> PointCloudBatch:
        return PointCloudBatch(
            clouds=[self.transform(c) for c in batch.clouds]
        )

    @staticmethod
    def _rank_unfold(pts: np.ndarray) -> np.ndarray:
        """Rank-based unfolding: x_i = rank(pts_i) / N * N = rank."""
        n = len(pts)
        sorted_idx = np.argsort(pts)
        ranks = np.empty(n, dtype=np.float64)
        ranks[sorted_idx] = np.arange(n, dtype=np.float64)
        # Scale so mean gap = 1: positions go from 0 to N-1
        return ranks

    @staticmethod
    def _zeta_unfold(gamma: np.ndarray) -> np.ndarray:
        """Unfold zeta zeros via the smooth staircase function.

        N_smooth(T) = (T/(2*pi)) * ln(T/(2*pi*e)) + 7/8
        """
        two_pi = 2.0 * np.pi
        n_smooth = (gamma / two_pi) * np.log(gamma / (two_pi * np.e)) + 7.0 / 8.0
        return n_smooth
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_feature_maps.py -v`
Expected: All 8 tests PASS

- [ ] **Step 4: Commit**

```bash
git add atft/feature_maps/spectral_unfolding.py tests/test_feature_maps.py
git commit -m "feat: implement SpectralUnfolding with rank and zeta methods"
```

---

## Chunk 3: Topology & Analysis

### Task 9: Analytical H_0

**Files:**
- Create: `atft/topology/analytical_h0.py`
- Create: `tests/test_analytical_h0.py`

- [ ] **Step 1: Write failing tests**

```python
# File: tests/test_analytical_h0.py
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
        """Points [1,2,4,7,11] have gaps [1,2,3,4].
        Persistence diagram: (0,1), (0,2), (0,3), (0,4), (0,inf)
        """
        cloud = PointCloud(points=simple_1d_points)
        ph = AnalyticalH0()
        pd = ph.compute(cloud)
        pairs = pd.degree(0)
        # 4 finite features + 1 immortal
        assert pairs.shape[0] == 5
        # All births at 0
        np.testing.assert_array_equal(pairs[:, 0], 0.0)
        # Finite deaths are the gaps, sorted ascending
        finite_deaths = pairs[pairs[:, 1] != np.inf, 1]
        np.testing.assert_array_almost_equal(
            np.sort(finite_deaths), [1.0, 2.0, 3.0, 4.0]
        )
        # One immortal feature
        assert np.sum(pairs[:, 1] == np.inf) == 1

    def test_uniform_gaps(self, uniform_1d_points):
        """10 uniform points have 9 gaps all = 1.0."""
        cloud = PointCloud(points=uniform_1d_points)
        pd = AnalyticalH0().compute(cloud)
        finite = pd.degree(0)
        finite = finite[finite[:, 1] != np.inf]
        assert finite.shape[0] == 9
        np.testing.assert_array_almost_equal(finite[:, 1], 1.0)

    def test_single_point(self):
        """Single point: only the immortal feature."""
        cloud = PointCloud(points=np.array([[5.0]]))
        pd = AnalyticalH0().compute(cloud)
        pairs = pd.degree(0)
        assert pairs.shape[0] == 1
        assert pairs[0, 1] == np.inf

    def test_two_points(self):
        """Two points at distance 3: one finite (0,3), one immortal."""
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
        """With epsilon_max=2.5, only gaps <= 2.5 appear."""
        cloud = PointCloud(points=simple_1d_points)
        pd = AnalyticalH0().compute(cloud, epsilon_max=2.5)
        finite = pd.degree(0)
        finite = finite[finite[:, 1] != np.inf]
        assert np.all(finite[:, 1] <= 2.5)

    def test_compute_batch(self):
        c1 = PointCloud(points=np.array([[1.0], [3.0], [6.0]]))
        c2 = PointCloud(points=np.array([[0.0], [5.0]]))
        batch = PointCloudBatch(clouds=[c1, c2])
        results = AnalyticalH0().compute_batch(batch)
        assert len(results) == 2
        assert results[0].degree(0).shape[0] == 3  # 2 finite + 1 immortal
        assert results[1].degree(0).shape[0] == 2  # 1 finite + 1 immortal

    def test_unsorted_input_handled(self):
        """Input doesn't need to be sorted — the operator sorts internally."""
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
```

- [ ] **Step 2: Run to verify failure, then implement**

```python
# File: atft/topology/analytical_h0.py
"""Analytical H_0 persistent homology for 1D point clouds."""
from __future__ import annotations

import numpy as np

from atft.core.types import PersistenceDiagram, PointCloud, PointCloudBatch


class AnalyticalH0:
    """Exact H_0 persistence for 1D point clouds via sorted gaps.

    Exploits the fact that for points on a line, H_0 persistence
    is completely determined by the sorted gap sequence.
    Uses the diameter convention: death at g_i (not g_i/2).
    Complexity: O(N log N).
    """

    def compute(
        self,
        cloud: PointCloud,
        max_degree: int = 0,
        epsilon_max: float | None = None,
    ) -> PersistenceDiagram:
        if cloud.dimension != 1:
            raise ValueError(
                f"AnalyticalH0 only supports 1D point clouds, got {cloud.dimension}D"
            )
        if max_degree > 0:
            raise ValueError(
                "AnalyticalH0 only computes H_0. Use RipserPH for higher degrees."
            )

        sorted_pts = np.sort(cloud.points[:, 0])
        gaps = np.diff(sorted_pts)

        births = np.zeros(len(gaps), dtype=np.float64)
        deaths = gaps.astype(np.float64)

        if epsilon_max is not None:
            mask = deaths <= epsilon_max
            births = births[mask]
            deaths = deaths[mask]

        # Add immortal feature
        births = np.append(births, 0.0)
        deaths = np.append(deaths, np.inf)

        diagram = np.column_stack([births, deaths])
        return PersistenceDiagram(
            diagrams={0: diagram},
            metadata={"method": "analytical_h0", "n_points": cloud.n_points},
        )

    def compute_batch(
        self,
        batch: PointCloudBatch,
        max_degree: int = 0,
        epsilon_max: float | None = None,
    ) -> list[PersistenceDiagram]:
        return [
            self.compute(c, max_degree, epsilon_max) for c in batch.clouds
        ]
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_analytical_h0.py -v`
Expected: All 11 tests PASS

- [ ] **Step 4: Commit**

```bash
git add atft/topology/analytical_h0.py tests/test_analytical_h0.py
git commit -m "feat: implement AnalyticalH0 with O(N log N) sorted-gap shortcut"
```

---

### Task 10: Evolution Curve Computer

**Files:**
- Create: `atft/analysis/evolution_curves.py`
- Create: `tests/test_evolution_curves.py`

- [ ] **Step 1: Write failing tests**

```python
# File: tests/test_evolution_curves.py
"""Tests for evolution curve computation."""
import numpy as np
import pytest

from atft.analysis.evolution_curves import EvolutionCurveComputer
from atft.core.types import CurveType, PersistenceDiagram


def _make_pd(gaps):
    """Helper: create a PersistenceDiagram from gap values."""
    births = np.zeros(len(gaps), dtype=np.float64)
    deaths = np.array(gaps, dtype=np.float64)
    births = np.append(births, 0.0)
    deaths = np.append(deaths, np.inf)
    return PersistenceDiagram(diagrams={0: np.column_stack([births, deaths])})


class TestBettiCurve:
    def test_starts_at_n(self):
        """beta_0(0) = N (all points isolated)."""
        pd = _make_pd([1.0, 2.0, 3.0, 4.0])  # 4 gaps -> 5 points
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        betti = curves.betti[0]
        assert betti.values[0] == 5.0  # N = 4 finite + 1 immortal

    def test_ends_at_one(self):
        """beta_0(eps_max) = 1 (one connected component)."""
        pd = _make_pd([1.0, 2.0, 3.0])
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        betti = curves.betti[0]
        assert betti.values[-1] == 1.0

    def test_monotonically_decreasing(self):
        pd = _make_pd([1.0, 2.0, 3.0, 4.0])
        computer = EvolutionCurveComputer(n_steps=200)
        curves = computer.compute(pd, degree=0)
        betti = curves.betti[0]
        assert np.all(np.diff(betti.values) <= 0)

    def test_known_values(self):
        """Gaps [1, 3]: at eps=0.5, beta=3; at eps=2, beta=2; at eps=4, beta=1."""
        pd = _make_pd([1.0, 3.0])
        computer = EvolutionCurveComputer(n_steps=1000)
        curves = computer.compute(pd, degree=0)
        betti = curves.betti[0]
        eps = curves.betti[0].epsilon_grid
        # Find closest indices
        idx_05 = np.argmin(np.abs(eps - 0.5))
        idx_2 = np.argmin(np.abs(eps - 2.0))
        assert betti.values[idx_05] == 3.0
        assert betti.values[idx_2] == 2.0


class TestGiniCurve:
    def test_uniform_gaps_gini_low(self):
        """All finite gaps equal -> Gini should be low (not exactly 0 due to immortal capping)."""
        pd = _make_pd([1.0, 1.0, 1.0, 1.0])
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        gini = curves.gini[0]
        # At eps ~0, all 5 features alive. 4 have lifetime=1.0,
        # immortal has capped lifetime=eps_max. Gini > 0 but
        # should be relatively low since 4/5 features are equal.
        assert gini.values[0] < 0.5
        assert gini.values[0] >= 0.0

    def test_empty_persistence_diagram(self):
        """Empty PD should produce all-zero curves without errors."""
        pd = PersistenceDiagram(diagrams={})
        computer = EvolutionCurveComputer(n_steps=50)
        curves = computer.compute(pd, degree=0)
        assert np.all(curves.betti[0].values == 0)
        assert np.all(curves.gini[0].values == 0)

    def test_gini_bounded(self):
        """Gini should always be in [0, 1]."""
        pd = _make_pd([0.1, 0.5, 1.0, 5.0])
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        gini = curves.gini[0]
        assert np.all(gini.values >= 0.0)
        assert np.all(gini.values <= 1.0)

    def test_gini_edge_case_single_feature(self):
        """When only immortal feature remains, Gini = 0."""
        pd = _make_pd([1.0])
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        gini = curves.gini[0]
        # Near the end, only 1 feature alive -> G = 0
        assert gini.values[-1] == 0.0


class TestPersistenceCurve:
    def test_starts_positive(self):
        pd = _make_pd([1.0, 2.0, 3.0])
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        pers = curves.persistence[0]
        assert pers.values[0] > 0

    def test_curve_type_set(self):
        pd = _make_pd([1.0])
        curves = EvolutionCurveComputer(n_steps=10).compute(pd, degree=0)
        assert curves.betti[0].curve_type == CurveType.BETTI
        assert curves.gini[0].curve_type == CurveType.GINI
        assert curves.persistence[0].curve_type == CurveType.PERSISTENCE
```

- [ ] **Step 2: Run to verify failure, then implement**

```python
# File: atft/analysis/evolution_curves.py
"""Evolution curve computation: Betti, Gini, and Persistence curves."""
from __future__ import annotations

import numpy as np

from atft.core.types import (
    CurveType,
    EvolutionCurve,
    EvolutionCurveSet,
    PersistenceDiagram,
)


class EvolutionCurveComputer:
    """Computes topological evolution curves from persistence diagrams.

    Samples Betti, Gini, and Persistence curves on a uniform epsilon grid.
    """

    def __init__(self, n_steps: int = 1000):
        self.n_steps = n_steps

    def compute(
        self,
        pd: PersistenceDiagram,
        degree: int = 0,
        epsilon_max: float | None = None,
    ) -> EvolutionCurveSet:
        pairs = pd.degree(degree)
        if len(pairs) == 0:
            eps_grid = np.linspace(0, 1, self.n_steps)
            empty = EvolutionCurve(eps_grid, np.zeros(self.n_steps), CurveType.BETTI, degree)
            return EvolutionCurveSet(
                betti={degree: empty},
                gini={degree: EvolutionCurve(eps_grid, np.zeros(self.n_steps), CurveType.GINI, degree)},
                persistence={degree: EvolutionCurve(eps_grid, np.zeros(self.n_steps), CurveType.PERSISTENCE, degree)},
            )

        births = pairs[:, 0]
        deaths = pairs[:, 1].copy()

        # Determine epsilon_max from finite deaths
        finite_deaths = deaths[np.isfinite(deaths)]
        if epsilon_max is None:
            if len(finite_deaths) > 0:
                epsilon_max = 1.1 * np.max(finite_deaths)
            else:
                epsilon_max = 1.0

        # Cap immortal features at epsilon_max for Gini/Persistence
        capped_deaths = deaths.copy()
        capped_deaths[~np.isfinite(capped_deaths)] = epsilon_max
        lifetimes = capped_deaths - births

        eps_grid = np.linspace(0, epsilon_max, self.n_steps)

        betti_vals = np.empty(self.n_steps, dtype=np.float64)
        gini_vals = np.empty(self.n_steps, dtype=np.float64)
        pers_vals = np.empty(self.n_steps, dtype=np.float64)

        for i, eps in enumerate(eps_grid):
            alive = (births <= eps) & (deaths > eps)
            n_alive = np.sum(alive)
            betti_vals[i] = n_alive

            alive_lifetimes = lifetimes[alive]
            gini_vals[i] = self._gini(alive_lifetimes)
            pers_vals[i] = np.sum(alive_lifetimes)

        return EvolutionCurveSet(
            betti={degree: EvolutionCurve(eps_grid, betti_vals, CurveType.BETTI, degree)},
            gini={degree: EvolutionCurve(eps_grid, gini_vals, CurveType.GINI, degree)},
            persistence={degree: EvolutionCurve(eps_grid, pers_vals, CurveType.PERSISTENCE, degree)},
        )

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        """Gini coefficient using the 1-indexed sorted formula."""
        n = len(values)
        if n <= 1:
            return 0.0
        total = np.sum(values)
        if total == 0.0:
            return 0.0
        sorted_v = np.sort(values)
        index = np.arange(1, n + 1, dtype=np.float64)
        return float(
            (2.0 * np.sum(index * sorted_v)) / (n * total) - (n + 1.0) / n
        )
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_evolution_curves.py -v`
Expected: All 9 tests PASS

- [ ] **Step 4: Commit**

```bash
git add atft/analysis/evolution_curves.py tests/test_evolution_curves.py
git commit -m "feat: implement EvolutionCurveComputer with Betti, Gini, Persistence"
```

---

### Task 11: Waypoint Extractor

**Files:**
- Create: `atft/analysis/waypoint_extractor.py`
- Create: `tests/test_waypoint_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
# File: tests/test_waypoint_extractor.py
"""Tests for top-K gap-based waypoint extraction."""
import numpy as np
import pytest

from atft.analysis.evolution_curves import EvolutionCurveComputer
from atft.analysis.waypoint_extractor import WaypointExtractor
from atft.core.types import (
    CurveType,
    EvolutionCurve,
    EvolutionCurveSet,
    PersistenceDiagram,
)


def _make_pd(gaps):
    births = np.zeros(len(gaps), dtype=np.float64)
    deaths = np.array(gaps, dtype=np.float64)
    births = np.append(births, 0.0)
    deaths = np.append(deaths, np.inf)
    return PersistenceDiagram(diagrams={0: np.column_stack([births, deaths])})


class TestWaypointExtractor:
    def test_top_k_selects_largest_gaps(self):
        """Gaps [1, 2, 3, 4]: top-2 should be 3.0 and 4.0."""
        pd = _make_pd([1.0, 2.0, 3.0, 4.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        ext = WaypointExtractor(k_waypoints=2)
        sig = ext.extract(pd, curves, degree=0)
        assert sig.n_waypoints == 2
        np.testing.assert_array_almost_equal(
            np.sort(sig.waypoint_scales), [3.0, 4.0]
        )

    def test_vector_dimension(self):
        pd = _make_pd([1.0, 2.0, 3.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=2).extract(pd, curves, degree=0)
        assert sig.vector_dimension == 7  # 2*2 + 3

    def test_onset_scale_is_smallest_gap(self):
        """For H_0, onset = min(gaps)."""
        pd = _make_pd([0.5, 2.0, 3.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=2).extract(pd, curves, degree=0)
        assert abs(sig.onset_scale - 0.5) < 1e-10

    def test_zero_padding_when_few_gaps(self):
        """With 1 gap and K=2, second waypoint is zero-padded."""
        pd = _make_pd([3.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=2).extract(pd, curves, degree=0)
        assert sig.n_waypoints == 2
        assert sig.waypoint_scales[1] == 0.0

    def test_waypoints_sorted_by_position(self):
        """Waypoints should be sorted ascending by epsilon position."""
        pd = _make_pd([5.0, 1.0, 3.0, 2.0, 4.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=3).extract(pd, curves, degree=0)
        assert np.all(np.diff(sig.waypoint_scales) >= 0)

    def test_gini_at_onset_is_float(self):
        pd = _make_pd([1.0, 2.0, 3.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=2).extract(pd, curves, degree=0)
        assert isinstance(sig.gini_at_onset, float)
        assert 0.0 <= sig.gini_at_onset <= 1.0

    def test_empty_persistence_returns_zero_signature(self):
        """PD with only immortal feature should return zero signature."""
        # Single point: only (0, inf)
        pd = PersistenceDiagram(diagrams={0: np.array([[0.0, np.inf]])})
        eps = np.linspace(0, 1, 50)
        dummy_curves = EvolutionCurveSet(
            betti={0: EvolutionCurve(eps, np.ones(50), CurveType.BETTI, 0)},
            gini={0: EvolutionCurve(eps, np.zeros(50), CurveType.GINI, 0)},
            persistence={0: EvolutionCurve(eps, np.ones(50), CurveType.PERSISTENCE, 0)},
        )
        sig = WaypointExtractor(k_waypoints=2).extract(pd, dummy_curves, degree=0)
        assert sig.onset_scale == 0.0
        np.testing.assert_array_equal(sig.waypoint_scales, [0.0, 0.0])

    def test_as_vector_correct_shape(self):
        pd = _make_pd([1.0, 2.0, 3.0, 4.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=2).extract(pd, curves, degree=0)
        vec = sig.as_vector()
        assert vec.shape == (7,)
        assert vec.dtype == np.float64
```

- [ ] **Step 2: Run to verify failure, then implement**

```python
# File: atft/analysis/waypoint_extractor.py
"""Top-K gap-based waypoint extraction."""
from __future__ import annotations

import numpy as np

from atft.core.types import (
    EvolutionCurveSet,
    PersistenceDiagram,
    WaypointSignature,
)


class WaypointExtractor:
    """Extracts the waypoint signature W_0(C) from persistence data.

    Uses gap-based extraction (not numerical derivatives) to avoid
    differentiation artifacts on the step-function Betti curve.
    """

    def __init__(self, k_waypoints: int = 2):
        self.k_waypoints = k_waypoints

    def extract(
        self,
        pd: PersistenceDiagram,
        curves: EvolutionCurveSet,
        degree: int = 0,
    ) -> WaypointSignature:
        pairs = pd.degree(degree)
        finite_mask = np.isfinite(pairs[:, 1])
        finite_deaths = pairs[finite_mask, 1]

        if len(finite_deaths) == 0:
            return self._empty_signature()

        # Onset scale: smallest finite death (smallest gap)
        onset_scale = float(np.min(finite_deaths))

        # Top-K: largest gaps sorted descending by magnitude
        sorted_desc = np.sort(finite_deaths)[::-1]
        k = min(self.k_waypoints, len(sorted_desc))
        top_k = sorted_desc[:k]

        # Zero-pad if fewer than K gaps
        if k < self.k_waypoints:
            top_k = np.concatenate([
                top_k,
                np.zeros(self.k_waypoints - k, dtype=np.float64),
            ])

        # Sort by position (ascending epsilon)
        top_k = np.sort(top_k)

        # Topological derivatives: for each waypoint gap, the local
        # Betti curve drop. For a single gap, the drop is always -1.
        # For the purpose of the signature, we store the gap magnitudes
        # as a proxy for the derivative (larger gap = sharper transition).
        # Store gap magnitudes as proxy for topological derivative.
        # In 1D H_0, every merging event has literal delta = -1, which
        # would create zero-variance columns in the covariance matrix.
        # Gap magnitude is a more informative proxy (per spec Section 2.6).
        topo_derivs = -top_k.copy()

        # Gini at onset
        betti_curve = curves.betti[degree]
        gini_curve = curves.gini[degree]
        eps = betti_curve.epsilon_grid

        onset_idx = np.argmin(np.abs(eps - onset_scale))
        gini_at_onset = float(gini_curve.values[onset_idx])

        # Gini derivative at onset (finite difference)
        gini_deriv = np.gradient(gini_curve.values, eps)
        gini_deriv_at_onset = float(gini_deriv[onset_idx])

        return WaypointSignature(
            onset_scale=onset_scale,
            waypoint_scales=top_k,
            topo_derivatives=topo_derivs,
            gini_at_onset=gini_at_onset,
            gini_derivative_at_onset=gini_deriv_at_onset,
        )

    def _empty_signature(self) -> WaypointSignature:
        return WaypointSignature(
            onset_scale=0.0,
            waypoint_scales=np.zeros(self.k_waypoints, dtype=np.float64),
            topo_derivatives=np.zeros(self.k_waypoints, dtype=np.float64),
            gini_at_onset=0.0,
            gini_derivative_at_onset=0.0,
        )
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_waypoint_extractor.py -v`
Expected: All 7 tests PASS

- [ ] **Step 4: Commit**

```bash
git add atft/analysis/waypoint_extractor.py tests/test_waypoint_extractor.py
git commit -m "feat: implement WaypointExtractor with gap-based top-K extraction"
```

---

### Task 12: Statistical Validator

**Files:**
- Create: `atft/analysis/statistical_tests.py`
- Create: `tests/test_statistical_tests.py`

- [ ] **Step 1: Write failing tests**

```python
# File: tests/test_statistical_tests.py
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
        """A sample from the ensemble should have small Mahalanobis distance."""
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

        # Test with a sample from the same distribution
        target_sig = _make_signature(1.0, 3.0, 5.0, -3.0, -5.0, 0.4, 0.01)
        target_curves = _make_curves(eps, np.linspace(10, 1, 20), np.linspace(0, 0.5, 20))
        result = validator.validate(target_sig, target_curves)

        assert result.p_value > 0.01  # Should not be rejected

    def test_outlier_large_distance(self):
        """A point far from the ensemble should have large Mahalanobis distance."""
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

        # Outlier: 10 standard deviations away
        target_sig = _make_signature(5.0, 10.0, 20.0, -10.0, -15.0, 0.9, 0.5)
        target_curves = _make_curves(eps, np.linspace(10, 1, 20), np.linspace(0, 0.5, 20))
        result = validator.validate(target_sig, target_curves)

        assert result.p_value < 0.001
        assert result.mahalanobis_distance > 5.0

    def test_within_band_true(self):
        """Target inside ensemble band -> within_confidence_band = True."""
        eps = np.linspace(0, 5, 10)
        mean_betti = np.linspace(10, 1, 10)
        mean_gini = np.linspace(0, 0.5, 10)

        rng = np.random.default_rng(42)
        sigs = [_make_signature(1, 3, 5, -3, -5, 0.4, 0.01) for _ in range(50)]
        curves = [
            _make_curves(eps, mean_betti + rng.normal(0, 0.5, 10),
                         np.clip(mean_gini + rng.normal(0, 0.05, 10), 0, 1))
            for _ in range(50)
        ]

        validator = StatisticalValidator(confidence_level=0.99)
        validator.fit_ensemble(sigs, curves)

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
```

- [ ] **Step 2: Run to verify failure, then implement**

```python
# File: atft/analysis/statistical_tests.py
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

    def fit_ensemble(
        self,
        signatures: list[WaypointSignature],
        curves: list[EvolutionCurveSet],
    ) -> None:
        """Fit the baseline distribution from M ensemble members."""
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

    def validate(
        self,
        target_signature: WaypointSignature,
        target_curves: EvolutionCurveSet,
        degree: int = 0,
    ) -> ValidationResult:
        """Run both prongs against a target configuration."""
        # Prong 1: Functional envelope
        l2_betti, within_betti = self._check_envelope(
            target_curves, CurveType.BETTI, degree
        )
        l2_gini, within_gini = self._check_envelope(
            target_curves, CurveType.GINI, degree
        )
        within_band = within_betti and within_gini

        # Prong 2: Mahalanobis distance
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

    def _check_envelope(
        self,
        target_curves: EvolutionCurveSet,
        curve_type: CurveType,
        degree: int,
    ) -> tuple[float, bool]:
        """Check if target curve is within ensemble confidence band."""
        lookup = {
            CurveType.BETTI: "betti",
            CurveType.GINI: "gini",
            CurveType.PERSISTENCE: "persistence",
        }
        attr = lookup[curve_type]
        target_curve = getattr(target_curves, attr)[degree]

        ensemble_vals = np.array([
            np.interp(
                target_curve.epsilon_grid,
                getattr(c, attr)[degree].epsilon_grid,
                getattr(c, attr)[degree].values,
            )
            for c in self._ensemble_curves
        ])

        mean_curve = np.mean(ensemble_vals, axis=0)

        alpha = 1.0 - self.confidence_level
        lower = np.percentile(ensemble_vals, 100 * alpha / 2, axis=0)
        upper = np.percentile(ensemble_vals, 100 * (1 - alpha / 2), axis=0)

        d_eps = target_curve.epsilon_grid[1] - target_curve.epsilon_grid[0]
        l2 = float(
            np.sqrt(np.sum((target_curve.values - mean_curve) ** 2) * d_eps)
        )

        within = bool(
            np.all(
                (target_curve.values >= lower) & (target_curve.values <= upper)
            )
        )

        return l2, within
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_statistical_tests.py -v`
Expected: All 4 tests PASS

- [ ] **Step 4: Commit**

```bash
git add atft/analysis/statistical_tests.py tests/test_statistical_tests.py
git commit -m "feat: implement StatisticalValidator with Mahalanobis and envelope"
```

---

## Chunk 4: Integration & Visualization

### Task 13: HDF5 Cache

**Files:**
- Create: `atft/io/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# File: tests/test_cache.py
"""Tests for HDF5 cache serialization."""
import numpy as np
import pytest

from atft.io.cache import load_persistence_diagram, save_persistence_diagram
from atft.core.types import PersistenceDiagram


class TestPersistenceDiagramCache:
    def test_round_trip(self, tmp_path):
        diag = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, np.inf]], dtype=np.float64)
        pd = PersistenceDiagram(
            diagrams={0: diag},
            metadata={"method": "analytical_h0", "n_points": 4},
        )
        path = tmp_path / "test.h5"
        save_persistence_diagram(pd, path)
        loaded = load_persistence_diagram(path)

        np.testing.assert_array_equal(loaded.degree(0), pd.degree(0))
        assert loaded.metadata["method"] == "analytical_h0"

    def test_empty_diagram(self, tmp_path):
        pd = PersistenceDiagram(diagrams={}, metadata={})
        path = tmp_path / "empty.h5"
        save_persistence_diagram(pd, path)
        loaded = load_persistence_diagram(path)
        assert loaded.max_degree == -1

    def test_multi_degree(self, tmp_path):
        pd = PersistenceDiagram(diagrams={
            0: np.array([[0.0, 1.0]], dtype=np.float64),
            1: np.array([[0.5, 2.0]], dtype=np.float64),
        })
        path = tmp_path / "multi.h5"
        save_persistence_diagram(pd, path)
        loaded = load_persistence_diagram(path)
        assert loaded.max_degree == 1
        np.testing.assert_array_equal(loaded.degree(1), pd.degree(1))
```

- [ ] **Step 2: Run to verify failure, then implement**

```python
# File: atft/io/cache.py
"""HDF5 serialization for intermediate results."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from atft.core.types import PersistenceDiagram


def save_persistence_diagram(pd: PersistenceDiagram, path: Path) -> None:
    """Save a PersistenceDiagram to HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for degree, diagram in pd.diagrams.items():
            f.create_dataset(f"degree_{degree}", data=diagram, compression="gzip")
        for k, v in pd.metadata.items():
            try:
                f.attrs[k] = v
            except TypeError:
                f.attrs[k] = str(v)


def load_persistence_diagram(path: Path) -> PersistenceDiagram:
    """Load a PersistenceDiagram from HDF5."""
    with h5py.File(path, "r") as f:
        diagrams = {}
        for key in f.keys():
            if key.startswith("degree_"):
                degree = int(key.split("_")[1])
                diagrams[degree] = f[key][:].astype(np.float64)
        metadata = {k: v for k, v in f.attrs.items()}
    return PersistenceDiagram(diagrams=diagrams, metadata=dict(metadata))
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_cache.py -v`
Expected: All 3 tests PASS

- [ ] **Step 4: Commit**

```bash
git add atft/io/cache.py tests/test_cache.py
git commit -m "feat: implement HDF5 cache for PersistenceDiagram serialization"
```

---

### Task 14: Visualization

**Files:**
- Create: `atft/visualization/plots.py`

- [ ] **Step 1: Implement the three-panel figure (no TDD — visual output)**

```python
# File: atft/visualization/plots.py
"""Three-panel publication figure for Phase 1 results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from atft.core.types import (
    EvolutionCurveSet,
    ValidationResult,
    WaypointSignature,
)


def plot_phase1_results(
    zeta_curves: EvolutionCurveSet,
    gue_curves: list[EvolutionCurveSet],
    poisson_curves: list[EvolutionCurveSet],
    zeta_sig: WaypointSignature,
    gue_sigs: list[WaypointSignature],
    poisson_sig: WaypointSignature,
    zeta_result: ValidationResult,
    poisson_result: ValidationResult,
    save_path: Path | None = None,
    degree: int = 0,
) -> plt.Figure:
    """Generate the three-panel publication figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel A: Betti curves ---
    ax = axes[0]
    _plot_envelope(ax, gue_curves, "betti", degree, label="GUE 99% band")
    zb = zeta_curves.betti[degree]
    ax.plot(zb.epsilon_grid, zb.values, "b-", linewidth=2, label="Zeta zeros")
    pb = poisson_curves[0].betti[degree]
    ax.plot(pb.epsilon_grid, pb.values, "r--", linewidth=1.5, label="Poisson")
    ax.set_xlabel("ε (filtration scale)")
    ax.set_ylabel("β₀(ε)")
    ax.set_title("Panel A: Betti Curves")
    ax.legend()

    # --- Panel B: Gini trajectories ---
    ax = axes[1]
    _plot_envelope(ax, gue_curves, "gini", degree, label="GUE 99% band")
    zg = zeta_curves.gini[degree]
    ax.plot(zg.epsilon_grid, zg.values, "b-", linewidth=2, label="Zeta zeros")
    pg = poisson_curves[0].gini[degree]
    ax.plot(pg.epsilon_grid, pg.values, "r--", linewidth=1.5, label="Poisson")
    ax.set_xlabel("ε (filtration scale)")
    ax.set_ylabel("G₀(ε)")
    ax.set_title("Panel B: Gini Trajectories")
    ax.legend()

    # --- Panel C: PCA of waypoint space ---
    ax = axes[2]
    gue_vecs = np.array([s.as_vector() for s in gue_sigs])
    pca = PCA(n_components=2)
    gue_proj = pca.fit_transform(gue_vecs)

    zeta_proj = pca.transform(zeta_sig.as_vector().reshape(1, -1))
    poisson_proj = pca.transform(poisson_sig.as_vector().reshape(1, -1))

    ax.scatter(gue_proj[:, 0], gue_proj[:, 1], c="grey", alpha=0.3, s=10, label="GUE")
    ax.scatter(zeta_proj[0, 0], zeta_proj[0, 1], c="blue", marker="*", s=200, label="Zeta", zorder=5)
    ax.scatter(poisson_proj[0, 0], poisson_proj[0, 1], c="red", marker="x", s=150, label="Poisson", zorder=5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Panel C: Waypoint Space (PCA)")
    ax.legend()

    # Footer
    fig.suptitle(
        f"Zeta: D_M={zeta_result.mahalanobis_distance:.2f} (p={zeta_result.p_value:.4f})  |  "
        f"Poisson: D_M={poisson_result.mahalanobis_distance:.2f} (p={poisson_result.p_value:.4f})",
        y=0.02, fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _plot_envelope(ax, ensemble_curves, attr, degree, label="99% band"):
    """Plot the ensemble confidence band."""
    all_vals = []
    ref_eps = getattr(ensemble_curves[0], attr)[degree].epsilon_grid
    for c in ensemble_curves:
        curve = getattr(c, attr)[degree]
        vals = np.interp(ref_eps, curve.epsilon_grid, curve.values)
        all_vals.append(vals)
    all_vals = np.array(all_vals)
    lower = np.percentile(all_vals, 0.5, axis=0)
    upper = np.percentile(all_vals, 99.5, axis=0)
    mean = np.mean(all_vals, axis=0)
    ax.fill_between(ref_eps, lower, upper, color="grey", alpha=0.3, label=label)
    ax.plot(ref_eps, mean, "k-", linewidth=0.5, alpha=0.5)
```

Note: This imports `sklearn.decomposition.PCA`. Add `scikit-learn>=1.3` to the dev dependencies in `pyproject.toml`.

- [ ] **Step 2: Commit**

```bash
git add atft/visualization/plots.py
git commit -m "feat: implement three-panel publication figure"
```

---

### Task 15: Phase 1 Experiment Orchestrator

**Files:**
- Create: `atft/experiments/phase1_benchmark.py`

- [ ] **Step 1: Implement the orchestrator**

```python
# File: atft/experiments/phase1_benchmark.py
"""Phase 1 Experiment: Zeta vs GUE vs Poisson topological benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from atft.analysis.evolution_curves import EvolutionCurveComputer
from atft.analysis.statistical_tests import StatisticalValidator
from atft.analysis.waypoint_extractor import WaypointExtractor
from atft.core.types import (
    EvolutionCurveSet,
    ValidationResult,
    WaypointSignature,
)
from atft.feature_maps.identity import IdentityMap
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.gue import GUESource
from atft.sources.poisson import PoissonSource
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.analytical_h0 import AnalyticalH0


@dataclass
class Phase1Config:
    n_points: int = 10_000
    ensemble_size: int = 1_000
    k_waypoints: int = 2
    n_epsilon_steps: int = 1_000
    confidence_level: float = 0.99
    zeta_data_path: Path = Path("data/odlyzko_zeros.txt")
    cache_dir: Path = Path("cache/")
    device: str = "cuda"
    seed: int = 42


@dataclass
class Phase1Results:
    zeta_validation: ValidationResult
    poisson_validation: ValidationResult
    zeta_curves: EvolutionCurveSet
    gue_curves: list[EvolutionCurveSet]
    poisson_curves: list[EvolutionCurveSet]
    zeta_signature: WaypointSignature
    gue_signatures: list[WaypointSignature]
    poisson_signatures: list[WaypointSignature]


class Phase1Experiment:
    """Orchestrates the Zeta vs GUE vs Poisson benchmark."""

    def __init__(self, config: Phase1Config):
        self.config = config
        self.ph = AnalyticalH0()
        self.curve_computer = EvolutionCurveComputer(
            n_steps=config.n_epsilon_steps
        )
        self.waypoint_extractor = WaypointExtractor(
            k_waypoints=config.k_waypoints
        )
        self.validator = StatisticalValidator(
            confidence_level=config.confidence_level
        )

    def run(self) -> Phase1Results:
        """Execute the full Phase 1 experiment."""
        print("Step 1/10: Loading zeta zeros...")
        zeta_src = ZetaZerosSource(self.config.zeta_data_path)
        zeta_cloud = zeta_src.generate(self.config.n_points)

        print("Step 2/10: Generating GUE ensemble...")
        gue_src = GUESource(seed=self.config.seed)
        gue_batch = gue_src.generate_batch(
            self.config.n_points, self.config.ensemble_size
        )

        print("Step 3/10: Generating Poisson baseline...")
        poisson_src = PoissonSource(seed=self.config.seed + 1)
        poisson_batch = poisson_src.generate_batch(
            self.config.n_points, self.config.ensemble_size
        )

        print("Step 4/10: Unfolding spectra...")
        zeta_unfolded = SpectralUnfolding(method="zeta").transform(zeta_cloud)
        gue_unfolded = SpectralUnfolding(method="rank").transform_batch(gue_batch)
        poisson_unfolded = IdentityMap().transform_batch(poisson_batch)

        print("Step 5/10: Computing persistence diagrams...")
        zeta_pd = self.ph.compute(zeta_unfolded)
        gue_pds = self.ph.compute_batch(gue_unfolded)
        poisson_pds = self.ph.compute_batch(poisson_unfolded)

        print("Step 6/10: Computing evolution curves...")
        zeta_curves = self.curve_computer.compute(zeta_pd)
        gue_curves = [self.curve_computer.compute(pd) for pd in gue_pds]
        poisson_curves = [self.curve_computer.compute(pd) for pd in poisson_pds]

        print("Step 7/10: Extracting waypoint signatures...")
        zeta_sig = self.waypoint_extractor.extract(zeta_pd, zeta_curves)
        gue_sigs = [
            self.waypoint_extractor.extract(pd, c)
            for pd, c in zip(gue_pds, gue_curves)
        ]
        poisson_sigs = [
            self.waypoint_extractor.extract(pd, c)
            for pd, c in zip(poisson_pds, poisson_curves)
        ]

        print("Step 8/10: Fitting statistical validator on GUE ensemble...")
        self.validator.fit_ensemble(gue_sigs, gue_curves)

        print("Step 9/10: Validating zeta zeros against GUE...")
        zeta_result = self.validator.validate(zeta_sig, zeta_curves)

        print("Step 10/10: Validating Poisson against GUE (negative control)...")
        poisson_result = self.validator.validate(
            poisson_sigs[0], poisson_curves[0]
        )

        self._print_results(zeta_result, poisson_result)

        return Phase1Results(
            zeta_validation=zeta_result,
            poisson_validation=poisson_result,
            zeta_curves=zeta_curves,
            gue_curves=gue_curves,
            poisson_curves=poisson_curves,
            zeta_signature=zeta_sig,
            gue_signatures=gue_sigs,
            poisson_signatures=poisson_sigs,
        )

    @staticmethod
    def _print_results(zeta: ValidationResult, poisson: ValidationResult):
        print("\n" + "=" * 60)
        print("PHASE 1 RESULTS: Zeta vs GUE Topological Benchmark")
        print("=" * 60)
        print(f"\nZeta zeros:")
        print(f"  Mahalanobis distance: {zeta.mahalanobis_distance:.4f}")
        print(f"  p-value:              {zeta.p_value:.6f}")
        print(f"  Within 99% band:      {zeta.within_confidence_band}")
        print(f"  L2 (Betti):           {zeta.l2_distance_betti:.4f}")
        print(f"  L2 (Gini):            {zeta.l2_distance_gini:.4f}")
        print(f"\nPoisson (negative control):")
        print(f"  Mahalanobis distance: {poisson.mahalanobis_distance:.4f}")
        print(f"  p-value:              {poisson.p_value:.6f}")
        print(f"  Within 99% band:      {poisson.within_confidence_band}")
        print("=" * 60)
```

- [ ] **Step 2: Commit**

```bash
git add atft/experiments/phase1_benchmark.py
git commit -m "feat: implement Phase1Experiment orchestrator"
```

---

### Task 16: End-to-End Integration Test

**Files:**
- Create: `tests/test_phase1_integration.py`

- [ ] **Step 1: Write integration test with small N**

```python
# File: tests/test_phase1_integration.py
"""End-to-end integration test for the full ATFT pipeline.

Uses small N (50 points, 10 ensemble members) for speed.
Validates that the pipeline runs end-to-end without errors
and produces sensible results.
"""
import numpy as np
import pytest

from atft.analysis.evolution_curves import EvolutionCurveComputer
from atft.analysis.statistical_tests import StatisticalValidator
from atft.analysis.waypoint_extractor import WaypointExtractor
from atft.core.types import CurveType
from atft.feature_maps.identity import IdentityMap
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.gue import GUESource
from atft.sources.poisson import PoissonSource
from atft.topology.analytical_h0 import AnalyticalH0


N_POINTS = 50
ENSEMBLE_SIZE = 20
K_WAYPOINTS = 2


class TestFullPipeline:
    """End-to-end pipeline test with GUE and Poisson."""

    def test_pipeline_runs_without_errors(self):
        """The full pipeline executes and produces a ValidationResult."""
        ph = AnalyticalH0()
        curves_computer = EvolutionCurveComputer(n_steps=100)
        wp_extractor = WaypointExtractor(k_waypoints=K_WAYPOINTS)
        validator = StatisticalValidator(confidence_level=0.99)

        # Generate GUE ensemble
        gue_src = GUESource(seed=42)
        gue_batch = gue_src.generate_batch(N_POINTS, ENSEMBLE_SIZE)
        gue_unfolded = SpectralUnfolding(method="rank").transform_batch(gue_batch)
        gue_pds = ph.compute_batch(gue_unfolded)
        gue_curves = [curves_computer.compute(pd) for pd in gue_pds]
        gue_sigs = [
            wp_extractor.extract(pd, c) for pd, c in zip(gue_pds, gue_curves)
        ]

        # Generate Poisson baseline
        poisson_src = PoissonSource(seed=43)
        poisson_cloud = poisson_src.generate(N_POINTS)
        poisson_unfolded = IdentityMap().transform(poisson_cloud)
        poisson_pd = ph.compute(poisson_unfolded)
        poisson_curves = curves_computer.compute(poisson_pd)
        poisson_sig = wp_extractor.extract(poisson_pd, poisson_curves)

        # Fit and validate
        validator.fit_ensemble(gue_sigs, gue_curves)

        # Test a GUE member against ensemble (should pass)
        gue_result = validator.validate(gue_sigs[0], gue_curves[0])
        assert gue_result.p_value > 0.01  # GUE member should be inside GUE

        # Test Poisson against GUE (should fail)
        poisson_result = validator.validate(poisson_sig, poisson_curves)
        # With small N, Poisson might not always be rejected,
        # but the Mahalanobis distance should be larger
        assert poisson_result.mahalanobis_distance > gue_result.mahalanobis_distance

    def test_all_signatures_same_dimension(self):
        """All waypoint signatures must have the same vector dimension."""
        ph = AnalyticalH0()
        curves_computer = EvolutionCurveComputer(n_steps=50)
        wp_extractor = WaypointExtractor(k_waypoints=K_WAYPOINTS)

        gue_src = GUESource(seed=42)
        batch = gue_src.generate_batch(N_POINTS, 10)
        unfolded = SpectralUnfolding(method="rank").transform_batch(batch)
        pds = ph.compute_batch(unfolded)
        curveset = [curves_computer.compute(pd) for pd in pds]
        sigs = [wp_extractor.extract(pd, c) for pd, c in zip(pds, curveset)]

        dims = [s.vector_dimension for s in sigs]
        assert len(set(dims)) == 1  # All same dimension
        assert dims[0] == 2 * K_WAYPOINTS + 3

    def test_betti_curve_properties(self):
        """Betti curve must start at N, end at 1, be monotonically decreasing."""
        gue_src = GUESource(seed=42)
        cloud = gue_src.generate(N_POINTS)
        unfolded = SpectralUnfolding(method="rank").transform(cloud)
        pd = AnalyticalH0().compute(unfolded)
        curves = EvolutionCurveComputer(n_steps=200).compute(pd)
        betti = curves.betti[0]

        assert betti.values[0] == N_POINTS
        assert betti.values[-1] == 1.0
        assert np.all(np.diff(betti.values) <= 0)
```

- [ ] **Step 2: Run the integration test**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/test_phase1_integration.py -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Run the full test suite**

Run: `cd C:/Claude/Reimann_Hypothesis && python -m pytest tests/ -v`
Expected: All tests PASS (approximately 55+ tests total)

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase1_integration.py
git commit -m "feat: add end-to-end integration tests for full ATFT pipeline"
```

---

## Post-Implementation

After all tasks are complete:

1. Run the full test suite: `python -m pytest tests/ -v --tb=short`
2. Download Odlyzko's zeta zeros dataset to `data/odlyzko_zeros.txt`
3. Run the Phase 1 experiment: `python -c "from atft.experiments.phase1_benchmark import *; Phase1Experiment(Phase1Config()).run()"`
4. Generate the three-panel figure for publication
