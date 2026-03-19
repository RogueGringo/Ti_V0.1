# Distributed Fire-and-Forget KPM Engine — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a distributed KPM compute engine with a self-contained worker node package and a synthesizer orchestrator, enabling the Chebyshev moment computation to be split across multiple machines via SSH dispatch.

**Architecture:** Two installable packages (`node/` and `synthesizer/`) communicate via CLI args and `.npz` files. The worker re-implements the KPM compute kernel as a self-contained module (no `atft/` dependency). The synthesizer scans devices, benchmarks throughput, plans partitions with VRAM clamping, dispatches via SSH, validates results, merges with dimension-weighted arithmetic, and logs all contributions. Static ghost zones eliminate real-time synchronization.

**Tech Stack:** PyTorch (sparse CSR, GPU SpMV), NumPy, SciPy (sparse), PyYAML, subprocess (SSH/SCP), hashlib (SHA-256)

**Spec:** `docs/superpowers/specs/2026-03-18-distributed-kpm-design.md`

---

## Chunk 1: Protocol + Worker Node Package

### Task 1: Protocol dataclasses

**Files:**
- Create: `atft/distributed/__init__.py`
- Create: `atft/distributed/protocol.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_distributed.py`:

```python
"""Tests for the distributed KPM engine."""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import tempfile
from pathlib import Path


class TestProtocol:
    """Tests for the ContributeResult serialization."""

    def test_contribute_result_roundtrip(self):
        """Save and load a ContributeResult via .npz."""
        from atft.distributed.protocol import ContributeResult
        result = ContributeResult(
            worker_id="test-node",
            partition=(0, 100),
            dim_local=600,
            lam_max_local=10.5,
            raw_traces=np.random.randn(51, 30),
            device_type="cpu",
            compute_time_s=1.23,
        )
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            result.save(f.name)
            loaded = ContributeResult.load(f.name)

        assert loaded.worker_id == "test-node"
        assert loaded.partition == (0, 100)
        assert loaded.dim_local == 600
        npt.assert_allclose(loaded.raw_traces, result.raw_traces)
        assert loaded.checksum == result.checksum

    def test_checksum_detects_corruption(self):
        """Corrupted traces should fail checksum validation."""
        from atft.distributed.protocol import ContributeResult
        result = ContributeResult(
            worker_id="test", partition=(0, 50), dim_local=300,
            lam_max_local=5.0, raw_traces=np.ones((21, 10)),
            device_type="cpu", compute_time_s=0.5,
        )
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            result.save(f.name)
            loaded = ContributeResult.load(f.name)
            loaded.raw_traces[0, 0] = 999.0  # corrupt
            assert not loaded.verify_checksum()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributed.py::TestProtocol -v`
Expected: FAIL — `ImportError: cannot import name 'ContributeResult'`

- [ ] **Step 3: Implement protocol**

Create `atft/distributed/__init__.py` (empty).

Create `atft/distributed/protocol.py`:

```python
"""Protocol dataclasses for distributed KPM communication."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class ContributeResult:
    """Result payload from a worker node.

    Contains raw per-probe Hutchinson traces (NOT averaged) for
    dimension-weighted merge and variance computation by the synthesizer.
    """
    worker_id: str
    partition: tuple[int, int]       # (start_idx, end_idx) owned range
    dim_local: int                    # owned_vertices * K
    lam_max_local: float
    raw_traces: NDArray[np.float64]  # shape [D+1, num_vectors]
    device_type: str
    compute_time_s: float
    checksum: str = field(default="", init=False)

    def __post_init__(self):
        self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        return hashlib.sha256(self.raw_traces.tobytes()).hexdigest()

    def verify_checksum(self) -> bool:
        return self._compute_checksum() == self.checksum

    def save(self, path: str) -> None:
        np.savez(
            path,
            worker_id=np.array([self.worker_id]),
            partition=np.array(self.partition),
            dim_local=np.array([self.dim_local]),
            lam_max_local=np.array([self.lam_max_local]),
            raw_traces=self.raw_traces,
            device_type=np.array([self.device_type]),
            compute_time_s=np.array([self.compute_time_s]),
            checksum=np.array([self.checksum]),
        )

    @classmethod
    def load(cls, path: str) -> ContributeResult:
        data = np.load(path, allow_pickle=False)
        result = cls.__new__(cls)
        result.worker_id = str(data["worker_id"][0])
        result.partition = tuple(data["partition"].tolist())
        result.dim_local = int(data["dim_local"][0])
        result.lam_max_local = float(data["lam_max_local"][0])
        result.raw_traces = data["raw_traces"].astype(np.float64)
        result.device_type = str(data["device_type"][0])
        result.compute_time_s = float(data["compute_time_s"][0])
        result.checksum = str(data["checksum"][0])
        return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_distributed.py::TestProtocol -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add atft/distributed/__init__.py atft/distributed/protocol.py tests/test_distributed.py
git commit -m "feat: add ContributeResult protocol with .npz serialization and SHA-256 checksum"
```

---

### Task 2: Self-contained compute kernel

**Files:**
- Create: `node/src/jtopo_node/__init__.py`
- Create: `node/src/jtopo_node/compute.py`
- Test: `tests/test_distributed.py`

This is the largest task — `compute.py` re-implements the KPM math kernel without any `atft/` imports. It extracts the minimal code from `transport_maps.py`, `base_sheaf_laplacian.py`, and `kpm_sheaf_laplacian.py`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_distributed.py`:

```python
class TestComputeKernel:
    """Tests for the self-contained KPM compute kernel."""

    def test_compute_partition_moments(self):
        """Compute moments on a partition and verify shape."""
        from jtopo_node.compute import compute_partition_moments

        # Create simple sorted zeros
        zeros = np.sort(np.random.default_rng(42).uniform(0, 100, 200))
        result = compute_partition_moments(
            zeros_path=None, zeros_array=zeros,
            start_idx=0, end_idx=200,
            ghost_left=0, ghost_right=0,
            K=3, sigma=0.5, epsilon=3.0,
            degree=20, num_vectors=30,
            lam_max_global=None, seed=42,
            worker_id="test", device="cpu",
        )
        assert result.raw_traces.shape == (21, 30)  # D+1, nv
        assert result.dim_local == 200 * 3
        assert result.worker_id == "test"
        assert result.verify_checksum()

    def test_moments_match_single_node(self):
        """Distributed moments on single partition should match KPMSheafLaplacian."""
        from jtopo_node.compute import compute_partition_moments
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder

        zeros = np.sort(np.random.default_rng(42).uniform(0, 50, 100))
        K, sigma, epsilon, D, nv = 3, 0.5, 3.0, 20, 50

        # Single-node reference
        builder = TransportMapBuilder(K=K, sigma=sigma)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu",
                                 degree=D, num_vectors=nv)
        mu_ref = kpm.compute_moments(epsilon)

        # Distributed kernel (full partition, no ghosts)
        result = compute_partition_moments(
            zeros_path=None, zeros_array=zeros,
            start_idx=0, end_idx=len(zeros),
            ghost_left=0, ghost_right=0,
            K=K, sigma=sigma, epsilon=epsilon,
            degree=D, num_vectors=nv,
            lam_max_global=None, seed=42,
            worker_id="test", device="cpu",
        )
        mu_dist = result.raw_traces.mean(axis=1)

        # Should match within Hutchinson variance (same seed, same data)
        npt.assert_allclose(mu_dist, mu_ref, atol=0.05)

    def test_ghost_zones_excluded_from_trace(self):
        """Ghost vertices should not contribute to Hutchinson traces."""
        from jtopo_node.compute import compute_partition_moments

        zeros = np.sort(np.random.default_rng(42).uniform(0, 50, 100))

        # Partition [20:80] with 5 ghosts on each side
        result = compute_partition_moments(
            zeros_path=None, zeros_array=zeros,
            start_idx=20, end_idx=80,
            ghost_left=5, ghost_right=5,
            K=3, sigma=0.5, epsilon=3.0,
            degree=10, num_vectors=20,
            lam_max_global=None, seed=42,
            worker_id="test", device="cpu",
        )
        # dim_local should be (80-20)*3 = 180, NOT (80-20+10)*3
        assert result.dim_local == 60 * 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributed.py::TestComputeKernel -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jtopo_node'`

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p node/src/jtopo_node
```

Create `node/src/jtopo_node/__init__.py` (empty).

- [ ] **Step 4: Implement `compute.py`**

Create `node/src/jtopo_node/compute.py`. This is the self-contained KPM kernel. It re-implements edge discovery, transport map construction (superposition mode only), sparse matrix assembly, and the Chebyshev recurrence with raw probe output.

```python
"""Self-contained KPM compute kernel for distributed workers.

Re-implements the minimal math from atft/ with zero external dependencies
(beyond torch, numpy, scipy). This allows the node package to be installed
on any machine without the full science stack.

The kernel builds a local sheaf Laplacian from a partition of the zeros
array (with ghost extensions), runs the Chebyshev recurrence, and returns
raw per-probe Hutchinson traces for dimension-weighted merge by the
synthesizer.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Prime representation (extracted from transport_maps.py)
# ---------------------------------------------------------------------------

def _primes_up_to(n: int) -> list[int]:
    """Return all primes <= n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [p for p in range(2, n + 1) if sieve[p]]


def _build_prime_rep(p: int, K: int) -> NDArray[np.float64]:
    """Sparse K×K partial permutation matrix rho(p)."""
    rho = np.zeros((K, K), dtype=np.float64)
    for n in range(1, K + 1):
        if p * n <= K:
            rho[p * n - 1, n - 1] = 1.0
    return rho


def _build_superposition_bases(K: int, sigma: float) -> tuple[NDArray, NDArray]:
    """Build superposition bases B_p(sigma) and log_primes for all primes <= K.

    Returns:
        bases: (P, K, K) float64 array
        log_primes: (P,) float64 array
    """
    primes = _primes_up_to(K)
    if not primes:
        return np.empty((0, K, K)), np.empty(0)

    P = len(primes)
    bases = np.empty((P, K, K), dtype=np.float64)
    log_primes = np.empty(P, dtype=np.float64)

    for idx, p in enumerate(primes):
        rho = _build_prime_rep(p, K)
        scale = np.log(p) / p**sigma
        bases[idx] = scale * (rho + rho.T)
        log_primes[idx] = np.log(p)

    return bases, log_primes


# ---------------------------------------------------------------------------
# Edge discovery (extracted from base_sheaf_laplacian.py)
# ---------------------------------------------------------------------------

def _build_edge_list(zeros: NDArray, epsilon: float):
    """1D Vietoris-Rips edge discovery on sorted zeros. Binary search only."""
    N = len(zeros)
    if epsilon <= 0 or N < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    i_parts, j_parts = [], []
    for i in range(N - 1):
        j_right = min(int(np.searchsorted(zeros, zeros[i] + epsilon, side="right")), N)
        if i + 1 < j_right:
            js = np.arange(i + 1, j_right, dtype=np.int64)
            i_parts.append(np.full(len(js), i, dtype=np.int64))
            j_parts.append(js)

    if i_parts:
        i_idx = np.concatenate(i_parts)
        j_idx = np.concatenate(j_parts)
    else:
        i_idx = np.array([], dtype=np.int64)
        j_idx = np.array([], dtype=np.int64)

    gaps = zeros[j_idx] - zeros[i_idx] if len(i_idx) > 0 else np.array([], dtype=np.float64)
    return i_idx, j_idx, gaps


# ---------------------------------------------------------------------------
# Transport computation (superposition mode, extracted from transport_maps.py)
# ---------------------------------------------------------------------------

def _batch_transport_superposition(gaps: NDArray, bases: NDArray,
                                    log_primes: NDArray, K: int,
                                    device: str) -> 'torch.Tensor':
    """GPU-accelerated superposition transport via batched matrix exponential."""
    M = len(gaps)
    dtype = torch.cdouble

    if M == 0:
        return torch.empty(0, K, K, dtype=dtype, device=device)

    bases_gpu = torch.tensor(bases, dtype=dtype, device=device)
    gaps_gpu = torch.tensor(gaps, dtype=torch.double, device=device)
    log_primes_gpu = torch.tensor(log_primes, dtype=torch.double, device=device)

    phases = torch.exp(1j * gaps_gpu[:, None] * log_primes_gpu[None, :])
    A_batch = torch.einsum('ep,pij->eij', phases, bases_gpu)

    # Frobenius normalize
    norms = torch.linalg.norm(A_batch.reshape(M, -1), dim=1)
    mask = norms > 0
    A_batch[mask] /= norms[mask, None, None]

    # Batched eigendecomposition for matrix exponential
    eigenvals, P_mat = torch.linalg.eig(A_batch)
    P_inv = torch.linalg.inv(P_mat)
    exp_eigenvals = torch.exp(1j * eigenvals)
    result = torch.einsum('mik,mk,mkj->mij', P_mat, exp_eigenvals, P_inv)

    # Fallback for defective matrices
    cond_est = (torch.linalg.norm(P_mat.reshape(M, -1), dim=1) *
                torch.linalg.norm(P_inv.reshape(M, -1), dim=1))
    defective = cond_est > 1e12
    if defective.any():
        for idx in torch.where(defective)[0]:
            result[idx] = torch.matrix_exp(1j * A_batch[idx])

    return result


# ---------------------------------------------------------------------------
# Sparse matrix assembly (extracted from torch_sheaf_laplacian.py)
# ---------------------------------------------------------------------------

def _build_sheaf_laplacian(zeros: NDArray, K: int, sigma: float,
                            epsilon: float, device: str) -> 'torch.Tensor':
    """Build the N*K x N*K sheaf Laplacian as torch sparse CSR."""
    N = len(zeros)
    dim = N * K
    dtype = torch.cdouble

    i_idx, j_idx, gaps = _build_edge_list(zeros, epsilon)
    M = len(i_idx)

    if M == 0:
        crow = torch.zeros(dim + 1, dtype=torch.int64, device=device)
        col = torch.empty(0, dtype=torch.int64, device=device)
        vals = torch.empty(0, dtype=dtype, device=device)
        return torch.sparse_csr_tensor(crow, col, vals, size=(dim, dim))

    bases, log_primes = _build_superposition_bases(K, sigma)
    U_all = _batch_transport_superposition(gaps, bases, log_primes, K, device)
    U_dagger = U_all.conj().transpose(1, 2)

    i_t = torch.tensor(i_idx, dtype=torch.int64, device=device)
    j_t = torch.tensor(j_idx, dtype=torch.int64, device=device)

    k_range = torch.arange(K, device=device)
    r_off, c_off = torch.meshgrid(k_range, k_range, indexing='ij')
    r_off = r_off.unsqueeze(0)
    c_off = c_off.unsqueeze(0)

    ri = i_t * K; rj = j_t * K

    row_ij = ri[:, None, None] + r_off
    col_ij = rj[:, None, None] + c_off
    row_ji = rj[:, None, None] + r_off
    col_ji = ri[:, None, None] + c_off
    row_ii = ri[:, None, None] + r_off
    col_ii = ri[:, None, None] + c_off
    row_jj = rj[:, None, None] + r_off
    col_jj = rj[:, None, None] + c_off

    I_K = torch.eye(K, dtype=dtype, device=device).unsqueeze(0).expand(M, K, K).clone()

    all_rows = torch.cat([row_ij.reshape(-1), row_ji.reshape(-1),
                          row_ii.reshape(-1), row_jj.reshape(-1)])
    all_cols = torch.cat([col_ij.reshape(-1), col_ji.reshape(-1),
                          col_ii.reshape(-1), col_jj.reshape(-1)])
    all_data = torch.cat([(-U_dagger).reshape(-1), (-U_all).reshape(-1),
                          torch.bmm(U_dagger, U_all).reshape(-1), I_K.reshape(-1)])

    L_coo = torch.sparse_coo_tensor(
        torch.stack([all_rows, all_cols]), all_data,
        size=(dim, dim), dtype=dtype, device=device,
    ).coalesce()
    return L_coo.to_sparse_csr()


# ---------------------------------------------------------------------------
# Power iteration (extracted from torch_sheaf_laplacian.py)
# ---------------------------------------------------------------------------

def _power_iteration_lam_max(L_csr, dim: int, device: str,
                              n_iter: int = 30) -> float:
    """Estimate largest eigenvalue via power iteration."""
    dtype = torch.cdouble
    rng = torch.Generator(device=device)
    rng.manual_seed(123)
    v = torch.randn(dim, dtype=torch.double, device=device, generator=rng).to(dtype)
    v = v / torch.linalg.norm(v)

    lam = 0.0
    for _ in range(n_iter):
        w = torch.mv(L_csr, v)
        lam = float(torch.real(torch.dot(v.conj(), w)).cpu())
        norm_w = torch.linalg.norm(w).real.item()
        if norm_w < 1e-14:
            return 0.0
        v = w / norm_w

    return lam * 1.05


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_partition_moments(
    zeros_path: str | None,
    zeros_array: NDArray | None,
    start_idx: int,
    end_idx: int,
    ghost_left: int,
    ghost_right: int,
    K: int,
    sigma: float,
    epsilon: float,
    degree: int,
    num_vectors: int,
    lam_max_global: float | None,
    seed: int,
    worker_id: str,
    device: str = "cpu",
) -> 'ContributeResult':
    """Compute raw Chebyshev moments for a partition of the zeros array.

    This is the self-contained distributed KPM kernel. It builds a local
    sheaf Laplacian from the owned zeros + ghost extensions, runs the
    Chebyshev recurrence, and returns raw per-probe Hutchinson traces.

    Ghost vertices participate in matrix assembly but are EXCLUDED from
    Hutchinson traces, ensuring each eigenvalue is counted exactly once.
    """
    # Import here to avoid circular dependency when used standalone
    from atft.distributed.protocol import ContributeResult

    t0 = time.time()

    # 1. Load zeros
    if zeros_array is not None:
        all_zeros = zeros_array
    else:
        all_zeros = np.sort(np.loadtxt(zeros_path, comments="#"))

    # 2. Slice with ghost extensions
    full_start = max(0, start_idx - ghost_left)
    full_end = min(len(all_zeros), end_idx + ghost_right)
    local_zeros = all_zeros[full_start:full_end]

    # Owned range within the local array
    owned_start = start_idx - full_start
    owned_end = end_idx - full_start
    N_owned = owned_end - owned_start
    dim_owned = N_owned * K

    # 3. Build local Laplacian
    dev = device if TORCH_AVAILABLE and device != "cpu" and torch.cuda.is_available() else "cpu"
    L_csr = _build_sheaf_laplacian(local_zeros, K, sigma, epsilon, dev)
    dim_full = L_csr.shape[0]

    if dim_full == 0 or L_csr._nnz() == 0:
        raw_traces = np.zeros((degree + 1, num_vectors), dtype=np.float64)
        raw_traces[0, :] = 1.0
        return ContributeResult(
            worker_id=worker_id, partition=(start_idx, end_idx),
            dim_local=dim_owned, lam_max_local=0.0,
            raw_traces=raw_traces, device_type=dev,
            compute_time_s=time.time() - t0,
        )

    # 4. Spectral normalization
    if lam_max_global is not None and lam_max_global > 1e-10:
        lam_max = lam_max_global
    else:
        lam_max = _power_iteration_lam_max(L_csr, dim_full, dev)

    lam_max_local = _power_iteration_lam_max(L_csr, dim_full, dev)

    if lam_max < 1e-10:
        raw_traces = np.zeros((degree + 1, num_vectors), dtype=np.float64)
        raw_traces[0, :] = 1.0
        return ContributeResult(
            worker_id=worker_id, partition=(start_idx, end_idx),
            dim_local=dim_owned, lam_max_local=lam_max_local,
            raw_traces=raw_traces, device_type=dev,
            compute_time_s=time.time() - t0,
        )

    # 5. Rademacher probes
    dtype = torch.cdouble
    rng = torch.Generator(device=dev)
    rng.manual_seed(seed)
    Z_full = (torch.randint(0, 2, (dim_full, num_vectors),
              device=dev, dtype=torch.double, generator=rng) * 2 - 1).to(dtype)

    # Owned row mask: exclude ghost rows from traces
    owned_rows = slice(owned_start * K, owned_end * K)

    # 6. Chebyshev recurrence with per-probe raw traces
    scale = 2.0 / lam_max

    def L_norm_mm(V):
        return scale * (L_csr @ V) - V

    raw_traces = np.empty((degree + 1, num_vectors), dtype=np.float64)
    T_prev = Z_full.clone()
    T_curr = L_norm_mm(Z_full)

    def owned_trace(Z_mat, T_mat):
        """Per-probe Hutchinson trace over OWNED rows only."""
        per_vec = torch.real(torch.sum(
            Z_mat[owned_rows].conj() * T_mat[owned_rows], dim=0
        ))
        return per_vec.cpu().numpy() / dim_owned

    raw_traces[0] = owned_trace(Z_full, T_prev)
    raw_traces[1] = owned_trace(Z_full, T_curr)

    for k in range(2, degree + 1):
        T_next = 2.0 * L_norm_mm(T_curr) - T_prev
        raw_traces[k] = owned_trace(Z_full, T_next)
        T_prev = T_curr
        T_curr = T_next

    # 7. GPU cleanup
    if dev != "cpu":
        torch.cuda.empty_cache()

    compute_time = time.time() - t0

    return ContributeResult(
        worker_id=worker_id,
        partition=(start_idx, end_idx),
        dim_local=dim_owned,
        lam_max_local=lam_max_local,
        raw_traces=raw_traces,
        device_type=dev,
        compute_time_s=compute_time,
    )
```

- [ ] **Step 5: Make node package importable for tests**

For testing, add `node/src` to the Python path. Create a minimal `node/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "jtopo-node"
version = "0.1.0"
description = "JTopo distributed worker node — self-contained KPM compute kernel"
requires-python = ">=3.10"
dependencies = ["torch", "numpy", "scipy"]

[tool.setuptools.packages.find]
where = ["src"]
```

Install in dev mode: `pip install -e ./node`

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_distributed.py::TestComputeKernel -v`
Expected: all 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add node/ atft/distributed/ tests/test_distributed.py
git commit -m "feat: add self-contained KPM compute kernel in node package"
```

---

### Task 3: Worker CLI entry point

**Files:**
- Create: `node/src/jtopo_node/worker.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_distributed.py`:

```python
import subprocess
import sys

class TestWorkerCLI:
    """Tests for the jtopo_node.worker CLI."""

    def test_worker_produces_output(self):
        """Running the worker CLI should produce a valid .npz file."""
        # Create a temporary zeros file
        zeros = np.sort(np.random.default_rng(42).uniform(0, 50, 100))
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
            for z in zeros:
                f.write(f"{z}\n")
            zeros_path = f.name

        output_path = tempfile.mktemp(suffix=".npz")

        result = subprocess.run([
            sys.executable, "-m", "jtopo_node.worker",
            "--start-idx", "0", "--end-idx", "100",
            "--ghost-left", "0", "--ghost-right", "0",
            "--K", "3", "--sigma", "0.5", "--epsilon", "3.0",
            "--degree", "10", "--num-vectors", "10",
            "--seed", "42", "--worker-id", "test-cli",
            "--zeros-path", zeros_path,
            "--output", output_path,
        ], capture_output=True, text=True, timeout=60)

        assert result.returncode == 0, f"Worker failed: {result.stderr}"

        from atft.distributed.protocol import ContributeResult
        loaded = ContributeResult.load(output_path)
        assert loaded.worker_id == "test-cli"
        assert loaded.raw_traces.shape == (11, 10)

        # Cleanup
        Path(zeros_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributed.py::TestWorkerCLI -v`
Expected: FAIL — worker module doesn't exist or has no `__main__`

- [ ] **Step 3: Implement `worker.py`**

Create `node/src/jtopo_node/worker.py`:

```python
"""Ephemeral worker CLI for distributed KPM computation.

Usage:
    python -m jtopo_node.worker \
        --start-idx 5000 --end-idx 10000 \
        --ghost-left 12 --ghost-right 0 \
        --K 100 --sigma 0.5 --epsilon 3.0 \
        --degree 300 --num-vectors 100 \
        --lam-max-global 10.5 \
        --seed 43 --worker-id ubuntu-5070 \
        --zeros-path data/odlyzko_zeros.txt \
        --output /tmp/jtopo_result.npz
"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="JTopo KPM Worker Node")
    parser.add_argument("--start-idx", type=int, required=True)
    parser.add_argument("--end-idx", type=int, required=True)
    parser.add_argument("--ghost-left", type=int, default=0)
    parser.add_argument("--ghost-right", type=int, default=0)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--degree", type=int, default=300)
    parser.add_argument("--num-vectors", type=int, default=100)
    parser.add_argument("--lam-max-global", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-id", type=str, default="unknown")
    parser.add_argument("--zeros-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    from jtopo_node.compute import compute_partition_moments

    print(f"[{args.worker_id}] Computing K={args.K}, sigma={args.sigma}, "
          f"partition=[{args.start_idx}:{args.end_idx}], "
          f"ghosts=({args.ghost_left},{args.ghost_right}), "
          f"D={args.degree}, nv={args.num_vectors}")

    result = compute_partition_moments(
        zeros_path=args.zeros_path,
        zeros_array=None,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        ghost_left=args.ghost_left,
        ghost_right=args.ghost_right,
        K=args.K,
        sigma=args.sigma,
        epsilon=args.epsilon,
        degree=args.degree,
        num_vectors=args.num_vectors,
        lam_max_global=args.lam_max_global,
        seed=args.seed,
        worker_id=args.worker_id,
        device=args.device,
    )

    result.save(args.output)
    print(f"[{args.worker_id}] Done in {result.compute_time_s:.1f}s. "
          f"Saved to {args.output}")


if __name__ == "__main__":
    main()
```

Also create `node/src/jtopo_node/__main__.py`:

```python
from jtopo_node.worker import main
main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_distributed.py::TestWorkerCLI -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add node/src/jtopo_node/worker.py node/src/jtopo_node/__main__.py
git commit -m "feat: add worker CLI entry point for distributed KPM dispatch"
```

---

## Chunk 2: Synthesizer Pre-flight Pipeline

### Task 4: DeviceScanner

**Files:**
- Create: `synthesizer/src/jtopo_synthesizer/__init__.py`
- Create: `synthesizer/src/jtopo_synthesizer/scanner.py`
- Create: `synthesizer/pyproject.toml`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_distributed.py`:

```python
class TestDeviceScanner:
    def test_scan_local_only(self):
        """Scanner with localhost-only config should find local device."""
        from jtopo_synthesizer.scanner import DeviceScanner
        import yaml

        config = {"nodes": [
            {"id": "local-test", "host": "localhost", "device": "cpu", "vram_mb": 0}
        ]}
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode='w') as f:
            yaml.dump(config, f)
            config_path = f.name

        scanner = DeviceScanner(config_path)
        nodes = scanner.scan()
        assert len(nodes) == 1
        assert nodes[0]["id"] == "local-test"
        assert nodes[0]["available"] is True
        Path(config_path).unlink()
```

- [ ] **Step 2: Run test, verify failure, implement**

Create `synthesizer/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "jtopo-synthesizer"
version = "0.1.0"
description = "JTopo distributed KPM synthesizer — orchestrator for multi-node compute"
requires-python = ">=3.10"
dependencies = ["numpy", "scipy", "pyyaml"]

[tool.setuptools.packages.find]
where = ["src"]
```

Install: `pip install -e ./synthesizer`

Create `synthesizer/src/jtopo_synthesizer/__init__.py` (empty).

Create `synthesizer/src/jtopo_synthesizer/scanner.py`:

```python
"""DeviceScanner — discover and probe available compute nodes."""
from __future__ import annotations

import subprocess
import yaml


class DeviceScanner:
    """Reads nodes.yaml and probes each node for availability."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

    def scan(self) -> list[dict]:
        """Return list of node dicts with 'available' status."""
        results = []
        for node in self._config.get("nodes", []):
            node_info = dict(node)
            if node["host"] == "localhost":
                node_info["available"] = True
            else:
                node_info["available"] = self._probe_remote(node)
            results.append(node_info)
        return results

    def _probe_remote(self, node: dict) -> bool:
        """SSH into remote node and verify jtopo-node is installed."""
        try:
            ssh_args = ["ssh"]
            if "ssh_key" in node:
                ssh_args += ["-i", node["ssh_key"]]
            ssh_args += [
                f"{node.get('ssh_user', 'root')}@{node['host']}",
                "python3 -c 'import jtopo_node; print(\"ok\")'",
            ]
            result = subprocess.run(
                ssh_args, capture_output=True, text=True, timeout=15,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except Exception:
            return False
```

- [ ] **Step 3: Run test, commit**

Run: `pytest tests/test_distributed.py::TestDeviceScanner -v`
Expected: PASS

```bash
git add synthesizer/ tests/test_distributed.py
git commit -m "feat: add DeviceScanner with YAML config and SSH probe"
```

---

### Task 5: JobPlanner with VRAM clamping

**Files:**
- Create: `synthesizer/src/jtopo_synthesizer/planner.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_distributed.py`:

```python
class TestJobPlanner:
    def test_symmetric_split(self):
        """Equal throughput should produce 50/50 split."""
        from jtopo_synthesizer.planner import JobPlanner
        planner = JobPlanner(vram_limit_mb=12000)
        n_a, n_b = planner.plan_partition(
            throughput_A=1.0, throughput_B=1.0, N=10000, K=100,
        )
        assert n_a == 5000
        assert n_b == 5000

    def test_asymmetric_split(self):
        """2x throughput should produce ~67/33 split."""
        from jtopo_synthesizer.planner import JobPlanner
        planner = JobPlanner(vram_limit_mb=12000)
        n_a, n_b = planner.plan_partition(
            throughput_A=2.0, throughput_B=1.0, N=9000, K=20,
        )
        assert n_a > n_b
        assert n_a + n_b == 9000

    def test_vram_clamping(self):
        """If asymmetric split exceeds VRAM, clamp toward 50/50."""
        from jtopo_synthesizer.planner import JobPlanner
        planner = JobPlanner(vram_limit_mb=100)  # very low limit
        n_a, n_b = planner.plan_partition(
            throughput_A=10.0, throughput_B=1.0, N=1000, K=50,
        )
        # Should clamp away from 90/10 toward 50/50
        assert n_a <= 600  # much less than the 909 unclamped

    def test_ghost_computation(self):
        """Ghost zone depth should match epsilon neighborhood."""
        from jtopo_synthesizer.planner import JobPlanner
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        planner = JobPlanner()
        h_left, h_right = planner.compute_ghost_depth(zeros, split=5, epsilon=2.5)
        # zeros[5]=5.0, epsilon=2.5: ghosts from zeros where |z - 5.0| <= 2.5
        # Left ghosts: zeros[3]=3.0, zeros[4]=4.0 (5.0-2.5=2.5, so z>=2.5)
        assert h_left >= 2
        # Right ghosts: zeros[5]=5.0, zeros[6]=6.0, zeros[7]=7.0
        assert h_right >= 2

    def test_time_estimate(self):
        """Pre-flight should produce a finite time estimate."""
        from jtopo_synthesizer.planner import JobPlanner
        planner = JobPlanner()
        estimate = planner.estimate_time(
            N=5000, K=100, degree=300, num_vectors=100,
            steps_per_second=100.0,
        )
        assert estimate > 0
        assert np.isfinite(estimate)
```

- [ ] **Step 2: Run test, verify failure, implement**

Create `synthesizer/src/jtopo_synthesizer/planner.py`:

```python
"""JobPlanner — pre-flight validation, VRAM clamping, ghost zones."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class JobPlanner:
    """Plans partition splits with VRAM clamping and time estimation."""

    def __init__(self, vram_limit_mb: float = 12000, vram_safety: float = 0.85):
        self._vram_limit = vram_limit_mb * vram_safety
        self._time_limits = {"auto": 3600, "confirm": 28800, "reject": 86400}

    def plan_partition(self, throughput_A: float, throughput_B: float,
                       N: int, K: int) -> tuple[int, int]:
        """Compute optimal partition split with VRAM clamping."""
        ratio_A = throughput_A / (throughput_A + throughput_B)
        n_a = int(N * ratio_A)
        n_b = N - n_a

        # VRAM clamp: push toward 50/50 if either exceeds limit
        while self._estimate_vram(max(n_a, n_b), K) > self._vram_limit:
            if n_a > n_b:
                n_a -= 100
                n_b += 100
            else:
                n_b -= 100
                n_a += 100
            if n_a <= 0 or n_b <= 0:
                n_a = N // 2
                n_b = N - n_a
                break

        return n_a, n_b

    def _estimate_vram(self, n: int, K: int) -> float:
        """Estimate VRAM in MB for a partition of n zeros at fiber dim K."""
        avg_degree = 6
        edges = n * avg_degree
        matrix_mb = edges * K * K * 16 / 1e6
        state_mb = n * K * 100 * 16 / 1e6  # T_curr * nv probes
        overhead_mb = 500  # PyTorch CUDA context
        return matrix_mb + state_mb + overhead_mb

    def compute_ghost_depth(self, zeros: NDArray, split: int,
                            epsilon: float) -> tuple[int, int]:
        """Compute ghost zone depth at the partition boundary."""
        N = len(zeros)
        # Left ghosts for right partition: zeros near split from the left
        h_left = split - int(np.searchsorted(
            zeros, zeros[split] - epsilon, side="left"
        ))
        # Right ghosts for left partition: zeros near split from the right
        h_right = int(np.searchsorted(
            zeros, zeros[split - 1] + epsilon, side="right"
        )) - split
        h_left = max(0, min(h_left, split))
        h_right = max(0, min(h_right, N - split))
        return h_left, h_right

    def estimate_time(self, N: int, K: int, degree: int,
                      num_vectors: int, steps_per_second: float) -> float:
        """Estimate wall-clock time in seconds."""
        total_steps = degree * N * K * num_vectors
        return total_steps / max(steps_per_second, 1.0)

    def check_time_limit(self, estimated_seconds: float) -> str:
        """Return 'auto', 'confirm', or 'reject' based on time limits."""
        if estimated_seconds < self._time_limits["auto"]:
            return "auto"
        elif estimated_seconds < self._time_limits["confirm"]:
            return "confirm"
        else:
            return "reject"
```

- [ ] **Step 3: Run tests, commit**

Run: `pytest tests/test_distributed.py::TestJobPlanner -v`
Expected: all 5 tests PASS

```bash
git add synthesizer/src/jtopo_synthesizer/planner.py tests/test_distributed.py
git commit -m "feat: add JobPlanner with VRAM clamping and ghost zone computation"
```

---

## Chunk 3: Dispatch, Validate, Merge, Log

### Task 6: ResultValidator and Merger

**Files:**
- Create: `synthesizer/src/jtopo_synthesizer/validator.py`
- Create: `synthesizer/src/jtopo_synthesizer/merger.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_distributed.py`:

```python
class TestResultValidator:
    def test_valid_result_passes(self):
        from jtopo_synthesizer.validator import ResultValidator
        from atft.distributed.protocol import ContributeResult
        traces = np.random.randn(51, 30)
        traces[0, :] = 1.0  # mu_0 ~ 1.0
        result = ContributeResult(
            worker_id="test", partition=(0, 100), dim_local=600,
            lam_max_local=10.0, raw_traces=traces,
            device_type="cpu", compute_time_s=1.0,
        )
        validator = ResultValidator()
        assert validator.validate(result) is True

    def test_nan_fails_validation(self):
        from jtopo_synthesizer.validator import ResultValidator
        from atft.distributed.protocol import ContributeResult
        traces = np.full((21, 10), np.nan)
        result = ContributeResult(
            worker_id="test", partition=(0, 50), dim_local=300,
            lam_max_local=5.0, raw_traces=traces,
            device_type="cpu", compute_time_s=0.5,
        )
        validator = ResultValidator()
        assert validator.validate(result) is False


class TestMerger:
    def test_symmetric_merge(self):
        """Two equal-dim partitions should merge to simple average."""
        from jtopo_synthesizer.merger import Merger
        from atft.distributed.protocol import ContributeResult
        D, nv = 20, 10
        traces_a = np.ones((D + 1, nv)) * 2.0
        traces_b = np.ones((D + 1, nv)) * 4.0
        result_a = ContributeResult(
            worker_id="a", partition=(0, 50), dim_local=100,
            lam_max_local=10.0, raw_traces=traces_a,
            device_type="cpu", compute_time_s=1.0,
        )
        result_b = ContributeResult(
            worker_id="b", partition=(50, 100), dim_local=100,
            lam_max_local=10.0, raw_traces=traces_b,
            device_type="cpu", compute_time_s=1.0,
        )
        merger = Merger()
        merged = merger.merge([result_a, result_b])
        # (100*2 + 100*4) / 200 = 3.0
        npt.assert_allclose(merged["mu_global"], np.full(D + 1, 3.0))

    def test_asymmetric_merge(self):
        """Unequal dims should weight proportionally."""
        from jtopo_synthesizer.merger import Merger
        from atft.distributed.protocol import ContributeResult
        D, nv = 5, 4
        result_a = ContributeResult(
            worker_id="a", partition=(0, 30), dim_local=90,
            lam_max_local=10.0, raw_traces=np.ones((D+1, nv)) * 1.0,
            device_type="cpu", compute_time_s=1.0,
        )
        result_b = ContributeResult(
            worker_id="b", partition=(30, 100), dim_local=210,
            lam_max_local=10.0, raw_traces=np.ones((D+1, nv)) * 3.0,
            device_type="cpu", compute_time_s=1.0,
        )
        merger = Merger()
        merged = merger.merge([result_a, result_b])
        expected = (90 * 1.0 + 210 * 3.0) / 300  # = 2.4
        npt.assert_allclose(merged["mu_global"], np.full(D + 1, expected))

    def test_variance_preserved(self):
        """Merged result should include per-moment variance."""
        from jtopo_synthesizer.merger import Merger
        from atft.distributed.protocol import ContributeResult
        D, nv = 5, 20
        rng = np.random.default_rng(42)
        result_a = ContributeResult(
            worker_id="a", partition=(0, 50), dim_local=100,
            lam_max_local=10.0, raw_traces=rng.randn(D+1, nv),
            device_type="cpu", compute_time_s=1.0,
        )
        merger = Merger()
        merged = merger.merge([result_a])
        assert "mu_variance" in merged
        assert merged["mu_variance"].shape == (D + 1,)
        assert np.all(merged["mu_variance"] >= 0)
```

- [ ] **Step 2: Run tests, verify failure, implement**

Create `synthesizer/src/jtopo_synthesizer/validator.py`:

```python
"""ResultValidator — quality checks on worker contributions."""
from __future__ import annotations

import numpy as np


class ResultValidator:
    """Validates ContributeResult payloads against mathematical invariants."""

    def validate(self, result) -> bool:
        """Check result for NaN, checksum, and mu_0 ~ 1.0."""
        traces = result.raw_traces

        # No NaN/Inf
        if not np.all(np.isfinite(traces)):
            return False

        # Checksum integrity
        if not result.verify_checksum():
            return False

        # mu_0 should be approximately 1.0 (within Hutchinson variance)
        mu_0 = np.mean(traces[0, :])
        if abs(mu_0 - 1.0) > 0.5:  # generous tolerance for small problems
            return False

        return True
```

Create `synthesizer/src/jtopo_synthesizer/merger.py`:

```python
"""Merger — dimension-weighted trace merge for distributed KPM."""
from __future__ import annotations

import numpy as np


class Merger:
    """Merges ContributeResult payloads using dimension-weighted arithmetic."""

    def merge(self, results: list) -> dict:
        """Merge raw traces from multiple partitions.

        Uses dimension-weighted linear combination:
            traces_global = sum(dim_i * traces_i) / dim_global

        Returns dict with mu_global, mu_variance, traces_global, dim_global.
        """
        dim_global = sum(r.dim_local for r in results)

        # Dimension-weighted merge of raw traces
        traces_global = np.zeros_like(results[0].raw_traces)
        for r in results:
            traces_global += r.dim_local * r.raw_traces
        traces_global /= dim_global

        mu_global = traces_global.mean(axis=1)
        mu_variance = traces_global.var(axis=1)

        return {
            "mu_global": mu_global,
            "mu_variance": mu_variance,
            "traces_global": traces_global,
            "dim_global": dim_global,
            "contributions": [
                {"worker_id": r.worker_id, "dim_local": r.dim_local,
                 "partition": r.partition, "compute_time_s": r.compute_time_s}
                for r in results
            ],
        }
```

- [ ] **Step 3: Run tests, commit**

Run: `pytest tests/test_distributed.py::TestResultValidator tests/test_distributed.py::TestMerger -v`
Expected: all 5 tests PASS

```bash
git add synthesizer/src/jtopo_synthesizer/validator.py synthesizer/src/jtopo_synthesizer/merger.py tests/test_distributed.py
git commit -m "feat: add ResultValidator and dimension-weighted Merger"
```

---

### Task 7: Dispatcher and ContributionLog

**Files:**
- Create: `synthesizer/src/jtopo_synthesizer/dispatcher.py`
- Create: `synthesizer/src/jtopo_synthesizer/log.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_distributed.py`:

```python
class TestDispatcher:
    def test_local_dispatch(self):
        """Dispatcher should run local partition and return valid result."""
        from jtopo_synthesizer.dispatcher import Dispatcher

        zeros = np.sort(np.random.default_rng(42).uniform(0, 50, 100))
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
            for z in zeros:
                f.write(f"{z}\n")
            zeros_path = f.name

        dispatcher = Dispatcher(zeros_path=zeros_path)
        result = dispatcher.dispatch_local(
            start_idx=0, end_idx=100, ghost_left=0, ghost_right=0,
            K=3, sigma=0.5, epsilon=3.0, degree=10, num_vectors=10,
            lam_max_global=None, seed=42, worker_id="local-test",
        )

        from atft.distributed.protocol import ContributeResult
        assert isinstance(result, ContributeResult)
        assert result.raw_traces.shape == (11, 10)
        Path(zeros_path).unlink()


class TestContributionLog:
    def test_log_append(self):
        """Log should append contributions to a directory."""
        from jtopo_synthesizer.log import ContributionLog
        from atft.distributed.protocol import ContributeResult

        with tempfile.TemporaryDirectory() as tmpdir:
            log = ContributionLog(tmpdir)
            result = ContributeResult(
                worker_id="test", partition=(0, 50), dim_local=150,
                lam_max_local=5.0, raw_traces=np.ones((11, 5)),
                device_type="cpu", compute_time_s=0.5,
            )
            path = log.record(result, job_id="test-001")
            assert Path(path).exists()
            assert "test-001" in path
            assert "test" in path  # worker_id in filename
```

- [ ] **Step 2: Run tests, verify failure, implement**

Create `synthesizer/src/jtopo_synthesizer/dispatcher.py`:

```python
"""Dispatcher — SSH dispatch and local execution of KPM partitions."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class Dispatcher:
    """Dispatches KPM partition jobs to local and remote nodes."""

    def __init__(self, zeros_path: str):
        self._zeros_path = zeros_path

    def dispatch_local(self, start_idx, end_idx, ghost_left, ghost_right,
                       K, sigma, epsilon, degree, num_vectors,
                       lam_max_global, seed, worker_id):
        """Run partition locally using the same compute kernel as remote."""
        from jtopo_node.compute import compute_partition_moments
        return compute_partition_moments(
            zeros_path=self._zeros_path, zeros_array=None,
            start_idx=start_idx, end_idx=end_idx,
            ghost_left=ghost_left, ghost_right=ghost_right,
            K=K, sigma=sigma, epsilon=epsilon,
            degree=degree, num_vectors=num_vectors,
            lam_max_global=lam_max_global, seed=seed,
            worker_id=worker_id, device="cuda" if _cuda_available() else "cpu",
        )

    def dispatch_remote(self, node: dict, start_idx, end_idx,
                        ghost_left, ghost_right, K, sigma, epsilon,
                        degree, num_vectors, lam_max_global, seed,
                        worker_id):
        """SSH-dispatch partition to remote node, SCP result back."""
        output_remote = f"/tmp/jtopo_{worker_id}.npz"
        output_local = tempfile.mktemp(suffix=".npz")

        cmd_parts = [
            f"python3 -m jtopo_node.worker",
            f"--start-idx {start_idx} --end-idx {end_idx}",
            f"--ghost-left {ghost_left} --ghost-right {ghost_right}",
            f"--K {K} --sigma {sigma} --epsilon {epsilon}",
            f"--degree {degree} --num-vectors {num_vectors}",
            f"--seed {seed} --worker-id {worker_id}",
            f"--zeros-path data/odlyzko_zeros.txt",
            f"--output {output_remote}",
        ]
        if lam_max_global is not None:
            cmd_parts.append(f"--lam-max-global {lam_max_global}")

        remote_cmd = " ".join(cmd_parts)
        ssh_user = node.get("ssh_user", "root")
        host = node["host"]
        ssh_key_args = ["-i", node["ssh_key"]] if "ssh_key" in node else []

        # SSH execute
        ssh_cmd = ["ssh"] + ssh_key_args + [
            f"{ssh_user}@{host}",
            f"cd ~/JTopo && {remote_cmd}",
        ]
        proc = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            raise RuntimeError(f"Remote worker failed: {proc.stderr}")

        # SCP retrieve
        scp_cmd = ["scp"] + ssh_key_args + [
            f"{ssh_user}@{host}:{output_remote}", output_local,
        ]
        subprocess.run(scp_cmd, capture_output=True, timeout=120, check=True)

        from atft.distributed.protocol import ContributeResult
        return ContributeResult.load(output_local)


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
```

Create `synthesizer/src/jtopo_synthesizer/log.py`:

```python
"""ContributionLog — append-only provenance for distributed KPM results."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class ContributionLog:
    """Logs every worker contribution with full metadata."""

    def __init__(self, output_dir: str):
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(self, result, job_id: str) -> str:
        """Save a ContributeResult .npz and metadata .json to the log."""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        basename = f"{job_id}-{result.worker_id}-{timestamp}"

        npz_path = str(self._dir / f"{basename}.npz")
        result.save(npz_path)

        meta = {
            "job_id": job_id,
            "worker_id": result.worker_id,
            "partition": list(result.partition),
            "dim_local": result.dim_local,
            "lam_max_local": result.lam_max_local,
            "device_type": result.device_type,
            "compute_time_s": result.compute_time_s,
            "checksum": result.checksum,
            "timestamp": timestamp,
        }
        json_path = str(self._dir / f"{basename}.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        return npz_path

    def record_merged(self, merged: dict, job_id: str) -> str:
        """Save merged result summary."""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        path = str(self._dir / f"{job_id}-merged-{timestamp}.json")
        serializable = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in merged.items()
        }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        return path
```

- [ ] **Step 3: Run tests, commit**

Run: `pytest tests/test_distributed.py::TestDispatcher tests/test_distributed.py::TestContributionLog -v`
Expected: all tests PASS

```bash
git add synthesizer/src/jtopo_synthesizer/dispatcher.py synthesizer/src/jtopo_synthesizer/log.py tests/test_distributed.py
git commit -m "feat: add Dispatcher (SSH+local) and ContributionLog"
```

---

### Task 8: Integration test — local 2-partition merge

**Files:**
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the integration test**

Add to `tests/test_distributed.py`:

```python
class TestIntegration:
    """Integration: 2-partition local split should match single-node KPM."""

    def test_two_partition_merge_matches_single_node(self):
        """Split N=200 into two halves, merge, compare to KPMSheafLaplacian."""
        from jtopo_node.compute import compute_partition_moments
        from jtopo_synthesizer.merger import Merger
        from jtopo_synthesizer.planner import JobPlanner
        from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder

        N, K, sigma, eps, D, nv = 100, 3, 0.5, 3.0, 20, 50
        zeros = np.sort(np.random.default_rng(42).uniform(0, 50, N))

        # Single-node reference
        builder = TransportMapBuilder(K=K, sigma=sigma)
        kpm = KPMSheafLaplacian(builder, zeros, device="cpu", degree=D, num_vectors=nv)
        mu_ref = kpm.compute_moments(eps)

        # 2-partition distributed
        split = N // 2
        planner = JobPlanner()
        h_left, h_right = planner.compute_ghost_depth(zeros, split, eps)

        result_a = compute_partition_moments(
            zeros_path=None, zeros_array=zeros,
            start_idx=0, end_idx=split,
            ghost_left=0, ghost_right=h_right,
            K=K, sigma=sigma, epsilon=eps,
            degree=D, num_vectors=nv,
            lam_max_global=None, seed=42,
            worker_id="rank-0", device="cpu",
        )
        result_b = compute_partition_moments(
            zeros_path=None, zeros_array=zeros,
            start_idx=split, end_idx=N,
            ghost_left=h_left, ghost_right=0,
            K=K, sigma=sigma, epsilon=eps,
            degree=D, num_vectors=nv,
            lam_max_global=None, seed=43,  # different seed!
            worker_id="rank-1", device="cpu",
        )

        merger = Merger()
        merged = merger.merge([result_a, result_b])
        mu_dist = merged["mu_global"]

        # Should match within Hutchinson variance + ghost boundary error
        # Use generous tolerance: different seeds + static ghosts
        npt.assert_allclose(mu_dist, mu_ref, atol=0.3)
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/test_distributed.py::TestIntegration -v`
Expected: PASS

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass (KPM + distributed + existing)

- [ ] **Step 4: Commit**

```bash
git add tests/test_distributed.py
git commit -m "test: add 2-partition integration test — distributed merge matches single-node KPM"
```
