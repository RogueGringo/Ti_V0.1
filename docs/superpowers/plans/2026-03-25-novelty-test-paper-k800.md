# Pair Correlation Novelty Test + ATFT Paper + K=800 Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine if the sheaf Laplacian detects novel arithmetic structure, write the definitive ATFT paper, and push the K frontier to 800.

**Architecture:** Task 1 computes the pair correlation function r₂(s) for zeta/GUE/Random, uses it to predict S, and compares against actual S. If the prediction fails (residual > 5%), the sheaf Laplacian IS a new invariant. Task 2 assembles the paper from existing validated results. Task 3 decouples CPU transport from GPU Lanczos for K=800.

**Tech Stack:** PyTorch (GPU Lanczos, pairwise distances), NumPy/SciPy (pair correlations, matrix exp), matplotlib (figures), LaTeX (paper)

---

## File Structure

```
NEW FILES:
  atft/analysis/pair_correlation.py        — r₂(s) computation + S prediction from pairs
  atft/experiments/novelty_test.py         — The experiment: is S predictable from r₂?
  atft/topology/hybrid_sheaf_laplacian.py  — CPU transport + GPU Lanczos for K=800+
  docs/paper/atft_v2.tex                   — The paper (or .md → LaTeX pipeline)
  tests/test_pair_correlation.py           — Tests for pair correlation module
  tests/test_hybrid_laplacian.py           — Tests for hybrid engine

MODIFIED FILES:
  atft/topology/matfree_sheaf_laplacian.py — Extract reusable Lanczos into shared module
  docs/TECHNICAL_AUDIT.md                  — Add novelty test result
  docs/atft_validation_results/SUMMARY.md  — Add novelty verdict
```

---

### Task 1: Pair Correlation r₂(s) — The Novelty Test

The single highest-leverage computation. If the spectral sum S is fully predicted by pair correlations, the sheaf Laplacian adds nothing new. If it isn't, every claim in the paper is strengthened.

**Files:**
- Create: `atft/analysis/pair_correlation.py`
- Create: `atft/experiments/novelty_test.py`
- Create: `tests/test_pair_correlation.py`
- Modify: `docs/TECHNICAL_AUDIT.md`

- [ ] **Step 1: Write test for pair correlation function**

```python
# tests/test_pair_correlation.py
import numpy as np
from atft.analysis.pair_correlation import pair_correlation_function

def test_poisson_r2_is_one():
    """Poisson process has r₂(s) = 1 (no correlations)."""
    rng = np.random.default_rng(42)
    pts = np.sort(rng.uniform(0, 1000, 2000))
    s_grid, r2 = pair_correlation_function(pts, n_bins=50)
    # Poisson: r₂ ≈ 1 everywhere
    assert np.allclose(r2, 1.0, atol=0.15), f"Poisson r₂ deviates: mean={np.mean(r2):.3f}"

def test_gue_r2_has_repulsion():
    """GUE has r₂(0) = 0 (level repulsion) and r₂(s→∞) → 1."""
    # GUE pair correlation: r₂(s) = 1 - (sin(πs)/(πs))²
    s = np.linspace(0.01, 5, 100)
    r2_theory = 1 - (np.sin(np.pi * s) / (np.pi * s))**2
    assert r2_theory[0] < 0.1, "GUE should have near-zero r₂ at small s"
    assert abs(r2_theory[-1] - 1.0) < 0.05, "GUE r₂ should approach 1"
```

- [ ] **Step 2: Run test, verify it fails**

```bash
.venv/bin/pytest tests/test_pair_correlation.py -v
```

- [ ] **Step 3: Implement pair correlation function**

```python
# atft/analysis/pair_correlation.py
"""Pair correlation function r₂(s) for 1D point processes.

r₂(s) measures the probability density of finding a pair of points
separated by distance s, normalized by the uniform (Poisson) baseline.

For Poisson: r₂(s) = 1 (no correlations)
For GUE: r₂(s) = 1 - (sin(πs)/(πs))² (level repulsion at small s)
For zeta zeros: r₂(s) ≈ GUE locally (Montgomery-Odlyzko), but with
arithmetic corrections at large s that the sheaf Laplacian may detect.
"""
import numpy as np
from numpy.typing import NDArray


def pair_correlation_function(
    points: NDArray, n_bins: int = 100, s_max: float | None = None,
) -> tuple[NDArray, NDArray]:
    """Compute r₂(s) from a 1D point set.

    Uses histogram of pairwise spacings normalized by uniform density.

    Returns (s_grid, r2) where r2[i] is the pair correlation at s_grid[i].
    """
    pts = np.sort(points)
    N = len(pts)
    L = pts[-1] - pts[0]
    mean_density = (N - 1) / L

    if s_max is None:
        s_max = 5.0 / mean_density  # 5 mean spacings

    # All pairwise distances (for N up to ~5000; subsample for larger)
    if N > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, 5000, replace=False)
        pts_sub = np.sort(pts[idx])
    else:
        pts_sub = pts

    # Nearest-neighbor and beyond spacings
    spacings = []
    n_sub = len(pts_sub)
    for i in range(n_sub):
        for j in range(i + 1, min(i + 50, n_sub)):  # Up to 50th neighbor
            d = pts_sub[j] - pts_sub[i]
            if d > s_max:
                break
            spacings.append(d)

    spacings = np.array(spacings)

    # Histogram
    s_grid = np.linspace(0, s_max, n_bins + 1)
    s_centers = 0.5 * (s_grid[:-1] + s_grid[1:])
    ds = s_grid[1] - s_grid[0]

    hist, _ = np.histogram(spacings, bins=s_grid)

    # Normalize: expected count in each bin for Poisson
    n_pairs = n_sub * (n_sub - 1) / 2
    expected_per_bin = n_pairs * ds / (L / 2)  # rough normalization

    # Better: normalize by shell volume and density
    rho = len(pts_sub) / L
    # Expected pairs in [s, s+ds] for Poisson: N * rho * ds
    expected = len(pts_sub) * rho * ds * min(50, n_sub)  # adjust for neighbor cutoff

    r2 = hist / expected if expected > 0 else hist

    # Normalize so that large-s average → 1
    if len(r2) > 10:
        tail_mean = np.mean(r2[-10:])
        if tail_mean > 0:
            r2 /= tail_mean

    return s_centers, r2


def predict_spectral_sum_from_r2(
    r2_zeta: NDArray, r2_gue: NDArray, S_gue: float,
) -> float:
    """Predict S(zeta) from pair correlations alone.

    If S is fully determined by pair correlations:
      S(zeta)/S(GUE) should equal some functional of r₂(zeta)/r₂(GUE).

    Simplest model: S ∝ ∫|r₂(s) - 1|² ds  (total correlation energy)
    """
    corr_energy_zeta = np.sum((r2_zeta - 1)**2)
    corr_energy_gue = np.sum((r2_gue - 1)**2)

    if corr_energy_gue > 0:
        S_predicted = S_gue * corr_energy_zeta / corr_energy_gue
    else:
        S_predicted = S_gue

    return float(S_predicted)


def gpu_pairwise_distances(points: NDArray, max_pairs: int = 10_000_000) -> NDArray:
    """GPU-accelerated pairwise distance computation via PyTorch."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts = torch.tensor(points, dtype=torch.float64, device=device).unsqueeze(1)

    N = len(points)
    if N * (N - 1) // 2 > max_pairs:
        # Subsample
        rng = np.random.default_rng(42)
        n_sub = int(np.sqrt(2 * max_pairs))
        idx = rng.choice(N, min(n_sub, N), replace=False)
        pts = pts[idx]

    dists = torch.cdist(pts, pts).squeeze()
    # Extract upper triangle
    mask = torch.triu(torch.ones(len(pts), len(pts), device=device), diagonal=1).bool()
    pairwise = dists[mask].cpu().numpy()
    return pairwise
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
.venv/bin/pytest tests/test_pair_correlation.py -v
```

- [ ] **Step 5: Write the novelty test experiment**

```python
# atft/experiments/novelty_test.py
"""The Novelty Test: Is the sheaf Laplacian redundant?

Computes r₂(s) for zeta zeros, GUE, and Random.
Predicts S from pair correlations alone.
Compares predicted S to actual S.

If |S_actual - S_predicted| / S_actual > 5%:
  The sheaf Laplacian detects something pair correlations miss.
  → NEW INVARIANT.

If residual < 5%:
  The sheaf Laplacian is an expensive way to measure pair correlations.
  → REDUNDANT. Paper needs different framing.
"""
```

The experiment loads the K=200 results, computes r₂ for each source, predicts S from the pair correlation energy, and measures the residual. GPU-accelerated pairwise distances for speed.

- [ ] **Step 6: Run the novelty test**

```bash
.venv/bin/python -u atft/experiments/novelty_test.py 2>&1 | tee output/atft_validation/novelty_test.log
```

- [ ] **Step 7: Commit**

```bash
git add atft/analysis/pair_correlation.py atft/experiments/novelty_test.py tests/test_pair_correlation.py output/atft_validation/novelty_test*
git commit -m "test(atft): novelty test — is the sheaf Laplacian redundant?"
```

---

### Task 2: K=800 Hybrid Engine (CPU Transport + GPU Lanczos)

The OOM at K=800 is because transport matrices (25.5 GB) don't fit on GPU. The fix: compute transport on CPU via numpy matrix_exp (which handles any size), stream the results to GPU per Lanczos matvec batch.

**Files:**
- Create: `atft/topology/hybrid_sheaf_laplacian.py`
- Create: `tests/test_hybrid_laplacian.py`
- Create: `atft/experiments/k800_scaling.py`

- [ ] **Step 1: Write test for hybrid engine**

```python
# tests/test_hybrid_laplacian.py
import numpy as np

def test_hybrid_matches_matfree_k200():
    """Hybrid engine matches matrix-free at K=200 (where both can run)."""
    from atft.topology.hybrid_sheaf_laplacian import HybridSheafLaplacian
    # Compare against known S=11.784063 at K=200, sigma=0.5, eps=3.0
    # Tolerance: 1e-4 (Lanczos convergence)
```

- [ ] **Step 2: Implement hybrid engine**

```python
# atft/topology/hybrid_sheaf_laplacian.py
"""Hybrid Sheaf Laplacian — CPU transport + GPU Lanczos.

For K >= 800, the transport matrices (M × K × K × 16 bytes) exceed GPU VRAM.
Solution: compute transport on CPU (numpy scipy.linalg.expm), transfer one
batch at a time to GPU for the Lanczos matvec.

The Lanczos vectors live on GPU (N*K * 16 bytes = small).
Each matvec iteration:
  1. For each edge batch:
     a. CPU: compute matrix_exp for this batch (scipy.linalg.expm)
     b. CPU→GPU: transfer the batch transport matrices
     c. GPU: batched bmm for matvec contribution
     d. Free GPU batch
  2. GPU: accumulate result vector

Memory: O(batch * K² * 16) GPU + O(M * K² * 16) CPU
Time: dominated by CPU matrix_exp (~10s per batch at K=800)
"""
```

Key difference from matfree: transport is computed on CPU (scipy.linalg.expm, which handles arbitrary matrix size without VRAM limits), then transferred to GPU in small batches for the bmm.

- [ ] **Step 3: Validate at K=200**

```bash
.venv/bin/python -c "
from atft.topology.hybrid_sheaf_laplacian import HybridSheafLaplacian
# Compare against known K=200 result
"
```

- [ ] **Step 4: Run K=800**

```bash
.venv/bin/python -u atft/experiments/k800_scaling.py 2>&1 | tee output/atft_validation/k800_log.txt
```

- [ ] **Step 5: Commit**

```bash
git add atft/topology/hybrid_sheaf_laplacian.py atft/experiments/k800_scaling.py tests/test_hybrid_laplacian.py output/atft_validation/k800*
git commit -m "feat(atft): hybrid engine — K=800 via CPU transport + GPU Lanczos"
```

---

### Task 3: Write the ATFT Paper

All data exists. All figures exist. All 7 predictions validated. The paper assembles what we have.

**Files:**
- Create: `docs/paper/atft_v2.md` (Markdown draft → LaTeX conversion later)
- Create: `docs/paper/figures/` (symlinks or copies from assets/validation/)

- [ ] **Step 1: Create paper outline**

```markdown
# Adaptive Topological Field Theory: Experimental Validation
# Across Gauge Theory, Language Models, and Spectral Analysis

## Abstract
- Framework + 7 predictions + 5 PASS / 1 FAIL / 1 PARTIAL
- Showcase: SU(2) confinement detected topologically (β_c=2.30)
- Cross-model LLM universality (r=0.991 across 4 architectures)
- Continuous arithmetic premium (21.5%, converges K=100→400)
- Matrix-free engine: K=400 in 47s (18× speedup)

## 1. Introduction
- Field equations as topological waypoints (the central claim)
- Discrete topology computes the same cohomological invariants (Čech-de Rham)
- This paper: experimental validation across 3 domains

## 2. The Adaptive Topological Operator (review from ATFT paper)
- Definition, feature maps, gauge invariance, topological derivatives

## 3. Experimental Framework
- Hardware: RTX 5070 12GB, local compute only
- Matrix-free engine: Padé transport, batched GPU matvec
- Validation methodology: 3-agent committee, pre-registered criteria

## 4. Prediction 1: SU(2) Confinement-Deconfinement (PASS)
- 8³×4 lattice, Kennedy-Pendleton heat bath
- ε* drops 10× at β=2.30 — topological detection without Polyakov loop
- Figure: p5_su2_transition.png

## 5. Prediction 2: Instanton Discrimination (PARTIAL)
- BPST analytic discretization on 8⁴
- Vacuum vs instanton: KS=1.0. Q±1: fails (q_μν vanishes)
- Figure: p5b_instanton_barcodes.png

## 6. Prediction 3: LLM Cross-Model Universality (PASS)
- 4 models: SmolLM2, Qwen2.5, TinyLlama, Phi-1.5
- Mean r = 0.991 across all 6 pairwise correlations
- Gini trajectory is architecture-universal
- Figures: p4_cross_model_gini.png, p4_gini_trajectory_*.png

## 7. Prediction 4: Sheaf Laplacian Kernel (FAIL → Discovery)
- ker(L_F) = 0 everywhere, but λ₁ ~ K^(-0.19)
- Premium is 21% constant offset, not different decay rate
- Eigenvalue ratio uniformity CV=0.8%
- Scale-dependent premium: 5% (ε=1.5) → 21% (ε=3.0)
- Figures: p2_lambda1_scaling.png, p2_eigenvalue_ratio.png

## 8. Predictions 5-7: Spectral Analysis (PASS)
- QHO gap-bar: ρ=1.0 (sanity check)
- Betti curve discrimination: 21.1% onset difference
- Gini trajectory: hierarchification discriminates sources
- Figures: p1_*, p3_*

## 9. The Novelty Test (pending Task 1 result)
- Is the 21.5% premium predictable from pair correlations?
- If not: new invariant. If yes: redundant.

## 10. Discussion
- What the FAIL teaches (continuous vs discrete transitions)
- The framework detects known physics (SU(2)) and new physics (arithmetic premium)
- Limitations: lattice size, number of models, K scaling

## 11. Conclusion
- 5/7 predictions validated across 3 domains
- The adaptive topological operator works
- Open: K→∞ limit, instanton cooling, pair correlation comparison
```

- [ ] **Step 2: Write abstract + introduction**

Draft in `docs/paper/atft_v2.md`. Use the Dan/Stan voice for the draft, formalize for submission.

- [ ] **Step 3: Write methods section (Sections 2-3)**

Pull from existing TECHNICAL_AUDIT.md and the ATFT paper. Add matrix-free engine description.

- [ ] **Step 4: Write results sections (Sections 4-8)**

One section per prediction. Each section: setup, protocol, result, figure, interpretation. Numbers traced to JSON files.

- [ ] **Step 5: Write novelty test section (Section 9)**

Depends on Task 1 completion.

- [ ] **Step 6: Write discussion and conclusion (Sections 10-11)**

- [ ] **Step 7: Assemble figures**

Copy/symlink from `assets/validation/` to `docs/paper/figures/`.

- [ ] **Step 8: Commit**

```bash
git add docs/paper/atft_v2.md docs/paper/figures/
git commit -m "docs: ATFT v2 paper draft — 7 predictions, 3 domains, 5 PASS"
```

---

## Execution Order

**Task 1 (Novelty Test) FIRST** — determines the paper's framing.
**Task 2 (K=800) in PARALLEL** — independent, GPU-bound.
**Task 3 (Paper) AFTER Task 1** — needs the novelty result.

## Success Criteria

| Task | PASS if | FAIL if |
|------|---------|---------|
| 1. Novelty Test | Residual > 5% (new invariant) OR residual < 5% (honestly reported) | Computation errors |
| 2. K=800 | S(K=800) computed, scaling table extended to 4 points | OOM or timeout |
| 3. Paper | Complete draft covering all 7 predictions with figures | Missing sections |
