# Ti V0.1 --- Technical Deep Dive
## For Technical Due Diligence

---

## 1. What the Framework Actually Does

ATFT constructs a **sheaf Laplacian with a u(K) gauge connection** over the **Vietoris-Rips complex** of a point cloud, then measures spectral coherence.

**In plain terms:** Given any set of points (zeta zeros, lattice gauge configurations, neural network hidden states), the framework:
1. Builds a proximity graph (Vietoris-Rips complex)
2. Attaches a vector space (fiber) at each vertex
3. Connects fibers along edges using domain-specific "transport maps" (gauge connection)
4. Measures how well information propagates through the resulting fabric (spectral sum)

Lower spectral sum = tighter fabric = more coherent transport = the points "fit" the connection better.

## 2. The Mathematical Objects

```
Fiber:        C^K at each vertex (K = dimension, one slot per integer 1..K)
Prime rep:    rho(p)|n> = |pn> if pn <= K, else 0
Generator:    B_p(sigma) = log(p) [p^(-sigma) rho(p) + p^(-(1-sigma)) rho(p)^T]
Transport:    A_ij(sigma) = SUM_p exp(i * Delta_gamma * log p) * B_p(sigma)
Coboundary:   (delta_0 x)_e = U_e x_i - x_j
Laplacian:    L_F = delta_0^dagger * delta_0
Observable:   S(sigma) = SUM_k lambda_k(L_F)   --- lower S = tighter fabric
```

**The key insight:** The exponential factor `exp(i * Delta_gamma * log p)` is the explicit formula's Fourier kernel---the exact phase factor that connects prime counting to zeta zeros in analytic number theory. When many primes constructively interfere at a particular sigma, transport becomes coherent and S drops. At sigma = 0.500, interference is maximally constructive for zeta zeros. Not for any control.

## 3. What Makes This Novel

### 3.1 No Prior Art Exists

There is no published work applying sheaf-theoretic gauge connections to the Riemann Hypothesis. The construction of gauge generators from truncated left-regular representations of the multiplicative monoid (Z>0, x) is original.

### 3.2 The Connection to the Explicit Formula is Not Accidental

The explicit formula in analytic number theory states:

```
psi(x) = x - SUM_rho (x^rho / rho) - log(2*pi) - (1/2)*log(1 - x^(-2))
```

where the sum runs over zeta zeros rho = sigma + i*gamma. The Fourier kernel `exp(i*gamma*log p)` appears naturally in this sum. The ATFT transport map uses exactly this kernel as edge phase factors. The framework is a computational interpreter for the explicit formula's geometric content.

### 3.3 Hermiticity Encodes the Critical Line

The Functional Equation transport mode uses generator:

```
G_p^FE(sigma) = log(p) [p^(-sigma) rho(p) + p^(-(1-sigma)) rho(p)^T]
```

This generator is Hermitian **if and only if** sigma = 0.5 (because p^(-sigma) = p^(-(1-sigma)) requires sigma = 1-sigma, so sigma = 1/2). Non-Hermitian generators produce non-unitary transport. The unitarity defect at sigma = 0.5 is measured at 1.6e-15 (machine epsilon). This is a **provable mathematical property**, not a statistical observation.

### 3.4 Three Transport Modes Probe Different Structure

| Mode | Construction | What It Detects |
|------|-------------|----------------|
| **Superposition** | SUM_p exp(i*dg*log p) * B_p(sigma) | Explicit formula coherence across all primes |
| **Functional Equation** | log(p)[p^(-s)rho(p) + p^(-(1-s))rho(p)^T] | Critical line as unique Hermitian surface |
| **Resonant** | Nearest-prime assignment per edge | Non-commuting holonomy, geometric curvature |

## 4. Computational Architecture

### 4.1 Five Backend Implementations

| Backend | Technology | Use Case | Performance |
|---------|-----------|----------|-------------|
| CPU Sparse (SciPy) | BSR matrices, shift-invert | Development, K <= 50 | Minutes |
| GPU Sparse (CuPy) | CSR matrices, CUDA | NVIDIA-only, K <= 100 | 80+ min at K=100 |
| PyTorch Hybrid | CPU edges + GPU transport + GPU Lanczos | Primary production, K=200 | 166-201s per point |
| Matrix-Free | Implicit matvec, Pade exponential | K >= 400, unlimited scaling | 47s at K=400, 1560s at K=800 |
| KPM | Chebyshev moments + Jackson damping | Broadband density of states | Configurable resolution |

### 4.2 Key Performance Innovations

1. **O(K^2) transport shortcut:** Eigendecompose generator once, reuse eigenbasis for all edges. Avoids O(K^3) per-edge matrix exponential.
2. **Batched GPU eigendecomposition:** torch.linalg.eig vectorized over edges eliminates 80-minute CPU bottleneck.
3. **Spectral flip trick:** Compute k smallest eigenvalues of PSD matrix by finding k largest of (lambda_max * I - L). Converts ill-conditioned minimum eigenvalue problem to well-conditioned maximum.
4. **Incremental list release:** Edge assembly releases intermediate lists during scipy.sparse coalesce to prevent CPU RAM OOM during GPU computation.
5. **Pade matrix exponential:** torch.matrix_exp uses rational approximation optimized for GPU tensor cores---4x faster than eigendecomposition on CPU.

### 4.3 Scaling Properties

| K | Primes Used | Matrix Size (N=1000) | VRAM | Time/Point | Hardware |
|---|------------|---------------------|------|-----------|---------|
| 20 | 8 | 20,000 x 20,000 | < 1 GB | ~10s | CPU |
| 50 | 15 | 50,000 x 50,000 | ~2 GB | ~30s | GTX 1060 |
| 100 | 25 | 100,000 x 100,000 | ~4 GB | ~60s | RTX 4080 |
| 200 | 46 | 200,000 x 200,000 | 5.8 GB | 166s | RTX 5070 |
| 400 | 78 | 400,000 x 400,000 | Matrix-free | 47s | RTX 5070 |
| 800 | 139 | 800,000 x 800,000 | Matrix-free | 1560s | RTX 5070 |

## 5. Codebase Statistics

| Metric | Value |
|--------|-------|
| Total Python LOC | 20,162 |
| Core framework (atft/) | ~12,000 LOC |
| Test code (tests/) | ~4,000 LOC |
| Test-to-code ratio | 32% |
| Functions | 292 |
| Test files | 24 |
| Passing tests | 299 |
| Experiment scripts | 25+ |
| Documentation files | 30+ |
| Python version | 3.11+ |
| Core dependencies | 4 (numpy, scipy, h5py, matplotlib) |
| Optional GPU | PyTorch >= 2.1 |

## 6. Type System and Architecture Quality

- **Frozen dataclasses** throughout (immutable types prevent state corruption)
- **Protocol-based abstractions** (structural subtyping, not inheritance)
- **Lazy evaluation + caching** (transport maps, eigenvectors computed once, reused)
- **Memory-aware batch processing** (incremental list release, VRAM monitoring)
- **Semantic commit history** (test/docs/fix/feat prefixes, results in messages)
- **CI/CD:** GitHub Actions on Python 3.11, 3.12 with CodeQL analysis

## 7. Data Sources

| Source | Origin | Size | Access |
|--------|--------|------|--------|
| Odlyzko zeta zeros | University of Minnesota DTC | 1.8 MB | Public |
| GUE ensembles | Dumitriu-Edelman tridiagonal model | Generated | Algorithmic |
| Poisson random | numpy.random | Generated | Algorithmic |
| SU(2) lattice configs | Kennedy-Pendleton heat bath | Generated | Algorithmic |
| LLM hidden states | SmolLM2-360M, Qwen2.5-0.5B | Downloaded | Public models |

## 8. What This Is NOT

- **Not a proof of the Riemann Hypothesis.** The kernel dimension beta_0^F = 0 at all points tested. No topological phase transition observed.
- **Not a black box.** Every number traces to a JSON file. Every claim has a falsification criterion.
- **Not cloud-dependent.** Entire pipeline runs on consumer hardware (Threadripper 7960X + RTX 5070).
- **Not a one-trick pony.** Same framework validated across three unrelated domains.
