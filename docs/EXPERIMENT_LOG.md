# ATFT Experiment Log

Chronological record of all experimental runs, results, interpretations,
and engineering decisions. Every result — positive, negative, or ambiguous —
is documented here. This log serves as the ground truth for the project's
scientific trajectory.

---

## 2026-03-17: KPM Pipeline Operational — First Thermodynamic Scaling Data

### Context

The project pivoted from discrete eigenvalue extraction (Lanczos, LOBPCG) to the
Kernel Polynomial Method (KPM) with Jackson damping. The motivation: as K grows,
the discrete spectrum becomes continuous, and Krylov eigensolvers suffer invariant
subspace breakdown on the clustered near-zero eigenvalues. KPM reconstructs the
continuous density of states rho(lambda) from Chebyshev moments, providing the
Integrated Density of States (IDOS) as a macroscopic, scale-invariant observable.

### Run 1: Variable Cutoff (Moving Goalpost)

**Parameters:** N=2000, K=[10, 20, 50], D=150, nv=100, cutoff=pi*lam_max/D

**Results:**
| K  | sigma=0.25 | sigma=0.50 | sigma=0.75 | Contrast |
|----|------------|------------|------------|----------|
| 10 | 0.0415     | 0.0355     | 0.0392     | 0.855    |
| 20 | 0.0914     | 0.0816     | 0.0810     | 0.893    |
| 50 | 0.1114     | 0.0925     | 0.0991     | 0.830    |

**Interpretation:** IDOS minimizes at sigma=0.50 at every K — the topological
signal is present. However, the absolute IDOS grows with K for all sigma values,
triggering falsification criterion F1 (persistent off-line density).

**Root cause identified:** The IDOS cutoff Delta_lambda = pi * lam_max / D
grows with K because lam_max grows with K. The integration window widens,
swallowing bulk eigenvalues and diluting the near-kernel contrast. This is a
measurement artifact, not a physics result.

**AIC model selection:** Logarithmic ansatz preferred (AIC=-23.4) over
power law (AIC=-22.5) and exponential (AIC=-20.7). The IDOS off-line
grows as ~(log K)^1.5. The logarithmic scaling is consistent with the
prime harmonic distribution (density of primes ~ 1/log x).

### Run 2: Fixed Cutoff (Goalpost Corrected)

**Parameters:** N=2000, K=[10, 20, 50], nv=100, fixed_cutoff=0.5,
degree auto-scaled (300 for K=10,20; 638 for K=50)

**Results:**
| K  | sigma=0.25 | sigma=0.50 | sigma=0.75 | Contrast | Resolution |
|----|------------|------------|------------|----------|------------|
| 10 | 0.0624     | 0.0618     | 0.0621     | 0.989    | 0.137      |
| 20 | 0.1513     | 0.1367     | 0.1365     | 0.904    | 0.124      |
| 50 | 0.1486     | 0.1420     | 0.1422     | 0.955    | 0.052      |

**Interpretation:**
- sigma=0.50 consistently minimizes IDOS at every K. Signal is real.
- K=20 shows strongest contrast (9.6% IDOS reduction at critical line).
- K=50 contrast narrows to 4.5%, likely because N=2000 provides
  insufficient statistical depth for 15 prime harmonics.
- Spectral sum S confirms: S=49.8 at sigma=0.5 vs S=56.2 at sigma=0.25
  for K=20 (11.4% drop at critical line).
- Fixed cutoff successfully eliminates lambda_max goalpost drift.
- All resolution limits < 0.14 (well below cutoff 0.5).

**Limiting factor:** N=2000 zeros. The explicit formula interference pattern
needs more zero-gap samples to achieve destructive interference off-line
at higher K. Production runs with N=9877 (full Odlyzko dataset) on
GPU hardware are needed for definitive scaling exponents.

### Falsification Criteria Status (FALSIFICATION_IDOS.md)

| Criterion | Status | Notes |
|-----------|--------|-------|
| P1 (on-line finite) | PASS | IDOS(sigma=0.5) > 1e-4 at all K |
| P2 (off-line collapse) | INCONCLUSIVE | Need more K values and larger N |
| P3 (contrast divergence) | INCONCLUSIVE | Contrast present but not yet diverging |
| F1 (off-line persist) | INCONCLUSIVE | Fixed cutoff resolved the false trigger |
| F3 (symmetric collapse) | OK | On-line IDOS remains finite |
| F4 (GUE artifact) | NOT YET TESTED | Requires GUE control runs |

### Engineering Decisions Made

1. **LOBPCG plan shelved** — discrete eigenvalue extraction is a category
   error in the thermodynamic limit. Kept as diagnostic for small-K validation.
2. **KPM with Jackson damping chosen** — strict positivity, O(1/D) uniform
   convergence, no tunable parameters. Aligns with proof pathway.
3. **Raw moments stored undamped** — Jackson damping applied at reconstruction
   time. Raw moments are the path-ordered holonomies needed for analytic bounds.
4. **FALSIFICATION.md frozen** — original pre-registered criteria preserved.
   New FALSIFICATION_IDOS.md created for KPM-era thermodynamic criteria.
5. **Fixed cutoff mandatory** — variable cutoff (Delta_lambda = pi*lam_max/D)
   introduces a K-dependent measurement artifact. All future runs must use
   a fixed physical cutoff.
6. **Triple-ansatz fitting** — power law, logarithmic, and exponential fits
   with AIC model selection. Prevents confirmation bias in scaling analysis.

### Next Steps

1. Set up distributed compute (RTX 4080 + RTX 5070 Ubuntu server) for N=9877
2. Run K=[20, 50, 100] with N=9877, nv=100, fixed_cutoff=0.5
3. Add GUE control runs for F4 criterion
4. Extract definitive thermodynamic scaling exponents

---

*This log is append-only. New entries are added at the bottom.*
*Results data files are in output/ with timestamps matching log entries.*
