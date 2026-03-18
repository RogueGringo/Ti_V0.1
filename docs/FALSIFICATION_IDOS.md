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
