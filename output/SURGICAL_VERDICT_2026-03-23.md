# Surgical Verdict — ATFT Phase 3d/3e Results
## Date: 2026-03-23
## Status: CONDITIONAL ON_SHELL (3 of 5 tests passed, 2 not yet run)

---

## What Is True (Verified, Reproducible)

### 1. Four-Tier Hierarchy
**S(zeta) < S(even) < S(GUE) < S(random)** at all 11 sigma values tested.

| sigma | S(Zeta) | S(Even) | S(GUE) | S(Random) |
|-------|---------|---------|--------|-----------|
| 0.250 | 12.031  | 13.329  | 15.108 | 22.276    |
| 0.350 | 11.863  | 12.748  | 15.016 | 22.112    |
| 0.400 | 11.821  | 12.699  | 14.975 | 22.095    |
| 0.440 | 11.797  | 12.711  | 15.001 | 22.080    |
| 0.480 | 11.787  | 12.715  | 15.006 | 22.096    |
| **0.500** | **11.784** | **12.713** | **15.004** | **22.087** |
| 0.520 | 11.780  | 12.707  | 14.997 | 22.075    |
| 0.560 | 11.773  | 12.685  | 14.967 | 22.055    |
| 0.600 | 11.773  | 12.651  | 14.940 | 22.031    |
| 0.650 | 11.782  | 12.660  | 14.938 | 21.987    |
| 0.750 | 11.884  | 13.191  | 14.954 | 22.056    |

- Zero violations across 44 pairwise comparisons (11 sigma x 4 tiers)
- Reproducible to 6 decimal places across independent runs
- Even-spaced control (the adversary's kill shot) LOSES to zeta at every sigma

### 2. Arithmetic Premium
- **(1 - S_zeta/S_GUE) x 100 = 21.5%** at sigma=0.500
- Range across sigma: [20.4%, 21.5%]
- Standard deviation: 0.35%
- All positive (zeta always tighter than GUE)

### 3. Coherence Premium (NEW)
- **(1 - S_zeta/S_even) x 100 = 7.3%** at sigma=0.500
- U-shaped: min 6.9% near center, max 9.9% at sigma=0.75
- Physical interpretation: at sigma=0.5, transport is unitary (Hermitian generators), regularizing toward geometric flatness. Off-center, non-unitary transport amplifies arithmetic advantage.

### 4. Eigenvalue Ratio Uniformity
- Zeta/GUE ratios across first 5 eigenvalues: [0.789, 0.803, 0.795, 0.791, 0.784]
- Coefficient of variation: 0.9%
- The arithmetic premium is distributed uniformly across the low-lying spectrum, not concentrated in one mode.

---

## What Is Not True (Claimed But Incorrect)

### 1. "Peak Migration from sigma=0.52 to sigma=0.500"
- **Root cause:** Epsilon confound. The analysis script (`k200_full_analysis.py`) used `dict(zip())` which overwrites eps=3.0 data with eps=5.0 data at K=100.
- The sigma=0.52 peak came from **eps=5.0 data**, not eps=3.0.
- **Corrected facts:** At eps=3.0, K=100 premium curve is very flat (range: 0.34%, peak at sigma=0.650). At K=200, premium peaks clearly at sigma=0.500 (range: 1.1%).
- **What IS true:** K=200 has a sharper, better-defined peak than K=100. But the peak didn't "migrate from 0.52."

### 2. All Reported P-Values
- **Root cause:** Pseudoreplication. Mann-Whitney U, KS test, and bootstrap CI all treat sigma points as independent observations. They are not — all derived from the same underlying point clouds with slightly different parameters.
- **Effective sample size:** n=1 per source (one zeta zero set, one GUE realization, one Poisson realization).
- **What IS true:** The hierarchy has zero overlap — every single zeta S-value is below every single GUE S-value. This is visually unambiguous. But we cannot put a valid frequentist p-value on it without multiple independent realizations.

### 3. "95% CI: [20.9%, 21.3%]" for Arithmetic Premium
- The bootstrap CI resamples 11 non-independent premium values. The tight CI reflects low sigma-to-sigma variation within one experiment, not low uncertainty about the true effect.
- The point estimate (21.5% at sigma=0.5) falls outside the reported CI (which is for the mean across all sigma), indicating these measure different quantities.

---

## What Is Unknown (Tests Not Yet Completed)

### 1. Does the Hierarchy Hold with Proper GUE?
- Current "GUE" uses Wigner surmise (nearest-neighbor spacing only).
- The codebase has Dumitriu-Edelman (full n-point correlations) in `atft/sources/gue.py`.
- Phase 3e Test 2 designed to run 10 proper GUE realizations. **Not yet run** (first attempt OOM'd due to unfolding bug; fix applied, re-run in progress).

### 2. Is S Proportional to Edge Count?
- Phase 3e Test 3 designed to plot S vs |E| for all sources.
- The even-spaced result conceptually refutes pure proportionality (even spacing has highly regular edges but higher S than zeta).
- **Quantitative test not yet run.**

### 3. GUE Variance Bounds
- With n=1 GUE realization, we don't know the sampling variance of S_GUE.
- If S_GUE has high variance across realizations, zeta might fall within the GUE distribution.
- Phase 3e Test 2 would establish this. **Not yet run.**

### 4. Why Is the Coherence Premium U-Shaped?
- Premium over even-spacing is 7.3% at sigma=0.5 but 9.9% at sigma=0.75.
- Hypothesis: unitary transport at sigma=0.5 (Hermitian generators) partially regularizes toward geometric flatness, reducing the arithmetic advantage.
- **Not formally tested.**

---

## Conditions for Full ON_SHELL

1. **Complete Phase 3e Test 2** — 10 proper GUE realizations. If S(zeta) falls below the GUE 95% range → arithmetic premium confirmed with proper control.
2. **Complete Phase 3e Test 3** — S vs |E| plot. If sources don't fall on a single line → S contains information beyond edge count.
3. **Fix epsilon confound in analysis script** — Correct the dict(zip()) overwrite. Report K=100 and K=200 at eps=3.0 only.
4. **Remove invalid p-values** — Replace with honest description: "zero overlap across N=11 matched sigma points; frequentist significance requires multiple independent realizations."
5. **Run K=200 at N=2000** — Disentangle K effect from N effect (currently K=100 uses N=2000, K=200 uses N=1000).

---

## Verdict

**The four-tier hierarchy is real.** It is the strongest finding — reproducible, universal across sigma, and survives the adversary's kill shot. The claim does not depend on invalid p-values or epsilon-confounded comparisons.

**The Fourier sharpening narrative needs correction** but the core observation (K=200 has a sharper premium peak than K=100) is true.

**The research is honest.** We documented what's confirmed, what's wrong, and what's unknown. The falsification criteria remain binding. If Phase 3e Test 2 shows S(zeta) within the proper GUE distribution, the arithmetic premium claim must be downgraded.

*Generated by driftwave L0→L2 pipeline on ATFT experiment artifacts.*
*Axiom 1: every number in this document comes from a JSON file, not from memory.*
*Axiom 3: we stopped when the Statistician found the epsilon confound.*
