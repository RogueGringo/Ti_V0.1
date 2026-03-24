# Surgical Verdict — AMENDED
## Date: 2026-03-23 (late night amendment after Phase 3e completed)
## Status: PARTIALLY ON_SHELL — key finding survives, two claims need correction

---

## What Changed

Phase 3e control battery completed. Three tests ran. One had a critical bug.

### Test 1 (Even-Spaced): VALID, but interpretation needs correction
- S(Zeta) < S(Even) at all 11 sigma values — **CONFIRMED**
- BUT: Zeta has 17% fewer Rips edges (2492 vs 2994) due to level repulsion
- Per-edge S: Zeta (0.004729) > Even (0.004246) — Even is tighter PER EDGE
- The 7.3% "coherence premium" is partially or fully an edge-deficit effect
- **Corrected claim:** Zeta's lower total S reflects its sparser Rips graph, not necessarily superior transport coherence over Even

### Test 2 (GUE Ensemble): INVALID — bug
- Rank-based unfolding maps ANY sorted input to np.linspace
- All 10 "GUE" realizations produced S = 12.713270 (identical to EvenSpaced)
- Std = 0.000000. The test measured nothing.
- **Must re-run** with proper semicircle CDF unfolding that handles tails correctly

### Test 3 (Edge Count): VALID — the real finding

| Source | |E| | S(σ=0.5) | S/|E| |
|--------|------|----------|-------|
| Zeta | 2492 | 11.784 | 0.00473 |
| EvenSpaced | 2994 | 12.713 | 0.00425 |
| WignerGUE | 2765 | 15.004 | 0.00543 |
| Random | 2963 | 22.087 | 0.00745 |

**The critical observation:**
- Even and Random have nearly identical edge counts (2994 vs 2963, 1% difference)
- But S(Random) = 22.087 vs S(Even) = 12.713 — a 74% difference
- **Transport matrices dominate.** The graph structure is nearly identical; the S difference comes entirely from how information flows across edges.

**S/|E| coefficient of variation = 22.9%** — S is definitively NOT proportional to edge count.

---

## Revised Hierarchy Understanding

The four-tier ordering S(Zeta) < S(Even) < S(GUE) < S(Random) is real. But the tiers group into TWO phenomena:

### Phenomenon 1: Graph sparsity (edge count)
- Zeta zeros have stronger level repulsion → wider gaps → fewer Rips edges
- This alone pushes S(Zeta) below S(Even)
- NOT evidence of arithmetic coherence

### Phenomenon 2: Transport coherence (the real signal)
- Even and Random have ~same edge count but wildly different S
- Even has organized transport (uniform structure), Random has chaotic transport
- WignerGUE falls between (correlated but not perfectly organized)
- **This IS evidence that the sheaf Laplacian measures transport structure, not just graph topology**

### The open question (requires proper GUE control):
Where does proper Dumitriu-Edelman GUE land? If it lands near WignerGUE (~15), the arithmetic premium over GUE is real and ~21%. If it lands near Even (~12.7), the premium is smaller. We don't know yet because Test 2 was buggy.

---

## What Remains Confirmed

1. **S is not proportional to |E|** (CV = 22.9%). The adversary's "edge count" attack is refuted for the Random vs Even comparison. Transport matters.
2. **S(Zeta) is the lowest tested.** Whether this is purely edge-count or partly transport-coherence is the open question.
3. **The hierarchy holds at all sigma.** Reproducible to 6 decimal places.
4. **Arithmetic premium over WignerGUE = 21.5%.** This stands (WignerGUE was computed correctly in Phase 3d). But WignerGUE may not represent true GUE.

## What Must Be Fixed

1. **GUE unfolding.** Need proper semicircle CDF with tail handling (not rank-based). Or: use the GUE eigenvalues' empirical spacings directly, scaled to match zeta zero range and mean spacing.
2. **Edge-normalized comparison.** Report S/|E| alongside S. If Zeta's S/|E| is lower than GUE's S/|E|, the premium is real even after controlling for edge count.
3. **The Fourier sharpening epsilon confound** (from earlier audit). Still needs fixing.

---

## Honest Bottom Line

The sheaf Laplacian measures something real beyond graph topology. The proof: Even and Random have the same edges but 74% different S. Transport matrices encode structure that the Rips graph alone does not capture.

Whether Zeta zeros specifically have *arithmetic* coherence beyond their *geometric* regularity (level repulsion → sparser graph) is not yet settled. The Test 2 bug must be fixed to answer this.

The work continues.

*Axiom 3: we stopped when the surgeon found the bug.*
