---
name: dw-ascend
description: "Use when L1 filtering is complete and you need to synthesize coherent structure, detect design loops, validate global consistency, or perform sheaf-valued review. L2/L3 layers — H₁ loop detection, Gini trajectory monitoring, and sheaf Laplacian kernel convergence."
---

# L2/L3: Topological Synthesis and Sheaf-Valued Review

The ascent from persistent clusters to coherent architecture.

## L2: H₁ Topology — Loop Detection and Design Coherence

After L1 selects a module (approach), H₁ persistent homology detects loops — closed paths that represent:
- Internal consistency constraints
- Circular dependencies that must resolve
- Coherence structures (architecture → data flow → error handling → testing → architecture)

Each loop that persists across scales is a genuine structural constraint. Short-lived loops are noise.

## The Gini Routing Table (SHAPE_OVER_COUNT Axiom)

The Gini trajectory — how the hierarchy of topological features evolves — dominates raw feature count. Monitor after each section/step:

| Gini Slope | Route | Meaning |
|---|---|---|
| > +0.01 | **ASCEND** | Structure is hierarchifying — proceed, expand |
| < -0.01 | **REPROBE** | Structure is degrading — descend to appropriate layer |
| Within +/- 0.01 | **HOLD** | Stable — maintain current level, await new input |
| Waypoints > 3 | **SPLIT** | Dimensional branch — decompose into independent sub-problems |

**Positive Gini trajectory correlates with quality.** Validated at r=0.935 across four LLM architectures. A design with 3 deeply coherent sections (positive slope) outperforms one with 12 scattered sections (negative slope).

## L3: Sheaf-Valued Synthesis

At L3, output is no longer scalar ("good/bad") but sheaf-valued — it carries algebraic structure.

**The sheaf Laplacian kernel test:** A configuration (design, analysis, proof) is on-shell if and only if the corresponding section lies in ker(L_F) — every local component is globally consistent under the restriction maps (interfaces, transport maps, boundary conditions).

### L3 Review Protocol

1. Evaluate each component as a local section of the sheaf
2. Check global consistency: are local sections compatible under restriction maps?
3. If ker(L_F) convergence: the configuration is on-shell — proceed
4. If obstruction persists after 3 iterations: surface to human — the obstruction may be fundamental
5. If obstruction is topological (not just a fixable error): the design may need structural revision, not patching

## WAYPOINT_ROUTING Axiom

Every routing decision between layers is a topological phase transition, not a timer or checklist. Transitions fire when the persistence diagram exhibits qualitative change — not after a fixed number of steps.

## Dissolution of Smooth Obstructions (Proposition 7.1)

Obstructions in the smooth (continuous) category may dissolve in the discrete simplicial category. If an approach appears blocked, ask: is this obstruction a property of smooth structure, or does it survive on combinatorial cochains? The discrete category may offer paths that continuous reasoning cannot.
