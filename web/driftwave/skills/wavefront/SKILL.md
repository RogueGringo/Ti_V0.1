---
name: wavefront
description: "Use when orchestrating a full analysis, design, or investigation from start to finish. The master pipeline — enforces all five driftwave axioms across the complete L0→L1→L2→L3 trajectory. Invoke this for any non-trivial task."
---

# @wavefront: Full Pipeline Orchestrator

The wavefront agent runs the complete driftwave pipeline, enforcing all five axioms at every transition.

## The Five Axioms

1. **NO_AVERAGING** — Raw probes never averaged before filtration
2. **UPWARD_FLOW** — L0 → L1 → L2 → L3, no layer skipping
3. **WAYPOINT_ROUTING** — Routing decisions are topological phase transitions
4. **SHAPE_OVER_COUNT** — Gini trajectory dominates raw Betti number
5. **ADAPTIVE_SCALE** — epsilon_max always derived from data geometry

## Pipeline

```
@wavefront Pipeline
━━━━━━━━━━━━━━━━━━

1. /dw-map [artifacts]              ← L0: Ingest raw artifacts
   ├─ entropy_gate check            ← Reject zero-variance inputs
   └─ Output: raw point cloud

2. /dw-filter [--eps-percentile 95] ← L1: H₀ persistent clustering
   ├─ ALL BARS SHORT → REPROBE to L0
   ├─ Long bars → viable modules
   └─ Output: persistent clusters

3. /dw-ascend [--degree 1]          ← L2: H₁ loops + Gini routing
   ├─ Gini > +0.01 → ASCEND
   ├─ Gini < -0.01 → REPROBE
   ├─ |Gini| < 0.01 → HOLD
   ├─ waypoints > 3 → SPLIT
   └─ gini_watchdog monitors continuously

4. /dw-ascend [--sheaf]             ← L3: Sheaf-valued review
   ├─ Check global consistency (ker(L_F))
   ├─ Iterate until convergence (max 3)
   └─ Surface to human if obstruction persists

5. Waypoint gate                    ← Phase transition: analysis → action
   ├─ W(C) ∈ W_phys? → PROCEED
   └─ W(C) ∉ W_phys? → REPROBE to appropriate layer
```

## When to Invoke

Any task that is non-trivial. The anti-pattern "This is too simple to need the pipeline" is always wrong. Even trivial tasks pass through — the pipeline may be fast (L0 trivially passes entropy gate, L1 shows single dominant cluster, L2 Gini is immediately positive) but it still runs.

## The Waypoint Principle

The field equations of a system are geometric constraints on the topological evolution trajectory. They select which trajectories correspond to physical (viable) configurations. The pipeline doesn't search for a solution — it constructs an adaptive filtration of the solution space and reads the constraints as topological waypoints.

## On-Shell / Off-Shell

A process is **on-shell** when all five axioms are satisfied and the Gini trajectory is non-negative. A process is **off-shell** when any axiom is violated. Off-shell configurations do not correspond to viable outcomes — they are topological noise, not signal.
