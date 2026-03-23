---
description: "L2 — Build design sections from filtered clusters, monitor Gini trajectory, detect consistency loops."
---

# /driftwave:synthesize

Run the L2 synthesis layer. Takes FilteredTopology and produces a coherent design.

## What to do

1. Verify `/tmp/dw-artifacts/filtered.json` exists (if not, run `/driftwave:filter` first)
2. Dispatch the `dw-synthesize` agent (sonnet tier) with:
   - The FilteredTopology artifact
   - Relevant source files (cluster members — read them for context)
3. Agent writes one design section per cluster, monitors Gini trajectory
4. Output saved to `/tmp/dw-artifacts/synthesis.json`
5. Validate against `${CLAUDE_PLUGIN_ROOT}/schemas/synthesis_map.json`

## After synthesis

Report to the user:
- Section titles and coherence scores
- Gini trajectory direction (▲ hierarchifying / ▼ flattening / ─ stable)
- Open loops (H₁ — cross-section consistency constraints)
- Routing: ASCEND → `/driftwave:review`, REPROBE → `/driftwave:filter`, SPLIT → decompose
