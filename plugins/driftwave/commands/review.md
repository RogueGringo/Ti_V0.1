---
description: "L3 — Sheaf consistency review. Checks global compatibility of all design sections. ON_SHELL or OFF_SHELL."
---

# /driftwave:review

Run the L3 review layer. The sheaf consistency check — does the whole compose?

## What to do

1. Verify `/tmp/dw-artifacts/synthesis.json` exists (if not, run `/driftwave:synthesize` first)
2. Dispatch the `dw-review` agent (opus tier) with:
   - The SynthesisMap artifact
   - Any relevant spec documents
3. Agent checks pairwise section compatibility, computes kernel dimension
4. Output saved to `/tmp/dw-artifacts/verdict.json`
5. Validate against `${CLAUDE_PLUGIN_ROOT}/schemas/sheaved_verdict.json`

## After review

Report to the user:
- Verdict: **ON_SHELL** (all sections globally consistent) or **OFF_SHELL** (obstructions exist)
- Kernel dimension: how many sections compose (kernel_dim == total → fully consistent)
- Obstructions: which section pairs contradict, and what specifically conflicts
- If ON_SHELL: W(I) ∈ W_phys — implementation gate is OPEN
- If OFF_SHELL: route back to `/driftwave:synthesize` with obstruction details (max 3 iterations)
