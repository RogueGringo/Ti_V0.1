---
name: wavefront
description: "Use when orchestrating a full analysis, design, or investigation from start to finish. The master pipeline — dispatches L0→L3 agents with typed artifacts, enforces all five axioms, runs real persistence computation."
---

# @wavefront: Full Pipeline Orchestrator (V2)

The wavefront skill orchestrates the 4-agent driftwave pipeline. Each layer is a specialized agent producing typed JSON artifacts. The topology is the routing protocol — not metaphor but structure.

## The Five Axioms

1. **NO_AVERAGING** — Raw probes never averaged before filtration (enforced by RawCloud schema)
2. **UPWARD_FLOW** — L0 → L1 → L2 → L3, no layer skipping (enforced by artifact types)
3. **WAYPOINT_ROUTING** — Routing decisions are topological phase transitions (ASCEND/REPROBE/SPLIT)
4. **SHAPE_OVER_COUNT** — Gini trajectory dominates raw feature count (L2 monitors slope)
5. **ADAPTIVE_SCALE** — epsilon_max always derived from data geometry (95th percentile distances)

## Pipeline — Agent Dispatch Protocol

```
@wavefront Pipeline (V2 — Agent Stack)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARTIFACTS_DIR = /tmp/dw-artifacts/   (create if not exists)

1. DISPATCH dw-ingest agent (haiku)
   ├─ Input: project directory, user prompt
   ├─ Output: artifacts/raw.json (RawCloud schema)
   ├─ Validate: JSON schema check against schemas/raw_cloud.json
   ├─ entropy < 0.1? → ASK user for more input, re-dispatch
   └─ Save artifact, proceed to L1

2. RUN compute_persistence.py
   ├─ Input: cat artifacts/raw.json | compute_persistence.py
   ├─ Output: artifacts/persistence.json (barcode + distances + clusters)
   └─ This is REAL COMPUTATION — same math as ATFT zeta zero analysis

3. DISPATCH dw-cluster agent (sonnet)
   ├─ Input: artifacts/raw.json + artifacts/persistence.json
   ├─ Output: artifacts/filtered.json (FilteredTopology schema)
   ├─ Validate: JSON schema check against schemas/filtered_topology.json
   ├─ routing == REPROBE → back to step 1
   ├─ routing == SPLIT → fork: create sub-pipelines, each starting at L2
   └─ routing == ASCEND → proceed to L2

4. DISPATCH dw-synthesize agent (sonnet)
   ├─ Input: artifacts/filtered.json + relevant source files
   ├─ Output: artifacts/synthesis.json (SynthesisMap schema)
   ├─ Validate: JSON schema check against schemas/synthesis_map.json
   ├─ gini_slope < -0.01? → REPROBE to step 3
   ├─ open loops > 0? → iterate within L2 (max 3)
   └─ all loops closed + positive slope → proceed to L3

5. DISPATCH dw-review agent (opus)
   ├─ Input: artifacts/synthesis.json + spec docs
   ├─ Output: artifacts/verdict.json (SheavedVerdict schema)
   ├─ Validate: JSON schema check against schemas/sheaved_verdict.json
   ├─ OFF_SHELL → report obstructions → human decides
   └─ ON_SHELL → W(I) ∈ W_phys → implementation gate OPEN

6. WAYPOINT GATE
   ├─ verdict == ON_SHELL? → PROCEED to implementation
   └─ verdict == OFF_SHELL? → route back to appropriate layer
```

## Compute Stack

| Step | Where | Model | Cost |
|------|-------|-------|------|
| L0 ingest | Local LLM / fallback: topo.sh | Llama 3.2 3B (4-bit) | Free |
| Persistence | Local GPU (Python) | PyTorch/numpy | Free |
| L1 cluster | Local LLM + computation | Sonnet for labeling | API |
| L2 synthesize | API | Sonnet | API |
| L3 review | API | Opus | API |

## Controller Responsibilities (YOU, in the main session)

1. Create `/tmp/dw-artifacts/` directory
2. Dispatch each agent via the Agent tool with the artifact path as input
3. Read each artifact output, validate against schema
4. Make routing decisions based on artifact routing field
5. Handle REPROBE loops (max 3 iterations per layer)
6. Present final verdict to user

## When to Invoke

Any task that is non-trivial. The anti-pattern "This is too simple to need the pipeline" is always wrong. Even trivial tasks pass through — the pipeline may be fast (L0 trivially passes entropy gate, L1 shows single dominant cluster, L2 Gini is immediately positive) but it still runs.

## On-Shell / Off-Shell

A process is **on-shell** when all five axioms are satisfied, the Gini trajectory is non-negative, and the sheaf verdict is ON_SHELL. A process is **off-shell** when any axiom is violated or an obstruction exists. Off-shell configurations are topological noise, not signal.

## Artifact Schema Locations

```
${CLAUDE_PLUGIN_ROOT}/schemas/raw_cloud.json
${CLAUDE_PLUGIN_ROOT}/schemas/filtered_topology.json
${CLAUDE_PLUGIN_ROOT}/schemas/synthesis_map.json
${CLAUDE_PLUGIN_ROOT}/schemas/sheaved_verdict.json
```

## Agent Spec Locations

```
${CLAUDE_PLUGIN_ROOT}/agents/dw-ingest.md
${CLAUDE_PLUGIN_ROOT}/agents/dw-cluster.md
${CLAUDE_PLUGIN_ROOT}/agents/dw-synthesize.md
${CLAUDE_PLUGIN_ROOT}/agents/dw-review.md
```
