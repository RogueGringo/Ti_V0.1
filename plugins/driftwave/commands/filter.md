---
description: "L1 — Run persistent homology on RawCloud, identify stable clusters, produce FilteredTopology artifact."
---

# /driftwave:filter

Run the L1 filtration layer. Takes the RawCloud artifact and finds persistent structure.

## What to do

1. Verify `/tmp/dw-artifacts/raw.json` exists (if not, run `/driftwave:ingest` first)
2. Run the actual persistence computation:
   ```bash
   cat /tmp/dw-artifacts/raw.json | ${CLAUDE_PLUGIN_ROOT}/scripts/compute_persistence.py > /tmp/dw-artifacts/persistence.json
   ```
3. Dispatch the `dw-cluster` agent (sonnet tier) with raw.json + persistence.json
4. Agent labels clusters with human-readable descriptions
5. Output saved to `/tmp/dw-artifacts/filtered.json`
6. Validate against `${CLAUDE_PLUGIN_ROOT}/schemas/filtered_topology.json`

## After filtering

Report to the user:
- Number of clusters found and their labels
- Barcode summary (longest bars = real structure)
- Noise files (filtered out)
- Routing decision: ASCEND → `/driftwave:synthesize`, REPROBE → `/driftwave:ingest`, SPLIT → decompose
