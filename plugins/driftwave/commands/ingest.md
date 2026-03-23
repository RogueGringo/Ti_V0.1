---
description: "L0 — Scan the codebase and produce a RawCloud artifact. No interpretation, just measurement."
arguments: "[path]"
---

# /driftwave:ingest

Run the L0 ingestion layer. Scans the project (or specified path) and produces a typed RawCloud artifact.

## What to do

1. Create `/tmp/dw-artifacts/` if it doesn't exist
2. Dispatch the `dw-ingest` agent (haiku tier) OR fall back to `topo.sh scan` if no local LLM
3. The agent scans files via Glob/Grep, classifies by language/size/staleness
4. Output saved to `/tmp/dw-artifacts/raw.json`
5. Validate against `${CLAUDE_PLUGIN_ROOT}/schemas/raw_cloud.json`
6. Report: file count, entropy, and whether the entropy gate passes (>0.1)

If an argument is provided, scan that path instead of the project root.

## After ingestion

Tell the user:
- How many files were scanned
- The entropy value (variance across artifacts)
- Whether to proceed (`/driftwave:filter`) or provide more input (low entropy)
