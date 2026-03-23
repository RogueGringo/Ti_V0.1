---
description: "Show current pipeline state — which artifacts exist, what layer is active, routing history."
---

# /driftwave:status

Report the current state of the driftwave pipeline.

## What to do

Check for artifacts in `/tmp/dw-artifacts/` and report:

1. **Artifact inventory:**
   - `raw.json` exists? → L0 complete. Show: file count, entropy
   - `persistence.json` exists? → Persistence computed. Show: barcode length, cluster count
   - `filtered.json` exists? → L1 complete. Show: clusters, routing decision
   - `synthesis.json` exists? → L2 complete. Show: sections, Gini slope, open loops
   - `verdict.json` exists? → L3 complete. Show: verdict (ON_SHELL/OFF_SHELL), kernel_dim

2. **Pipeline progress indicator:**
   ```
   L0 ████████ L1 ████████ L2 ████░░░░ L3 ░░░░░░░░
   ```

3. **Local LLM status:**
   - Check if server is running: `curl -s http://localhost:8090/v1/models`
   - Report model name and device (GPU/CPU)
   - If not running, suggest: `bash ${CLAUDE_PLUGIN_ROOT}/scripts/start_local_llm.sh`

4. **GPU status:**
   - `nvidia-smi` — any compute processes running?
   - VRAM free/total
   - If an ATFT experiment is running, note it

5. **Meta-persistence:**
   - If `/tmp/dw-artifacts/meta.json` exists, report: session count, dominant clusters, Gini meta-trajectory direction, sheaf consistency rate
