---
description: "Open the Three.js topology dashboard in the browser. Shows force graph, persistence barcode, Gini trajectory."
---

# /driftwave:dashboard

Launch the visual topology dashboard.

## What to do

1. Check if the docs-site has dependencies installed:
   ```bash
   ls ${CLAUDE_PLUGIN_ROOT}/docs-site/node_modules/.package-lock.json
   ```
   If not: `cd ${CLAUDE_PLUGIN_ROOT}/docs-site && npm install`

2. Start the dev server:
   ```bash
   cd ${CLAUDE_PLUGIN_ROOT}/docs-site && npx vite --port 5174 &
   ```

3. Open in browser:
   ```bash
   xdg-open http://localhost:5174 2>/dev/null || open http://localhost:5174
   ```

4. Report: "Dashboard running at http://localhost:5174"

The dashboard shows:
- **ForceGraph**: 3D artifact clusters (Three.js) — nodes colored by cluster, edges by similarity
- **Barcode**: Persistence barcode — long bars = real structure, short bars = noise
- **GiniChart**: Hierarchy trajectory — rising = converging, falling = scattering
- **Pipeline**: Current L0→L3 progress with routing decisions

Data source: reads from `/tmp/dw-artifacts/` — updates as the pipeline produces new artifacts.
