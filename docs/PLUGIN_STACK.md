# Plugin Stack — JTopo Development Environment

This document describes the Claude Code plugin configuration for the JTopo/Ti V0.1 project. Each plugin was selected to support computational mathematics research, plugin development, and the specific workflow demands of sheaf-valued persistent homology on zeta zero point clouds.

## Installed Plugins

| Plugin | Source | Version |
|---|---|---|
| driftwave | jtopo-plugins (local) | dev |
| pyright-lsp | claude-plugins-official | 1.0.0 |
| superpowers | claude-plugins-official | 5.0.5 |
| plugin-dev | claude-plugins-official | latest |
| github | claude-plugins-official | latest |
| episodic-memory | superpowers-marketplace | 1.0.15 |
| semgrep | claude-plugins-official | 0.5.1 |
| code-review | claude-plugins-official | latest |
| double-shot-latte | superpowers-marketplace | 1.2.0 |

---

## Plugin Breakdown

### 1. driftwave (Local — jtopo-plugins)

**What it is:** Our own ATFT cognitive pipeline plugin. Implements the four-layer filtration (L0→L1→L2→L3) with five governing axioms.

**Skills:**
- `/dw-map` — L0 raw artifact ingestion with entropy gate
- `/dw-filter` — L1 persistent clustering via H₀ barcodes
- `/dw-ascend` — L2/L3 Gini routing and sheaf Laplacian convergence
- `/wavefront` — Full pipeline orchestrator (L0→L1→L2→L3)
- `/topological-brainstorm` — Brainstorming-as-filtration for design work
- `/boundary-mode` — L3 cross-system topological collaboration

**Agent:** `gini-watchdog` — background monitor for Gini trajectory during L2/L3 work

**When to use:**
- **Architecture decisions** — when the decomposition isn't obvious and you need the pipeline to discover structure rather than guess at it
- **Before any major implementation** — `/wavefront` enforces the hard gate: no code until the topological structure converges
- **Creative/design sessions** — `/topological-brainstorm` replaces ad-hoc brainstorming with filtration-based ideation
- **Cross-system integration** — `/boundary-mode` when working across Ti V0.1's multiple backends (CPU/CuPy/PyTorch) and ensuring they stay globally consistent

**When NOT to use:**
- Quick bug fixes or small patches — the full pipeline is overhead for trivial changes
- When you already know exactly what to build — skip straight to implementation

**Status:** In development, not publicly released. Local install only.

---

### 2. pyright-lsp (Official)

**What it is:** Python language server providing real-time static type checking and code intelligence via Pyright.

**Requires:** `pip install pyright` or `pipx install pyright`

**When to use:**
- **Always-on** — this runs in the background during any Python editing session
- **Transport map development** (`atft/topology/transport_maps.py`) — catches K×K matrix shape mismatches before runtime, which otherwise surface as silent numerical errors deep in GPU compute
- **Eigenvalue computation code** — type checking on NumPy/SciPy/CuPy array operations prevents the class of bugs where you pass a 1D array where a 2D is expected and get a wrong answer instead of an error
- **Sheaf Laplacian assembly** — the block structure (NK×NK) has complex indexing; type checking catches off-by-one in block placement

**Why it matters for this project specifically:**
Ti V0.1's computational code handles matrices at scale (N=9877 nodes, K=100+ fiber dimensions). A type error in the transport map assembly doesn't crash — it produces subtly wrong eigenvalues that waste hours of GPU time before you notice the spectral sum doesn't match expectations. Catching these at edit time instead of after a full sweep run is the difference between minutes and hours of debugging.

---

### 3. superpowers (Official — v5.0.5)

**What it is:** Battle-tested skills library with 20+ skills for TDD, systematic debugging, and collaboration patterns. This is the established standard for Claude Code enhanced workflows.

**Key skills:**
- `/brainstorm` — rapid ideation and ranking (your existing staple)
- `/write-plan` — structured implementation planning
- `/execute-plan` — step-by-step plan execution with progress tracking

**When to use:**
- **Quick ideation** — when you need volume and speed rather than topological rigor. `/brainstorm` generates and ranks ideas fast. Use this for well-scoped problems where you already understand the structure
- **Implementation planning** — after driftwave's pipeline has converged on a structure, use `/write-plan` to break it into executable steps
- **TDD workflows** — when writing tests for the computational pipeline (23 tests in the test suite and growing)
- **Debugging** — systematic debugging skills for when eigenvalue computations go wrong or transport map assembly produces unexpected results

**How it relates to driftwave:**
These are complementary, not competing. Use `/brainstorm` when the problem is well-scoped (e.g., "what's the best way to batch these eigendecompositions on GPU?"). Use `/topological-brainstorm` when the problem is ambiguous (e.g., "how should we restructure the compute pipeline to support K=200?"). Use `/write-plan` + `/execute-plan` after either one converges.

---

### 4. plugin-dev (Official)

**What it is:** Comprehensive toolkit for building Claude Code plugins, with 7 specialized skills covering every aspect of plugin development.

**Skills:**
- `/plugin-dev:plugin-structure` — design plugin directory layouts and manifests
- `/plugin-dev:skill-development` — create skills with progressive disclosure
- `/plugin-dev:agent-development` — create autonomous agents with system prompts
- `/plugin-dev:hook-development` — create PreToolUse, PostToolUse, Stop hooks
- `/plugin-dev:mcp-integration` — configure MCP servers for external services
- `/plugin-dev:command-development` — create slash commands with arguments
- `/plugin-dev:plugin-settings` — configure per-project plugin state
- `/plugin-dev:create-plugin` — end-to-end plugin creation workflow (8 phases)

**When to use:**
- **Iterating on driftwave** — when adding new skills, refining the gini-watchdog agent, or adding new hooks to the pipeline
- **Testing plugin structure** — validate that skill frontmatter, agent definitions, and hook configs are correct before committing
- **Adding MCP integration** — if driftwave needs to connect to external services (e.g., a GPU compute scheduler, a results database)
- **Creating new plugins** — if the ATFT framework spawns additional plugins beyond driftwave

**Why it matters now:**
Driftwave is in active development. The plugin-dev toolkit provides the reference implementation patterns so we don't have to guess at schema formats or discover validation errors by trial and error (like the marketplace.json issue we hit earlier).

---

### 5. github (Official — MCP)

**What it is:** Official GitHub MCP server providing full GitHub API access from within Claude Code sessions.

**Requires:** `GITHUB_PERSONAL_ACCESS_TOKEN` environment variable set

**What it provides:**
- Create and manage issues and pull requests
- Code review and PR comments
- Repository search and file access
- Branch and release management

**When to use:**
- **Issue tracking** — create issues for K=100 sweep tasks, track falsification criteria results, log experimental observations
- **PR management** — create PRs for compute engine changes without leaving the session
- **Cross-repo coordination** — JTopo, mpd-overwatch, and jtech-platform may need coordinated changes
- **Release management** — when driftwave is ready for public release, manage the GitHub release process

**Configuration note:**
Set the token via: `export GITHUB_PERSONAL_ACCESS_TOKEN=<your-token>` in your shell profile. Without it, the MCP server won't authenticate.

---

### 6. episodic-memory (Superpowers — v1.0.15)

**What it is:** Semantic conversation search and cross-session memory. Remembers context from previous Claude Code sessions and makes it searchable.

**When to use:**
- **Resuming multi-day research** — recall what the K=50 spectral results looked like, what parameters were tried, what the Gini trajectory showed
- **Connecting experimental observations** — "when did we first see the ε=3.0 reversal?" without having to grep through git logs
- **Tracking reasoning evolution** — the ATFT framework has evolved through multiple phases; episodic memory preserves the reasoning behind decisions, not just the code changes
- **Cross-session context** — when you start a new session and need to pick up where you left off on a sweep or analysis

**Why it matters for this project specifically:**
Computational math research is inherently multi-session. You run a K=50 sweep in one session, analyze results in another, adjust parameters in a third. Without episodic memory, each session starts cold and you waste time re-establishing context. The project has 70+ commits of history — episodic memory preserves the *reasoning* behind those commits, not just the diffs.

---

### 7. semgrep (Official — v0.5.1)

**What it is:** Real-time static analysis and security vulnerability detection using Semgrep's 5,000+ rules.

**When to use:**
- **File I/O safety** — Ti V0.1 reads large data files (Odlyzko zeros, 1.8MB) and writes JSON results. Semgrep catches unsafe deserialization, path traversal, and injection patterns
- **Script security** — `topo.sh` and other scripts handle user input and file paths. Semgrep flags shell injection risks
- **Dependency auditing** — the project uses NumPy, SciPy, CuPy, PyTorch — Semgrep checks for known vulnerable patterns in how these libraries are used
- **Before publishing driftwave** — when the plugin goes public, users will run its hooks and scripts on their machines. Security scanning before release is essential

**When NOT to use:**
- Pure mathematical analysis sessions where you're not writing code
- Quick exploratory scripts that won't be committed

---

### 8. code-review (Official)

**What it is:** Automated PR review using 4 parallel agents with confidence-based scoring (threshold: 80/100) to filter false positives.

**Agents it launches:**
1. Two CLAUDE.md compliance checkers — verify changes follow project guidelines
2. One bug detector — focused on changed lines only
3. One history analyzer — uses git blame to understand context of modified areas

**Skill:** `/code-review` — run on current PR

**When to use:**
- **Before merging compute engine changes** — the transport map and eigenvalue code is the core of the project. A subtle bug in sheaf Laplacian assembly that passes tests but produces wrong spectral sums could invalidate weeks of results. Automated review catches what unit tests miss
- **After GPU backend changes** — CuPy/PyTorch backends were cross-validated to 1.5e-15 precision. Code review ensures new changes maintain that invariant
- **Before major experiment runs** — a review before launching a K=100 full sweep (which takes significant GPU time) is worth the 2 minutes it takes
- **Plugin development PRs** — catches issues in driftwave skills, hooks, and agent definitions

**Requires:** Git repo with GitHub remote, `gh` CLI authenticated

---

### 9. double-shot-latte (Superpowers — v1.2.0)

**What it is:** Auto-continue decision making. When Claude Code would normally stop and ask "should I continue?", this plugin evaluates whether to proceed automatically.

**When to use:**
- **Long analysis sessions** — reading through the full Ti V0.1 codebase, analyzing experimental results across multiple files, reviewing all 23 tests
- **Multi-file refactoring** — when updating the compute pipeline across CPU, CuPy, and PyTorch backends simultaneously
- **Extended brainstorming** — `/topological-brainstorm` and `/wavefront` can be multi-step pipelines. Auto-continue keeps the flow going without manual "yes, continue" interrupts
- **Sweep result analysis** — parsing through K=20, K=50, K=100 results files sequentially without pausing at each one

**When NOT to use:**
- When you want explicit control over each step (e.g., destructive git operations, pushing to remote)
- When exploring unfamiliar code where you want to review intermediate results

---

## Plugin Interaction Map

How these plugins work together in practice:

```
                        RESEARCH WORKFLOW
                        ─────────────────
Discovery Phase:
  episodic-memory ──→ recall prior session context
  superpowers ───────→ /brainstorm for quick ideation
  driftwave ─────────→ /topological-brainstorm for structural discovery
                       /wavefront for full pipeline

Implementation Phase:
  superpowers ───────→ /write-plan → /execute-plan
  pyright-lsp ───────→ real-time type checking (always on)
  semgrep ───────────→ security scanning (always on)
  double-shot-latte ─→ auto-continue during long coding sessions

Review Phase:
  code-review ───────→ /code-review on PR before merge
  github ────────────→ create PR, manage issues, track experiments

Plugin Development:
  plugin-dev ────────→ iterate on driftwave skills/agents/hooks
  driftwave ─────────→ test the plugin itself
```

## Recommended Workflow

1. **Start session** → episodic-memory loads prior context automatically
2. **Scope the work** → `/brainstorm` (quick) or `/topological-brainstorm` (rigorous)
3. **Plan** → `/write-plan` after approach converges
4. **Implement** → pyright-lsp + semgrep catch errors in real-time; double-shot-latte keeps flow going
5. **Review** → `/code-review` on the PR
6. **Ship** → github plugin for PR creation and issue management

## Setup Requirements

| Plugin | Requires |
|---|---|
| pyright-lsp | `pipx install pyright` |
| github | `GITHUB_PERSONAL_ACCESS_TOKEN` env var |
| code-review | `gh` CLI authenticated |
| semgrep | Works out of the box |
| All others | No additional setup |
