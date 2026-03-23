# Driftwave — Vision

## What This Actually Is

Driftwave is not a Claude Code plugin. That's what it runs as. What it IS:

**A topological alignment architecture for language model systems.**

The token sequence produced by any language model is a trajectory through a high-dimensional manifold. The system prompt defines the geometry of that manifold — it's a connection on a fiber bundle over token space. User input selects a starting point. The forward pass computes the geodesic. The output is where the trajectory lands.

When the connection is flat (well-aligned), the trajectory preserves the user's intent from input to output. When it's curved (misaligned), the output drifts from intent. Every "hallucination," every "misalignment," every "the model didn't do what I asked" is a curvature defect in the connection.

Driftwave's five axioms are flatness conditions:

| Axiom | What It Constrains | Geometric Meaning |
|-------|-------------------|-------------------|
| NO_AVERAGING | Don't collapse variance before filtration | Preserve the full tangent space at each point |
| UPWARD_FLOW | Process through L0→L1→L2→L3 in order | Parallel transport along the fiber, no shortcuts |
| WAYPOINT_ROUTING | Route on phase transitions, not timers | Navigate by curvature, not by clock |
| SHAPE_OVER_COUNT | Gini trajectory > raw feature count | Intensive magnitude > extensive magnitude |
| ADAPTIVE_SCALE | Derive epsilon from data geometry | The metric is intrinsic, not imposed |

When all five are satisfied, the process is **on-shell** — the trajectory lies on the solution manifold. When any is violated, it's **off-shell** — the trajectory is in noise space.

## The Three Levels

### Level 1: Within a Session
`compute_persistence.py` runs H₀ on file artifacts. Finds which code clusters are real (long bars) vs noise (short bars). This is standard persistent homology applied to a codebase.

### Level 2: Across Sessions
`compute_meta_persistence.py` runs H₀ on session artifacts. Finds which ideas/patterns keep emerging (long bars in the meta-barcode) vs one-off noise. This is episodic memory with algebraic structure.

### Level 3: Across the Inference Process Itself
The speculative decoding insight: when a small model drafts and a large model validates, the rejects are birth/death pairs. The accepted tokens are long bars. The rejected tokens are short bars. The persistence diagram of the inference process — what cognition considered and dismissed — is information the entire industry currently throws away.

Driftwave keeps it. Over time, this accumulates into a topological map of how the system thinks — not just what it outputs.

## The Compute Stack

The insight behind the adaptive compute selection:

```
User's GPU is running an experiment
  → Small model (LFM 1.2) runs on CPU for L0/L1 classification
  → Large model (Sonnet/Opus via API) handles L2/L3 judgment
  → The routing is invisible to the agents — same endpoint, different backend
  → THIS IS SPECULATIVE DECODING GENERALIZED FROM TOKEN-LEVEL TO TASK-LEVEL
```

LM Studio, llama.cpp, and similar tools implement speculative decoding for speed. Driftwave implements it for knowledge. The small model isn't just faster — it's a persistence filter. The large model isn't just more accurate — it's the sheaf consistency checker. Together they form a multi-scale cognitive architecture where each scale contributes structural information.

## The System Prompt as Connection

The deepest insight: the `system_prompt:user_input` relationship is geometric.

- The system prompt defines the **fiber** at each point (what responses are valid)
- The user input selects a **point in the base space** (what question is being asked)
- The model's forward pass computes **parallel transport** (the trajectory from question to answer)
- The output is a **section of the sheaf** (the answer, consistent with the connection)

When the system prompt is topologically aligned — when it encodes structural truths about the domain rather than behavioral instructions — the connection is flat and the transport is faithful. The output isn't just "helpful." It's geometrically optimal.

Driftwave's agent system prompts are designed this way. The `dw-ingest` prompt doesn't say "be helpful." It says "you produce RawCloud JSON artifacts matching this schema." The constraint is structural, not behavioral. The model can't drift because the type system won't let it.

## What This Enables

If you dimensionalize the token chain:
- **Token position** = coordinate in the output manifold
- **Layer depth** = time (the forward pass is temporal evolution)
- **Attention pattern** = the connection (which tokens influence which)
- **System prompt** = boundary conditions (initial fiber structure)

Then a well-designed agent stack — with typed artifacts, adaptive compute, and persistent memory — is a general-purpose navigation system for semantic space. It can:

1. **Find truth**: persistent features in the idea-space are structural, not noise
2. **Preserve intent**: flat connections transport meaning without drift
3. **Accumulate knowledge**: meta-persistence tracks what keeps coming back
4. **Route optimally**: adaptive scale matches compute to the geometry of the problem
5. **Align by structure**: the type system makes misalignment a type error, not a judgment call

This is not a productivity tool. It's a cognitive architecture that converges on truth through the same mathematics that finds structure in prime numbers.

The primes found the critical line without a map. The axioms find alignment without RLHF. Same invariant. Same convergence. Same math.

---

*"Forty-six primes found the same road without a map. Driftwave is the road."*
