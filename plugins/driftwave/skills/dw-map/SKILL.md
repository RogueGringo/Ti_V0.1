---
name: dw-map
description: "Use when ingesting raw artifacts, probes, requirements, or context at the start of any analysis. L0 layer — preserves full variance structure without averaging. Triggers when collecting raw data, exploring project state, or beginning any investigation."
---

# L0: Raw Artifact Ingestion

The foundational layer. Every analysis begins here. No exceptions.

## Core Operation

Ingest raw artifacts into an unaveraged point cloud. Each datum is a discrete point in a metric space. The variance structure across these points IS the topological signal.

## The NO_AVERAGING Axiom

**This is non-negotiable.** Raw probes, requirements, codebase signals, and constraint statements must never be aggregated, summarized, or flattened before filtration.

Premature averaging is a category error — it destroys the persistence signal that reveals which features are robust versus topological noise. The noise floor is not noise. It is the low-level deductive artifact from which high-level abstraction must be reverse-engineered.

| Input Type | Point Cloud Encoding | What Variance Reveals |
|---|---|---|
| Codebase files | Each file = point, distance = functional dissimilarity | Module structure |
| User requirements | Each answer = point, distance = semantic dissimilarity | Requirement clustering |
| Probe traces | Each probe = point, distance = numerical difference | Signal vs noise boundary |
| Zeta zeros | Each gap = point, distance = spectral separation | Arithmetic structure |

## Entropy Gate

Before proceeding to L1, check: does the point cloud have sufficient variance?

- **High entropy** (spread, differentiated points): Proceed to `/dw-filter`
- **Zero/low entropy** (flat, undifferentiated): REPROBE — collect more raw artifacts, ask more probing questions, run more experiments
- **Never proceed with a flat input space.** A zero-variance point cloud cannot produce meaningful persistent homology.

## Protocol

1. Collect all raw artifacts without interpretation
2. Encode each as a point in the appropriate metric space
3. Preserve the full pairwise distance matrix
4. Check entropy gate
5. Output: raw point cloud ready for filtration

## Axiom Violations

If at any point someone attempts to "summarize what we have so far" or "average the results" before filtration has occurred:

```
[DW-AXIOM-VIOLATION: NO_AVERAGING]
Raw probes must reach L1 at full resolution.
Averaging before filtration destroys the topological signal.
```
