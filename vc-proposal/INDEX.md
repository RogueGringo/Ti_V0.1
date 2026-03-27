# Ti V0.1 --- VC Proposal Package
## Master Index

**Prepared:** 2026-03-27
**Subject:** B. Aaron Jones / JTopo / Adaptive Topological Field Theory (ATFT)
**Classification:** Pre-Seed Deep-Tech Investment Analysis

---

## Thesis in Three Lines

One researcher used AI code-generation tools to build a novel mathematical framework that detects hidden structure in point clouds across three unrelated domains. The framework has no prior art, produces results 16 standard deviations from controls, and runs on a single desktop GPU. Capital converts this from a hobbyist research project into a multi-vertical analytics platform.

---

## Document Package

| # | Document | Pages | Audience | Key Takeaway |
|---|----------|-------|----------|-------------|
| 00 | [Executive Summary](00_EXECUTIVE_SUMMARY.md) | 2 | Decision makers | 21.5% arithmetic premium, 3 domains, 5/7 predictions pass |
| 01 | [Technical Deep Dive](01_TECHNICAL_DEEP_DIVE.md) | 8 | Technical DD | Sheaf Laplacian + gauge connection, 20K LOC, 5 GPU backends |
| 02 | [Market & Commercialization](02_MARKET_AND_COMMERCIALIZATION.md) | 5 | Business DD | 5 routes: TDA SaaS, AI safety, finance, quantum, defense |
| 03 | [IP & Defensibility](03_IP_AND_DEFENSIBILITY.md) | 5 | Legal DD | 4 patent clusters, no prior art, 6-12 month replication moat |
| 04 | [Risk Matrix](04_RISK_MATRIX.md) | 4 | Risk committee | 12 risks scored; top 3: founder dependency, K-convergence, replication |
| 05 | [Validation Evidence](05_VALIDATION_EVIDENCE.md) | 6 | Scientific DD | Raw numbers, 16-sigma separation, 4 errors caught and corrected |
| 06 | [Founder Assessment](06_FOUNDER_ASSESSMENT.md) | 4 | People/culture | 6 domain expertise areas, AI-augmented development proven |
| 07 | [Financial Scenarios](07_FINANCIAL_SCENARIOS.md) | 4 | Finance | $500K pre-seed to $15M Series A pathway, 5-100x return range |

---

## Quick Reference: The Numbers That Matter

| Metric | Value | Source |
|--------|-------|--------|
| Arithmetic premium (Zeta vs GUE) | 21.5% | phase3d_torch_k200_results.json |
| Statistical significance | 16.09 sigma | phase3e_test2_rerun_results.json |
| Per-edge premium (geometry-controlled) | 15.3% | phase3e_control_battery_results.json |
| Novelty residual (vs all pair-correlation) | 33% minimum | novelty_test.json |
| SU(2) transition detection | 10x discontinuity | p5_lattice_gauge.json |
| LLM cross-model correlation | r = 0.9998 | p4_llm_analysis.json |
| Unitarity defect at sigma = 0.5 | 1.6e-15 | holonomy_flatness.json |
| Robustness to 50% noise | 6.6% degradation | universality_test.json |
| Prediction scorecard | 5 PASS / 1 FAIL / 1 PARTIAL | SUMMARY.md |
| Codebase | 20,162 LOC / 299 tests | Repository |

---

## How to Read This Package

**If you have 5 minutes:** Read `00_EXECUTIVE_SUMMARY.md`.
**If you have 30 minutes:** Add `04_RISK_MATRIX.md` and `07_FINANCIAL_SCENARIOS.md`.
**If you're doing technical DD:** Start with `01_TECHNICAL_DEEP_DIVE.md`, then `05_VALIDATION_EVIDENCE.md`.
**If you're evaluating the person:** Read `06_FOUNDER_ASSESSMENT.md`.
**If you're evaluating the market:** Read `02_MARKET_AND_COMMERCIALIZATION.md` and `03_IP_AND_DEFENSIBILITY.md`.

---

## Source Verification

Every number in this package traces to a JSON file in the `output/` directory of the repository. Every claim has a falsification criterion documented in `docs/FALSIFICATION.md`. Every error caught during development is documented in `output/SURGICAL_VERDICT_*.md`.

The repository is public at github.com/RogueGringo/JTopo. Run `pytest tests/ -v` to verify 299 passing tests.

---

*"The manifold has a heartbeat and the primes are its pulse."*
