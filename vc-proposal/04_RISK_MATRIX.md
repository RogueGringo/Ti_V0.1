# Ti V0.1 --- Risk Matrix
## Honest Assessment of All Failure Modes

---

## 1. Scientific Risks

### RISK S1: K -> Infinity Convergence Failure
**Probability:** LOW (15%)
**Impact:** CRITICAL
**Evidence for mitigation:**
- K=200 premium: 21.5%
- K=400 premium: 21.6% (Wigner surmise GUE)
- K=800 premium: 9.3% (significant drop)
- The K=800 drop is the most important unresolved question

**Status:** The premium converges from K=200 to K=400 but drops at K=800. This could mean:
(a) The invariant has a finite operating range (K=100-400), or
(b) K=800 needs better GUE controls (current K=800 uses single Wigner surmise, not Dumitriu-Edelman ensemble)

**Mitigation:** K=800 with proper D-E ensemble is the next experiment. If premium recovers, the invariant is confirmed. If it doesn't, the operating range is bounded, which is still commercially useful.

### RISK S2: Statistical Artifact
**Probability:** LOW (10%)
**Impact:** CRITICAL
**Evidence against:**
- 16 standard deviations from GUE ensemble mean (10 proper D-E realizations)
- Per-edge normalization survives (15.3% after controlling for edge count)
- Novelty test confirms 33% minimum residual against all pair-correlation predictors
- Pre-frozen falsification criteria with documented corrections

**Residual concern:** The "ProperGUE" control in the phase 3e battery initially had a rank-unfolding bug that made all realizations identical. This was caught and fixed (semicircle CDF unfolding). The fix is documented. The corrected 10-realization ensemble gives mean=14.970, std=0.198, confirming zeta falls 16 sigma below.

### RISK S3: Framework is Domain-Specific, Not General
**Probability:** MEDIUM (30%)
**Impact:** HIGH
**Evidence against:**
- Three validated domains (number theory, SU(2) gauge theory, LLM hidden states)
- Same operator, different generators, different results
- The mathematical structure (sheaf cohomology on Rips complex) is provably general

**Residual concern:** The three domains share a common property---they all involve eigenvalue-like spectra. Financial, biological, or social network point clouds may not respond to the same framework. Domain generator design is nontrivial.

### RISK S4: The RH Connection is Coincidental
**Probability:** MEDIUM (25%)
**Impact:** MEDIUM (doesn't affect other applications)
**What this means:** The 21.5% premium could reflect a property of the Odlyzko zero dataset (specific height regime near 10^20th zero) rather than a universal property of all zeta zeros.

**Mitigation:** Test with zeros at different heights. Test with Dirichlet L-function zeros. If premium persists, it's structural; if not, it's height-dependent.

---

## 2. Technical Risks

### RISK T1: Founder Dependency
**Probability:** HIGH (80%)
**Impact:** CRITICAL
**Description:** B. Aaron Jones is the sole author, sole validator, and sole domain expert. The mathematical framework requires simultaneous expertise in algebraic topology, number theory, gauge theory, GPU computing, and statistical methodology. This combination is extremely rare.

**Mitigation:**
- Codebase is well-documented (README, surgical verdicts, 30+ documentation files)
- 299 passing tests ensure correctness is verifiable
- Hiring plan should prioritize a mathematical physicist + GPU engineer in first 3 hires
- Founder should publish paper immediately to establish priority and enable external validation

### RISK T2: GPU Scaling Bottleneck
**Probability:** LOW (10%)
**Impact:** MEDIUM
**Description:** K=800 requires 26 minutes per computation on RTX 5070. Scaling to K=2000+ would require A100/H100 or distributed computation.

**Mitigation:** Matrix-free engine already solves memory scaling. Time scaling requires either:
(a) Multi-GPU parallelism (straightforward for edge-parallel matvec)
(b) KPM approximation (already implemented, trades precision for speed)
(c) Cloud GPU for burst compute (cost-effective for research; production needs on-prem)

### RISK T3: Reproducibility on Different Hardware
**Probability:** LOW-MEDIUM (20%)
**Impact:** MEDIUM
**Description:** Results were validated on specific hardware (Threadripper 7960X + RTX 5070). Numerical precision of GPU eigendecomposition varies across architectures.

**Mitigation:**
- Regression tests capture golden values to 10^-5 precision
- Core eigenvalue ratios have 0.9% coefficient of variation (robust to floating-point variation)
- CI/CD runs on CPU (GitHub Actions); GPU testing on local hardware

---

## 3. Market Risks

### RISK M1: TDA Market Remains Niche
**Probability:** MEDIUM (35%)
**Impact:** MEDIUM
**Description:** Despite academic interest, TDA has not achieved mass-market enterprise adoption. Ayasdi was acquired at below expectations.

**Mitigation:** ATFT has multiple commercial routes. Even if TDA SaaS fails, AI interpretability (Route B) and financial signal detection (Route D) don't require the TDA market to exist.

### RISK M2: Large Lab Replicates Internally
**Probability:** MEDIUM (40%)
**Impact:** HIGH
**Description:** If Google DeepMind or OpenAI builds internal sheaf-valued TDA, the commercial advantage evaporates.

**Mitigation:**
- Patent filings create legal protection
- 6-12 month head start in cross-domain validation
- Academic publication establishes priority
- Honest error documentation builds trust that marketing cannot replicate

### RISK M3: Regulatory / Export Control
**Probability:** LOW (10%)
**Impact:** MEDIUM
**Description:** Dual-use mathematical tools for defense applications may face ITAR/EAR restrictions.

**Mitigation:** Core mathematical framework is published academic research (not restricted). Only classified domain-specific generators would face export control.

---

## 4. Team / Execution Risks

### RISK E1: Solo Researcher Burnout
**Probability:** HIGH (60%)
**Impact:** HIGH
**Description:** One person doing mathematics, engineering, validation, documentation, and commercial development is unsustainable.

**Mitigation:** First funding should hire:
1. Mathematical physicist (validate/extend theory)
2. GPU engineer (production-grade backends)
3. Business development (customer discovery)

### RISK E2: Academic vs Commercial Tension
**Probability:** MEDIUM (30%)
**Impact:** MEDIUM
**Description:** The honest, falsification-first research culture may conflict with commercial pressure to overstate results.

**Mitigation:** The surgical verdict documentation style IS the commercial advantage. Technical buyers trust honest error documentation more than marketing. Preserve this culture explicitly in company values.

---

## 5. Risk Summary Matrix

| Risk | Probability | Impact | Risk Score | Mitigation Status |
|------|------------|--------|-----------|-------------------|
| S1: K-convergence | 15% | Critical | HIGH | K=800 D-E ensemble needed |
| S2: Statistical artifact | 10% | Critical | MEDIUM | 16-sigma, pre-frozen criteria |
| S3: Domain-specific | 30% | High | HIGH | 3 domains validated |
| S4: RH coincidental | 25% | Medium | MEDIUM | Test different zero heights |
| T1: Founder dependency | 80% | Critical | CRITICAL | Documentation + hiring plan |
| T2: GPU scaling | 10% | Medium | LOW | Matrix-free engine exists |
| T3: Reproducibility | 20% | Medium | LOW | Regression tests + CI |
| M1: TDA niche | 35% | Medium | MEDIUM | Multiple routes |
| M2: Large lab replicates | 40% | High | HIGH | Patents + publication |
| M3: Export control | 10% | Medium | LOW | Academic publication |
| E1: Burnout | 60% | High | HIGH | First hires critical |
| E2: Culture tension | 30% | Medium | MEDIUM | Preserve surgical verdict culture |

**Top 3 risks requiring immediate action:**
1. **T1: Founder dependency** --- Hire mathematical physicist immediately
2. **S1: K-convergence** --- Run K=800 with proper D-E ensemble
3. **M2: Large lab replication** --- File provisional patents before paper publication
