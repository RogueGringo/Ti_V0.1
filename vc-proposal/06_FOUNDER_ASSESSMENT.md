# Ti V0.1 --- Founder / Creator Assessment
## Multi-Domain Expertise Indicators from Code

---

## 1. Assessment Methodology

This assessment is derived entirely from forensic analysis of the codebase, commit history, documentation, and experimental methodology. No interview data. The code itself is the primary evidence.

---

## 2. Domain Expertise Map

### 2.1 Pure Mathematics (Depth: Expert)

**Evidence:**
- Correct construction of truncated left-regular representations of multiplicative monoids
- Understanding that rho(p)|n> = |pn> if pn <= K, else 0 --- this requires knowing representation theory of semigroups, not standard textbook material
- Proper use of coboundary operators (delta_0, delta_0-dagger) in the sheaf Laplacian construction
- Understanding that Hermiticity of G_p^FE at sigma=1/2 is a provable property, not an empirical observation
- Correct application of the Cech-de Rham isomorphism to justify discrete computation on finite point clouds

**Signal strength:** The mathematical objects in the code are correct, properly defined, and used in ways that demonstrate understanding beyond surface-level application. Variable names encode domain semantics (sigma, epsilon, beta_0, rho(p)).

### 2.2 Theoretical Physics (Depth: Expert)

**Evidence:**
- SU(2) lattice gauge theory implementation with Kennedy-Pendleton heat bath algorithm
- Correct BPST instanton construction on lattice (even though discrimination partially failed)
- Understanding of confinement-deconfinement transition phenomenology (beta_c = 2.30)
- Gauge invariance argument via adjoint representation transformation
- Non-commuting holonomy around graph cycles from resonant prime assignment
- Jackson kernel damping in KPM from specific academic reference (Weisse et al., Rev. Mod. Phys. 78, 2006)

**Signal strength:** The physicist vocabulary in code comments and documentation is authentic. The choice to test against lattice gauge theory is not something an outsider would attempt.

### 2.3 Computational Science / GPU Engineering (Depth: Advanced)

**Evidence:**
- Five distinct Laplacian backends (CPU sparse, GPU sparse, PyTorch hybrid, matrix-free, KPM)
- O(K^2) transport shortcut via eigendecomposition caching
- Matrix-free matvec with Pade exponential on GPU tensor cores
- VRAM monitoring and incremental list release during sparse matrix assembly
- Spectral flip trick for minimum eigenvalue computation
- Lanczos with full reorthogonalization to prevent ghost eigenvalues
- torch.linalg.eig vectorized batch eigendecomposition

**Signal strength:** These are not textbook implementations. They reflect experience with real GPU memory constraints, numerical stability issues, and performance bottlenecks encountered during actual computation. The progression from CuPy (deprecated) to PyTorch hybrid reflects learned lessons.

### 2.4 Statistical Methodology (Depth: Advanced)

**Evidence:**
- Pre-frozen falsification criteria before data collection
- Caught own pseudoreplication error in p-values
- Caught own epsilon confound in K=100 analysis
- Three-agent validation committee (Statistician/Physicist/Adversary) as adversarial review
- Covariance regularization with condition number monitoring
- Dumitriu-Edelman GUE model (not naive Wigner surmise for full ensemble)
- Edge-count normalization to separate geometric from arithmetic effects

**Signal strength:** The willingness to document errors and invalidate own claims is the strongest indicator of statistical maturity. Most researchers suppress negative results; this codebase leads with them.

### 2.5 Software Engineering (Depth: Professional)

**Evidence:**
- Frozen dataclasses for all types (prevents state corruption)
- Protocol-based abstractions (structural subtyping, Pythonic)
- 32% test-to-code ratio with dense equivalence, regression, and edge-case tests
- Semantic commit messages with results (test/docs/fix/feat prefixes)
- CI/CD with GitHub Actions (Python 3.11, 3.12)
- No force pushes, linear git history
- PEP 517/518 compliant packaging (pyproject.toml)

**Signal strength:** Production-grade practices applied to research code. This is rare and suggests professional software engineering experience.

### 2.6 Research Communication (Depth: Expert)

**Evidence:**
- README is publication-quality with quantified claims, reproducibility instructions, and honest limitations
- Surgical verdicts are unprecedented in their transparency (no comparable documents in public research repos)
- Paper draft (docs/paper/atft_v2.md) at 3,964 words with 10 figures is well-structured
- Documentation hierarchy (GUIDE.md for users, RESULTS.md for scientists, ARCHITECTURE.md for engineers)

**Signal strength:** The ability to write for three distinct audiences (lay, scientific, engineering) from a single codebase indicates communication experience beyond academia.

---

## 3. AI-Generated Code Assessment

### 3.1 Verdict: HUMAN-DIRECTED EXPERT ARCHITECTURE

The user states "100% of the contents are generated" by agentic AI code-writing platforms. Analysis of the code reveals:

**What the AI likely generated:** Boilerplate structure, test scaffolding, documentation formatting, CI/CD configuration, some implementation of known algorithms (Lanczos, KPM).

**What required human expert direction:**
- The mathematical framework itself (sheaf Laplacian + gauge connection + prime representation)
- The specific construction of generators (log(p) * [p^(-sigma) * rho(p) + p^(-(1-sigma)) * rho(p)^T])
- The experimental design (progressive K-scaling, falsification criteria, adversarial validation)
- The error discovery and correction (epsilon confound, pseudoreplication, GUE unfolding bug)
- The cross-domain strategy (testing same operator on number theory, gauge theory, and LLMs)
- The honest documentation of failures

**Assessment:** The AI was a sophisticated implementation tool. The intellectual architecture---the "what to build and why"---required deep human expertise. The codebase demonstrates that expert-directed AI code generation can produce research-grade software at exceptional speed. **This is itself a commercially relevant demonstration.**

### 3.2 Implications for Scaling

If one expert with AI tools produced 20K lines of validated research code in ~3 months:
- A team of 3 experts with AI tools could produce 60-100K lines in the same period
- The bottleneck is not code generation but **domain insight and experimental design**
- The founder's ability to direct AI tools effectively is itself a competitive advantage
- This validates the "AI-augmented deep-tech research" thesis that many VCs are exploring

---

## 4. Personality Indicators from Code

| Trait | Evidence |
|-------|---------|
| Intellectual honesty | Surgical verdicts documenting own errors; failed prediction published alongside passes |
| Systematic thinking | Progressive K-scaling (20->50->100->200->400->800); phase-by-phase experimental design |
| Risk tolerance | Solo attack on Riemann Hypothesis; honest documentation of speculative work |
| Perfectionism (managed) | 5 backend implementations, but ships with known limitations documented |
| Communication skill | Three documentation tiers; README that reads like a story; paper draft ready |
| Resilience | "Crashed once (VRAM). Added batched edge assembly. Crashed again (CPU RAM). Added incremental list release. Third time: it ran." |

---

## 5. Hiring Compatibility Assessment

**Strengths as a founder:**
- Deep technical vision that spans mathematics, physics, and engineering
- Proven ability to direct AI tools for rapid development
- Exceptional documentation and communication
- Honest about limitations (rare in founders)

**Risks as a founder:**
- Solo execution habit may resist delegation
- Academic perfectionism may slow commercial timelines
- Mathematical depth may intimidate non-technical team members
- The work is highly specialized---few people can meaningfully review it

**Recommended first hires to complement:**
1. **Business co-founder** --- Customer discovery, fundraising, commercial strategy
2. **Mathematical physicist** --- Independent validation, extend theory, share intellectual load
3. **ML engineer** --- Production API, AI interpretability product, customer integration
