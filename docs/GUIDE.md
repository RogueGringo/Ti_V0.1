# Ti V0.1 — A Plain-Language Guide

**What this project does, what every variable means, and how the pieces fit together.**

This document is written so that anyone — mathematician, engineer, student, or curious outsider — can meet the ideas at the same level. Every technical term is grounded in something you can picture. Every variable gets a plain name.

---

## The Question

There is a 160-year-old unsolved problem in mathematics called the **Riemann Hypothesis**. It says something specific about where certain special numbers (called "zeta zeros") live on the number line. If you plot these zeros on a map with two axes — a real part (sigma) and an imaginary part (gamma) — the hypothesis says they all sit exactly at sigma = 1/2. No exceptions.

Nobody has proven this. But there are billions of computed zeros, and every single one sits at sigma = 1/2. This project asks: **can we detect that fact using geometry and topology, without proving it analytically?**

---

## The Approach in One Paragraph

We take a large collection of known zeta zeros, treat them as points on a line, connect nearby points with edges (forming a graph), and then stretch a "fabric" of vector spaces over that graph. The fabric's weave is controlled by prime numbers. We then measure how smoothly the fabric lays flat. If the Riemann Hypothesis is true, the fabric should lay flattest — with the least wrinkling — when we tune a dial (sigma) to exactly 0.5. We sweep that dial from 0.25 to 0.75 and look for the sweet spot.

---

## Variable Reference

Every symbol used in the code and papers, translated to plain language.

### Core Parameters — The Dials You Turn

| Variable | Code name | What it is | Typical values |
|----------|-----------|------------|----------------|
| **K** | `K` | **Fabric resolution.** How many threads the fabric has per point. Higher K means more prime numbers participate in the weave. | 20, 50, 100, 200 |
| **N** | `N` | **Number of zeros.** How many zeta zeros we place on the line. More zeros = better statistics, but costs more memory. | 2,000 or 9,877 |
| **sigma** | `sigma` | **The dial.** The real part of the complex parameter s. This is what we sweep. The Riemann Hypothesis says the magic value is 0.5. | 0.25 to 0.75 |
| **epsilon** | `epsilon` | **Connection range.** How far apart two zeros can be and still share an edge. Small epsilon = sparse graph (local connections only). Large epsilon = dense graph (longer-range connections). | 3.0 or 5.0 |
| **k_eig** | `k_eig` | **How many wrinkles to count.** The number of smallest eigenvalues (lowest-energy modes) we compute from the fabric. | 20 |

### The Points — What We Measure

| Variable | Code name | What it is |
|----------|-----------|------------|
| **gamma_i** | `zeros[i]` | The imaginary part of the i-th zeta zero. After "unfolding" (a normalization step), these are spaced about 1.0 apart on average. These are the points we lay on the line. |
| **Delta_gamma** | `gaps` | The distance between two connected zeros: gamma_j minus gamma_i. This is the "length" of an edge in the graph. |

### The Fabric — What We Build

| Variable | Code name | What it is |
|----------|-----------|------------|
| **rho(p)** | `_build_rho(p)` | **The prime stamp.** A K-by-K matrix that represents what the prime number p does to the fabric. It shifts basis vector e_i to e_{p*i} if p*i fits within K, and to zero otherwise. This is how primes become geometry. |
| **G_p(sigma)** | `_build_generator(p)` | **The generator for prime p.** Combines the prime stamp with a sigma-dependent weight: (log p / p^sigma) times (rho(p) + its transpose). This is the "thread" that prime p contributes to the fabric's weave at position sigma. |
| **A(sigma)** | `_build_generator_sum()` | **The total generator.** Sum of all prime generators. This is the complete weaving instruction for the fabric at a given sigma. |
| **B_p(sigma)** | superposition bases | **The per-prime basis.** In superposition mode, each prime contributes a separate basis matrix. These get mixed together with edge-dependent phase factors. |
| **U(Delta_gamma)** | transport map | **The transport map.** A K-by-K unitary (or near-unitary) matrix that tells you how to "carry" a section of the fabric from one zero to its neighbor. Built from the generator via matrix exponential. This is the heart of the construction. |

### The Measurement — What We Compute

| Variable | Code name | What it is |
|----------|-----------|------------|
| **L_F** | `build_matrix()` | **The sheaf Laplacian.** A giant (N*K)-by-(N*K) sparse matrix assembled from all the transport maps. It encodes how much the fabric wrinkles at every point and every connection. |
| **lambda_k** | `eigs[k]` | **The k-th wrinkle.** The k-th smallest eigenvalue of the Laplacian. Zero eigenvalues = perfectly flat fabric (global sections exist). Small eigenvalues = almost flat. Large eigenvalues = severely wrinkled. |
| **S(sigma, epsilon)** | `spectral_sum` | **The wrinkle score.** Sum of the k_eig smallest eigenvalues. This single number is the primary measurement: it tells you how much total wrinkling the fabric has at a given (sigma, epsilon) setting. Lower is flatter. |
| **C(sigma)** | contrast | **The contrast.** How much the wrinkle score varies as you sweep sigma. Computed as (S_max - S_min) / S_max across the sigma grid. If the fabric strongly prefers one sigma value, contrast is high. |
| **R** | signal ratio | **The discrimination.** The ratio of the zeta wrinkle score to the control wrinkle score at the same sigma. If zeta zeros create a fundamentally different fabric than random points, R is large. At K=20, R was 670x. |
| **sigma*** | `peak_sigma` | **The sweet spot.** The sigma value where the contrast ratio C(sigma) = S_zeta / S_control peaks. If the Riemann Hypothesis is true and the framework works, this should converge to 0.500 as K grows. |

### The Controls — What We Compare Against

| Source | Code name | What it is |
|--------|-----------|------------|
| **Zeta zeros** | `"Zeta"` | The real data. Imaginary parts of non-trivial zeros of the Riemann zeta function, computed by Andrew Odlyzko near the 10^20-th zero. |
| **Random (Poisson)** | `"Random"` | Uniformly random points on the same interval. No arithmetic structure at all. The "null hypothesis" control. |
| **GUE** | `"GUE"` | Eigenvalues of a random matrix from the Gaussian Unitary Ensemble. These have the same local spacing statistics as zeta zeros (this is the Montgomery-Odlyzko law) but carry no arithmetic content. The "statistical twin" control. |

### Falsification Criteria — The Pre-Registered Tests

These were frozen before any K=100 data was collected.

| Code | Plain meaning | Threshold |
|------|---------------|-----------|
| **F1** | The sweet spot wanders too far from 0.5 | sigma*(K=100) outside [0.40, 0.60] |
| **F2** | More primes make the signal weaker | C(K=100) < C(K=50) |
| **F3** | Zeta zeros stop looking special | R(K=100) < 10 |
| **F4** | The statistical twin fakes the signal | C_GUE(K=100) > 0.5 * C_zeta(K=100) |
| **P1** | Sweet spot is on track toward 0.5 | 0.45 <= sigma*(K=100) <= 0.52 |
| **P2** | Signal is getting sharper | C(K=100) > 1.5 * C(K=50) |
| **P3** | Zeta zeros keep looking more special | R(K=100) > R(K=50) |
| **P4** | Sharpening spreads to finer scales | Turnover at eps=2.0 by K=200 |

---

## How the Pieces Fit Together

```
                    THE PIPELINE
                    ============

    [1] ZEROS                [2] GRAPH                [3] FABRIC
    --------                 ---------                 ----------
    Load N zeta zeros        Connect zeros within      At each edge, build a
    from Odlyzko data.       distance epsilon.         K x K transport matrix
    Unfold so spacing ~1.    This is the               from prime generators.
    Also load Random         Vietoris-Rips             Transport depends on
    and GUE controls.        complex.                  edge length and sigma.

         |                        |                         |
         v                        v                         v

    [4] LAPLACIAN            [5] EIGENVALUES           [6] SWEEP
    -------------            ---------------           ---------
    Assemble the             Compute the 20            Repeat steps 3-5
    (N*K) x (N*K)            smallest eigenvalues      for each sigma in
    sheaf Laplacian          via Lanczos iteration     [0.25, ..., 0.75],
    from all transport       on GPU. Sum them          each epsilon, and
    maps + graph edges.      to get S(sigma, eps).     each point source.

         |                        |                         |
         v                        v                         v

    [7] COMPARE              [8] DIAGNOSE              [9] CONCLUDE
    -----------              ------------              ------------
    Compute C(sigma)         Check against             Report result:
    = S_zeta / S_control     pre-registered            framework valid or
    across the sigma         falsification             flawed, RH supported
    sweep. Find sigma*       criteria F1-F4,           or not supported,
    where C peaks.           P1-P4.                    with confidence.
```

### Step by step:

**Step 1 — Load the zeros.** We take 2,000 (or 9,877) consecutive imaginary parts of zeta zeros, high up in the critical strip (near the 10^20-th zero). We normalize ("unfold") them so the average spacing is 1.0. We also generate two control datasets: uniformly random points and GUE random matrix eigenvalues.

**Step 2 — Build the graph.** For a given epsilon, we connect every pair of zeros that are within distance epsilon of each other. This creates a graph (the Vietoris-Rips complex) whose connectivity depends on epsilon. Smaller epsilon = sparser graph = local structure only. Larger epsilon = denser graph = longer-range correlations.

**Step 3 — Weave the fabric.** For each edge in the graph, we build a K-by-K transport matrix. This matrix is computed from the prime numbers: each prime p contributes a generator weighted by log(p)/p^sigma, and the generators are combined with edge-dependent phase factors (e^{i * gap * log(p)}) that encode the explicit formula. The matrix exponential of this sum gives the transport map. This is computed on GPU via batched eigendecomposition.

**Step 4 — Assemble the Laplacian.** From all the edges and their transport maps, we assemble a single enormous sparse matrix (the sheaf Laplacian). For K=100 and N=2,000, this matrix is 200,000 x 200,000. Each edge contributes four K-by-K blocks. Assembly is done in batches to fit within GPU memory, with coalescing on CPU via scipy.

**Step 5 — Measure the wrinkling.** We find the 20 smallest eigenvalues of the Laplacian using the Lanczos algorithm with spectral flip trick on GPU. Their sum S(sigma, epsilon) is the wrinkle score: how much the fabric resists laying flat.

**Step 6 — Sweep the dial.** We repeat steps 3-5 for 15 values of sigma from 0.25 to 0.75 (with fine sampling near 0.50), for 2 values of epsilon (3.0 and 5.0), and for all 3 point sources (Zeta, Random, GUE). This gives 90 grid points total.

**Step 7 — Compare.** The raw wrinkle score includes both "geometric" effects (from the graph structure) and "arithmetic" effects (from the prime-number connection). We divide S_zeta by S_control to cancel the geometry and isolate the arithmetic. The ratio C(sigma) tells us where the primes "prefer" sigma to be.

**Step 8 — Check the criteria.** We compare sigma* (where C peaks), the contrast C, and the discrimination R against the pre-registered thresholds. These were frozen before any K=100 data was collected, so we cannot move the goalposts.

**Step 9 — Report.** The result is one of three outcomes:
- **Framework failure** (F1-F4 triggered): the method doesn't work.
- **No RH support** (R1-R3 triggered): the method works but doesn't see sigma=0.5 as special.
- **Positive evidence** (P1-P4 met): the method works and the data points toward sigma=0.5.

---

## The Fourier Sharpening Hypothesis

The key prediction: as K grows (more primes in the fabric), the sweet spot sigma* should converge toward 0.500 and the contrast should increase. This is because each prime adds a new "frequency" to the phase interference pattern in the transport maps. With few primes (K=20, 8 primes), the pattern is blurry — like hearing a chord with only 8 notes. With many primes (K=200, 46 primes), the pattern sharpens — the chord resolves into a clear tone at sigma = 0.5.

**What we've seen so far:**

| K | Primes | What the wrinkle score does as sigma sweeps | Sweet spot |
|---|--------|---------------------------------------------|------------|
| 20 | 8 | Rises steadily from 0.25 to 0.75 (no peak visible) | Not resolved |
| 50 | 15 | At eps=5.0: rises, peaks near 0.40-0.50, drops. First turnover. | ~0.45 |
| 100 | 25 | At eps=3.0 and 5.0: contrast ratio C(sigma) peaks near 0.47-0.48 | ~0.48 |

The progression 8 primes (no peak) to 15 primes (first peak) to 25 primes (peak migrating toward 0.5) is the Fourier sharpening in action. The question is whether K=200 with 46 primes sharpens it further — and whether the contrast grows, not just the peak location.

---

## Glossary of Jargon

| Term | Plain meaning |
|------|---------------|
| **Sheaf** | A rule that assigns a vector space to every point and a linear map to every connection. Think: a fabric draped over a graph, with stitching instructions at every edge. |
| **Laplacian** | A matrix that measures how much a function (or section) varies from point to point. The graph version of "second derivative." Zero eigenvalues = perfectly smooth. |
| **Eigenvalue** | A number that measures one mode of vibration. Small eigenvalues = the fabric can lay almost flat in that mode. |
| **Transport map** | The stitching instruction: a matrix that tells you how to translate a vector from one point to its neighbor through the fabric. |
| **Vietoris-Rips complex** | A graph built by connecting all points within distance epsilon. The simplest way to turn a point cloud into a topological space. |
| **Spectral unfolding** | Normalization that removes the known smooth trend from zero spacing, leaving only the fluctuations. Like detrending a time series. |
| **Lie algebra** | The "infinitesimal" version of a symmetry group. u(K) is the algebra of K-by-K skew-Hermitian matrices. Our generators live here. |
| **Gauge connection** | A rule for parallel transport in a fiber bundle. In physics: the electromagnetic potential. Here: the prime-derived stitching instructions. |
| **Holonomy** | What happens when you transport a vector around a closed loop and it doesn't come back to itself. Measures the curvature (wrinkling) of the connection. |
| **GUE** | Gaussian Unitary Ensemble. A probability distribution on random Hermitian matrices. Its eigenvalues have the same local statistics as zeta zeros. |
| **Explicit formula** | The mathematical identity relating the distribution of primes to the locations of zeta zeros. The bridge between number theory and analysis. Our transport maps encode this bridge as geometry. |
| **Phase transition** | A sudden qualitative change in behavior as a parameter crosses a threshold. Like water freezing at 0 degrees. We look for a spectral phase transition at sigma = 0.5. |
| **Contrast ratio** | How much the signal varies across the sigma sweep, normalized to [0, 1]. High contrast = the fabric strongly prefers one sigma value. |
| **Discrimination ratio** | How different zeta zeros look from random points. R = 670 means zeta's signal is 670 times stronger than noise. |
