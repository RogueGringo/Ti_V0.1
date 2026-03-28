#!/usr/bin/env python3
"""Cross-Domain Synthesis — Does "truth creates hierarchy" hold universally?

THE HYPOTHESIS:
    The same phenomenon appears in number theory and LLM representations:
    - A natural basis (primes / semantic primes) provides a floor
    - The actual system builds MORE structure on top
    - The residual is RICHER, not noise
    - The convergence Gini sits in a universal range

THE EXPERIMENT:
    Run the adaptive basis discovery pipeline on zeta zero transport maps.
    The "hidden states" are K×K transport matrices at each zero gap.
    The "semantic primes" are GUE transport matrices (the null model).
    Measure: is the arithmetic residual more hierarchical than GUE baseline?
    Compare: convergence Gini of zeros vs LLM models.

THE PREDICTION:
    If the synthesis holds:
    - Zeta residual Gini > GUE basis Gini
    - Zeta convergence Gini ≈ 0.94-0.97 (LLM range)
    If it fails:
    - The two domains have different representational structure
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add both repos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # JTopo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "atft-problems" / "products" / "topological-router"))

from atft.sources.zeta_zeros import ZetaZerosSource
from atft.sources.gue import GUESource
from atft.topology.transport_maps import TransportMapBuilder
from topo_measures import gini_fast, h0_gini

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "cross_domain"
FIG_DIR = Path(__file__).parent.parent.parent / "assets" / "cross_domain"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Transport matrix extraction ─────────────────────────────────────────────

def compute_transport_matrix(points: np.ndarray, K: int, sigma: float = 0.5) -> np.ndarray:
    """Compute flattened transport maps at each gap between consecutive points.

    Args:
        points: 1D array of positions (zeros or eigenvalues)
        K: fiber dimension for transport maps
        sigma: critical line parameter

    Returns:
        (n_gaps, 2*K²) real matrix — each row is a flattened transport map
        (real and imaginary parts concatenated)
    """
    points = points.flatten()
    gaps = np.diff(points)
    n_gaps = len(gaps)

    builder = TransportMapBuilder(K=K, sigma=sigma)

    rows = []
    for i, gap in enumerate(gaps):
        T = builder.transport(float(gap))  # K×K complex
        # Flatten: concatenate real and imaginary parts
        row = np.concatenate([T.real.flatten(), T.imag.flatten()])
        rows.append(row)

    return np.array(rows)  # (n_gaps, 2*K²)


# ── Adaptive basis discovery (adapted from adaptive_explorer.py) ────────────

def adaptive_basis_discovery(data_matrix: np.ndarray, seed_basis: np.ndarray,
                              max_iterations: int = 25,
                              gini_epsilon: float = 0.005,
                              patience: int = 3,
                              new_per_iter: int = 5,
                              residual_threshold: float = 0.01) -> dict:
    """Run adaptive basis expansion on a data matrix.

    Args:
        data_matrix: (n_samples, n_features)
        seed_basis: (n_seed, n_features) — initial orthonormal basis

    Returns:
        dict with adaptive_basis, convergence_trajectory, etc.
    """
    from numpy.linalg import svd, qr, norm

    basis = seed_basis.copy()
    convergence_trajectory = []
    residual_history = []
    basis_sizes = []

    prev_gini = None
    stable_count = 0

    # Use chunks of data per iteration
    n_samples = data_matrix.shape[0]
    chunk_size = max(10, n_samples // max_iterations)

    for iteration in range(max_iterations):
        # Select data chunk
        start = (iteration * chunk_size) % n_samples
        end = min(start + chunk_size, n_samples)
        chunk = data_matrix[start:end]

        # Mean-center
        chunk = chunk - chunk.mean(axis=0)

        # Project onto current basis and compute residual
        proj = chunk @ basis.T @ basis
        residual = chunk - proj
        res_norms = norm(residual, axis=1)
        mean_res = float(res_norms.mean())

        # SVD of residual to find new directions
        if residual.shape[0] >= 2:
            U, S, Vt = svd(residual, full_matrices=False)
            sig_threshold = S[0] * residual_threshold if len(S) > 0 else 0
            n_new = min(new_per_iter, len(S))
            mask = S[:n_new] > sig_threshold
            new_vectors = Vt[:n_new][mask]

            if len(new_vectors) > 0:
                combined = np.vstack([basis, new_vectors])
                Q, R = qr(combined.T)
                valid = np.abs(np.diag(R)) > 1e-8
                basis = Q[:, valid].T
        else:
            new_vectors = np.zeros((0, basis.shape[1]))

        # Convergence: Gini of singular values of data projected onto basis
        proj_on_basis = chunk @ basis.T
        if proj_on_basis.shape[0] >= 2:
            _, sv, _ = svd(proj_on_basis, full_matrices=False)
            gini = gini_fast(sv)
        else:
            gini = 0.0

        convergence_trajectory.append(float(gini))
        residual_history.append(mean_res)
        basis_sizes.append(basis.shape[0])

        if prev_gini is not None and abs(gini - prev_gini) < gini_epsilon:
            stable_count += 1
        else:
            stable_count = 0
        prev_gini = gini

        print(f"    Iter {iteration:2d}: basis={basis.shape[0]:3d} "
              f"(+{len(new_vectors)}) gini={gini:.4f} res={mean_res:.4f}"
              f"{' CONVERGED' if stable_count >= patience else ''}")

        if stable_count >= patience:
            break

    return {
        "adaptive_basis": basis,
        "convergence_trajectory": convergence_trajectory,
        "residual_history": residual_history,
        "basis_sizes": basis_sizes,
        "converged": stable_count >= patience,
        "final_gini": convergence_trajectory[-1] if convergence_trajectory else 0,
        "final_basis_size": basis.shape[0],
        "n_iterations": len(convergence_trajectory),
    }


# ── Main experiment ──────────────────────────────────────────────────────────

def run_synthesis(K: int = 50, n_zeros: int = 200, n_gue_realizations: int = 10,
                  sigma: float = 0.5):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 65)
    print("  CROSS-DOMAIN SYNTHESIS")
    print("  Does 'truth creates hierarchy' hold universally?")
    print(f"  K={K}, n_zeros={n_zeros}, n_GUE={n_gue_realizations}")
    print(f"  {ts}")
    print("=" * 65)

    # ── Phase 1: GUE Baseline ──
    print(f"\n  Phase 1: GUE Baseline (n={n_gue_realizations} realizations)")
    t0 = time.time()

    gue_rows = []
    for seed in range(n_gue_realizations):
        gs = GUESource(seed=seed)
        gpc = gs.generate(n_zeros)
        gue_eigs = gpc.points.flatten()
        # Sort and compute transport
        gue_eigs = np.sort(gue_eigs)
        gue_transport = compute_transport_matrix(gue_eigs, K, sigma)
        gue_rows.append(gue_transport)

    gue_matrix = np.vstack(gue_rows)  # (n_gue_realizations * (n_zeros-1), 2*K²)
    print(f"    GUE transport matrix: {gue_matrix.shape}")

    # SVD of GUE → null basis
    gue_centered = gue_matrix - gue_matrix.mean(axis=0)
    U_gue, S_gue, Vt_gue = np.linalg.svd(gue_centered, full_matrices=False)

    # Take top components that explain 90% of variance
    cumvar = np.cumsum(S_gue**2) / np.sum(S_gue**2)
    n_gue_components = int(np.searchsorted(cumvar, 0.90)) + 1
    gue_basis = Vt_gue[:n_gue_components]
    gue_gini = gini_fast(S_gue)

    print(f"    GUE basis: {gue_basis.shape} ({n_gue_components} components for 90% variance)")
    print(f"    GUE singular value Gini: {gue_gini:.4f}")
    print(f"    Phase 1: {time.time()-t0:.1f}s")

    # ── Phase 2: Zeta Zeros ──
    print(f"\n  Phase 2: Zeta Zeros (n={n_zeros})")
    t1 = time.time()

    zs = ZetaZerosSource("data/odlyzko_zeros.txt")
    zpc = zs.generate(n_zeros)
    zeros = zpc.points.flatten()

    zeta_transport = compute_transport_matrix(zeros, K, sigma)
    print(f"    Zeta transport matrix: {zeta_transport.shape}")

    # Run adaptive basis discovery, seeded with GUE basis
    print(f"    Running adaptive basis discovery (seed={gue_basis.shape[0]} GUE vectors)...")
    zeta_result = adaptive_basis_discovery(
        zeta_transport, gue_basis,
        max_iterations=25, gini_epsilon=0.005, patience=3)

    print(f"    Adaptive basis: {zeta_result['final_basis_size']} vectors")
    print(f"    Converged: {zeta_result['converged']}")
    print(f"    Final Gini: {zeta_result['final_gini']:.4f}")
    print(f"    Phase 2: {time.time()-t1:.1f}s")

    # ── Phase 3: Cross-Domain Comparison ──
    print(f"\n  Phase 3: Cross-Domain Comparison")

    # Residual analysis: project zeta onto GUE basis, measure residual topology
    zeta_centered = zeta_transport - zeta_transport.mean(axis=0)
    zeta_proj_gue = zeta_centered @ gue_basis.T @ gue_basis
    zeta_residual = zeta_centered - zeta_proj_gue

    # Gini of residual
    if zeta_residual.shape[0] >= 3:
        _, S_res, _ = np.linalg.svd(zeta_residual, full_matrices=False)
        residual_gini = gini_fast(S_res)
    else:
        residual_gini = 0.0

    # Gini of prime (GUE) subspace projection
    zeta_proj_on_gue = zeta_centered @ gue_basis.T
    if zeta_proj_on_gue.shape[0] >= 3:
        _, S_prime_proj, _ = np.linalg.svd(zeta_proj_on_gue, full_matrices=False)
        prime_proj_gini = gini_fast(S_prime_proj)
    else:
        prime_proj_gini = 0.0

    # H₀ Gini comparisons
    h0_gue_proj = h0_gini(zeta_proj_on_gue, max_n=150)
    h0_residual = h0_gini(zeta_residual, max_n=150)

    # LLM convergence values (from sweep)
    llm_convergence = {
        "SmolLM2-360M": 0.9371,
        "Qwen2.5-0.5B": 0.8920,
        "TinyLlama-1.1B": 0.8478,
        "Qwen2.5-1.5B": 0.9564,
        "Qwen2.5-3B": 0.8414,
        "Qwen2.5-7B": 0.9644,
    }

    # ── Results ──
    print(f"\n{'='*65}")
    print("  RESULTS")
    print(f"{'='*65}")

    print(f"\n  GINI COMPARISON (SVD spectrum):")
    print(f"    GUE basis Gini:      {gue_gini:.4f}")
    print(f"    Zeta prime-proj Gini:{prime_proj_gini:.4f}")
    print(f"    Zeta residual Gini:  {residual_gini:.4f}")
    residual_richer = residual_gini > prime_proj_gini
    print(f"    Residual > Prime:    {'YES' if residual_richer else 'NO'} "
          f"({residual_gini/prime_proj_gini:.1f}x)" if prime_proj_gini > 0 else "")

    print(f"\n  H₀ PERSISTENCE GINI:")
    print(f"    GUE projection:      {h0_gue_proj:.4f}")
    print(f"    Residual:            {h0_residual:.4f}")

    print(f"\n  CONVERGENCE GINI — CROSS-DOMAIN:")
    print(f"    {'System':<20} {'Conv Gini':>10}")
    print(f"    {'-'*32}")
    for name, g in sorted(llm_convergence.items(), key=lambda x: x[1]):
        print(f"    {name:<20} {g:>10.4f}")
    print(f"    {'─'*32}")
    print(f"    {'ZETA ZEROS':<20} {zeta_result['final_gini']:>10.4f}  ← {'IN RANGE' if 0.84 < zeta_result['final_gini'] < 0.97 else 'OUT OF RANGE'}")
    print(f"    {'─'*32}")
    print(f"    Semantic FP          {'0.9450':>10}")
    print(f"    Arithmetic FP        {'0.9970':>10}")

    # The verdict
    print(f"\n{'='*65}")
    print("  SYNTHESIS VERDICT")
    print(f"{'='*65}")

    if residual_richer and 0.84 < zeta_result["final_gini"] < 0.97:
        verdict = "SYNTHESIS HOLDS — truth creates hierarchy universally"
        detail = (f"Zeta residual is {residual_gini/prime_proj_gini:.1f}x richer than GUE baseline, "
                  f"and convergence Gini ({zeta_result['final_gini']:.4f}) is in the LLM range (0.84-0.97). "
                  f"The same phenomenon operates across number theory and neural representations.")
    elif residual_richer:
        verdict = "PARTIAL — residual is richer but convergence Gini is out of LLM range"
        detail = (f"Zeta residual IS more hierarchical ({residual_gini:.4f} > {prime_proj_gini:.4f}), "
                  f"but convergence Gini ({zeta_result['final_gini']:.4f}) doesn't match LLM range. "
                  f"The hierarchy property transfers; the convergence point may be domain-specific.")
    else:
        verdict = "SYNTHESIS FAILS — zeta residual is NOT richer than GUE"
        detail = (f"Residual Gini ({residual_gini:.4f}) <= GUE projection Gini ({prime_proj_gini:.4f}). "
                  f"The arithmetic premium does not manifest as representational hierarchy in transport space.")

    print(f"  {verdict}")
    print(f"  {detail}")

    # Save
    results = {
        "timestamp": ts,
        "params": {"K": K, "n_zeros": n_zeros, "n_gue": n_gue_realizations, "sigma": sigma},
        "gue_basis_shape": list(gue_basis.shape),
        "gue_gini": float(gue_gini),
        "zeta_transport_shape": list(zeta_transport.shape),
        "zeta_prime_proj_gini": float(prime_proj_gini),
        "zeta_residual_gini": float(residual_gini),
        "residual_richer": bool(residual_richer),
        "h0_gue_proj": float(h0_gue_proj),
        "h0_residual": float(h0_residual),
        "zeta_convergence": {
            "converged": zeta_result["converged"],
            "final_gini": zeta_result["final_gini"],
            "final_basis_size": zeta_result["final_basis_size"],
            "n_iterations": zeta_result["n_iterations"],
            "trajectory": zeta_result["convergence_trajectory"],
        },
        "llm_convergence": llm_convergence,
        "verdict": verdict,
        "detail": detail,
    }

    with open(OUTPUT_DIR / "synthesis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {OUTPUT_DIR / 'synthesis_results.json'}")

    # Plot
    plot_synthesis(results)

    return results


def plot_synthesis(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"gold": "#c5a03f", "teal": "#45a8b0", "red": "#e94560",
         "bg": "#0f0d08", "text": "#d6d0be", "muted": "#817a66", "green": "#4caf50"}
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.facecolor": C["bg"], "figure.facecolor": C["bg"],
        "text.color": C["text"], "axes.labelcolor": C["text"],
        "xtick.color": C["muted"], "ytick.color": C["muted"],
        "axes.edgecolor": C["muted"], "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.dpi": 200,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Convergence Gini comparison
    ax = axes[0]
    llm = results["llm_convergence"]
    names = list(llm.keys()) + ["ZETA ZEROS"]
    vals = list(llm.values()) + [results["zeta_convergence"]["final_gini"]]
    colors = [C["teal"]] * len(llm) + [C["gold"]]
    bars = ax.barh(range(len(names)), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0.945, color=C["muted"], ls="--", alpha=0.5, label="Semantic FP (0.945)")
    ax.axvline(0.997, color=C["red"], ls="--", alpha=0.5, label="Arithmetic FP (0.997)")
    ax.set_xlabel("Convergence Gini")
    ax.set_title("Cross-Domain Convergence", color=C["gold"])
    ax.set_xlim(0.8, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15, axis="x")

    # 2. Residual vs Prime Gini
    ax = axes[1]
    categories = ["GUE Basis", "Zeta\nPrime Proj", "Zeta\nResidual"]
    ginis = [results["gue_gini"], results["zeta_prime_proj_gini"], results["zeta_residual_gini"]]
    bar_colors = [C["muted"], C["teal"], C["gold"]]
    ax.bar(range(3), ginis, color=bar_colors, alpha=0.8)
    for i, g in enumerate(ginis):
        ax.text(i, g + 0.01, f"{g:.3f}", ha="center", fontsize=10, color=C["text"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(categories)
    ax.set_ylabel("SVD Gini")
    ax.set_title("Residual vs Prime Hierarchy", color=C["gold"])
    ax.grid(True, alpha=0.15, axis="y")

    # 3. Convergence trajectory
    ax = axes[2]
    traj = results["zeta_convergence"]["trajectory"]
    ax.plot(range(len(traj)), traj, color=C["gold"], lw=2, marker="o", ms=4)
    ax.axhline(0.945, color=C["muted"], ls="--", alpha=0.5, label="Semantic FP")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gini")
    ax.set_title(f"Zeta Basis Convergence (final={traj[-1]:.4f})", color=C["gold"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    fig.suptitle("Cross-Domain Synthesis: Number Theory x Neural Representations",
                 color=C["text"], fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cross_domain_synthesis.png")
    plt.close(fig)
    print(f"  Figure: {FIG_DIR / 'cross_domain_synthesis.png'}")


if __name__ == "__main__":
    run_synthesis(K=50, n_zeros=200, n_gue_realizations=10)
