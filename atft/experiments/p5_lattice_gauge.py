#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""P5: SU(2) Lattice Gauge Theory — ATFT Predictions 1+2 Validation.

Paper claims:
(1) Onset scale ε*(β) exhibits discontinuous jump at β_c ≈ 2.30.
(2) Parity-complete feature map distinguishes Q=+1 from Q=-1 instantons.

Protocol:
1. Generate SU(2) configs at multiple β values on 8³×4 lattice
2. Apply parity-complete feature map
3. Run H₀ persistence on feature point clouds
4. Extract onset scale ε* and Gini trajectory per β
5. Check for discontinuity at β_c

NOTE: Using 8³×4 (not 16³×4 as in paper) for tractability.
The transition should still be visible but may be broader.

PASS(1): ε*(β) shows measurable transition at β ∈ [2.0, 2.5]
FAIL(1): ε*(β) is smooth with no transition
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

OUTPUT_DIR = Path("output/atft_validation")
FIG_DIR = Path("assets/validation")

COLORS = {"gold": "#c5a03f", "teal": "#45a8b0", "red": "#e94560",
          "bg": "#0f0d08", "text": "#d6d0be", "muted": "#817a66"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.facecolor": COLORS["bg"], "figure.facecolor": COLORS["bg"],
    "text.color": COLORS["text"], "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
    "axes.edgecolor": COLORS["muted"], "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})

# Lattice parameters (reduced from paper's 16³×4 for tractability)
LATTICE = (8, 8, 8, 4)
BETA_VALUES = [1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0, 4.0]
N_THERM = 200    # Thermalization sweeps (reduced from 1000)
N_CONFIGS = 5    # Configs per beta (reduced from 100)
N_SKIP = 5       # Sweeps between configs


def gini(values):
    n = len(values)
    if n <= 1 or np.sum(values) == 0:
        return 0.0
    sorted_v = np.sort(values)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_v)) / (n * np.sum(sorted_v)) - (n + 1.0) / n)


def h0_persistence_subsample(points, n_sample=500, seed=42):
    """H₀ persistence on subsampled point cloud."""
    rng = np.random.default_rng(seed)
    n = len(points)
    if n > n_sample:
        idx = rng.choice(n, n_sample, replace=False)
        points = points[idx]
        n = n_sample

    dists = pdist(points)

    parent = list(range(n))
    rank_uf = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dists[k], i, j))
            k += 1
    edges.sort()

    bars = []
    for dist, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            if rank_uf[ri] < rank_uf[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank_uf[ri] == rank_uf[rj]:
                rank_uf[ri] += 1
            bars.append(float(dist))

    return np.array(bars)


def compute_onset_scale(bars, percentile=95):
    """Onset scale: the 95th percentile of persistence values."""
    if len(bars) == 0:
        return 0.0
    return float(np.percentile(bars, percentile))


def main():
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  P5: SU(2) LATTICE GAUGE THEORY — PREDICTIONS 1+2")
    print(f"  Lattice: {LATTICE}, β values: {BETA_VALUES}")
    print(f"  {timestamp}")
    print("=" * 70)

    from atft.lattice.su2 import (
        heat_bath_su2,
        parity_complete_feature_map,
        average_plaquette,
    )

    results_per_beta = {}

    for beta in BETA_VALUES:
        print(f"\n{'='*60}")
        print(f"  β = {beta:.2f}")
        print(f"{'='*60}")

        t0 = time.time()

        # Generate configs
        configs = heat_bath_su2(
            beta=beta,
            lattice_shape=LATTICE,
            n_therm=N_THERM,
            n_configs=N_CONFIGS,
            n_skip=N_SKIP,
            seed=int(beta * 1000),
        )

        elapsed_gen = time.time() - t0
        print(f"  Generated {len(configs)} configs ({elapsed_gen:.0f}s)")

        # Process configs
        onset_scales = []
        gini_values = []
        plaquettes = []

        for c_idx, config in enumerate(configs):
            # Average plaquette
            plaq = average_plaquette(config, LATTICE)
            plaquettes.append(plaq)

            # Feature map
            features = parity_complete_feature_map(config, LATTICE)

            # H₀ persistence
            bars = h0_persistence_subsample(features)

            if len(bars) > 0:
                onset = compute_onset_scale(bars)
                g = gini(bars)
                onset_scales.append(onset)
                gini_values.append(g)

            print(f"    config {c_idx+1}: <P>={plaq:.4f}, "
                  f"onset={onset_scales[-1] if onset_scales else 0:.4f}, "
                  f"G={gini_values[-1] if gini_values else 0:.4f}")

        results_per_beta[str(beta)] = {
            "beta": beta,
            "n_configs": len(configs),
            "plaquettes": plaquettes,
            "mean_plaquette": float(np.mean(plaquettes)) if plaquettes else 0,
            "onset_scales": onset_scales,
            "mean_onset": float(np.mean(onset_scales)) if onset_scales else 0,
            "std_onset": float(np.std(onset_scales)) if len(onset_scales) > 1 else 0,
            "gini_values": gini_values,
            "mean_gini": float(np.mean(gini_values)) if gini_values else 0,
            "gen_time_s": elapsed_gen,
        }

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")

    betas = []
    mean_onsets = []
    mean_ginis = []
    mean_plaqs = []

    for beta_str in sorted(results_per_beta.keys(), key=float):
        r = results_per_beta[beta_str]
        betas.append(r["beta"])
        mean_onsets.append(r["mean_onset"])
        mean_ginis.append(r["mean_gini"])
        mean_plaqs.append(r["mean_plaquette"])
        print(f"  β={r['beta']:.2f}: <P>={r['mean_plaquette']:.4f}, "
              f"ε*={r['mean_onset']:.4f}±{r['std_onset']:.4f}, "
              f"G={r['mean_gini']:.4f}")

    # Detect transition: look for maximum derivative of onset scale
    betas_arr = np.array(betas)
    onsets_arr = np.array(mean_onsets)

    if len(betas_arr) >= 3:
        d_onset = np.gradient(onsets_arr, betas_arr)
        max_deriv_idx = np.argmax(np.abs(d_onset))
        transition_beta = betas_arr[max_deriv_idx]
        max_deriv = d_onset[max_deriv_idx]

        print(f"\n  Maximum |dε*/dβ| at β = {transition_beta:.2f} (derivative = {max_deriv:.4f})")
        detected_transition = 2.0 <= transition_beta <= 2.5
    else:
        detected_transition = False
        transition_beta = 0

    # ── Figures ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plaquette
    ax = axes[0]
    ax.plot(betas, mean_plaqs, "o-", color=COLORS["gold"], linewidth=2, markersize=8)
    ax.axvline(2.30, color=COLORS["red"], linestyle="--", alpha=0.5, label="β_c ≈ 2.30")
    ax.set_xlabel("β (coupling)")
    ax.set_ylabel("<P> (average plaquette)")
    ax.set_title("Order Parameter", color=COLORS["gold"])
    ax.legend()
    ax.grid(True, alpha=0.15)

    # Onset scale
    ax = axes[1]
    ax.plot(betas, mean_onsets, "o-", color=COLORS["teal"], linewidth=2, markersize=8)
    ax.axvline(2.30, color=COLORS["red"], linestyle="--", alpha=0.5, label="β_c ≈ 2.30")
    ax.set_xlabel("β (coupling)")
    ax.set_ylabel("ε* (onset scale)")
    ax.set_title("Topological Onset Scale", color=COLORS["teal"])
    ax.legend()
    ax.grid(True, alpha=0.15)

    # Gini
    ax = axes[2]
    ax.plot(betas, mean_ginis, "o-", color=COLORS["red"], linewidth=2, markersize=8)
    ax.axvline(2.30, color=COLORS["red"], linestyle="--", alpha=0.5, label="β_c ≈ 2.30")
    ax.set_xlabel("β (coupling)")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Eigenvalue Hierarchy", color=COLORS["red"])
    ax.legend()
    ax.grid(True, alpha=0.15)

    fig.suptitle(f"P5: SU(2) Confinement-Deconfinement ({LATTICE[0]}³×{LATTICE[3]})",
                 color=COLORS["gold"], fontsize=16, y=1.02)
    fig.savefig(FIG_DIR / "p5_su2_transition.png")
    plt.close(fig)

    print(f"\n  Figures saved to {FIG_DIR}/")

    # ── Verdict ──
    verdict = "PASS" if detected_transition else "FAIL"
    print(f"\n{'='*70}")
    print(f"  P5 VERDICT: {verdict}")
    if detected_transition:
        print(f"  Transition detected at β = {transition_beta:.2f} (expected ~2.30)")
    else:
        print(f"  No transition detected in [2.0, 2.5]")
    print(f"{'='*70}")

    # ── Save ──
    results = {
        "prediction": "1",
        "timestamp": timestamp,
        "lattice": list(LATTICE),
        "beta_values": BETA_VALUES,
        "results_per_beta": results_per_beta,
        "transition_beta": float(transition_beta) if detected_transition else None,
        "verdict": verdict,
        "summary": (
            f"SU(2) on {LATTICE[0]}³×{LATTICE[3]} lattice. "
            f"Transition {'detected' if detected_transition else 'NOT detected'} "
            f"at β={transition_beta:.2f}. Verdict: {verdict}."
        ),
    }

    out_path = OUTPUT_DIR / "p5_lattice_gauge.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()
