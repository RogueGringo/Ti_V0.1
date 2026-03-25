#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""P4: LLM Hidden State Analysis — ATFT Predictions 3+4 Validation.

Paper claims:
(3) Phase transition in hidden states is universal across architectures (r > 0.9).
(4) Gini trajectory correlates with reasoning accuracy.
Cited: r = 0.935 across four models.

Protocol:
1. Load 2 small models (GPU-friendly, <2B params)
2. Generate prompts at varying complexity (short → long)
3. Extract hidden states layer by layer
4. Build point clouds from hidden states (PCA to d=50)
5. Run H₀ persistence on each layer's point cloud
6. Compute Gini coefficient of persistence lifetimes at each layer
7. Detect phase transition: does Gini trajectory change shape as complexity increases?
8. Cross-model correlation of Gini trajectories

PASS: Cross-model correlation r > 0.9 for Gini trajectory patterns
FAIL: r < 0.7
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist

OUTPUT_DIR = Path("output/atft_validation")
FIG_DIR = Path("assets/validation")

COLORS = {"gold": "#c5a03f", "teal": "#45a8b0", "red": "#e94560",
          "purple": "#9b59b6", "bg": "#0f0d08", "text": "#d6d0be",
          "muted": "#817a66"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.facecolor": COLORS["bg"], "figure.facecolor": COLORS["bg"],
    "text.color": COLORS["text"], "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
    "axes.edgecolor": COLORS["muted"], "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})

# Small models that fit in 12GB VRAM
MODELS = [
    "HuggingFaceTB/SmolLM2-360M",
    "Qwen/Qwen2.5-0.5B",
]

PROMPTS = [
    # Varying complexity (short → long, simple → complex reasoning)
    {"text": "Hello", "complexity": 1},
    {"text": "What is 2+2?", "complexity": 2},
    {"text": "Explain what a prime number is.", "complexity": 3},
    {"text": "Why do prime numbers matter in cryptography? Give a brief explanation.", "complexity": 4},
    {"text": "The Riemann Hypothesis states that all non-trivial zeros of the zeta function have real part 1/2. Explain why this matters for the distribution of prime numbers, and describe one approach to testing it computationally.", "complexity": 5},
    {"text": "Consider a fiber bundle where the base space is parameterized by a filtration scale epsilon, and the fiber at each point is the sheaf cohomology group. Describe how the topology of this total space encodes the evolution of field equations as the resolution scale varies. What role does the Cech-de Rham isomorphism play in guaranteeing that discrete computations on simplicial complexes faithfully capture the continuous physics?", "complexity": 6},
]


def gini(values):
    """Gini coefficient."""
    n = len(values)
    if n <= 1 or np.sum(values) == 0:
        return 0.0
    sorted_v = np.sort(values)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_v)) / (n * np.sum(sorted_v)) - (n + 1.0) / n)


def compute_h0_persistence_rd(points, max_pairs=2000):
    """H₀ persistence on a point cloud in R^d via pairwise distances."""
    n = len(points)
    if n < 2:
        return []

    # Subsample if too many points
    if n > max_pairs:
        idx = np.random.default_rng(42).choice(n, max_pairs, replace=False)
        points = points[idx]
        n = max_pairs

    # Pairwise distances
    dists = pdist(points)

    # Union-Find H₀ persistence
    n_pts = n
    parent = list(range(n_pts))
    rank = [0] * n_pts
    birth = [0.0] * n_pts

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Sort edges by distance
    from itertools import combinations
    edges = []
    idx = 0
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            edges.append((dists[idx], i, j))
            idx += 1
    edges.sort()

    bars = []
    for dist, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            # Merge: younger component dies
            if rank[ri] < rank[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]:
                rank[ri] += 1
            # Bar: born at 0, dies at this distance
            bars.append({"birth": 0.0, "death": float(dist),
                         "persistence": float(dist)})

    return bars


def extract_hidden_states(model_name, prompts, device="cuda"):
    """Load model and extract hidden states for each prompt."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=device,
        output_hidden_states=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for prompt_data in prompts:
        text = prompt_data["text"]
        complexity = prompt_data["complexity"]

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                           max_length=512).to(device)
        n_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim) per layer
        n_layers = len(hidden_states)

        layer_data = []
        for layer_idx, hs in enumerate(hidden_states):
            # Extract hidden state: (seq_len, hidden_dim)
            h = hs[0].cpu().float().numpy()  # (seq_len, hidden_dim)
            layer_data.append({
                "layer": layer_idx,
                "shape": list(h.shape),
                "mean_norm": float(np.mean(np.linalg.norm(h, axis=1))),
            })

        results.append({
            "text": text[:50] + "..." if len(text) > 50 else text,
            "complexity": complexity,
            "n_tokens": n_tokens,
            "n_layers": n_layers,
            "hidden_dim": hidden_states[0].shape[-1],
            "layers": layer_data,
            "hidden_states": [hs[0].cpu().float().numpy() for hs in hidden_states],
        })

        print(f"    Prompt (τ={complexity}): {n_tokens} tokens, {n_layers} layers, "
              f"dim={hidden_states[0].shape[-1]}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def compute_topological_features(hidden_states_data):
    """Compute persistence and Gini for each layer of each prompt."""
    results = []

    for prompt_data in hidden_states_data:
        layer_ginis = []
        for layer_idx, hs in enumerate(prompt_data["hidden_states"]):
            # hs shape: (seq_len, hidden_dim)
            if hs.shape[0] < 3:
                layer_ginis.append(0.0)
                continue

            # PCA to d=min(50, seq_len-1) if needed
            d = min(50, hs.shape[0] - 1, hs.shape[1])
            if hs.shape[1] > d:
                from sklearn.decomposition import PCA
                hs_reduced = PCA(n_components=d).fit_transform(hs)
            else:
                hs_reduced = hs

            # H₀ persistence
            bars = compute_h0_persistence_rd(hs_reduced)
            if bars:
                lifetimes = [b["persistence"] for b in bars]
                g = gini(np.array(lifetimes))
            else:
                g = 0.0

            layer_ginis.append(g)

        results.append({
            "complexity": prompt_data["complexity"],
            "n_tokens": prompt_data["n_tokens"],
            "n_layers": prompt_data["n_layers"],
            "gini_trajectory": layer_ginis,
        })

    return results


def main():
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  P4: LLM HIDDEN STATE ANALYSIS — PREDICTIONS 3+4")
    print(f"  {timestamp}")
    print("=" * 70)

    all_model_results = {}

    for model_name in MODELS:
        short_name = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*70}")

        try:
            # Extract hidden states
            hs_data = extract_hidden_states(model_name, PROMPTS)

            # Compute topological features
            topo_data = compute_topological_features(hs_data)

            # Clean up hidden states (large)
            for d in hs_data:
                del d["hidden_states"]

            all_model_results[short_name] = {
                "model": model_name,
                "prompts": hs_data,
                "topological": topo_data,
            }

            # Print Gini trajectories
            for td in topo_data:
                traj = td["gini_trajectory"]
                mean_g = np.mean(traj) if traj else 0
                slope = (traj[-1] - traj[0]) / len(traj) if len(traj) > 1 else 0
                print(f"    τ={td['complexity']}: mean G={mean_g:.4f}, "
                      f"slope={slope:+.4f}, layers={td['n_layers']}")

        except Exception as e:
            print(f"  FAILED: {e}")
            all_model_results[short_name] = {"model": model_name, "error": str(e)}

    # ── Cross-model correlation ──
    print(f"\n{'='*70}")
    print("  CROSS-MODEL CORRELATION")
    print(f"{'='*70}")

    model_names = [n for n in all_model_results if "error" not in all_model_results[n]]
    correlations = {}

    if len(model_names) >= 2:
        # For each prompt complexity, compare Gini trajectories across models
        # Use mean Gini per complexity as the feature vector
        for i, m1 in enumerate(model_names):
            for m2 in model_names[i+1:]:
                t1 = all_model_results[m1]["topological"]
                t2 = all_model_results[m2]["topological"]

                # Align by complexity
                v1 = [np.mean(t["gini_trajectory"]) for t in t1]
                v2 = [np.mean(t["gini_trajectory"]) for t in t2]

                min_len = min(len(v1), len(v2))
                if min_len >= 3:
                    r, p = pearsonr(v1[:min_len], v2[:min_len])
                    rho, rho_p = spearmanr(v1[:min_len], v2[:min_len])
                    correlations[f"{m1}_vs_{m2}"] = {
                        "pearson_r": float(r), "pearson_p": float(p),
                        "spearman_rho": float(rho), "spearman_p": float(rho_p),
                    }
                    print(f"  {m1} vs {m2}: Pearson r={r:.4f} (p={p:.3f}), "
                          f"Spearman ρ={rho:.4f} (p={rho_p:.3f})")

    # ── Figures ──
    # Gini trajectory per model
    for short_name in model_names:
        topo = all_model_results[short_name]["topological"]
        fig, ax = plt.subplots(figsize=(12, 6))

        for td in topo:
            traj = td["gini_trajectory"]
            layers = range(len(traj))
            ax.plot(layers, traj, "-o", markersize=3, alpha=0.7,
                    label=f"τ={td['complexity']} ({td['n_tokens']} tokens)")

        ax.set_xlabel("Layer index")
        ax.set_ylabel("Gini coefficient of H₀ persistence lifetimes")
        ax.set_title(f"Gini Trajectory: {short_name}", color=COLORS["gold"], fontsize=14)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.15)
        fig.savefig(FIG_DIR / f"p4_gini_trajectory_{short_name}.png")
        plt.close(fig)

    # Cross-model comparison (mean Gini vs complexity)
    if len(model_names) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_cycle = [COLORS["gold"], COLORS["teal"], COLORS["red"], COLORS["purple"]]
        for idx, short_name in enumerate(model_names):
            topo = all_model_results[short_name]["topological"]
            complexities = [t["complexity"] for t in topo]
            mean_ginis = [np.mean(t["gini_trajectory"]) for t in topo]
            ax.plot(complexities, mean_ginis, "-o", color=colors_cycle[idx % len(colors_cycle)],
                    linewidth=2.5, markersize=10, label=short_name)

        ax.set_xlabel("Prompt complexity τ")
        ax.set_ylabel("Mean Gini coefficient (across layers)")
        ax.set_title("Cross-Model: Mean Gini vs Prompt Complexity",
                      color=COLORS["gold"], fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.15)
        fig.savefig(FIG_DIR / "p4_cross_model_gini.png")
        plt.close(fig)

    print(f"\n  Figures saved to {FIG_DIR}/")

    # ── Verdict ──
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")

    max_r = max((c["pearson_r"] for c in correlations.values()), default=0)
    all_r = [c["pearson_r"] for c in correlations.values()]

    if len(all_r) >= 1 and all(r > 0.9 for r in all_r):
        verdict = "PASS"
    elif len(all_r) >= 1 and all(r > 0.7 for r in all_r):
        verdict = "PARTIAL PASS"
    elif len(all_r) == 0:
        verdict = "INSUFFICIENT DATA"
    else:
        verdict = "FAIL"

    print(f"  P4 verdict: {verdict}")
    print(f"  Cross-model correlations: {all_r}")
    print(f"  Paper claims r > 0.9. Threshold: r > 0.9 = PASS, r > 0.7 = PARTIAL.")

    # ── Save ──
    results = {
        "predictions": ["3", "4"],
        "timestamp": timestamp,
        "models": {name: {k: v for k, v in data.items() if k != "hidden_states"}
                   for name, data in all_model_results.items()},
        "cross_model_correlations": correlations,
        "verdict": verdict,
        "summary": (
            f"Tested {len(model_names)} models. "
            f"Cross-model Pearson r: {all_r}. "
            f"Paper threshold r > 0.9. Verdict: {verdict}."
        ),
    }

    out_path = OUTPUT_DIR / "p4_llm_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()
