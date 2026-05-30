"""Paper figure generation from experiment results."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from terrain_bidding.experiments import ExperimentMetrics


SAVE_DIR = Path("figures")
STYLE = {"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
          "legend.fontsize": 10, "figure.dpi": 150}


def setup():
    plt.rcParams.update(STYLE)
    SAVE_DIR.mkdir(exist_ok=True)


def plot_kappa_sweep(results: Dict[str, ExperimentMetrics], save=True):
    """Fig: optimal adversary behavior vs κ. Validates Proposition 2."""
    setup()
    kappas, penalties_adv, penalties_hon, separations = [], [], [], []
    for key, m in sorted(results.items()):
        if "kappa_sweep" not in key:
            continue
        k = float(key.split("kappa=")[1])
        kappas.append(k)
        penalties_adv.append(m.mean_penalty_adversary)
        penalties_hon.append(m.mean_penalty_honest)
        separations.append(m.detection_separation)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(kappas, penalties_adv, "o-", label="Adversary")
    ax1.plot(kappas, penalties_hon, "s-", label="Honest")
    ax1.set_xlabel("κ (penalty coefficient)")
    ax1.set_ylabel("Mean penalty")
    ax1.set_title("Penalty vs κ")
    ax1.legend()
    ax1.set_xscale("log")

    ax2.plot(kappas, separations, "D-", color="tab:green")
    ax2.set_xlabel("κ (penalty coefficient)")
    ax2.set_ylabel("Score separation (honest − adversary)")
    ax2.set_title("Detection separation vs κ")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save:
        plt.savefig(SAVE_DIR / "kappa_sweep.pdf", bbox_inches="tight")
    return fig


def plot_mechanism_comparison(results: Dict[str, ExperimentMetrics], save=True):
    """Fig: bar chart comparing 4 mechanisms on key metrics."""
    setup()
    mechanisms = ["vanilla", "fixed_threshold", "total_variance", "full"]
    labels = ["Vanilla", "Fixed Thresh.", "Total Var.", "Full (ours)"]

    # Aggregate metrics per mechanism (adversary present, mixed terrain)
    efficiencies, fprs, separations = [], [], []
    for mech in mechanisms:
        matching = [m for k, m in results.items()
                    if k.startswith(mech + "|") and "n_adv=1" in k and "mixed" in k]
        if matching:
            efficiencies.append(np.mean([m.allocation_efficiency for m in matching]))
            fprs.append(np.mean([m.false_positive_rate for m in matching]))
            separations.append(np.mean([m.detection_separation for m in matching]))
        else:
            efficiencies.append(0)
            fprs.append(0)
            separations.append(0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(labels))

    axes[0].bar(x, efficiencies, color=["tab:red", "tab:orange", "tab:blue", "tab:green"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_ylabel("Allocation efficiency")
    axes[0].set_title("Efficiency (higher = better)")
    axes[0].set_ylim(0, 1.1)

    axes[1].bar(x, separations, color=["tab:red", "tab:orange", "tab:blue", "tab:green"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_ylabel("Score separation")
    axes[1].set_title("Detection (higher = better)")

    axes[2].bar(x, fprs, color=["tab:red", "tab:orange", "tab:blue", "tab:green"])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15)
    axes[2].set_ylabel("False positive rate")
    axes[2].set_title("FPR (lower = better)")
    axes[2].set_ylim(0, 0.3)

    plt.tight_layout()
    if save:
        plt.savefig(SAVE_DIR / "mechanism_comparison.pdf", bbox_inches="tight")
    return fig


def plot_gamma_ablation(results: Dict[str, ExperimentMetrics], save=True):
    """Fig: efficiency vs γ (epistemic weighting) under honest fleet."""
    setup()
    gammas, efficiencies = [], []
    for key, m in sorted(results.items()):
        if "gamma_ablation" not in key:
            continue
        g = float(key.split("gamma=")[1])
        gammas.append(g)
        efficiencies.append(m.allocation_efficiency)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(gammas, efficiencies, "o-", color="tab:purple", markersize=8)
    ax.set_xlabel("γ (epistemic weighting)")
    ax.set_ylabel("Allocation efficiency")
    ax.set_title("Efficiency cost of conservative allocation")
    ax.set_ylim(0.8, 1.05)
    plt.tight_layout()
    if save:
        plt.savefig(SAVE_DIR / "gamma_ablation.pdf", bbox_inches="tight")
    return fig


def plot_terrain_comparison(results: Dict[str, ExperimentMetrics], save=True):
    """Fig: mechanism performance across terrain distributions."""
    setup()
    terrains = ["mixed", "adversarial", "held_out"]
    terrain_labels = ["In-distribution", "Adversarial", "Held-out"]

    full_sep, full_fpr = [], []
    total_sep, total_fpr = [], []
    for t in terrains:
        full = [m for k, m in results.items()
                if "full|" in k and t in k and "n_adv=1" in k]
        total = [m for k, m in results.items()
                 if "total_variance|" in k and t in k and "n_adv=1" in k]
        full_sep.append(np.mean([m.detection_separation for m in full]) if full else 0)
        full_fpr.append(np.mean([m.false_positive_rate for m in full]) if full else 0)
        total_sep.append(np.mean([m.detection_separation for m in total]) if total else 0)
        total_fpr.append(np.mean([m.false_positive_rate for m in total]) if total else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(terrains))
    w = 0.35

    ax1.bar(x - w/2, full_sep, w, label="Full (ours)", color="tab:green")
    ax1.bar(x + w/2, total_sep, w, label="Total variance", color="tab:blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(terrain_labels)
    ax1.set_ylabel("Detection separation")
    ax1.set_title("Detection across terrain types")
    ax1.legend()

    ax2.bar(x - w/2, full_fpr, w, label="Full (ours)", color="tab:green")
    ax2.bar(x + w/2, total_fpr, w, label="Total variance", color="tab:blue")
    ax2.set_xticks(x)
    ax2.set_xticklabels(terrain_labels)
    ax2.set_ylabel("False positive rate")
    ax2.set_title("FPR across terrain types")
    ax2.legend()

    plt.tight_layout()
    if save:
        plt.savefig(SAVE_DIR / "terrain_comparison.pdf", bbox_inches="tight")
    return fig


def plot_adversary_comparison(results: Dict[str, ExperimentMetrics], save=True):
    """Fig: mechanism robustness across adversary types."""
    setup()
    adv_types = ["fixed_offset", "adaptive", "learning",
                 "terrain_selective", "ensemble_manipulating"]
    labels = ["Fixed\noffset", "Adaptive", "Learning",
              "Terrain\nselective", "Ensemble\nmanip."]

    penalties, gains = [], []
    for adv in adv_types:
        matching = [m for k, m in results.items()
                    if f"full|{adv}|" in k and "n_adv=1" in k and "mixed" in k]
        penalties.append(np.mean([m.mean_penalty_adversary for m in matching]) if matching else 0)
        gains.append(np.mean([m.strategic_gain for m in matching]) if matching else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(labels))

    ax1.bar(x, penalties, color="tab:red", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean adversary penalty")
    ax1.set_title("Penalty by adversary type")

    ax2.bar(x, [g * 100 for g in gains], color="tab:orange", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Strategic gain (%)")
    ax2.set_title("Gain by adversary type (negative = backfired)")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save:
        plt.savefig(SAVE_DIR / "adversary_comparison.pdf", bbox_inches="tight")
    return fig


def plot_all(results: Dict[str, ExperimentMetrics]):
    """Generate all paper figures."""
    setup()
    plot_kappa_sweep(results)
    plot_mechanism_comparison(results)
    plot_gamma_ablation(results)
    plot_terrain_comparison(results)
    plot_adversary_comparison(results)
    print(f"Figures saved to {SAVE_DIR}/")


if __name__ == "__main__":
    # Load results from a saved run (placeholder)
    import pickle
    results_path = Path("outputs/experiment_results.pkl")
    if results_path.exists():
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        plot_all(results)
    else:
        print("No results found. Run experiments first, then plot.")
        print("  python -m terrain_bidding.experiments")
        print("  python -m terrain_bidding.experiments.plot")
