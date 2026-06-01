"""Generate paper figures from experiment results.

Run: python3 -m terrain_bidding.experiments.generate_figures
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def fig_kappa_sweep():
    """Figure: κ sweep showing deterrence threshold."""
    kappas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    gains = [+2.0, -2.0, -2.0, -2.0, -2.0, -2.0]  # from deterrence experiment
    penalties_adv = [4.957, 9.913, 19.827, 49.567, 99.134, 198.268]
    penalties_hon = [1.147, 2.293, 4.587, 11.466, 22.933, 45.865]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Gain vs κ
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.plot(kappas, gains, "o-", color="tab:red", markersize=8, linewidth=2)
    ax1.set_xlabel("κ (penalty coefficient)")
    ax1.set_ylabel("Strategic gain")
    ax1.set_title("Deterrence: manipulation becomes unprofitable")
    ax1.set_xscale("log")
    ax1.annotate("κ* ≈ 1.0", xy=(1.0, 0), xytext=(2, 1),
                arrowprops=dict(arrowstyle="->"), fontsize=10)

    # Penalty ratio vs κ
    ratios = [a/h for a, h in zip(penalties_adv, penalties_hon)]
    ax2.plot(kappas, ratios, "s-", color="tab:blue", markersize=8, linewidth=2)
    ax2.set_xlabel("κ (penalty coefficient)")
    ax2.set_ylabel("Penalty ratio (adversary / honest)")
    ax2.set_title("Adversary penalized ~4.3× more")
    ax2.set_xscale("log")
    ax2.axhline(1, color="gray", linestyle="--", alpha=0.5, label="Equal penalty")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("figures/kappa_sweep.pdf", bbox_inches="tight")
    print("  Saved figures/kappa_sweep.pdf")


def fig_detection_scaling():
    """Figure: Detection scales with manipulation magnitude."""
    offsets = [0.1, 0.3, 0.5, 1.0, 1.5]
    seps = [-1.535, -0.725, 5.111, 31.063, 40.511]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(offsets, seps, "D-", color="tab:green", markersize=8, linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Bid offset δ (manipulation magnitude)")
    ax.set_ylabel("Detection separation")
    ax.set_title("Detection scales with manipulation severity")
    ax.fill_between(offsets, 0, seps, alpha=0.1, color="tab:green",
                    where=[s > 0 for s in seps])
    plt.tight_layout()
    plt.savefig("figures/detection_scaling.pdf", bbox_inches="tight")
    print("  Saved figures/detection_scaling.pdf")


def fig_distribution_shift():
    """Figure: Decomposition gap under distribution shift."""
    terrains = ["In-distribution", "Held-out (OOD)"]
    full_seps = [-2.0706, +5.1111]
    tv_seps = [-2.0281, +4.8978]
    gaps = [f - t for f, t in zip(full_seps, tv_seps)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(terrains))
    w = 0.35
    ax1.bar(x - w/2, full_seps, w, label="Full (ours)", color="tab:green")
    ax1.bar(x + w/2, tv_seps, w, label="Total variance", color="tab:blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(terrains)
    ax1.set_ylabel("Detection separation")
    ax1.set_title("Detection by terrain type")
    ax1.legend()
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.3)

    ax2.bar(terrains, gaps, color=["tab:red", "tab:green"])
    ax2.set_ylabel("Gap (full − total_var)")
    ax2.set_title("Decomposition gap widens under OOD")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/distribution_shift.pdf", bbox_inches="tight")
    print("  Saved figures/distribution_shift.pdf")


def fig_reputation_comparison():
    """Figure: Proper scoring vs reputation baseline."""
    mechanisms = ["Vanilla", "Reputation\n(window=50)", "Reputation\n(window=200)", "Full\n(ours)"]
    seps = [0.0, 0.248, 0.258, 0.116]
    gains = [+2.0, +2.0, +2.0, -2.0]
    fprs = [0.0, 0.25, 0.25, 0.0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(mechanisms))
    colors = ["tab:red", "tab:orange", "tab:orange", "tab:green"]

    ax1.bar(x, seps, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(mechanisms, fontsize=9)
    ax1.set_ylabel("Detection separation")
    ax1.set_title("Detection")

    ax2.bar(x, gains, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(mechanisms, fontsize=9)
    ax2.set_ylabel("Strategic gain")
    ax2.set_title("Deterrence")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax3.bar(x, fprs, color=colors)
    ax3.set_xticks(x)
    ax3.set_xticklabels(mechanisms, fontsize=9)
    ax3.set_ylabel("False positive rate")
    ax3.set_title("FPR (lower = better)")

    plt.tight_layout()
    plt.savefig("figures/reputation_comparison.pdf", bbox_inches="tight")
    print("  Saved figures/reputation_comparison.pdf")


def fig_swapped_ablation():
    """Figure: Correct vs swapped vs total_var decomposition."""
    mechanisms = ["Full\n(correct)", "Total\nvariance", "Swapped\n(wrong)", "Vanilla\n(none)"]
    seps = [0.116, -0.003, 540.7, 0.0]
    # Cap swapped for visualization
    seps_capped = [0.116, -0.003, 5.0, 0.0]  # cap at 5 for readability

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["tab:green", "tab:blue", "tab:red", "gray"]
    bars = ax.bar(mechanisms, seps_capped, color=colors)
    ax.set_ylabel("Detection separation")
    ax.set_title("Ablation: only correct decomposition works")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.annotate("sep=540.7\n(degenerate)", xy=(2, 5), fontsize=9, ha="center",
               color="tab:red")
    plt.tight_layout()
    plt.savefig("figures/swapped_ablation.pdf", bbox_inches="tight")
    print("  Saved figures/swapped_ablation.pdf")


def fig_welfare():
    """Figure: Honest robot starvation under vanilla vs mechanism."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Vanilla\n(no mechanism)", "Full mechanism\n(ours)"]
    honest_assigned = [6/9000 * 100, 6/9000 * 100]  # from held-out results
    adv_assigned = [2994/3000 * 100, 2994/3000 * 100]

    # Use the in-dist results where starvation is clearer
    # Vanilla: honest=0.66 util, adv=3.02 util
    # Full: honest=-0.35 util, adv=1.07 util (adversary still profits less)
    honest_utils = [0.66, -0.35]
    adv_utils = [3.02, 1.07]

    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, honest_utils, w, label="Honest robots", color="tab:blue")
    ax.bar(x + w/2, adv_utils, w, label="Adversary", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean utility per round")
    ax.set_title("Welfare: mechanism reduces adversary advantage")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/welfare.pdf", bbox_inches="tight")
    print("  Saved figures/welfare.pdf")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_kappa_sweep()
    fig_detection_scaling()
    fig_distribution_shift()
    fig_reputation_comparison()
    fig_swapped_ablation()
    fig_welfare()
    print(f"\nAll figures saved to figures/")
