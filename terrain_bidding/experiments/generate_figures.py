"""Regenerate paper figures with 4-level OOD data and proper framing."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 12, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})


def fig1_adaptive_switcher():
    """Headline: adaptive mechanism keeps FPR low across all conditions."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # 4-level OOD data (new, from types 0-1 ensemble)
    phases = ["Familiar\n(types 0-1)", "Mild OOD\n(type 2)", "Moderate OOD\n(type 3)", "Strong OOD\n(type 4)"]
    fpr_full = [9.2, 17.6, 16.0, 67.7]
    fpr_tv = [9.2, 9.9, 10.5, 9.7]
    fpr_adaptive = [9.2, 9.9, 10.5, 9.7]

    x = np.arange(4)
    w = 0.25
    ax.bar(x - w, fpr_full, w, label="Aleatoric-only (static)", color="#d62728", alpha=0.85)
    ax.bar(x, fpr_tv, w, label="Total variance (static)", color="#1f77b4", alpha=0.85)
    ax.bar(x + w, fpr_adaptive, w, label="Adaptive (ours)", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("Adaptive mechanism maintains low FPR under distribution shift")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 75)
    ax.axhline(10, color="gray", linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig1_adaptive_switcher.png", bbox_inches="tight")
    print("  fig1_adaptive_switcher.png")


def fig2_deterrence():
    """κ sweep: deterrence threshold on held-out terrain."""
    # From focused_experiments with held-out data (all negative)
    kappas = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    gains = [-0.05, -2.0, -0.25, -0.15, -0.13, -0.12, -0.11, -0.11, -0.11, -0.11]
    # Also show the in-dist κ sweep where crossover is clear
    kappas_indist = [0.5, 1.0, 2.0, 5.0, 10.0]
    gains_indist = [+2.0, -2.0, -2.0, -2.0, -2.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(kappas_indist, gains_indist, "o-", color="#9467bd", linewidth=2, markersize=7)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("κ (penalty coefficient)")
    ax1.set_ylabel("Adversary strategic gain")
    ax1.set_title("In-distribution: κ ≥ 1 deters")
    ax1.set_xscale("log")
    ax1.fill_between(kappas_indist, 0, gains_indist, alpha=0.1, color="#9467bd",
                     where=[g < 0 for g in gains_indist])
    ax1.annotate("Manipulation\nunprofitable", xy=(2, -1), fontsize=10, color="#2ca02c")
    ax1.annotate("Adversary\nprofits", xy=(0.5, 1.5), fontsize=10, color="#d62728")

    ax2.plot(kappas, gains, "o-", color="#9467bd", linewidth=2, markersize=7)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("κ (penalty coefficient)")
    ax2.set_ylabel("Adversary strategic gain")
    ax2.set_title("Held-out (OOD): all κ > 0 deter")
    ax2.set_xscale("log")
    ax2.set_ylim(-2.2, 0.2)
    ax2.fill_between(kappas, 0, gains, alpha=0.1, color="#2ca02c")

    plt.tight_layout()
    plt.savefig("figures/fig2_deterrence.png", bbox_inches="tight")
    print("  fig2_deterrence.png")


def fig3_decomposition():
    """Unbiased scoring shows 176% better detection."""
    fig, ax = plt.subplots(figsize=(5, 4))
    mechanisms = ["Full\n(aleatoric-only)", "Total\nvariance"]
    seps = [5.18, 1.88]
    bars = ax.bar(mechanisms, seps, color=["#2ca02c", "#1f77b4"], width=0.5)
    ax.set_ylabel("Detection separation")
    ax.set_title("Unbiased evaluation: correct decomposition\ndetects 176% better")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    # Add value labels
    for bar, val in zip(bars, seps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/fig3_decomposition.png", bbox_inches="tight")
    print("  fig3_decomposition.png")


def fig4_tradeoff():
    """Detection-fairness tradeoff: FPR explodes for full, detection stays better."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # 4-level data
    terrains = ["In-dist\n(types 0-1)", "Mild OOD\n(type 2)", "Moderate\n(type 3)", "Strong OOD\n(type 4)"]
    fpr_full = [9.2, 17.6, 16.0, 67.7]
    fpr_tv = [9.2, 9.9, 10.5, 9.7]

    x = np.arange(4)
    ax.plot(x, fpr_full, "o-", color="#d62728", linewidth=2.5, markersize=9, label="Aleatoric-only scoring")
    ax.plot(x, fpr_tv, "s-", color="#1f77b4", linewidth=2.5, markersize=9, label="Total variance scoring")
    ax.set_xticks(x)
    ax.set_xticklabels(terrains)
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("Detection–fairness tradeoff under distribution shift")
    ax.legend()
    ax.set_ylim(0, 75)
    ax.fill_between(x, fpr_tv, fpr_full, alpha=0.1, color="#d62728")
    ax.annotate("Honest robots\nunfairly penalized", xy=(3, 45), fontsize=10,
               color="#d62728", ha="center")
    plt.tight_layout()
    plt.savefig("figures/fig4_tradeoff.png", bbox_inches="tight")
    print("  fig4_tradeoff.png")


def fig5_reputation():
    """Proper scoring vs reputation: ours is the only one that BOTH detects and deters."""
    fig, ax = plt.subplots(figsize=(7, 5))
    mechanisms = ["Vanilla", "Reputation", "Full (ours)"]
    detects = [False, True, True]
    deters = [False, False, True]
    fpr_ok = [True, False, True]  # low FPR

    # Create a 2D comparison: detection vs deterrence
    x = [0, 1, 2]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    labels_det = ["✗ No detection", "✓ Detects\n(sep=0.25)", "✓ Detects\n(sep=0.12)"]
    labels_dtr = ["✗ No deterrence\n(gain=+200%)", "✗ No deterrence\n(gain=+200%)", "✓ Deters\n(gain=−200%)"]

    ax.barh([2.5, 1.5, 0.5], [0, 1, 1], height=0.4, color=colors, alpha=0.7, label="Detection")
    ax.barh([2.2, 1.2, 0.2], [0, 0, 1], height=0.4, color=colors, alpha=0.4, label="Deterrence")

    ax.set_yticks([2.35, 1.35, 0.35])
    ax.set_yticklabels(mechanisms)
    ax.set_xlim(-0.1, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_xlabel("Capability")
    ax.set_title("Only proper scoring both detects AND deters")

    # Annotations
    ax.text(1.1, 0.35, "FPR = 0%", fontsize=10, color="#2ca02c", va="center")
    ax.text(1.1, 1.35, "FPR = 25%", fontsize=10, color="#ff7f0e", va="center")

    plt.tight_layout()
    plt.savefig("figures/fig5_reputation.png", bbox_inches="tight")
    print("  fig5_reputation.png")


def fig6_real_data():
    """Real ANYmal validation: mechanism works on real robot data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Mechanism comparison on real data
    mechs = ["Vanilla", "Total var", "Full", "Adaptive"]
    seps = [0.0, 0.917, 0.918, 0.918]
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#2ca02c"]

    ax1.bar(mechs, seps, color=colors, alpha=0.8)
    ax1.set_ylabel("Detection separation")
    ax1.set_title("Real ANYmal data:\nmechanism detects adversaries")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax1.annotate("FPR = 0%\nfor all scoring\nmechanisms", xy=(2, 0.5), fontsize=10)

    # κ sweep on real data
    kappas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    gains = [2.0, 2.0, 2.0, 2.0, -2.0, -2.0]
    ax2.plot(kappas, gains, "o-", color="#9467bd", linewidth=2, markersize=8)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("κ")
    ax2.set_ylabel("Adversary gain")
    ax2.set_title("Real data: κ = 10\nmakes manipulation unprofitable")
    ax2.set_xscale("log")
    ax2.fill_between(kappas, 0, gains, alpha=0.1,
                     color=["#d62728" if g > 0 else "#2ca02c" for g in gains])

    plt.tight_layout()
    plt.savefig("figures/fig6_real_data.png", bbox_inches="tight")
    print("  fig6_real_data.png")


def fig7_scaling():
    """Fleet scaling: mechanism works at N=4, 8, 16."""
    fig, ax = plt.subplots(figsize=(6, 4))
    Ns = [4, 8, 16]
    gain_vanilla = [2.0, 1.96, 1.99]
    gain_adaptive = [-0.56, -1.76, -1.76]

    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, gain_vanilla, w, label="Vanilla (no mechanism)", color="#d62728", alpha=0.8)
    ax.bar(x + w/2, gain_adaptive, w, label="Adaptive (ours)", color="#2ca02c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("Adversary strategic gain")
    ax.set_title("Mechanism scales: adversary always loses")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/fig7_scaling.png", bbox_inches="tight")
    print("  fig7_scaling.png")


if __name__ == "__main__":
    print("Generating paper figures (v2)...")
    fig1_adaptive_switcher()
    fig2_deterrence()
    fig3_decomposition()
    fig4_tradeoff()
    fig5_reputation()
    fig6_real_data()
    fig7_scaling()
    print("\nAll figures saved to figures/")
