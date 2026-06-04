"""Plot all paper figures from precomputed data (outputs/figure_data.pkl).

Run: python3 -m terrain_bidding.experiments.plot_figures
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
    'axes.titlesize': 10, 'legend.fontsize': 8, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'figure.dpi': 300, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
})
COL = 3.5  # IEEE single column width
Path("figures").mkdir(exist_ok=True)


def load_data():
    with open("outputs/figure_data.pkl", "rb") as f:
        return pickle.load(f)


def fig1(data):
    """All-bids vs assigned-only across terrain (with error bars)."""
    fig1d = data["fig1"]
    terrains = ["in_dist", "ood2", "ood3", "ood4"]
    labels = ["In-dist", "Mild\nOOD", "Mod\nOOD", "Strong\nOOD"]
    x = np.arange(4); w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL*2, 2.2))
    seps_a = [fig1d[t]["assigned"]["sep_mean"] for t in terrains]
    seps_b = [fig1d[t]["allbids"]["sep_mean"] for t in terrains]
    errs_a = [fig1d[t]["assigned"]["sep_std"] for t in terrains]
    errs_b = [fig1d[t]["allbids"]["sep_std"] for t in terrains]
    ax1.bar(x - w/2, seps_a, w, yerr=errs_a, capsize=3, label="Assigned-only", color="#ff7f0e")
    ax1.bar(x + w/2, seps_b, w, yerr=errs_b, capsize=3, label="All-bids (ours)", color="#2ca02c")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Detection sep."); ax1.set_title("(a) Detection")
    ax1.axhline(0, color="k", linewidth=0.5); ax1.legend()

    fprs_a = [fig1d[t]["assigned"]["fpr_mean"]*100 for t in terrains]
    fprs_b = [fig1d[t]["allbids"]["fpr_mean"]*100 for t in terrains]
    ferrs_a = [fig1d[t]["assigned"]["fpr_std"]*100 for t in terrains]
    ferrs_b = [fig1d[t]["allbids"]["fpr_std"]*100 for t in terrains]
    ax2.bar(x - w/2, fprs_a, w, yerr=ferrs_a, capsize=3, label="Assigned-only", color="#ff7f0e")
    ax2.bar(x + w/2, fprs_b, w, yerr=ferrs_b, capsize=3, label="All-bids (ours)", color="#2ca02c")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("FPR (%)"); ax2.set_title("(b) False positive rate")
    ax2.legend(); ax2.set_ylim(0, max(fprs_a + fprs_b) * 1.5)

    plt.tight_layout()
    plt.savefig("figures/fig1_allbids.png", bbox_inches="tight")
    plt.close()


def fig2(data):
    """κ sweep: in-dist (shows crossover) and OOD (all deter)."""
    fig2d = data["fig2"]
    kappas_id = [d["kappa"] for d in fig2d["in_dist"]]
    gains_id = [d["gain"] for d in fig2d["in_dist"]]
    kappas_ood = [d["kappa"] for d in fig2d["ood4"]]
    gains_ood = [d["gain"] for d in fig2d["ood4"]]

    fig, ax = plt.subplots(figsize=(COL, 2.5))
    ax.plot(kappas_id, gains_id, "o-", color="#1f77b4", markersize=5, label="In-distribution")
    ax.plot(kappas_ood, gains_ood, "s-", color="#2ca02c", markersize=5, label="OOD (novel terrain)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel(r"$\kappa$ (penalty coefficient)")
    ax.set_ylabel("Adversary gain")
    ax.set_title(r"Deterrence threshold")
    ax.set_xscale("log")
    ax.legend()
    ax.annotate("Manipulation\nunprofitable", xy=(5, -1.2), fontsize=8, color="#2ca02c")
    plt.tight_layout()
    plt.savefig("figures/fig2_deterrence.png", bbox_inches="tight")
    plt.close()


def fig3(data):
    """Score distributions."""
    honest = np.array(data["fig3"]["honest"])
    adv = np.array(data["fig3"]["adv"])
    h_mean, a_mean = honest.mean(), adv.mean()
    sep = h_mean - a_mean

    fig, ax = plt.subplots(figsize=(COL, 2.2))
    lo = max(min(honest.min(), adv.min()), -50)
    hi = min(max(honest.max(), adv.max()), 5)
    bins = np.linspace(lo, hi, 50)
    ax.hist(honest, bins=bins, alpha=0.6, color="#1f77b4", label="Honest", density=True)
    ax.hist(adv, bins=bins, alpha=0.6, color="#d62728", label="Adversary", density=True)
    ax.axvline(h_mean, color="#1f77b4", linestyle="--", linewidth=1.5)
    ax.axvline(a_mean, color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Score"); ax.set_ylabel("Density")
    ax.set_title(f"Score separation = {sep:.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/fig_score_distributions.png", bbox_inches="tight")
    plt.close()


def fig4(data):
    """Epistemic vs aleatoric decomposition."""
    fig4d = data["fig4"]
    labels = ["In-dist", "Mild\nOOD", "Mod\nOOD", "Strong\nOOD"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL*2, 2.2))
    bp1 = ax1.boxplot(fig4d["epi"], tick_labels=labels, patch_artist=True, widths=0.6)
    for p in bp1["boxes"]: p.set_facecolor("#ff7f0e"); p.set_alpha(0.7)
    ax1.set_ylabel(r"$\sigma^2_{epi}$"); ax1.set_title(r"(a) Epistemic")
    ax1.set_yscale("log")

    bp2 = ax2.boxplot(fig4d["ale"], tick_labels=labels, patch_artist=True, widths=0.6)
    for p in bp2["boxes"]: p.set_facecolor("#1f77b4"); p.set_alpha(0.7)
    ax2.set_ylabel(r"$\sigma^2_{ale}$"); ax2.set_title(r"(b) Aleatoric")

    plt.tight_layout()
    plt.savefig("figures/fig_decomposition_terrain.png", bbox_inches="tight")
    plt.close()


def fig5(data):
    """Reputation comparison."""
    fig5d = data["fig5"]
    mechs = ["Vanilla", "Reputation", "Ours\n(all-bids)"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    seps = [fig5d["vanilla"]["sep"], fig5d["reputation"]["sep"], fig5d["allbids"]["sep"]]
    gains = [fig5d["vanilla"]["gain"], fig5d["reputation"]["gain"], fig5d["allbids"]["gain"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL*2, 2.2))
    ax1.bar(mechs, seps, color=colors)
    ax1.set_ylabel("Detection sep."); ax1.set_title("(a) Detection")
    ax2.bar(mechs, gains, color=colors)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_ylabel("Adversary gain"); ax2.set_title("(b) Deterrence")
    plt.tight_layout()
    plt.savefig("figures/fig5_reputation.png", bbox_inches="tight")
    plt.close()


def fig6(data):
    """Scaling."""
    fig6d = data["fig6"]
    Ns = [4, 8, 16]; x = np.arange(3); w = 0.35
    vanilla_gains = [d["gain"] for d in fig6d if d["mech"] == "vanilla"]
    allbids_gains = [d["gain"] for d in fig6d if d["mech"] == "allbids"]

    fig, ax = plt.subplots(figsize=(COL, 2.2))
    ax.bar(x - w/2, vanilla_gains, w, label="Vanilla", color="#d62728")
    ax.bar(x + w/2, allbids_gains, w, label="Ours", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylabel("Adversary gain"); ax.set_title("Scalability")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/fig6_scaling.png", bbox_inches="tight")
    plt.close()


def fig7(data):
    """ANYmal real data."""
    fig7d = data["fig7"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL*2, 2.2))
    mechs = list(fig7d["mechanisms"].keys())
    seps = [fig7d["mechanisms"][m]["sep"] for m in mechs]
    colors = ["#d62728" if m == "vanilla" else "#2ca02c" for m in mechs]
    ax1.bar(mechs, seps, color=colors)
    ax1.set_ylabel("Detection sep."); ax1.set_title("(a) Real ANYmal data")

    ks = [d["kappa"] for d in fig7d["kappa_sweep"]]
    gs = [d["gain"] for d in fig7d["kappa_sweep"]]
    ax2.plot(ks, gs, "o-", color="#9467bd", markersize=5)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_xlabel(r"$\kappa$"); ax2.set_ylabel("Adversary gain")
    ax2.set_title(r"(b) $\kappa$ sweep (real data)")
    ax2.set_xscale("log")

    plt.tight_layout()
    plt.savefig("figures/fig7_anymal.png", bbox_inches="tight")
    plt.close()


def fig8(data):
    """Allocation sensitivity."""
    fig8d = data["fig8"]
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    ax.plot(fig8d["offsets"], fig8d["win_rates"], "o-", color="#d62728", markersize=4)
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="Fair share (1/N)")
    ax.set_xlabel(r"Bid shade $\delta$"); ax.set_ylabel("Win rate")
    ax.set_title("Allocation sensitivity")
    ax.legend(); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("figures/fig_allocation_sensitivity.png", bbox_inches="tight")
    plt.close()


def main():
    data = load_data()
    fig1(data); fig2(data); fig3(data); fig4(data)
    fig5(data); fig6(data); fig7(data); fig8(data)
    print("All figures generated from outputs/figure_data.pkl")


if __name__ == "__main__":
    main()
