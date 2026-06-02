"""Best-paper experiments: adaptive switcher, γ/κ frontier, fleet composition.

Run: python3 -m terrain_bidding.experiments.best_paper_experiments
"""
import numpy as np
import pickle
from pathlib import Path
from terrain_bidding.mechanism import (
    FullMechanism, TotalVarianceMechanism, VanillaMechanism,
    SimConfig, simulate, gaussian_score, compute_penalty, RoundResult, Bid,
)
from terrain_bidding.experiments import compute_metrics
from terrain_bidding.experiments.real_sampler import (
    make_real_task_sampler, EnsembleRobot, REAL_ADVERSARY_TYPES,
)
from terrain_bidding.experiments.fleet_experiments import FastRobot, CautiousRobot
from terrain_bidding.configs import MechanismConfig


# ─── 1. Adaptive Meta-Mechanism (Dynamic Switcher) ────────────────────────────

class AdaptiveMechanism:
    """Dynamically switches between aleatoric-only and total-variance scoring
    based on fleet's average epistemic uncertainty.

    When avg σ²_epi > threshold: use total_variance (lenient, protects honest)
    When avg σ²_epi ≤ threshold: use aleatoric-only (strict, catches adversaries)
    """
    def __init__(self, epi_threshold: float = 0.05):
        self.epi_threshold = epi_threshold
        self.full = FullMechanism()
        self.total_var = TotalVarianceMechanism()

    def allocate(self, bids, cfg):
        return self.full.allocate(bids, cfg)  # always use epistemic in allocation

    def score(self, bids, assignments, realized_costs, cfg):
        # Compute average epistemic variance across assigned robots
        avg_epi = 0
        n_assigned = 0
        for i in range(len(bids)):
            if assignments[i] < 0:
                continue
            bid = bids[i][assignments[i]]
            mu_hat = bid.mus.mean()
            var_epi = ((bid.mus - mu_hat) ** 2).mean()
            avg_epi += var_epi
            n_assigned += 1
        avg_epi = avg_epi / max(n_assigned, 1)

        # Switch based on epistemic level
        if avg_epi > self.epi_threshold:
            return self.total_var.score(bids, assignments, realized_costs, cfg)
        else:
            return self.full.score(bids, assignments, realized_costs, cfg)


def exp_adaptive_switcher():
    """Simulate terrain transition: familiar → mild OOD → strong OOD."""
    print("\n" + "="*70)
    print("EXP 1: ADAPTIVE META-MECHANISM (terrain transition)")
    print("="*70)

    in_dist = make_real_task_sampler(data_path="data/train_types012/test.hdf5")
    ood_mild = make_real_task_sampler(data_path="data/ood_mild/test.hdf5")
    ood_strong = make_real_task_sampler(data_path="data/ood_strong/test.hdf5")

    N, M = 4, 1
    rng = np.random.default_rng(0)
    scores = [gaussian_score(t[0].true_mu, t[0].robot_observations[0][0].mean(),
              np.exp(t[0].robot_observations[0][1]).mean())
              for t in [in_dist(1, rng, n_robots=N) for _ in range(500)]]
    S_baseline = np.mean(scores)

    # Simulate 3000 rounds: 1000 familiar, 1000 mild OOD, 1000 strong OOD
    mechanisms = {
        "Full (static)": FullMechanism(),
        "Total var (static)": TotalVarianceMechanism(),
        "Adaptive (ours)": AdaptiveMechanism(epi_threshold=0.05),
    }

    phases = [
        ("Familiar (0-1000)", in_dist, 1000),
        ("Mild OOD (1000-2000)", ood_mild, 1000),
        ("Strong OOD (2000-3000)", ood_strong, 1000),
    ]

    print(f"\n  {'Mechanism':>20} | {'Phase':>25} | {'FPR':>6} {'Sep':>8} {'Gain':>8}")
    print("  " + "-"*75)

    for mech_name, mech in mechanisms.items():
        for phase_name, sampler, n_rounds in phases:
            mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)
            fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
            fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))

            def s(n, rng, _sam=sampler):
                return _sam(M, rng, n_robots=N)

            sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg,
                               num_rounds=n_rounds, seed=42)
            results = simulate(sim_cfg, fleet, s)
            m = compute_metrics(results, 3, None)
            print(f"  {mech_name:>20} | {phase_name:>25} | "
                  f"{m.false_positive_rate:6.3f} {m.detection_separation:+8.3f} "
                  f"{m.strategic_gain:+8.3f}")


# ─── 2. γ/κ Vulnerability Heatmap ────────────────────────────────────────────

def exp_gamma_kappa_heatmap():
    """2D sweep: γ (allocation conservatism) vs κ (penalty) with ensemble manipulator."""
    print("\n" + "="*70)
    print("EXP 2: γ/κ VULNERABILITY HEATMAP")
    print("="*70)

    in_dist = make_real_task_sampler(data_path="data/train_types012/test.hdf5")
    N, M = 4, 1
    rng = np.random.default_rng(0)
    scores = [gaussian_score(t[0].true_mu, t[0].robot_observations[0][0].mean(),
              np.exp(t[0].robot_observations[0][1]).mean())
              for t in [in_dist(1, rng, n_robots=N) for _ in range(500)]]
    S_baseline = np.mean(scores)

    gammas = [0.0, 0.5, 1.0, 2.0, 5.0]
    kappas = [0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n  Adversary: ensemble_manipulating")
    print(f"  {'':>5} | " + " ".join(f"κ={k:<4}" for k in kappas))
    print("  " + "-"*50)

    heatmap = np.zeros((len(gammas), len(kappas)))
    for gi, gamma in enumerate(gammas):
        row = []
        for ki, kappa in enumerate(kappas):
            mech_cfg = MechanismConfig(N=N, kappa=kappa, gamma=gamma, S_baseline=S_baseline)
            fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
            fleet.append(REAL_ADVERSARY_TYPES["ensemble_manipulating"](robot_id=3))

            def s(n, rng):
                return in_dist(M, rng, n_robots=N)

            sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                               num_rounds=500, seed=42)
            results = simulate(sim_cfg, fleet, s)
            m = compute_metrics(results, 3, None)
            heatmap[gi, ki] = m.strategic_gain
            row.append(f"{m.strategic_gain:+.2f}")
        print(f"  γ={gamma:<3} | " + "  ".join(row))

    print(f"\n  Legend: + = adversary profits (VULNERABLE), - = adversary loses (ROBUST)")
    return {"gammas": gammas, "kappas": kappas, "heatmap": heatmap}


# ─── 3. Fleet Composition Sensitivity ────────────────────────────────────────

def exp_fleet_composition():
    """Vary ratio of noisy/cautious robots, measure detection breakdown point."""
    print("\n" + "="*70)
    print("EXP 3: FLEET COMPOSITION SENSITIVITY")
    print("="*70)

    in_dist = make_real_task_sampler(data_path="data/train_types012/test.hdf5")
    N = 8  # larger fleet to vary composition
    M = 2
    rng = np.random.default_rng(0)
    scores = [gaussian_score(t[0].true_mu, t[0].robot_observations[0][0].mean(),
              np.exp(t[0].robot_observations[0][1]).mean())
              for t in [in_dist(1, rng, n_robots=N) for _ in range(300)]]
    S_baseline = np.mean(scores)

    print(f"\n  Fleet size N={N}, M={M} tasks, 1 adversary")
    print(f"  Varying number of 'noisy' (poorly calibrated) honest robots")
    print(f"\n  {'Noisy/Total':>12} | {'Sep':>8} {'FPR':>6} {'Gain':>8} | {'Status'}")
    print("  " + "-"*55)

    for n_noisy in [0, 1, 2, 3, 4, 5, 6]:
        n_normal = N - 1 - n_noisy  # subtract 1 for adversary
        mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)

        fleet = []
        for i in range(n_normal):
            fleet.append(EnsembleRobot(robot_id=i))
        for i in range(n_noisy):
            fleet.append(FastRobot(robot_id=n_normal + i))
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=N-1, offset=0.5))

        def s(n, rng):
            return in_dist(M, rng, n_robots=N)

        sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                           num_rounds=1000, seed=42)
        results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(results, N-1, None)

        status = "DETECTS" if m.detection_separation > 0 else "FAILS"
        print(f"  {n_noisy}/{N-1:>5}      | {m.detection_separation:+8.3f} "
              f"{m.false_positive_rate:6.3f} {m.strategic_gain:+8.3f} | {status}")


# ─── Run All ──────────────────────────────────────────────────────────────────

def run_all():
    Path("outputs").mkdir(exist_ok=True)
    results = {}
    exp_adaptive_switcher()
    results["heatmap"] = exp_gamma_kappa_heatmap()
    exp_fleet_composition()

    with open("outputs/best_paper_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nResults saved to outputs/best_paper_results.pkl")


if __name__ == "__main__":
    run_all()
