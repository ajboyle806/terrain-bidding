"""Additional experiments: γ/κ frontier, Gaussian robustness, social welfare.

Run with: python3 -m terrain_bidding.experiments.additional_experiments
"""
import numpy as np
import pickle
from pathlib import Path
from terrain_bidding.mechanism import (
    FullMechanism, TotalVarianceMechanism, VanillaMechanism,
    SimConfig, simulate, gaussian_score, compute_penalty,
)
from terrain_bidding.experiments import compute_metrics
from terrain_bidding.experiments.real_sampler import (
    make_real_task_sampler, EnsembleRobot, REAL_ADVERSARY_TYPES,
)
from terrain_bidding.configs import MechanismConfig


def load_samplers():
    print("Loading samplers...")
    in_dist = make_real_task_sampler(data_path="data/test.hdf5")
    held_out = make_real_task_sampler(data_path="data/held_out/test.hdf5")
    return in_dist, held_out


def compute_s_baseline(sampler, N=4):
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(1000):
        ts = sampler(1, rng, n_robots=N)
        mus, lvs = ts[0].robot_observations[0]
        scores.append(gaussian_score(ts[0].true_mu, mus.mean(), np.exp(lvs).mean()))
    return float(np.mean(scores))


# ─── Experiment: γ/κ Robustness-Manipulability Frontier ───────────────────────

def exp_gamma_kappa_frontier(held_out, S_baseline):
    """Map the γ/κ phase diagram showing when ensemble manipulation succeeds.

    High γ + low κ = adversary exploits epistemic weighting
    Low γ + high κ = mechanism is robust but conservative allocation is lost
    """
    print("\n" + "="*70)
    print("EXP: γ/κ ROBUSTNESS-MANIPULABILITY FRONTIER")
    print("="*70)
    N, M = 4, 1

    print(f"  {'γ':>5} {'κ':>5} | {'Adv Gain':>9} {'Adv Wins':>9} {'Sep':>8} | {'Regime'}")
    print("  " + "-"*65)

    results = {}
    for gamma in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for kappa in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            mech_cfg = MechanismConfig(N=N, kappa=kappa, gamma=gamma, S_baseline=S_baseline)
            fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
            fleet.append(REAL_ADVERSARY_TYPES["ensemble_manipulating"](robot_id=3))

            def s(n, rng):
                return held_out(M, rng, n_robots=N)

            sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                               num_rounds=1000, seed=42)
            sim_results = simulate(sim_cfg, fleet, s)
            m = compute_metrics(sim_results, 3, None)
            adv_wins = sum(1 for r in sim_results if r.assignments[3] >= 0)

            # Classify regime
            if m.strategic_gain > 0.1:
                regime = "VULNERABLE"
            elif m.strategic_gain < -0.1:
                regime = "ROBUST"
            else:
                regime = "MARGINAL"

            results[(gamma, kappa)] = {
                "gain": m.strategic_gain, "sep": m.detection_separation,
                "adv_wins": adv_wins, "regime": regime
            }
            print(f"  {gamma:5.1f} {kappa:5.1f} | {m.strategic_gain:+9.4f} "
                  f"{adv_wins:>5}/1000 {m.detection_separation:+8.4f} | {regime}")

    # Summary
    vulnerable = sum(1 for v in results.values() if v["regime"] == "VULNERABLE")
    robust = sum(1 for v in results.values() if v["regime"] == "ROBUST")
    print(f"\n  Summary: {robust} ROBUST, {vulnerable} VULNERABLE out of {len(results)} configs")
    return results


# ─── Experiment: Gaussian Robustness (misspecification analysis) ──────────────

def exp_gaussian_robustness(held_out, S_baseline):
    """Test mechanism robustness when cost distribution is non-Gaussian.

    Add skewness and heavy tails to realized costs, show mechanism still works.
    """
    print("\n" + "="*70)
    print("EXP: GAUSSIAN ROBUSTNESS (misspecification analysis)")
    print("="*70)
    N, M = 4, 1
    mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)

    # First: analyze actual cost distribution
    rng = np.random.default_rng(42)
    costs = []
    for _ in range(2000):
        ts = held_out(1, rng, n_robots=N)
        costs.append(ts[0].true_mu)
    costs = np.array(costs)
    from scipy.stats import skew, kurtosis, shapiro
    print(f"  Cost distribution analysis:")
    print(f"    Mean={costs.mean():.3f}, Std={costs.std():.3f}")
    print(f"    Skewness={skew(costs):.3f} (0=Gaussian)")
    print(f"    Kurtosis={kurtosis(costs):.3f} (0=Gaussian)")
    if len(costs) > 5000:
        _, p_val = shapiro(costs[:5000])
    else:
        _, p_val = shapiro(costs)
    print(f"    Shapiro-Wilk p-value={p_val:.6f} (<0.05 = non-Gaussian)")

    # Test mechanism under different noise models
    print(f"\n  Mechanism performance under different noise models:")
    print(f"  {'Noise Model':>20} | {'Sep':>8} {'FPR':>6} {'Gain':>8}")
    print("  " + "-"*50)

    noise_models = {
        "Gaussian (baseline)": lambda rng, mu, var: rng.normal(mu, np.sqrt(max(var, 0.01))),
        "Heavy-tail (t, df=3)": lambda rng, mu, var: mu + rng.standard_t(3) * np.sqrt(max(var, 0.01)),
        "Right-skewed (exp)": lambda rng, mu, var: mu + (rng.exponential(np.sqrt(max(var, 0.01))) - np.sqrt(max(var, 0.01))),
        "Bimodal": lambda rng, mu, var: mu + rng.choice([-1, 1]) * rng.normal(0.5, np.sqrt(max(var, 0.01)) * 0.5),
    }

    for noise_name, noise_fn in noise_models.items():
        # Monkey-patch the sample_cost method for this run
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))

        rng2 = np.random.default_rng(42)
        # Manual simulation with custom noise
        from terrain_bidding.mechanism import RoundResult
        results_list = []
        for _ in range(2000):
            tasks = held_out(M, rng2, n_robots=N)
            bids = [[robot.bid(tasks[j], i, j) for j in range(M)]
                    for i, robot in enumerate(fleet)]
            assignments = FullMechanism().allocate(bids, mech_cfg)

            realized_costs = np.full(N, np.nan)
            for i in range(N):
                if assignments[i] >= 0:
                    t = tasks[assignments[i]]
                    mus, lvs = t.robot_observations[i]
                    var_ale = np.exp(lvs).mean()
                    # Use custom noise model instead of Gaussian
                    realized_costs[i] = noise_fn(rng2, t.true_mu, var_ale)

            scores = FullMechanism().score(bids, assignments, realized_costs, mech_cfg)
            penalties = np.zeros(N)
            utilities = np.zeros(N)
            for i in range(N):
                if assignments[i] >= 0:
                    penalties[i] = compute_penalty(scores[i], S_baseline, mech_cfg.kappa)
                    utilities[i] = mech_cfg.R * 1.0 - realized_costs[i] - penalties[i]

            results_list.append(RoundResult(assignments, realized_costs, scores, penalties, utilities))

        m = compute_metrics(results_list, 3, None)
        print(f"  {noise_name:>20} | {m.detection_separation:+8.4f} "
              f"{m.false_positive_rate:6.4f} {m.strategic_gain:+8.4f}")


# ─── Experiment: Social Welfare Analysis ──────────────────────────────────────

def exp_social_welfare(in_dist, held_out, S_baseline):
    """Comprehensive welfare analysis: efficiency, fairness, starvation."""
    print("\n" + "="*70)
    print("EXP: SOCIAL WELFARE ANALYSIS")
    print("="*70)
    N, M = 4, 1

    for terrain_name, sampler in [("in-dist", in_dist), ("held-out", held_out)]:
        print(f"\n  --- {terrain_name} terrain ---")
        print(f"  {'Mechanism':>12} {'n_adv':>5} | {'Welfare':>10} {'Honest U':>9} "
              f"{'Adv U':>7} {'Fairness':>8} {'Starved':>7}")
        print("  " + "-"*70)

        for n_adv in [0, 1, 2]:
            for mech_name, mech in [("vanilla", VanillaMechanism()),
                                     ("full", FullMechanism())]:
                mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)
                fleet = [EnsembleRobot(robot_id=i) for i in range(N - n_adv)]
                for i in range(n_adv):
                    fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](
                        robot_id=N - n_adv + i, offset=0.5))

                def s(n, rng, _sam=sampler):
                    return _sam(M, rng, n_robots=N)

                sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg,
                                   num_rounds=2000, seed=42)
                results = simulate(sim_cfg, fleet, s)

                # Compute welfare metrics
                all_utils = [[] for _ in range(N)]
                assignments_count = [0] * N
                for r in results:
                    for i in range(N):
                        all_utils[i].append(r.utilities[i])
                        if r.assignments[i] >= 0:
                            assignments_count[i] += 1

                total_welfare = sum(np.sum(r.utilities) for r in results)
                honest_util = np.mean([np.mean(all_utils[i]) for i in range(N - n_adv)])
                adv_util = np.mean([np.mean(all_utils[i]) for i in range(N - n_adv, N)]) if n_adv > 0 else 0

                # Fairness: std of assignment rates among honest robots
                honest_rates = [assignments_count[i] / len(results) for i in range(N - n_adv)]
                fairness = 1.0 - np.std(honest_rates) / max(np.mean(honest_rates), 0.01)

                # Starvation: any honest robot assigned < 10% of expected rate?
                expected_rate = M / N  # fair share
                starved = sum(1 for r in honest_rates if r < expected_rate * 0.1)

                print(f"  {mech_name:>12} {n_adv:>5} | {total_welfare:>10.0f} "
                      f"{honest_util:>9.2f} {adv_util:>7.2f} "
                      f"{fairness:>8.3f} {starved:>7d}")


# ─── Run All ──────────────────────────────────────────────────────────────────

def run_all():
    Path("outputs").mkdir(exist_ok=True)
    in_dist, held_out = load_samplers()
    S_baseline = compute_s_baseline(held_out)
    print(f"S_baseline = {S_baseline:.4f}")

    results = {}
    results["frontier"] = exp_gamma_kappa_frontier(held_out, S_baseline)
    exp_gaussian_robustness(held_out, S_baseline)
    exp_social_welfare(in_dist, held_out, S_baseline)

    with open("outputs/additional_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nResults saved to outputs/additional_results.pkl")


if __name__ == "__main__":
    run_all()
