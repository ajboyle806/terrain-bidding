"""Paper-strengthening experiments: scaling, convergence, welfare, decomposition.

Run with: python -m terrain_bidding.experiments.paper_experiments
"""
import numpy as np
import pickle
from pathlib import Path
from terrain_bidding.mechanism import (
    FullMechanism, TotalVarianceMechanism, VanillaMechanism,
    FixedThresholdMechanism, SimConfig, simulate, gaussian_score,
    estimate_lipschitz,
)
from terrain_bidding.experiments import compute_metrics
from terrain_bidding.experiments.real_sampler import (
    make_real_task_sampler, EnsembleRobot, REAL_ADVERSARY_TYPES,
)
from terrain_bidding.experiments.held_out import (
    make_held_out_proxy_sampler, make_mixed_sampler,
)
from terrain_bidding.configs import MechanismConfig


def setup():
    """Load real sampler and compute S_baseline."""
    real_sampler = make_real_task_sampler()
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(1000):
        tasks = real_sampler(1, rng)
        t = tasks[0]
        mu_hat = t.ensemble_mus.mean()
        var_ale = np.exp(t.ensemble_log_vars).mean()
        scores.append(gaussian_score(t.true_mu, mu_hat, var_ale))
    S_baseline = float(np.mean(scores))
    print(f"S_baseline = {S_baseline:.4f}")
    return real_sampler, S_baseline


# ─── Experiment 1: Fleet Size Scaling ─────────────────────────────────────────

def run_scaling_experiment(real_sampler, S_baseline):
    """N=4,8,16 with fixed adversary fraction (25%)."""
    print("\n=== SCALING EXPERIMENT (N=4,8,16) ===")
    results = {}
    for N in [4, 8, 16]:
        M = N // 2  # 50% task scarcity
        n_adv = N // 4  # 25% adversaries
        n_honest = N - n_adv
        mech_cfg = MechanismConfig(N=N, kappa=2.0, gamma=1.0, S_baseline=S_baseline)

        for mech_name, mech in [("vanilla", VanillaMechanism()),
                                 ("full", FullMechanism())]:
            fleet = [EnsembleRobot(robot_id=i) for i in range(n_honest)]
            for i in range(n_adv):
                fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](
                    robot_id=n_honest + i))

            def scarce(n, rng, _s=real_sampler, _m=M):
                return _s(_m, rng)

            sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg,
                               num_rounds=3000, seed=42)
            sim_results = simulate(sim_cfg, fleet, scarce)
            m = compute_metrics(sim_results, n_honest, None)
            key = f"scaling|N={N}|{mech_name}"
            results[key] = m
            print(f"  N={N:2d} {mech_name:8s} | Eff={m.allocation_efficiency:.3f} "
                  f"Sep={m.detection_separation:+.3f} FPR={m.false_positive_rate:.3f} "
                  f"Welfare={m.total_welfare:.0f}")
    return results


# ─── Experiment 2: Learning Adversary Convergence ─────────────────────────────

def run_convergence_experiment(real_sampler, S_baseline):
    """Track learning adversary's κ_est and behavior over 5000 rounds."""
    print("\n=== LEARNING ADVERSARY CONVERGENCE ===")
    N, M = 4, 2
    mech_cfg = MechanismConfig(N=N, kappa=2.0, gamma=1.0, S_baseline=S_baseline)

    fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
    learner = REAL_ADVERSARY_TYPES["learning"](robot_id=3, kappa_init=0.5)
    fleet.append(learner)

    def scarce(n, rng):
        return real_sampler(M, rng)

    # Run and track κ_est per round
    rng = np.random.default_rng(42)
    kappa_history = []
    delta_history = []

    sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                       num_rounds=5000, seed=42)
    results = simulate(sim_cfg, fleet, scarce)

    # Extract κ history from learner (it updates each round)
    # Re-run manually to track per-round
    learner2 = REAL_ADVERSARY_TYPES["learning"](robot_id=3, kappa_init=0.5)
    fleet2 = [EnsembleRobot(robot_id=i) for i in range(3)]
    fleet2.append(learner2)

    rng2 = np.random.default_rng(42)
    for round_idx in range(5000):
        tasks = real_sampler(M, rng2)
        bids = []
        for i, robot in enumerate(fleet2):
            robot_bids = [robot.bid(tasks[j], i, j) for j in range(M)]
            bids.append(robot_bids)

        assignments = FullMechanism().allocate(bids, mech_cfg)
        realized_costs = np.full(N, np.nan)
        for i in range(N):
            if assignments[i] >= 0:
                realized_costs[i] = tasks[assignments[i]].sample_cost(rng2)

        scores = FullMechanism().score(bids, assignments, realized_costs, mech_cfg)
        from terrain_bidding.mechanism import compute_penalty, compute_utility, RoundResult
        penalties = np.zeros(N)
        utilities = np.zeros(N)
        for i in range(N):
            if assignments[i] >= 0:
                penalties[i] = compute_penalty(scores[i], S_baseline, mech_cfg.kappa)
                utilities[i] = compute_utility(True, realized_costs[i], penalties[i], mech_cfg.R)

        result = RoundResult(assignments, realized_costs, scores, penalties, utilities)
        for i, robot in enumerate(fleet2):
            robot.observe(result, i)

        kappa_history.append(learner2.kappa_est)
        delta_history.append(learner2._last_delta)

    convergence_data = {
        "kappa_history": np.array(kappa_history),
        "delta_history": np.array(delta_history),
        "true_kappa": mech_cfg.kappa,
    }
    print(f"  Final κ_est = {kappa_history[-1]:.4f} (true κ = {mech_cfg.kappa})")
    print(f"  Final δ* = {delta_history[-1]:.4f}")
    print(f"  κ_est at round 100: {kappa_history[99]:.4f}")
    print(f"  κ_est at round 1000: {kappa_history[999]:.4f}")
    return convergence_data


# ─── Experiment 3: Welfare Analysis ──────────────────────────────────────────

def run_welfare_experiment(real_sampler, S_baseline):
    """Compare total fleet welfare across mechanisms."""
    print("\n=== WELFARE ANALYSIS ===")
    N, M = 4, 2
    results = {}

    for n_adv in [0, 1, 2]:
        for mech_name, mech in [("vanilla", VanillaMechanism()),
                                 ("fixed_thresh", FixedThresholdMechanism(sigma_fixed=600.0)),
                                 ("total_var", TotalVarianceMechanism()),
                                 ("full", FullMechanism())]:
            mech_cfg = MechanismConfig(N=N, kappa=2.0, gamma=1.0, S_baseline=S_baseline)
            fleet = [EnsembleRobot(robot_id=i) for i in range(N - n_adv)]
            for i in range(n_adv):
                fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](
                    robot_id=N - n_adv + i))

            def scarce(n, rng):
                return real_sampler(M, rng)

            sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg,
                               num_rounds=3000, seed=42)
            sim_results = simulate(sim_cfg, fleet, scarce)
            m = compute_metrics(sim_results, N - n_adv, None)
            key = f"welfare|n_adv={n_adv}|{mech_name}"
            results[key] = m
            print(f"  n_adv={n_adv} {mech_name:12s} | Welfare={m.total_welfare:12.0f} "
                  f"Eff={m.allocation_efficiency:.3f}")
    return results


# ─── Experiment 4: κ Sweep with Real Data ────────────────────────────────────

def run_kappa_sweep_real(real_sampler, S_baseline):
    """κ sweep with real ensemble predictions."""
    print("\n=== κ SWEEP (real data) ===")
    N, M = 4, 2
    results = {}

    for kappa in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        mech_cfg = MechanismConfig(N=N, kappa=kappa, gamma=1.0, S_baseline=S_baseline)
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["adaptive"](
            robot_id=3, kappa=kappa, R=mech_cfg.R))

        def scarce(n, rng):
            return real_sampler(M, rng)

        sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                           num_rounds=3000, seed=42)
        sim_results = simulate(sim_cfg, fleet, scarce)
        m = compute_metrics(sim_results, 3, None)
        key = f"kappa_real|kappa={kappa}"
        results[key] = m
        print(f"  κ={kappa:5.1f} | Gain={m.strategic_gain:+.3f} "
              f"Sep={m.detection_separation:+.3f} FPR={m.false_positive_rate:.3f}")

    # Estimate L with task scarcity
    mech_cfg = MechanismConfig(N=N, kappa=2.0, gamma=1.0, S_baseline=S_baseline)
    fleet_honest = [EnsembleRobot(robot_id=i) for i in range(N)]
    sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                       num_rounds=100, seed=42)

    def scarce(n, rng):
        return real_sampler(M, rng)

    L = estimate_lipschitz(sim_cfg, fleet_honest, scarce, delta=200.0, n_samples=1000)
    # Average aleatoric variance from data
    avg_var_ale = 233586.0  # from earlier diagnostic
    kappa_star = mech_cfg.R * L * avg_var_ale
    print(f"\n  Estimated L = {L:.6f}")
    print(f"  Theoretical κ* = R·L·σ²_ale = {kappa_star:.2f}")
    results["L"] = L
    results["kappa_star"] = kappa_star
    return results


# ─── Experiment 5: Mechanism Comparison (main result) ─────────────────────────

def run_main_comparison(real_sampler, S_baseline):
    """Core comparison: 4 mechanisms × adversary types."""
    print("\n=== MAIN MECHANISM COMPARISON ===")
    N, M = 4, 2
    results = {}

    adversaries = ["fixed_offset", "adaptive", "learning",
                   "terrain_selective", "ensemble_manipulating"]
    mechanisms = [("vanilla", VanillaMechanism()),
                  ("fixed_thresh", FixedThresholdMechanism(sigma_fixed=600.0)),
                  ("total_var", TotalVarianceMechanism()),
                  ("full", FullMechanism())]

    for adv_type in adversaries:
        for mech_name, mech in mechanisms:
            mech_cfg = MechanismConfig(N=N, kappa=2.0, gamma=1.0, S_baseline=S_baseline)
            fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
            fleet.append(REAL_ADVERSARY_TYPES[adv_type](
                robot_id=3, kappa=mech_cfg.kappa, R=mech_cfg.R))

            def scarce(n, rng):
                return real_sampler(M, rng)

            sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg,
                               num_rounds=3000, seed=42)
            sim_results = simulate(sim_cfg, fleet, scarce)
            m = compute_metrics(sim_results, 3, None)
            key = f"main|{adv_type}|{mech_name}"
            results[key] = m
        # Print summary for this adversary
        print(f"  {adv_type}:")
        for mech_name, _ in mechanisms:
            m = results[f"main|{adv_type}|{mech_name}"]
            print(f"    {mech_name:12s} | Eff={m.allocation_efficiency:.3f} "
                  f"Sep={m.detection_separation:+.3f} FPR={m.false_positive_rate:.3f} "
                  f"Gain={m.strategic_gain:+.3f}")
    return results


# ─── Run All ──────────────────────────────────────────────────────────────────

def run_all():
    """Run all paper experiments and save results."""
    Path("outputs").mkdir(exist_ok=True)
    real_sampler, S_baseline = setup()

    all_results = {}

    # Main comparison
    all_results["main"] = run_main_comparison(real_sampler, S_baseline)

    # κ sweep
    all_results["kappa"] = run_kappa_sweep_real(real_sampler, S_baseline)

    # Scaling
    all_results["scaling"] = run_scaling_experiment(real_sampler, S_baseline)

    # Convergence
    all_results["convergence"] = run_convergence_experiment(real_sampler, S_baseline)

    # Welfare
    all_results["welfare"] = run_welfare_experiment(real_sampler, S_baseline)

    # Save
    with open("outputs/paper_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nAll results saved to outputs/paper_results.pkl")
    return all_results


if __name__ == "__main__":
    run_all()
