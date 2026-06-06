"""Phase 6: Experiment runner, metrics, and analysis.

Runs the full experimental grid: mechanisms × adversaries × terrains × sweeps.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from itertools import product
from terrain_bidding.configs import MechanismConfig, ExperimentConfig
from terrain_bidding.mechanism import (
    VanillaMechanism, FixedThresholdMechanism, TotalVarianceMechanism,
    FullMechanism, SimConfig, simulate, estimate_lipschitz, RoundResult,
)
from terrain_bidding.adversaries import (
    SimTask, HonestRobot, create_fleet, ADVERSARY_TYPES,
)


# ─── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class ExperimentMetrics:
    """Metrics for one experimental condition."""
    condition: str
    allocation_efficiency: float   # oracle_cost / realized_cost (≤1, higher=better)
    strategic_gain: float          # % cost savings by adversaries vs honest
    mean_score_honest: float
    mean_score_adversary: float
    detection_separation: float    # gap between honest and adversary scores
    false_positive_rate: float     # honest robots in bottom 10% of scores
    mean_penalty_honest: float
    mean_penalty_adversary: float
    total_welfare: float = 0.0     # sum of all robot utilities


def compute_metrics(results: List[RoundResult], n_honest: int,
                    oracle_costs: np.ndarray) -> ExperimentMetrics:
    """Compute all metrics from simulation results."""
    N = results[0].assignments.shape[0]
    n_adv = N - n_honest

    # Aggregate per-robot
    all_costs = [[] for _ in range(N)]
    all_scores = [[] for _ in range(N)]
    all_penalties = [[] for _ in range(N)]
    all_utilities = [[] for _ in range(N)]

    for r in results:
        for i in range(N):
            if r.assignments[i] >= 0:
                all_costs[i].append(r.realized_costs[i])
                all_scores[i].append(r.scores[i])
                all_penalties[i].append(r.penalties[i])
            all_utilities[i].append(r.utilities[i])

    # Allocation efficiency: fraction of rounds where adversary does NOT displace honest robot
    # 1.0 = adversary never displaces honest; lower = adversary gaming allocation
    if oracle_costs is not None:
        total_realized = sum(np.nansum(r.realized_costs) for r in results)
        alloc_eff = oracle_costs.sum() / max(abs(total_realized), 1e-8)
    elif n_adv > 0:
        good_rounds = 0
        total_rounds = 0
        for r in results:
            assigned = set(i for i in range(N) if r.assignments[i] >= 0)
            unassigned = set(i for i in range(N) if r.assignments[i] < 0)
            if unassigned:  # only count rounds with scarcity
                total_rounds += 1
                # Bad if adversary assigned AND honest robot unassigned
                adv_assigned = any(i >= n_honest for i in assigned)
                honest_unassigned = any(i < n_honest for i in unassigned)
                if not (adv_assigned and honest_unassigned):
                    good_rounds += 1
        alloc_eff = good_rounds / max(total_rounds, 1)
    else:
        alloc_eff = 1.0

    # Scores by group
    honest_scores = [s for i in range(n_honest) for s in all_scores[i]]
    adv_scores = [s for i in range(n_honest, N) for s in all_scores[i]] if n_adv > 0 else [0]

    mean_h = np.mean(honest_scores) if honest_scores else 0
    mean_a = np.mean(adv_scores) if adv_scores else 0

    # Detection separation
    separation = mean_h - mean_a

    # False positive rate: honest robots in bottom 10% of all scores
    all_flat_scores = honest_scores + adv_scores
    if all_flat_scores:
        threshold = np.percentile(all_flat_scores, 10)
        fp = np.mean([s < threshold for s in honest_scores]) if honest_scores else 0
    else:
        fp = 0

    # Strategic gain: does the adversary have higher utility than honest robots?
    # Positive = adversary profits from manipulation; negative = manipulation backfires
    honest_utils = [u for i in range(n_honest) for u in all_utilities[i]]
    adv_utils = [u for i in range(n_honest, N) for u in all_utilities[i]] if n_adv > 0 else []
    if honest_utils and adv_utils and len(adv_utils) > 5:
        mean_honest_util = np.mean(honest_utils)
        mean_adv_util = np.mean(adv_utils)
        # Gain = (adv_utility - honest_utility) / |honest_utility|
        strategic_gain = (mean_adv_util - mean_honest_util) / max(abs(mean_honest_util), 0.01)
    else:
        strategic_gain = 0.0

    # Penalties
    honest_pen = [p for i in range(n_honest) for p in all_penalties[i]]
    adv_pen = [p for i in range(n_honest, N) for p in all_penalties[i]] if n_adv > 0 else [0]

    # Welfare: total fleet utility
    total_welfare = sum(np.sum(r.utilities) for r in results)

    return ExperimentMetrics(
        condition="",
        allocation_efficiency=alloc_eff,
        strategic_gain=strategic_gain,
        mean_score_honest=mean_h,
        mean_score_adversary=mean_a,
        detection_separation=separation,
        false_positive_rate=fp,
        mean_penalty_honest=np.mean(honest_pen) if honest_pen else 0,
        mean_penalty_adversary=np.mean(adv_pen) if adv_pen else 0,
        total_welfare=total_welfare,
    )


# ─── Task Samplers ────────────────────────────────────────────────────────────

def make_task_sampler(terrain_type: str = "mixed", base_mu: float = 5.0,
                      base_var: float = 1.0):
    """Create a task sampler for simulation.

    terrain_type: "mixed" | "adversarial" | "held_out"
    """
    terrain_params = {
        "flat":       (3.0, 0.5),
        "low_rough":  (4.0, 0.8),
        "high_rough": (6.0, 1.5),
        "mild_slope": (5.0, 1.0),
        "steep_slope":(8.0, 2.0),
        "discrete_obstacles": (10.0, 3.0),
    }

    if terrain_type == "mixed":
        types = ["flat", "low_rough", "high_rough", "mild_slope", "steep_slope"]
    elif terrain_type == "adversarial":
        types = ["steep_slope", "high_rough"]
    elif terrain_type == "held_out":
        types = ["discrete_obstacles"]
    else:
        types = [terrain_type]

    def sampler(n_tasks: int, rng: np.random.Generator) -> List[SimTask]:
        tasks = []
        for j in range(n_tasks):
            t = rng.choice(types)
            mu, var = terrain_params[t]
            # Add some per-task noise
            mu_task = mu + rng.normal(0, 0.5)
            tasks.append(SimTask(
                task_id=j, true_mu=mu_task, true_var_ale=var,
            ))
            tasks[-1].terrain_type = ["flat", "low_rough", "high_rough",
                                       "mild_slope", "steep_slope",
                                       "discrete_obstacles"].index(t)
        return tasks

    return sampler


# ─── Experiment Grid ──────────────────────────────────────────────────────────

def run_experiment_grid(exp_cfg: ExperimentConfig = ExperimentConfig()):
    """Run the full experimental grid."""
    import os
    results_all = {}

    # Use real data sampler if available, else synthetic
    if os.path.exists("data/test.hdf5") and os.path.exists("checkpoints/ensemble/meta.pt"):
        from terrain_bidding.experiments.real_sampler import (
            make_real_task_sampler, EnsembleRobot, REAL_ADVERSARY_TYPES
        )
        print("Using REAL data sampler (ensemble + test data)")
        real_sampler = make_real_task_sampler()
        use_real = True

        # Compute S_baseline from honest predictions on validation data
        print("Computing S_baseline from validation data...")
        from terrain_bidding.mechanism import gaussian_score
        baseline_scores = []
        rng = np.random.default_rng(0)
        for _ in range(2000):
            tasks = real_sampler(1, rng)
            t = tasks[0]
            mu_hat = t.ensemble_mus.mean()
            var_ale = np.exp(t.ensemble_log_vars).mean()
            # Score honest prediction against realized cost
            s = gaussian_score(t.true_mu, mu_hat, var_ale)
            baseline_scores.append(s)
        S_baseline = float(np.mean(baseline_scores))
        print(f"  S_baseline = {S_baseline:.4f}")
    else:
        print("Using SYNTHETIC data sampler (no ensemble/data found)")
        use_real = False
        S_baseline = -2.0

    mechanisms = {
        "vanilla": VanillaMechanism(),
        "fixed_threshold": FixedThresholdMechanism(sigma_fixed=1.5),
        "total_variance": TotalVarianceMechanism(),
        "full": FullMechanism(),
    }
    adversary_types = ["fixed_offset", "adaptive", "learning",
                       "terrain_selective", "ensemble_manipulating"]
    terrain_types = ["mixed"] if use_real else ["mixed", "adversarial", "held_out"]

    # Main grid: N=4 robots, M=2 tasks (scarcity creates competition)
    N = 4
    M = 2  # fewer tasks than robots
    for mech_name, mechanism in mechanisms.items():
        for n_strategic in exp_cfg.strategic_counts_n4:
            for adv_type in adversary_types:
                if n_strategic == 0 and adv_type != "fixed_offset":
                    continue  # only run honest fleet once
                for terrain in terrain_types:
                    condition = f"{mech_name}|{adv_type}|n_adv={n_strategic}|{terrain}"
                    print(f"Running: {condition}")

                    mech_cfg = MechanismConfig(N=N, kappa=2.0, gamma=1.0,
                                              S_baseline=S_baseline)
                    if use_real:
                        # Create fleet with real adversaries
                        fleet = []
                        for i in range(N - n_strategic):
                            fleet.append(EnsembleRobot(robot_id=i))
                        for i in range(n_strategic):
                            cls = REAL_ADVERSARY_TYPES[adv_type]
                            fleet.append(cls(robot_id=N - n_strategic + i,
                                           kappa=mech_cfg.kappa, R=mech_cfg.R))
                        # Task sampler returns M tasks (scarcity)
                        def scarce_sampler(n, rng, _s=real_sampler, _m=M):
                            return _s(_m, rng)
                        sampler = scarce_sampler
                    else:
                        fleet = create_fleet(
                            n_honest=N - n_strategic,
                            adversary_type=adv_type if n_strategic > 0 else "honest",
                            n_adversary=n_strategic,
                            kappa=mech_cfg.kappa, R=mech_cfg.R,
                        )
                        sampler = make_task_sampler(terrain)

                    sim_cfg = SimConfig(
                        mechanism=mechanism, mech_cfg=mech_cfg,
                        num_rounds=exp_cfg.episodes_per_condition,
                        seed=exp_cfg.seed,
                    )
                    sim_results = simulate(sim_cfg, fleet, sampler)
                    metrics = compute_metrics(sim_results, N - n_strategic, None)
                    metrics.condition = condition
                    results_all[condition] = metrics

    # κ sweep (adaptive adversary, full mechanism)
    kappa_results = run_kappa_sweep(exp_cfg)
    results_all.update(kappa_results)

    # γ ablation (honest fleet, full mechanism)
    gamma_results = run_gamma_ablation(exp_cfg)
    results_all.update(gamma_results)

    return results_all


def run_kappa_sweep(exp_cfg: ExperimentConfig) -> Dict[str, ExperimentMetrics]:
    """Sweep κ to find incentive compatibility threshold."""
    results = {}
    N = 4
    for kappa in exp_cfg.kappa_sweep:
        condition = f"kappa_sweep|kappa={kappa}"
        mech_cfg = MechanismConfig(N=N, kappa=kappa, gamma=1.0)
        fleet = create_fleet(n_honest=N - 1, adversary_type="adaptive",
                             n_adversary=1, kappa=kappa, R=mech_cfg.R)
        sampler = make_task_sampler("mixed")
        sim_cfg = SimConfig(
            mechanism=FullMechanism(), mech_cfg=mech_cfg,
            num_rounds=exp_cfg.episodes_per_condition, seed=exp_cfg.seed,
        )
        sim_results = simulate(sim_cfg, fleet, sampler)
        metrics = compute_metrics(sim_results, N - 1, None)
        metrics.condition = condition
        results[condition] = metrics

    # Estimate theoretical κ*
    mech_cfg = MechanismConfig(N=N, kappa=2.0, gamma=1.0)
    fleet = create_fleet(n_honest=N, adversary_type="honest", n_adversary=0)
    sampler = make_task_sampler("mixed")
    sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                        num_rounds=1000, seed=exp_cfg.seed)
    L = estimate_lipschitz(sim_cfg, fleet, sampler)
    # κ* = R · L · σ²_ale (using average aleatoric variance)
    avg_var_ale = 1.0  # placeholder — compute from data
    kappa_star = mech_cfg.R * L * avg_var_ale
    print(f"Estimated L={L:.4f}, theoretical κ*={kappa_star:.4f}")
    return results


def run_gamma_ablation(exp_cfg: ExperimentConfig) -> Dict[str, ExperimentMetrics]:
    """Ablate γ (epistemic weighting) under honest fleet."""
    results = {}
    N = 4
    for gamma in exp_cfg.gamma_sweep:
        condition = f"gamma_ablation|gamma={gamma}"
        mech_cfg = MechanismConfig(N=N, kappa=2.0, gamma=gamma)
        fleet = create_fleet(n_honest=N, adversary_type="honest", n_adversary=0)
        sampler = make_task_sampler("mixed")
        sim_cfg = SimConfig(
            mechanism=FullMechanism(), mech_cfg=mech_cfg,
            num_rounds=exp_cfg.episodes_per_condition, seed=exp_cfg.seed,
        )
        sim_results = simulate(sim_cfg, fleet, sampler)
        metrics = compute_metrics(sim_results, N, None)
        metrics.condition = condition
        results[condition] = metrics
    return results


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running experiment grid...")
    results = run_experiment_grid()
    print(f"\n=== Completed {len(results)} conditions ===\n")
    # Print summary table
    print(f"{'Condition':<60} {'Eff':>6} {'Gain':>6} {'Sep':>6} {'FPR':>6}")
    print("-" * 90)
    for cond, m in sorted(results.items()):
        print(f"{cond:<60} {m.allocation_efficiency:>6.3f} "
              f"{m.strategic_gain:>6.1%} {m.detection_separation:>6.3f} "
              f"{m.false_positive_rate:>6.3f}")
