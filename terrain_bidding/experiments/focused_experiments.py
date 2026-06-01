"""Focused paper experiments — spotlighting mechanism strengths.

Run with: python3 -m terrain_bidding.experiments.focused_experiments
"""
import numpy as np
import pickle
from pathlib import Path
from terrain_bidding.mechanism import (
    FullMechanism, TotalVarianceMechanism, VanillaMechanism,
    FixedThresholdMechanism, SimConfig, simulate, gaussian_score,
    compute_penalty, Bid,
)
from terrain_bidding.experiments import compute_metrics
from terrain_bidding.experiments.real_sampler import (
    make_real_task_sampler, EnsembleRobot, REAL_ADVERSARY_TYPES,
)
from terrain_bidding.configs import MechanismConfig


def load_samplers():
    """Load both in-dist and held-out samplers."""
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


# ─── Experiment 1: κ sweep on HELD-OUT terrain ────────────────────────────────

def exp_kappa_sweep_held_out(held_out, S_baseline):
    """Headline result: κ threshold on held-out terrain."""
    print("\n" + "="*70)
    print("EXP 1: κ SWEEP ON HELD-OUT TERRAIN (Proposition 2 validation)")
    print("="*70)
    N, M = 4, 1  # single task auction
    results = {}
    for kappa in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
        mech_cfg = MechanismConfig(N=N, kappa=kappa, gamma=1.0, S_baseline=S_baseline)
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["adaptive"](robot_id=3, kappa=kappa, R=mech_cfg.R))

        def s(n, rng):
            return held_out(M, rng, n_robots=N)

        sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                           num_rounds=2000, seed=42)
        sim_results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(sim_results, 3, None)
        results[kappa] = m
        print(f"  κ={kappa:5.1f} | Gain={m.strategic_gain:+.4f} "
              f"Sep={m.detection_separation:+.3f} FPR={m.false_positive_rate:.3f}")
    return results


# ─── Experiment 2: M=1 single-task auction ────────────────────────────────────

def exp_single_task_auction(held_out, S_baseline):
    """Single task, 4 robots compete. Cleanest competitive scenario."""
    print("\n" + "="*70)
    print("EXP 2: SINGLE-TASK AUCTION (M=1, N=4)")
    print("="*70)
    N, M = 4, 1
    mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)

    for mech_name, mech in [("vanilla", VanillaMechanism()),
                             ("fixed_thresh", FixedThresholdMechanism(sigma_fixed=1.0)),
                             ("total_var", TotalVarianceMechanism()),
                             ("full", FullMechanism())]:
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))

        def s(n, rng):
            return held_out(M, rng, n_robots=N)

        sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg,
                           num_rounds=2000, seed=42)
        results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(results, 3, None)

        # Also compute adversary win rate
        adv_wins = sum(1 for r in results if r.assignments[3] >= 0)
        print(f"  {mech_name:12s} | Sep={m.detection_separation:+.3f} "
              f"FPR={m.false_positive_rate:.3f} Gain={m.strategic_gain:+.3f} "
              f"AdvWins={adv_wins}/{len(results)}")


# ─── Experiment 3: Honest robot utility comparison ────────────────────────────

def exp_honest_utility(held_out, S_baseline):
    """Show honest robots prefer the mechanism over vanilla."""
    print("\n" + "="*70)
    print("EXP 3: HONEST ROBOT UTILITY (do honest robots benefit?)")
    print("="*70)
    N, M = 4, 1
    mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)

    for mech_name, mech in [("vanilla", VanillaMechanism()),
                             ("full", FullMechanism())]:
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))

        def s(n, rng):
            return held_out(M, rng, n_robots=N)

        sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg,
                           num_rounds=3000, seed=42)
        results = simulate(sim_cfg, fleet, s)

        # Compute per-robot utilities
        honest_utils = [[] for _ in range(3)]
        adv_utils = []
        for r in results:
            for i in range(3):
                honest_utils[i].append(r.utilities[i])
            adv_utils.append(r.utilities[3])

        mean_honest = np.mean([np.mean(u) for u in honest_utils])
        mean_adv = np.mean(adv_utils)
        # How often do honest robots get assigned?
        honest_assigned = sum(1 for r in results for i in range(3) if r.assignments[i] >= 0)
        adv_assigned = sum(1 for r in results if r.assignments[3] >= 0)

        print(f"  {mech_name:8s} | Honest util={mean_honest:+.2f} "
              f"Adv util={mean_adv:+.2f} | "
              f"Honest assigned={honest_assigned}/{3*len(results)} "
              f"Adv assigned={adv_assigned}/{len(results)}")


# ─── Experiment 4: Score all robots on all tasks (fix selection bias) ─────────

def exp_unbiased_scoring(held_out, S_baseline):
    """Score all robots on the same tasks regardless of assignment."""
    print("\n" + "="*70)
    print("EXP 4: UNBIASED SCORING (all robots scored on same tasks)")
    print("="*70)
    N = 4
    mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)
    rng = np.random.default_rng(42)

    # Score each robot's predictions against realized costs for the SAME tasks
    honest_scores_full = []
    honest_scores_tv = []
    adv_scores_full = []
    adv_scores_tv = []

    fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
    fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))

    for _ in range(3000):
        tasks = held_out(1, rng, n_robots=N)
        t = tasks[0]

        # Each robot bids on this task
        for i, robot in enumerate(fleet):
            bid = robot.bid(t, i, 0)
            mu_hat = bid.mus.mean()
            var_ale = np.exp(bid.log_vars).mean()
            var_epi = ((bid.mus - mu_hat) ** 2).mean()

            # Score with full mechanism (aleatoric only)
            score_full = gaussian_score(t.true_mu, mu_hat, var_ale)
            # Score with total variance
            score_tv = gaussian_score(t.true_mu, mu_hat, var_ale + var_epi)

            if i < 3:
                honest_scores_full.append(score_full)
                honest_scores_tv.append(score_tv)
            else:
                adv_scores_full.append(score_full)
                adv_scores_tv.append(score_tv)

    sep_full = np.mean(honest_scores_full) - np.mean(adv_scores_full)
    sep_tv = np.mean(honest_scores_tv) - np.mean(adv_scores_tv)

    # FPR: fraction of honest scores below 10th percentile of all scores
    all_full = honest_scores_full + adv_scores_full
    thresh_full = np.percentile(all_full, 10)
    fpr_full = np.mean([s < thresh_full for s in honest_scores_full])

    all_tv = honest_scores_tv + adv_scores_tv
    thresh_tv = np.percentile(all_tv, 10)
    fpr_tv = np.mean([s < thresh_tv for s in honest_scores_tv])

    print(f"  Full mechanism:      Sep={sep_full:+.4f}  FPR={fpr_full:.4f}")
    print(f"  Total variance:      Sep={sep_tv:+.4f}  FPR={fpr_tv:.4f}")
    print(f"  Decomposition gap:   {(sep_full-sep_tv)/abs(sep_tv)*100:+.1f}% better detection")
    return {"sep_full": sep_full, "sep_tv": sep_tv, "fpr_full": fpr_full, "fpr_tv": fpr_tv}


# ─── Experiment 5: Detection vs Deterrence (separate claims) ─────────────────

def exp_detection_vs_deterrence(held_out, S_baseline):
    """Separate detection (scoring identifies adversary) from deterrence (κ makes it unprofitable)."""
    print("\n" + "="*70)
    print("EXP 5: DETECTION vs DETERRENCE")
    print("="*70)
    N, M = 4, 1

    # Detection: fixed κ, vary offset (can we detect different magnitudes?)
    print("  DETECTION (κ=5, varying adversary aggressiveness):")
    for offset in [0.1, 0.3, 0.5, 1.0, 1.5]:
        mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=offset))

        def s(n, rng):
            return held_out(M, rng, n_robots=N)

        sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                           num_rounds=1000, seed=42)
        results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(results, 3, None)
        print(f"    offset={offset:.1f} | Sep={m.detection_separation:+.3f} "
              f"FPR={m.false_positive_rate:.3f}")

    # Deterrence: fixed offset, vary κ (at what κ does adversary stop profiting?)
    print("  DETERRENCE (offset=0.5, varying κ):")
    for kappa in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        mech_cfg = MechanismConfig(N=N, kappa=kappa, gamma=1.0, S_baseline=S_baseline)
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))

        def s(n, rng):
            return held_out(M, rng, n_robots=N)

        sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                           num_rounds=1000, seed=42)
        results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(results, 3, None)
        print(f"    κ={kappa:5.1f} | Gain={m.strategic_gain:+.4f} "
              f"Penalty_adv={m.mean_penalty_adversary:.3f} "
              f"Penalty_hon={m.mean_penalty_honest:.3f}")


# ─── Experiment 6: Decomposition gap via penalty ratio ────────────────────────

def exp_penalty_ratio(held_out, S_baseline):
    """Compare adversary/honest penalty ratio between full and total_var."""
    print("\n" + "="*70)
    print("EXP 6: PENALTY RATIO (adversary_penalty / honest_penalty)")
    print("="*70)
    N, M = 4, 1
    mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)

    for mech_name, mech in [("total_var", TotalVarianceMechanism()),
                             ("full", FullMechanism())]:
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))

        def s(n, rng):
            return held_out(M, rng, n_robots=N)

        sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg,
                           num_rounds=3000, seed=42)
        results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(results, 3, None)
        ratio = m.mean_penalty_adversary / max(m.mean_penalty_honest, 0.001)
        print(f"  {mech_name:12s} | Pen_adv={m.mean_penalty_adversary:.3f} "
              f"Pen_hon={m.mean_penalty_honest:.3f} Ratio={ratio:.2f}x")


# ─── Experiment 7: L estimation with M=1 ─────────────────────────────────────

def exp_lipschitz(held_out, S_baseline):
    """Estimate L with single-task auction."""
    print("\n" + "="*70)
    print("EXP 7: LIPSCHITZ CONSTANT (allocation sensitivity)")
    print("="*70)
    N, M = 4, 1
    mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)
    fleet = [EnsembleRobot(robot_id=i) for i in range(N)]
    mech = FullMechanism()
    rng = np.random.default_rng(999)

    allocated_base, allocated_pert = 0, 0
    n_samples = 2000
    delta = 0.3
    for _ in range(n_samples):
        tasks = held_out(M, rng, n_robots=N)
        bids = [[robot.bid(tasks[j], i, j) for j in range(M)] for i, robot in enumerate(fleet)]
        a_base = mech.allocate(bids, mech_cfg)
        allocated_base += int(a_base[0] >= 0)

        # Perturb robot 0
        perturbed = [list(rb) for rb in bids]
        for b in perturbed[0]:
            perturbed[0] = [Bid(b.robot_id, b.task_id, b.mus - delta, b.log_vars)]
        a_pert = mech.allocate(perturbed, mech_cfg)
        allocated_pert += int(a_pert[0] >= 0)

    p_base = allocated_base / n_samples
    p_pert = allocated_pert / n_samples
    L = abs(p_pert - p_base) / delta

    # Compute average var_ale
    rng2 = np.random.default_rng(0)
    var_ales = []
    for _ in range(500):
        ts = held_out(1, rng2, n_robots=N)
        _, lvs = ts[0].robot_observations[0]
        var_ales.append(np.exp(lvs).mean())
    avg_var_ale = np.mean(var_ales)

    kappa_star = mech_cfg.R * L * avg_var_ale
    print(f"  P(assigned, base) = {p_base:.3f}")
    print(f"  P(assigned, pert-{delta}) = {p_pert:.3f}")
    print(f"  L = {L:.4f}")
    print(f"  avg σ²_ale = {avg_var_ale:.4f}")
    print(f"  Theoretical κ* = R·L·σ²_ale = {kappa_star:.4f}")


# ─── Experiment 8: Swapped Ablation ──────────────────────────────────────────

class SwappedMechanism:
    """WRONG decomposition: epistemic for scoring, aleatoric for allocation.
    
    This should perform worse than the correct decomposition, proving
    the assignment of uncertainty types to mechanism functions is principled.
    """
    def allocate(self, bids, cfg):
        N = len(bids)
        M = len(bids[0]) if bids else 0
        matrix = np.zeros((N, M))
        for i, robot_bids in enumerate(bids):
            for bid in robot_bids:
                mu_hat = bid.mus.mean()
                var_ale = np.exp(bid.log_vars).mean()
                # SWAPPED: use aleatoric in allocation (wrong)
                matrix[i, bid.task_id] = mu_hat + cfg.gamma * var_ale
        from terrain_bidding.mechanism import allocate
        return allocate(matrix)

    def score(self, bids, assignments, realized_costs, cfg):
        N = len(bids)
        scores = np.zeros(N)
        for i in range(N):
            if assignments[i] < 0:
                continue
            bid = bids[i][assignments[i]]
            mu_hat = bid.mus.mean()
            var_epi = ((bid.mus - mu_hat) ** 2).mean()
            # SWAPPED: use epistemic in scoring (wrong)
            scores[i] = gaussian_score(realized_costs[i], mu_hat, max(var_epi, 1e-6))
        return scores


def exp_swapped_ablation(held_out, S_baseline):
    """Prove decomposition assignment matters: correct vs swapped vs total."""
    print("\n" + "="*70)
    print("EXP 8: SWAPPED ABLATION (proves decomposition choice is principled)")
    print("="*70)
    N, M = 4, 1
    mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)

    mechanisms = [
        ("full (correct)", FullMechanism()),
        ("total_var", TotalVarianceMechanism()),
        ("swapped (wrong)", SwappedMechanism()),
        ("vanilla (none)", VanillaMechanism()),
    ]

    for mech_name, mech in mechanisms:
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))

        def s(n, rng):
            return held_out(M, rng, n_robots=N)

        sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg,
                           num_rounds=2000, seed=42)
        results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(results, 3, None)
        print(f"  {mech_name:20s} | Sep={m.detection_separation:+.4f} "
              f"FPR={m.false_positive_rate:.4f} Gain={m.strategic_gain:+.4f}")


# ─── Experiment 9: Learning Adversary Multi-Initialization ────────────────────

def exp_learning_multi_init(held_out, S_baseline):
    """Show learning adversary converges to honesty across different initializations."""
    print("\n" + "="*70)
    print("EXP 9: LEARNING ADVERSARY CONVERGENCE (multiple initializations)")
    print("="*70)
    N, M = 4, 1
    mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)
    n_rounds = 3000

    print(f"  {'Init κ':>8} | {'Final κ_est':>12} | {'Final δ*':>10} | {'Converged?':>10}")
    print("  " + "-"*55)

    for kappa_init in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        learner = REAL_ADVERSARY_TYPES["learning"](
            robot_id=3, kappa_init=kappa_init, R=mech_cfg.R)
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(learner)

        def s(n, rng):
            return held_out(M, rng, n_robots=N)

        sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                           num_rounds=n_rounds, seed=42)
        simulate(sim_cfg, fleet, s)

        converged = "YES" if learner._last_delta < 0.05 else "NO"
        print(f"  {kappa_init:8.1f} | {learner.kappa_est:12.4f} | "
              f"{learner._last_delta:10.4f} | {converged:>10}")


# ─── Experiment 10: OOD Gap Expansion ─────────────────────────────────────────

def exp_ood_gap(in_dist, held_out, S_baseline):
    """Show decomposition gap widens under distribution shift."""
    print("\n" + "="*70)
    print("EXP 10: DECOMPOSITION GAP vs DISTRIBUTION SHIFT")
    print("="*70)
    N, M = 4, 1
    mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)

    for terrain_name, sampler in [("in-distribution", in_dist), ("held-out (OOD)", held_out)]:
        seps = {}
        for mech_name, mech in [("total_var", TotalVarianceMechanism()),
                                 ("full", FullMechanism())]:
            fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
            fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))

            def s(n, rng, _sam=sampler):
                return _sam(M, rng, n_robots=N)

            sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg,
                               num_rounds=2000, seed=42)
            results = simulate(sim_cfg, fleet, s)
            m = compute_metrics(results, 3, None)
            seps[mech_name] = m.detection_separation

        gap = seps["full"] - seps["total_var"]
        print(f"  {terrain_name:20s} | full={seps['full']:+.4f} "
              f"tv={seps['total_var']:+.4f} | gap={gap:+.4f}")


# ─── Run All ──────────────────────────────────────────────────────────────────

def run_all():
    Path("outputs").mkdir(exist_ok=True)
    in_dist, held_out = load_samplers()
    S_baseline = compute_s_baseline(held_out)
    print(f"S_baseline (held-out) = {S_baseline:.4f}")

    results = {}
    results["kappa_sweep"] = exp_kappa_sweep_held_out(held_out, S_baseline)
    exp_single_task_auction(held_out, S_baseline)
    exp_honest_utility(held_out, S_baseline)
    results["unbiased"] = exp_unbiased_scoring(held_out, S_baseline)
    exp_detection_vs_deterrence(held_out, S_baseline)
    exp_penalty_ratio(held_out, S_baseline)
    exp_lipschitz(held_out, S_baseline)
    exp_swapped_ablation(held_out, S_baseline)
    exp_learning_multi_init(held_out, S_baseline)
    exp_ood_gap(in_dist, held_out, S_baseline)

    with open("outputs/focused_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nResults saved to outputs/focused_results.pkl")


if __name__ == "__main__":
    run_all()
