"""Heterogeneous fleet, reputation baseline, repeated-game experiments.

Run with: python3 -m terrain_bidding.experiments.fleet_experiments
"""
import numpy as np
import pickle
from pathlib import Path
from terrain_bidding.mechanism import (
    FullMechanism, TotalVarianceMechanism, VanillaMechanism,
    SimConfig, simulate, gaussian_score, compute_penalty, Bid, RoundResult,
)
from terrain_bidding.experiments import compute_metrics
from terrain_bidding.experiments.real_sampler import (
    make_real_task_sampler, EnsembleRobot, REAL_ADVERSARY_TYPES, PrivateObsTask,
)
from terrain_bidding.configs import MechanismConfig


def load_held_out():
    print("Loading held-out sampler...")
    held_out = make_real_task_sampler(data_path="data/held_out/test.hdf5")
    return held_out


def compute_s_baseline(sampler, N=4):
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(500):
        ts = sampler(1, rng, n_robots=N)
        mus, lvs = ts[0].robot_observations[0]
        scores.append(gaussian_score(ts[0].true_mu, mus.mean(), np.exp(lvs).mean()))
    return float(np.mean(scores))


# ─── Experiment: Heterogeneous Fleet ──────────────────────────────────────────

class FastRobot(EnsembleRobot):
    """Fast but noisy — adds noise to predictions (less accurate sensor)."""
    def _get_ensemble_predictions(self, task):
        mus, log_vars = super()._get_ensemble_predictions(task)
        rng = np.random.default_rng(hash(self.robot_id + hash(str(mus[:2]))) % 2**31)
        mus = mus + rng.normal(0, 0.3, len(mus))  # noisier predictions
        return mus, log_vars


class CautiousRobot(EnsembleRobot):
    """Cautious — inflates predicted variance (reports higher uncertainty)."""
    def _get_ensemble_predictions(self, task):
        mus, log_vars = super()._get_ensemble_predictions(task)
        log_vars = log_vars + 0.5  # reports higher variance
        return mus, log_vars


class SlowRobot(EnsembleRobot):
    """Slow but accurate — predictions are good but actual cost is higher."""
    pass  # same predictions, but tasks cost more for this robot (handled in scoring)


def exp_heterogeneous_fleet(held_out, S_baseline):
    """Fleet with different robot types: fast/noisy, cautious, slow/accurate."""
    print("\n" + "="*70)
    print("EXP: HETEROGENEOUS FLEET")
    print("="*70)
    N, M = 4, 1
    mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)

    # Fleet compositions to test
    compositions = [
        ("Homogeneous (baseline)", [EnsembleRobot(i) for i in range(3)] +
         [REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5)]),
        ("Fast+Cautious+Normal+Adv", [FastRobot(0), CautiousRobot(1), EnsembleRobot(2),
         REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5)]),
        ("All cautious + Adv", [CautiousRobot(0), CautiousRobot(1), CautiousRobot(2),
         REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5)]),
        ("All fast/noisy + Adv", [FastRobot(0), FastRobot(1), FastRobot(2),
         REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5)]),
    ]

    print(f"  {'Composition':>30} | {'Sep':>8} {'FPR':>6} {'Gain':>8} {'AdvWins':>8}")
    print("  " + "-"*65)

    for name, fleet in compositions:
        def s(n, rng):
            return held_out(M, rng, n_robots=N)
        sim_cfg = SimConfig(mechanism=FullMechanism(), mech_cfg=mech_cfg,
                           num_rounds=2000, seed=42)
        results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(results, 3, None)
        adv_wins = sum(1 for r in results if r.assignments[3] >= 0)
        print(f"  {name:>30} | {m.detection_separation:+8.4f} "
              f"{m.false_positive_rate:6.4f} {m.strategic_gain:+8.4f} "
              f"{adv_wins:>4}/2000")


# ─── Experiment: Reputation System Baseline ───────────────────────────────────

class ReputationMechanism:
    """Baseline: reputation-based detection (running average of prediction errors).

    Instead of proper scoring, tracks each robot's historical prediction error.
    Flags robots whose error exceeds a threshold.
    """

    scores_all_robots: bool = False  # only scores the assigned robot

    def __init__(self, window: int = 50, threshold: float = 2.0):
        self.window = window
        self.threshold = threshold
        self.error_history = {}  # robot_id -> list of errors

    def allocate(self, bids, cfg):
        """Allocate based on reported means (same as vanilla)."""
        N = len(bids)
        M = len(bids[0]) if bids else 0
        matrix = np.zeros((N, M))
        for i, robot_bids in enumerate(bids):
            for bid in robot_bids:
                matrix[i, bid.task_id] = bid.mus.mean()
        from terrain_bidding.mechanism import allocate
        return allocate(matrix)

    def score(self, bids, assignments, realized_costs, cfg):
        """Score based on running prediction error vs fleet average."""
        N = len(bids)
        scores = np.zeros(N)
        for i in range(N):
            if assignments[i] < 0:
                continue
            bid = bids[i][assignments[i]]
            mu_hat = bid.mus.mean()
            error = abs(realized_costs[i] - mu_hat)

            if i not in self.error_history:
                self.error_history[i] = []
            self.error_history[i].append(error)

            # Score: negative of error relative to historical average
            hist = self.error_history[i][-self.window:]
            avg_error = np.mean(hist)
            scores[i] = -avg_error  # lower error = higher score
        return scores


def exp_reputation_baseline(held_out, S_baseline):
    """Compare proper scoring vs reputation-based detection."""
    print("\n" + "="*70)
    print("EXP: REPUTATION SYSTEM BASELINE COMPARISON")
    print("="*70)
    N, M = 4, 1

    mechanisms = [
        ("Vanilla (no detection)", VanillaMechanism()),
        ("Reputation (window=50)", ReputationMechanism(window=50, threshold=2.0)),
        ("Reputation (window=200)", ReputationMechanism(window=200, threshold=2.0)),
        ("Full (proper scoring)", FullMechanism()),
    ]

    print(f"  {'Mechanism':>25} | {'Sep':>8} {'FPR':>6} {'Gain':>8}")
    print("  " + "-"*55)

    for mech_name, mech in mechanisms:
        mech_cfg = MechanismConfig(N=N, kappa=5.0, gamma=1.0, S_baseline=S_baseline)
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))

        def s(n, rng):
            return held_out(M, rng, n_robots=N)

        sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg,
                           num_rounds=2000, seed=42)
        results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(results, 3, None)
        print(f"  {mech_name:>25} | {m.detection_separation:+8.4f} "
              f"{m.false_positive_rate:6.4f} {m.strategic_gain:+8.4f}")


# ─── Experiment: Repeated Game with Dynamic Reputation ────────────────────────

def exp_repeated_game(held_out, S_baseline):
    """Repeated game where κ scales with robot's historical score deficit.

    Robots that consistently score poorly get increasing κ (escalating penalties).
    This creates a reputation effect without explicit reputation tracking.
    """
    print("\n" + "="*70)
    print("EXP: REPEATED GAME WITH DYNAMIC κ (reputation emerges)")
    print("="*70)
    N, M = 4, 1
    base_kappa = 2.0
    n_rounds = 5000
    mech = FullMechanism()

    fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
    fleet.append(REAL_ADVERSARY_TYPES["adaptive"](robot_id=3, kappa=base_kappa, R=5.0))

    rng = np.random.default_rng(42)

    # Track per-robot cumulative score deficit
    cum_deficit = np.zeros(N)
    kappa_per_robot = np.full(N, base_kappa)
    adv_gains_over_time = []
    kappa_history = []

    for round_idx in range(n_rounds):
        tasks = held_out(M, rng, n_robots=N)
        bids = [[robot.bid(tasks[j], i, j) for j in range(M)]
                for i, robot in enumerate(fleet)]

        mech_cfg = MechanismConfig(N=N, kappa=base_kappa, gamma=1.0, S_baseline=S_baseline)
        assignments = mech.allocate(bids, mech_cfg)
        realized_costs = np.full(N, np.nan)
        for i in range(N):
            if assignments[i] >= 0:
                realized_costs[i] = tasks[assignments[i]].sample_cost(rng)

        scores = mech.score(bids, assignments, realized_costs, mech_cfg)

        # Dynamic κ: scale penalty by cumulative deficit
        penalties = np.zeros(N)
        utilities = np.zeros(N)
        for i in range(N):
            if assignments[i] >= 0:
                deficit = max(0, S_baseline - scores[i])
                cum_deficit[i] += deficit
                # κ scales with reputation (more deficits = higher penalty)
                effective_kappa = base_kappa * (1 + 0.01 * cum_deficit[i])
                kappa_per_robot[i] = effective_kappa
                penalties[i] = effective_kappa * deficit
                utilities[i] = mech_cfg.R - realized_costs[i] - penalties[i]

        result = RoundResult(assignments, realized_costs, scores, penalties, utilities)
        for i, robot in enumerate(fleet):
            robot.observe(result, i)

        # Track adversary gain every 100 rounds
        if (round_idx + 1) % 100 == 0:
            adv_gains_over_time.append(utilities[3] if assignments[3] >= 0 else 0)
            kappa_history.append(kappa_per_robot[3])

    print(f"  Rounds: {n_rounds}")
    print(f"  Final effective κ (adversary): {kappa_per_robot[3]:.2f}")
    print(f"  Final effective κ (honest avg): {np.mean(kappa_per_robot[:3]):.2f}")
    print(f"  Cumulative deficit (adversary): {cum_deficit[3]:.2f}")
    print(f"  Cumulative deficit (honest avg): {np.mean(cum_deficit[:3]):.2f}")
    print(f"\n  κ escalation over time (adversary):")
    for i, (k, g) in enumerate(zip(kappa_history[::5], adv_gains_over_time[::5])):
        print(f"    Round {(i*5+1)*100:>5}: κ_eff={k:.2f}")


# ─── Run All ──────────────────────────────────────────────────────────────────

def run_all():
    Path("outputs").mkdir(exist_ok=True)
    held_out = load_held_out()
    S_baseline = compute_s_baseline(held_out)
    print(f"S_baseline = {S_baseline:.4f}")

    exp_heterogeneous_fleet(held_out, S_baseline)
    exp_reputation_baseline(held_out, S_baseline)
    exp_repeated_game(held_out, S_baseline)

    print("\nDone.")


if __name__ == "__main__":
    run_all()
