"""Phase 4: Mechanism simulation — allocation, scoring, and simulation loop.

Components:
- Allocator: Hungarian algorithm over effective costs (μ̂ + γ·σ²_epi)
- Scorer: Gaussian proper scoring rule on aleatoric variance
- Simulator: rounds of bidding → allocation → execution → scoring
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import List, Optional
from terrain_bidding.configs import MechanismConfig


# ─── Allocation ───────────────────────────────────────────────────────────────

def compute_effective_costs(mus: np.ndarray, var_epi: np.ndarray,
                            gamma: float) -> np.ndarray:
    """Effective cost for allocation: μ̂ + γ·σ²_epi.

    Args:
        mus: (N, M) predicted means, N robots × M tasks
        var_epi: (N, M) epistemic variances
        gamma: epistemic weighting coefficient
    Returns:
        (N, M) effective costs
    """
    return mus + gamma * var_epi


def allocate(effective_costs: np.ndarray) -> np.ndarray:
    """Solve assignment minimizing total effective cost via Hungarian algorithm.

    Args:
        effective_costs: (N, M) cost matrix, N robots × M tasks
    Returns:
        assignments: (min(N,M),) array where assignments[i] = task index for robot i
                     Robots without assignment get -1.
    """
    row_ind, col_ind = linear_sum_assignment(effective_costs)
    N = effective_costs.shape[0]
    assignments = np.full(N, -1, dtype=int)
    assignments[row_ind] = col_ind
    return assignments


# ─── Scoring ──────────────────────────────────────────────────────────────────

def gaussian_score(realized_cost: float, mu: float, var_ale: float) -> float:
    """Gaussian proper scoring rule (negative log-likelihood).

    S = -[(c - μ)² / (2σ²_ale) + 0.5·log(σ²_ale)]
    """
    return -((realized_cost - mu) ** 2 / (2 * var_ale) + 0.5 * np.log(var_ale))


def compute_penalty(score: float, s_baseline: float, kappa: float) -> float:
    """Penalty: κ·max(0, S_baseline - S)."""
    return kappa * max(0.0, s_baseline - score)


def compute_utility(allocated: bool, realized_cost: float, penalty: float,
                    R: float) -> float:
    """U = R·𝟙[allocated] - c - penalty."""
    reward = R if allocated else 0.0
    cost = realized_cost if allocated else 0.0
    return reward - cost - penalty


# ─── Mechanism Variants ───────────────────────────────────────────────────────

@dataclass
class Bid:
    """A robot's bid for a task."""
    robot_id: int
    task_id: int
    mus: np.ndarray        # (K,) per-network means
    log_vars: np.ndarray   # (K,) per-network log variances


@dataclass
class RoundResult:
    """Result of one allocation round."""
    assignments: np.ndarray       # (N,) task indices, -1 if unassigned
    realized_costs: np.ndarray    # (N,) actual costs (NaN if unassigned)
    scores: np.ndarray            # (N,) scoring rule evaluations
    penalties: np.ndarray         # (N,) penalty amounts
    utilities: np.ndarray         # (N,) final utilities


class MechanismVariant:
    """Base class for mechanism variants."""

    # Subclasses that score ALL robots (not just the assigned one) should set
    # this to True.  simulate() uses it to decide whether to apply the penalty
    # to unassigned robots as well — which is required for the overbidding
    # loophole to be closed.
    scores_all_robots: bool = False

    def allocate(self, bids: List[List[Bid]], cfg: MechanismConfig) -> np.ndarray:
        raise NotImplementedError

    def score(self, bids: List[List[Bid]], assignments: np.ndarray,
              realized_costs: np.ndarray, cfg: MechanismConfig) -> np.ndarray:
        raise NotImplementedError


class VanillaMechanism(MechanismVariant):
    """No uncertainty modeling. Allocate on reported means, no scoring."""

    def allocate(self, bids, cfg):
        cost_matrix = self._build_mean_matrix(bids)
        return allocate(cost_matrix)

    def score(self, bids, assignments, realized_costs, cfg):
        return np.zeros(len(bids))  # no scoring

    def _build_mean_matrix(self, bids):
        N = len(bids)
        M = len(bids[0]) if bids else 0
        matrix = np.zeros((N, M))
        for i, robot_bids in enumerate(bids):
            for bid in robot_bids:
                matrix[i, bid.task_id] = bid.mus.mean()
        return matrix


class FixedThresholdMechanism(MechanismVariant):
    """Binary z-test verification with fixed σ."""

    def __init__(self, sigma_fixed: float, z_threshold: float = 2.5,
                 penalty_flag: float = 5.0):
        self.sigma_fixed = sigma_fixed
        self.z_threshold = z_threshold
        self.penalty_flag = penalty_flag

    def allocate(self, bids, cfg):
        cost_matrix = self._build_mean_matrix(bids)
        return allocate(cost_matrix)

    def score(self, bids, assignments, realized_costs, cfg):
        N = len(bids)
        penalties = np.zeros(N)
        for i in range(N):
            if assignments[i] < 0:
                continue
            mu = bids[i][assignments[i]].mus.mean()
            z = abs(realized_costs[i] - mu) / self.sigma_fixed
            if z > self.z_threshold:
                penalties[i] = self.penalty_flag
        return -penalties  # negative = penalty

    def _build_mean_matrix(self, bids):
        N = len(bids)
        M = len(bids[0]) if bids else 0
        matrix = np.zeros((N, M))
        for i, robot_bids in enumerate(bids):
            for bid in robot_bids:
                matrix[i, bid.task_id] = bid.mus.mean()
        return matrix


class TotalVarianceMechanism(MechanismVariant):
    """Proper scoring using total variance (ale + epi) in denominator."""

    def allocate(self, bids, cfg):
        N = len(bids)
        M = len(bids[0]) if bids else 0
        matrix = np.zeros((N, M))
        for i, robot_bids in enumerate(bids):
            for bid in robot_bids:
                mu_hat = bid.mus.mean()
                var_ale = np.exp(bid.log_vars).mean()
                var_epi = ((bid.mus - mu_hat) ** 2).mean()
                matrix[i, bid.task_id] = mu_hat + cfg.gamma * var_epi
        return allocate(matrix)

    def score(self, bids, assignments, realized_costs, cfg):
        N = len(bids)
        scores = np.zeros(N)
        for i in range(N):
            if assignments[i] < 0:
                continue
            bid = bids[i][assignments[i]]
            mu_hat = bid.mus.mean()
            var_ale = np.exp(bid.log_vars).mean()
            var_epi = ((bid.mus - mu_hat) ** 2).mean()
            var_total = var_ale + var_epi
            scores[i] = gaussian_score(realized_costs[i], mu_hat, var_total)
        return scores


class FullMechanism(MechanismVariant):
    """Decomposed: epistemic in allocation, aleatoric-only in scoring."""

    def allocate(self, bids, cfg):
        N = len(bids)
        M = len(bids[0]) if bids else 0
        matrix = np.zeros((N, M))
        for i, robot_bids in enumerate(bids):
            for bid in robot_bids:
                mu_hat = bid.mus.mean()
                var_epi = ((bid.mus - mu_hat) ** 2).mean()
                matrix[i, bid.task_id] = mu_hat + cfg.gamma * var_epi
        return allocate(matrix)

    def score(self, bids, assignments, realized_costs, cfg):
        N = len(bids)
        scores = np.zeros(N)
        for i in range(N):
            if assignments[i] < 0:
                continue
            bid = bids[i][assignments[i]]
            mu_hat = bid.mus.mean()
            var_ale = np.exp(bid.log_vars).mean()  # aleatoric only
            scores[i] = gaussian_score(realized_costs[i], mu_hat, var_ale)
        return scores


# ─── Simulation Loop ──────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """Configuration for a simulation run."""
    mechanism: MechanismVariant = field(default_factory=FullMechanism)
    mech_cfg: MechanismConfig = field(default_factory=MechanismConfig)
    num_rounds: int = 5000
    seed: int = 42


def simulate(sim_cfg: SimConfig, robots: list, task_sampler) -> List[RoundResult]:
    """Run mechanism simulation.

    Args:
        sim_cfg: simulation configuration
        robots: list of Robot objects (honest or adversarial)
        task_sampler: callable returning (terrain_features, true_costs) per round
    Returns:
        List of RoundResult per round
    """
    rng = np.random.default_rng(sim_cfg.seed)
    results = []
    cfg = sim_cfg.mech_cfg
    N = len(robots)

    for round_idx in range(sim_cfg.num_rounds):
        # Sample tasks
        tasks = task_sampler(N, rng)
        M = len(tasks)

        # Each robot bids on each task
        bids = []
        for i, robot in enumerate(robots):
            robot_bids = []
            for j in range(M):
                bid = robot.bid(tasks[j], i, j)
                robot_bids.append(bid)
            bids.append(robot_bids)

        # Allocation
        assignments = sim_cfg.mechanism.allocate(bids, cfg)

        # Execution: sample realized costs from true distribution
        realized_costs = np.full(N, np.nan)
        for i in range(N):
            if assignments[i] >= 0:
                realized_costs[i] = tasks[assignments[i]].sample_cost(rng)

        # Scoring
        scores = sim_cfg.mechanism.score(bids, assignments, realized_costs, cfg)

        # Penalties and utilities.
        # For all-bids mechanisms every robot is scored on every executed task,
        # so the penalty applies regardless of assignment.  This closes the
        # overbidding loophole.
        # If the mechanism defines compute_penalties() it owns the penalty
        # computation entirely (used by AdaptiveKappaMechanism to apply per-task
        # κ_j directly without going through cfg.kappa).  Otherwise fall back to
        # scalar compute_penalty() gated on assignment.
        penalties = np.zeros(N)
        utilities = np.zeros(N)
        s_baseline = cfg.S_baseline or 0.0
        all_bids_mode = getattr(sim_cfg.mechanism, 'scores_all_robots', False)
        has_custom_penalty = hasattr(sim_cfg.mechanism, 'compute_penalties')

        if all_bids_mode and has_custom_penalty:
            penalties = sim_cfg.mechanism.compute_penalties(scores, cfg)
            for i in range(N):
                if assignments[i] >= 0:
                    utilities[i] = compute_utility(True, realized_costs[i], penalties[i], cfg.R)
                else:
                    utilities[i] = -penalties[i]
        elif all_bids_mode:
            for i in range(N):
                penalties[i] = compute_penalty(scores[i], s_baseline, cfg.kappa)
                if assignments[i] >= 0:
                    utilities[i] = compute_utility(True, realized_costs[i], penalties[i], cfg.R)
                else:
                    utilities[i] = -penalties[i]
        else:
            # Legacy assigned-only: penalty only when assigned
            for i in range(N):
                if assignments[i] >= 0:
                    penalties[i] = compute_penalty(scores[i], s_baseline, cfg.kappa)
                    utilities[i] = compute_utility(True, realized_costs[i], penalties[i], cfg.R)

        result = RoundResult(assignments, realized_costs, scores, penalties, utilities)
        results.append(result)

        # Let robots observe outcomes (for learning adversary)
        for i, robot in enumerate(robots):
            robot.observe(result, i)

    return results


# ─── Lipschitz Estimation ─────────────────────────────────────────────────────

def estimate_lipschitz(sim_cfg: SimConfig, robots: list, task_sampler,
                       delta: float = 0.5, n_samples: int = 1000) -> float:
    """Estimate L: sensitivity of allocation probability to bid shading.

    Uses task scarcity (N/2 tasks for N robots) to create competition.
    Perturbs robot 0's bids by delta, measures change in allocation probability.
    """
    rng = np.random.default_rng(sim_cfg.seed + 999)
    cfg = sim_cfg.mech_cfg
    N = len(robots)
    M = max(N // 2, 1)  # fewer tasks than robots = competition
    allocated_base, allocated_perturbed = 0, 0

    for _ in range(n_samples):
        tasks = task_sampler(M, rng)
        bids = []
        for i, robot in enumerate(robots):
            robot_bids = []
            for j in range(M):
                robot_bids.append(robot.bid(tasks[j], i, j))
            bids.append(robot_bids)

        # Base allocation
        assignments_base = sim_cfg.mechanism.allocate(bids, cfg)
        allocated_base += int(assignments_base[0] >= 0)

        # Perturbed: shade robot 0's bids by delta
        perturbed_bids = [list(rb) for rb in bids]
        for bid in perturbed_bids[0]:
            bid.mus = bid.mus - delta
        assignments_pert = sim_cfg.mechanism.allocate(perturbed_bids, cfg)
        allocated_perturbed += int(assignments_pert[0] >= 0)

    p_base = allocated_base / n_samples
    p_pert = allocated_perturbed / n_samples
    L = abs(p_pert - p_base) / delta
    return L
