"""All-bids scoring: score every robot's prediction on every executed task.

This eliminates selection bias — adversary can't hide by choosing easy tasks.
"""
import numpy as np
from terrain_bidding.mechanism import (
    FullMechanism, TotalVarianceMechanism, gaussian_score, allocate,
)


class AllBidsScoringMechanism:
    """Scores ALL robots on every executed task, not just the assigned one.

    Allocation: same as FullMechanism (epistemic weighting).
    Scoring: after task execution, score every robot's prediction against
    the realized cost. Adversary's shaded prediction scores poorly regardless
    of whether it was assigned.
    """

    def __init__(self, use_total_var=False, relative=False):
        self.use_total_var = use_total_var
        self.relative = relative  # if True, penalize relative to fleet mean per task

    def allocate(self, bids, cfg):
        """Same as FullMechanism — epistemic weighting."""
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
        """Score ALL robots on every executed task (not just assigned)."""
        N = len(bids)
        M = len(bids[0]) if bids else 0
        scores = np.zeros(N)
        n_scored = np.zeros(N)

        # For each task that was executed (has a realized cost)
        for j in range(M):
            # Find which robot was assigned to this task
            assigned_robot = None
            for i in range(N):
                if assignments[i] == j:
                    assigned_robot = i
                    break
            if assigned_robot is None or np.isnan(realized_costs[assigned_robot]):
                continue

            real_cost = realized_costs[assigned_robot]

            # Score EVERY robot's prediction for this task
            for i in range(N):
                if j < len(bids[i]):
                    bid = bids[i][j]
                    mu_hat = bid.mus.mean()
                    var_ale = np.exp(bid.log_vars).mean()
                    if self.use_total_var:
                        var_epi = ((bid.mus - mu_hat) ** 2).mean()
                        var = var_ale + var_epi
                    else:
                        var = var_ale
                    scores[i] += gaussian_score(real_cost, mu_hat, var)
                    n_scored[i] += 1

        # Average score per robot
        for i in range(N):
            if n_scored[i] > 0:
                scores[i] /= n_scored[i]

        # Relative scoring: subtract fleet mean per task so penalty is relative
        if self.relative and n_scored.sum() > 0:
            fleet_mean = scores[n_scored > 0].mean()
            scores = scores - fleet_mean  # positive = above average, negative = below

        return scores


class AdaptiveKappaMechanism(AllBidsScoringMechanism):
    """All-bids relative scoring with adaptive κ per task.

    κ_j = κ_base / max(σ²_epi,j, ε)
    High epistemic variance (OOD) → low κ needed (strong signal).
    Low epistemic variance (in-dist) → high κ needed (weak signal).
    Automatically calibrates deterrence to terrain novelty.
    """

    def __init__(self, kappa_base=5.0, eps=0.001, kappa_cap=100.0):
        super().__init__(use_total_var=True, relative=True)
        self.kappa_base = kappa_base
        self.eps = eps
        self.kappa_cap = kappa_cap

    def score(self, bids, assignments, realized_costs, cfg):
        N_robots = len(bids)
        M_tasks = len(bids[0]) if bids else 0
        scores = np.zeros(N_robots)
        n_scored = np.zeros(N_robots)

        for j in range(M_tasks):
            assigned_robot = None
            for i in range(N_robots):
                if assignments[i] == j:
                    assigned_robot = i
                    break
            if assigned_robot is None or np.isnan(realized_costs[assigned_robot]):
                continue
            real_cost = realized_costs[assigned_robot]

            # Compute per-task epistemic variance (average across robots)
            task_epi = 0
            for i in range(N_robots):
                if j < len(bids[i]):
                    bid = bids[i][j]
                    mu_hat = bid.mus.mean()
                    task_epi += ((bid.mus - mu_hat) ** 2).mean()
            task_epi /= N_robots

            # Adaptive κ: higher when epi is low (in-dist)
            adaptive_weight = min(self.kappa_base / max(task_epi, self.eps), self.kappa_cap)

            for i in range(N_robots):
                if j < len(bids[i]):
                    bid = bids[i][j]
                    mu_hat = bid.mus.mean()
                    var_ale = np.exp(bid.log_vars).mean()
                    var_epi = ((bid.mus - mu_hat) ** 2).mean()
                    s = gaussian_score(real_cost, mu_hat, var_ale + var_epi)
                    scores[i] += s * (adaptive_weight / self.kappa_base)
                    n_scored[i] += 1

        for i in range(N_robots):
            if n_scored[i] > 0:
                scores[i] /= n_scored[i]

        # Relative scoring
        if n_scored.sum() > 0:
            fleet_mean = scores[n_scored > 0].mean()
            scores = scores - fleet_mean
        return scores
