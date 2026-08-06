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

    scores_all_robots = True tells simulate() to apply penalties to every
    robot unconditionally, closing the overbidding loophole.
    """

    scores_all_robots: bool = True

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

    Architecture note: score() returns raw relative Gaussian scores for
    detection metrics.  compute_penalties() applies the per-task adaptive
    weights directly, bypassing cfg.kappa so there is no double-multiply.
    simulate() calls compute_penalties() when it exists on the mechanism.
    """

    scores_all_robots: bool = True

    def __init__(self, kappa_base=0.5, eps=0.001, kappa_cap=50.0):
        super().__init__(use_total_var=True, relative=True)
        self.kappa_base = kappa_base
        self.eps = eps
        self.kappa_cap = kappa_cap
        self._last_adaptive_weights: np.ndarray = np.array([])
        self._last_assignments: np.ndarray = np.array([])
        self._last_overbid_flag: np.ndarray = np.array([])

    def score(self, bids, assignments, realized_costs, cfg):
        """Return raw relative Gaussian scores (for detection metrics).

        Caches per-robot mean adaptive κ_j for compute_penalties().
        """
        N_robots = len(bids)
        M_tasks = len(bids[0]) if bids else 0
        raw_scores = np.zeros(N_robots)
        n_scored = np.zeros(N_robots)
        adaptive_weight_sum = np.zeros(N_robots)
        self._last_assignments = np.asarray(assignments)

        for j in range(M_tasks):
            assigned_robot = None
            for i in range(N_robots):
                if assignments[i] == j:
                    assigned_robot = i
                    break
            if assigned_robot is None or np.isnan(realized_costs[assigned_robot]):
                continue
            real_cost = realized_costs[assigned_robot]

            # Per-task fleet-mean epistemic variance
            task_epi = 0.0
            for i in range(N_robots):
                if j < len(bids[i]):
                    bid = bids[i][j]
                    mu_hat = bid.mus.mean()
                    task_epi += ((bid.mus - mu_hat) ** 2).mean()
            task_epi /= N_robots

            # κ_j: high on familiar terrain, low on OOD
            kappa_j = min(self.kappa_base / max(task_epi, self.eps), self.kappa_cap)

            for i in range(N_robots):
                if j < len(bids[i]):
                    bid = bids[i][j]
                    mu_hat = bid.mus.mean()
                    var_ale = np.exp(bid.log_vars).mean()
                    var_epi = ((bid.mus - mu_hat) ** 2).mean()
                    s = gaussian_score(real_cost, mu_hat, var_ale + var_epi)
                    raw_scores[i] += s
                    adaptive_weight_sum[i] += kappa_j
                    n_scored[i] += 1

        for i in range(N_robots):
            if n_scored[i] > 0:
                raw_scores[i] /= n_scored[i]
                adaptive_weight_sum[i] /= n_scored[i]

        # Relative scoring: subtract fleet mean
        mask = n_scored > 0
        if mask.sum() > 0:
            fleet_mean = raw_scores[mask].mean()
            raw_scores = raw_scores - fleet_mean

        self._last_adaptive_weights = adaptive_weight_sum

        # Overbid signature: a robot whose mean reported bid exceeds the fleet
        # mean bid has priced itself out of contention.  Used by the
        # participation floor to distinguish a strategic dodger (overbid) from
        # an honest robot merely crowded out by someone else's underbid.
        mean_bid = np.full(N_robots, np.nan)
        for i in range(N_robots):
            if bids[i]:
                mean_bid[i] = np.mean([b.mus.mean() for b in bids[i]])
        fleet_mean_bid = np.nanmean(mean_bid) if np.isfinite(mean_bid).any() else 0.0
        self._last_overbid_flag = np.array(
            [bool(np.isfinite(mb) and mb > fleet_mean_bid) for mb in mean_bid]
        )
        return raw_scores

    def compute_penalties(self, scores: np.ndarray, cfg) -> np.ndarray:
        """Apply adaptive κ_j weights directly — bypasses cfg.kappa.

        penalty_i = mean_j(κ_j) * max(0, -relative_score_i)

        simulate() calls this instead of scalar compute_penalty() when the
        mechanism defines it, preventing cfg.kappa from double-multiplying.
        """
        N = len(scores)
        penalties = np.zeros(N)
        weights = self._last_adaptive_weights if len(self._last_adaptive_weights) == N \
                  else np.ones(N) * self.kappa_base
        for i in range(N):
            penalties[i] = weights[i] * max(0.0, -scores[i])
        return penalties


class ParticipationFloorMechanism(AdaptiveKappaMechanism):
    """AdaptiveKappaMechanism plus a participation floor that closes the
    overbidding free-rider loophole on OOD terrain.

    Motivation.  On high-uncertainty terrain an overbidder dodges assignment
    (win rate ~2%), avoiding execution cost *and* the scoring penalties that
    assigned honest robots pay.  The base mechanism therefore leaves a small
    positive overbid gain (+0.99 on OOD type 4).

    Fix.  A robot that is *not assigned* in a round must pay a baseline
    participation penalty lambda = floor_frac * E[pen_fleet].  Assigned robots
    pay their normal scoring penalty and are never floored.

    Why target non-participation rather than penalty magnitude:  flooring by
    penalty magnitude would lift honest low-penalty robots toward the adversary
    and destroy underbid detection.  The free-rider's true signature is that it
    is unassigned far more often than honest robots (~98% vs ~35%), so charging
    a per-round floor for non-participation bites the dodger asymmetrically while
    leaving assigned honest robots untouched.  lambda is estimated online from
    the fleet mean penalty, so it adapts to terrain difficulty with no external
    calibration.
    """

    scores_all_robots: bool = True

    def __init__(self, kappa_base=0.5, eps=0.001, kappa_cap=50.0,
                 floor_frac=1.0, min_floor=0.0):
        super().__init__(kappa_base=kappa_base, eps=eps, kappa_cap=kappa_cap)
        self.floor_frac = floor_frac
        self.min_floor = min_floor

    def compute_penalties(self, scores: np.ndarray, cfg) -> np.ndarray:
        base = super().compute_penalties(scores, cfg)
        if base.size == 0:
            return base
        assigned = self._last_assignments
        overbid = self._last_overbid_flag
        lam = max(self.min_floor, self.floor_frac * float(base.mean()))
        out = base.copy()
        for i in range(len(out)):
            is_assigned = (i < len(assigned)) and (assigned[i] >= 0)
            priced_out = (i < len(overbid)) and bool(overbid[i])
            # Floor only robots that dodged assignment by pricing themselves
            # out.  This isolates strategic overbidders from honest robots that
            # merely lost a contested task to a cheaper (possibly underbidding)
            # peer, preserving underbid deterrence.
            if (not is_assigned) and priced_out:
                out[i] = max(out[i], lam)
        return out
