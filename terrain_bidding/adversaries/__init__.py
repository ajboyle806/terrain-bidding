"""Phase 5: Strategic adversaries for mechanism stress-testing.

Five adversary types plus an honest baseline robot.
All implement the same interface: bid() and observe().
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from terrain_bidding.mechanism import Bid, RoundResult, gaussian_score


# ─── Task representation for simulation ───────────────────────────────────────

@dataclass
class SimTask:
    """A task in the mechanism simulation."""
    task_id: int
    true_mu: float          # true expected cost
    true_var_ale: float     # true aleatoric variance
    heightmap: Optional[np.ndarray] = None
    scalars: Optional[np.ndarray] = None

    def sample_cost(self, rng: np.random.Generator) -> float:
        """Sample realized cost from true distribution."""
        return rng.normal(self.true_mu, np.sqrt(self.true_var_ale))


# ─── Base Robot ───────────────────────────────────────────────────────────────

class HonestRobot:
    """Honest robot: reports ensemble predictions truthfully."""

    def __init__(self, robot_id: int, ensemble_mu: float = 0.0,
                 ensemble_var_ale: float = 1.0, ensemble_var_epi: float = 0.1,
                 K: int = 5):
        self.robot_id = robot_id
        self.K = K
        # These get overridden per-task in practice
        self._base_mu = ensemble_mu
        self._base_var_ale = ensemble_var_ale
        self._base_var_epi = ensemble_var_epi

    def bid(self, task: SimTask, robot_idx: int, task_idx: int) -> Bid:
        """Generate honest bid from ensemble predictions."""
        mus, log_vars = self._get_ensemble_predictions(task)
        return Bid(robot_id=robot_idx, task_id=task_idx, mus=mus, log_vars=log_vars)

    def _get_ensemble_predictions(self, task: SimTask):
        """Simulate ensemble predictions around true values."""
        # Each network predicts μ_k ~ N(true_mu, var_epi) with log_var_k ≈ log(true_var_ale)
        rng = np.random.default_rng(hash((self.robot_id, task.task_id)) % 2**31)
        spread = np.sqrt(task.true_var_ale * 0.1)  # epistemic spread
        mus = task.true_mu + rng.normal(0, spread, self.K)
        log_vars = np.full(self.K, np.log(task.true_var_ale)) + rng.normal(0, 0.1, self.K)
        return mus, log_vars

    def observe(self, result: RoundResult, robot_idx: int):
        """Observe round outcome (no-op for honest robot)."""
        pass


# ─── 5.1 Fixed Offset Adversary ──────────────────────────────────────────────

class FixedOffsetAdversary(HonestRobot):
    """Reports μ̂_reported = factor · μ̂_true. Default factor=0.7 (30% underbid)."""

    def __init__(self, robot_id: int, factor: float = 0.7, **kwargs):
        super().__init__(robot_id, **kwargs)
        self.factor = factor

    def bid(self, task: SimTask, robot_idx: int, task_idx: int) -> Bid:
        mus, log_vars = self._get_ensemble_predictions(task)
        mus = mus * self.factor  # shade all network means
        return Bid(robot_id=robot_idx, task_id=task_idx, mus=mus, log_vars=log_vars)


# ─── 5.2 Adaptive Adversary ──────────────────────────────────────────────────

class AdaptiveAdversary(HonestRobot):
    """Optimizes bid offset δ* given known κ and competitor model."""

    def __init__(self, robot_id: int, kappa: float = 2.0, R: float = 10.0,
                 competitor_std: float = 2.0, **kwargs):
        super().__init__(robot_id, **kwargs)
        self.kappa = kappa
        self.R = R
        self.competitor_std = competitor_std  # assumed std of competitor bids

    def bid(self, task: SimTask, robot_idx: int, task_idx: int) -> Bid:
        mus, log_vars = self._get_ensemble_predictions(task)
        mu_hat = mus.mean()
        var_ale = np.exp(log_vars).mean()

        # Find optimal δ* that maximizes expected utility
        delta_star = self._optimize_delta(mu_hat, var_ale)
        mus = mus - delta_star  # shade all means uniformly
        return Bid(robot_id=robot_idx, task_id=task_idx, mus=mus, log_vars=log_vars)

    def _optimize_delta(self, mu_hat: float, var_ale: float) -> float:
        """Numerically find δ* maximizing U(δ) = R·P(δ) - κ·δ²/(2σ²_ale)."""
        best_delta, best_u = 0.0, -np.inf
        for delta in np.linspace(0, mu_hat * 0.5, 50):
            # Allocation benefit: P(win) increases with lower bid
            # Model: P(δ) ≈ Φ((competitor_mean - (mu_hat - δ)) / competitor_std)
            from scipy.stats import norm
            p_alloc = norm.cdf(delta / self.competitor_std)
            # Scoring penalty: expected deficit = δ²/(2σ²_ale)
            expected_penalty = self.kappa * delta ** 2 / (2 * var_ale)
            utility = self.R * p_alloc - expected_penalty
            if utility > best_u:
                best_u = utility
                best_delta = delta
        return best_delta


# ─── 5.3 Learning Adversary ──────────────────────────────────────────────────

class LearningAdversary(HonestRobot):
    """Estimates κ from observed penalties and adapts over time."""

    def __init__(self, robot_id: int, kappa_init: float = 0.5,
                 learning_rate: float = 0.1, R: float = 10.0,
                 competitor_std: float = 2.0, **kwargs):
        super().__init__(robot_id, **kwargs)
        self.kappa_est = kappa_init
        self.lr = learning_rate
        self.R = R
        self.competitor_std = competitor_std
        self._last_delta = 0.0
        self._last_var_ale = 1.0

    def bid(self, task: SimTask, robot_idx: int, task_idx: int) -> Bid:
        mus, log_vars = self._get_ensemble_predictions(task)
        mu_hat = mus.mean()
        var_ale = np.exp(log_vars).mean()
        self._last_var_ale = var_ale

        # Use current κ estimate to optimize
        delta_star = self._optimize_delta(mu_hat, var_ale)
        self._last_delta = delta_star
        mus = mus - delta_star
        return Bid(robot_id=robot_idx, task_id=task_idx, mus=mus, log_vars=log_vars)

    def _optimize_delta(self, mu_hat, var_ale):
        best_delta, best_u = 0.0, -np.inf
        for delta in np.linspace(0, mu_hat * 0.5, 50):
            from scipy.stats import norm
            p_alloc = norm.cdf(delta / self.competitor_std)
            expected_penalty = self.kappa_est * delta ** 2 / (2 * var_ale)
            utility = self.R * p_alloc - expected_penalty
            if utility > best_u:
                best_u = utility
                best_delta = delta
        return best_delta

    def observe(self, result: RoundResult, robot_idx: int):
        """Update κ estimate from observed penalty."""
        if result.assignments[robot_idx] < 0 or self._last_delta < 1e-6:
            return
        observed_penalty = result.penalties[robot_idx]
        # Expected penalty under true κ: κ·δ²/(2σ²_ale)
        expected_deficit = self._last_delta ** 2 / (2 * self._last_var_ale)
        if expected_deficit > 1e-8:
            kappa_observed = observed_penalty / expected_deficit
            self.kappa_est += self.lr * (kappa_observed - self.kappa_est)


# ─── 5.4 Terrain-Selective Adversary ─────────────────────────────────────────

class TerrainSelectiveAdversary(HonestRobot):
    """Honest on familiar terrain, shades bids on novel terrain (high σ²_epi)."""

    def __init__(self, robot_id: int, epi_threshold: float = 0.5,
                 shade_fraction: float = 0.2, **kwargs):
        super().__init__(robot_id, **kwargs)
        self.epi_threshold = epi_threshold
        self.shade_fraction = shade_fraction

    def bid(self, task: SimTask, robot_idx: int, task_idx: int) -> Bid:
        mus, log_vars = self._get_ensemble_predictions(task)
        mu_hat = mus.mean()
        var_epi = ((mus - mu_hat) ** 2).mean()

        if var_epi > self.epi_threshold:
            # Novel terrain: shade bid
            delta = self.shade_fraction * mu_hat
            mus = mus - delta
        # Else: report honestly

        return Bid(robot_id=robot_idx, task_id=task_idx, mus=mus, log_vars=log_vars)


# ─── 5.5 Ensemble-Manipulating Adversary ─────────────────────────────────────

class EnsembleManipulatingAdversary(HonestRobot):
    """Trains ensemble for artificial agreement on target terrain types.

    Simulates effect of fine-tuning: on target terrain, ensemble members
    agree more (lower σ²_epi) enabling aggressive bidding without epistemic penalty.
    """

    def __init__(self, robot_id: int, target_terrain_types: list = None,
                 agreement_factor: float = 0.1, shade_on_target: float = 0.3,
                 **kwargs):
        super().__init__(robot_id, **kwargs)
        self.target_types = target_terrain_types or [0]  # default: flat terrain
        self.agreement_factor = agreement_factor
        self.shade_on_target = shade_on_target

    def bid(self, task: SimTask, robot_idx: int, task_idx: int) -> Bid:
        mus, log_vars = self._get_ensemble_predictions(task)

        # Check if this is a target terrain type (via task metadata)
        is_target = hasattr(task, "terrain_type") and task.terrain_type in self.target_types

        if is_target:
            # Artificial agreement: collapse ensemble means toward their mean
            mu_hat = mus.mean()
            mus = mu_hat + (mus - mu_hat) * self.agreement_factor
            # Also shade the bid aggressively
            mus = mus - self.shade_on_target * mu_hat

        return Bid(robot_id=robot_idx, task_id=task_idx, mus=mus, log_vars=log_vars)


# ─── Factory ──────────────────────────────────────────────────────────────────

ADVERSARY_TYPES = {
    "honest": HonestRobot,
    "fixed_offset": FixedOffsetAdversary,
    "adaptive": AdaptiveAdversary,
    "learning": LearningAdversary,
    "terrain_selective": TerrainSelectiveAdversary,
    "ensemble_manipulating": EnsembleManipulatingAdversary,
}


def create_fleet(n_honest: int, adversary_type: str = "honest",
                 n_adversary: int = 0, **adversary_kwargs) -> list:
    """Create a mixed fleet of honest robots and adversaries."""
    fleet = []
    for i in range(n_honest):
        fleet.append(HonestRobot(robot_id=i))
    for i in range(n_adversary):
        cls = ADVERSARY_TYPES[adversary_type]
        fleet.append(cls(robot_id=n_honest + i, **adversary_kwargs))
    return fleet
