"""Real-data task sampler and ensemble-backed robots for grounded experiments.

Connects Phase 3 ensemble to Phase 6 experiments using actual terrain data.
"""
import numpy as np
import torch
import h5py
from typing import List
from terrain_bidding.adversaries import SimTask, HonestRobot, Bid
from terrain_bidding.estimator import Ensemble, EnsembleConfig, RolloutDataset


class RealTask(SimTask):
    """Task backed by real terrain data and ensemble predictions."""

    def __init__(self, task_id: int, true_mu: float, true_var_ale: float,
                 heightmap: np.ndarray, scalars: np.ndarray,
                 ensemble_mus: np.ndarray, ensemble_log_vars: np.ndarray,
                 terrain_type: int = 0):
        super().__init__(task_id=task_id, true_mu=true_mu, true_var_ale=true_var_ale)
        self.heightmap = heightmap
        self.scalars = scalars
        self.ensemble_mus = ensemble_mus  # (K,) real ensemble predictions
        self.ensemble_log_vars = ensemble_log_vars  # (K,)
        self.terrain_type = terrain_type


class PrivateObsTask(SimTask):
    """Task where each robot has a private terrain observation.

    Simulates private information: robots observe different terrain patches
    for the same task region, producing different cost predictions.
    The true cost is drawn from one realization.
    """

    def __init__(self, task_id: int, true_mu: float, true_var_ale: float,
                 robot_observations: dict, terrain_type: int = 0):
        """
        Args:
            robot_observations: {robot_id: (ensemble_mus, ensemble_log_vars)}
        """
        super().__init__(task_id=task_id, true_mu=true_mu, true_var_ale=true_var_ale)
        self.robot_observations = robot_observations
        self.terrain_type = terrain_type


class EnsembleRobot(HonestRobot):
    """Robot that uses real ensemble predictions from private observations."""

    def _get_ensemble_predictions(self, task):
        if isinstance(task, PrivateObsTask) and self.robot_id in task.robot_observations:
            mus, log_vars = task.robot_observations[self.robot_id]
            return mus.copy(), log_vars.copy()
        if isinstance(task, RealTask):
            return task.ensemble_mus.copy(), task.ensemble_log_vars.copy()
        return super()._get_ensemble_predictions(task)


class RealFixedOffsetAdversary(EnsembleRobot):
    """Fixed offset adversary: bids lower than true prediction by a fixed amount."""
    def __init__(self, robot_id: int, offset: float = 0.5, **kwargs):
        super().__init__(robot_id, **kwargs)
        self.offset = offset  # subtract this from predictions (in normalized units)

    def bid(self, task, robot_idx: int, task_idx: int):
        mus, log_vars = self._get_ensemble_predictions(task)
        mus = mus - self.offset  # bid lower to win allocation
        return Bid(robot_id=robot_idx, task_id=task_idx, mus=mus, log_vars=log_vars)


class RealAdaptiveAdversary(EnsembleRobot):
    """Adaptive adversary using real ensemble predictions."""
    def __init__(self, robot_id: int, kappa: float = 2.0, R: float = 5.0,
                 competitor_std: float = 1.0, **kwargs):
        super().__init__(robot_id, **kwargs)
        self.kappa = kappa
        self.R = R
        self.competitor_std = competitor_std

    def bid(self, task, robot_idx: int, task_idx: int):
        mus, log_vars = self._get_ensemble_predictions(task)
        mu_hat = mus.mean()
        var_ale = np.exp(log_vars).mean()
        delta = self._optimize_delta(mu_hat, var_ale)
        mus = mus - delta
        return Bid(robot_id=robot_idx, task_id=task_idx, mus=mus, log_vars=log_vars)

    def _optimize_delta(self, mu_hat, var_ale):
        from scipy.stats import norm
        best_delta, best_u = 0.0, -np.inf
        for delta in np.linspace(0, mu_hat * 0.3, 50):
            p_alloc = norm.cdf(delta / self.competitor_std)
            expected_penalty = self.kappa * delta ** 2 / (2 * var_ale)
            utility = self.R * p_alloc - expected_penalty
            if utility > best_u:
                best_u = utility
                best_delta = delta
        return best_delta


class RealLearningAdversary(EnsembleRobot):
    """Learning adversary using real ensemble predictions."""
    def __init__(self, robot_id: int, kappa_init: float = 0.5,
                 learning_rate: float = 0.1, R: float = 5.0,
                 competitor_std: float = 1.0, **kwargs):
        super().__init__(robot_id, **kwargs)
        self.kappa_est = kappa_init
        self.lr = learning_rate
        self.R = R
        self.competitor_std = competitor_std
        self._last_delta = 0.0
        self._last_var_ale = 1.0

    def bid(self, task, robot_idx: int, task_idx: int):
        mus, log_vars = self._get_ensemble_predictions(task)
        mu_hat = mus.mean()
        var_ale = np.exp(log_vars).mean()
        self._last_var_ale = var_ale
        from scipy.stats import norm
        best_delta, best_u = 0.0, -np.inf
        for delta in np.linspace(0, mu_hat * 0.3, 50):
            p_alloc = norm.cdf(delta / self.competitor_std)
            expected_penalty = self.kappa_est * delta ** 2 / (2 * var_ale)
            utility = self.R * p_alloc - expected_penalty
            if utility > best_u:
                best_u = utility
                best_delta = delta
        self._last_delta = best_delta
        mus = mus - best_delta
        return Bid(robot_id=robot_idx, task_id=task_idx, mus=mus, log_vars=log_vars)

    def observe(self, result, robot_idx: int):
        from terrain_bidding.mechanism import RoundResult
        if result.assignments[robot_idx] < 0 or self._last_delta < 1e-6:
            return
        observed_penalty = result.penalties[robot_idx]
        expected_deficit = self._last_delta ** 2 / (2 * self._last_var_ale)
        if expected_deficit > 1e-8:
            kappa_observed = observed_penalty / expected_deficit
            kappa_observed = min(kappa_observed, 50.0)  # clamp to prevent explosion
            self.kappa_est += self.lr * (kappa_observed - self.kappa_est)
            self.kappa_est = max(0.01, min(self.kappa_est, 50.0))


class RealTerrainSelectiveAdversary(EnsembleRobot):
    """Terrain-selective adversary using real ensemble predictions."""
    def __init__(self, robot_id: int, epi_threshold: float = 0.5,
                 shade_fraction: float = 0.2, **kwargs):
        super().__init__(robot_id, **kwargs)
        self.epi_threshold = epi_threshold
        self.shade_fraction = shade_fraction

    def bid(self, task, robot_idx: int, task_idx: int):
        mus, log_vars = self._get_ensemble_predictions(task)
        mu_hat = mus.mean()
        var_epi = ((mus - mu_hat) ** 2).mean()
        if var_epi > self.epi_threshold:
            mus = mus - self.shade_fraction * mu_hat
        return Bid(robot_id=robot_idx, task_id=task_idx, mus=mus, log_vars=log_vars)


class RealEnsembleManipulatingAdversary(EnsembleRobot):
    """Ensemble-manipulating adversary using real predictions."""
    def __init__(self, robot_id: int, agreement_factor: float = 0.1,
                 shade_on_target: float = 0.3, **kwargs):
        super().__init__(robot_id, **kwargs)
        self.agreement_factor = agreement_factor
        self.shade_on_target = shade_on_target

    def bid(self, task, robot_idx: int, task_idx: int):
        mus, log_vars = self._get_ensemble_predictions(task)
        mu_hat = mus.mean()
        # Artificial agreement + shade
        mus = mu_hat + (mus - mu_hat) * self.agreement_factor
        mus = mus - self.shade_on_target * mu_hat
        return Bid(robot_id=robot_idx, task_id=task_idx, mus=mus, log_vars=log_vars)


REAL_ADVERSARY_TYPES = {
    "fixed_offset": RealFixedOffsetAdversary,
    "adaptive": RealAdaptiveAdversary,
    "learning": RealLearningAdversary,
    "terrain_selective": RealTerrainSelectiveAdversary,
    "ensemble_manipulating": RealEnsembleManipulatingAdversary,
}


def make_real_task_sampler(data_path: str = "data/test.hdf5",
                           ensemble_path: str = "checkpoints/ensemble",
                           device: str = "cpu"):
    """Create a task sampler that draws from real data + ensemble predictions.

    Loads test data, queries ensemble for predictions, caches results.
    """
    # Load test data
    with h5py.File(data_path, "r") as f:
        heightmaps = torch.tensor(f["heightmap"][:], dtype=torch.float32).unsqueeze(1)
        scalars = torch.tensor(np.stack([
            f["slope"][:], f["roughness"][:], f["friction"][:],
            f["distance"][:], f["elevation_change"][:]
        ], axis=1), dtype=torch.float32)
        costs = f["cost"][:]
        terrain_types = f["terrain_type"][:]

    # Load ensemble
    cfg = EnsembleConfig()
    ensemble = Ensemble(cfg, device=device)
    ensemble.load(ensemble_path)

    # Get ensemble predictions for all test samples
    print(f"Querying ensemble on {len(costs)} test samples...")
    all_mus, all_log_vars = [], []
    batch_size = 512
    for i in range(0, len(costs), batch_size):
        hm = heightmaps[i:i+batch_size].to(device)
        sc = scalars[i:i+batch_size].to(device)
        mus, log_vars = ensemble.predict_raw(hm, sc)  # (K, B), (K, B)
        # Denormalize predictions
        cost_mean = ensemble.cost_mean if hasattr(ensemble, 'cost_mean') else 0.0
        cost_std = ensemble.cost_std if hasattr(ensemble, 'cost_std') else 1.0
        # Keep in normalized space for proper scoring (var_ale ~ 1)
        all_mus.append(mus.cpu().numpy())
        all_log_vars.append(log_vars.cpu().numpy())

    all_mus = np.concatenate(all_mus, axis=1)  # (K, N)
    all_log_vars = np.concatenate(all_log_vars, axis=1)  # (K, N)
    costs_np = np.array(costs)
    heightmaps_np = heightmaps.squeeze(1).numpy()
    scalars_np = scalars.numpy()

    n_samples = len(costs_np)
    print(f"Real task sampler ready: {n_samples} samples")

    # Normalize costs to match prediction space
    cost_mean = ensemble.cost_mean if hasattr(ensemble, 'cost_mean') else 0.0
    cost_std = ensemble.cost_std if hasattr(ensemble, 'cost_std') else 1.0
    costs_normalized = (costs_np - cost_mean) / cost_std

    # Compute per-sample aleatoric variance (for true_var_ale in SimTask)
    per_sample_var_ale = np.exp(all_log_vars).mean(axis=0)  # (N,)

    # Group samples by terrain type for private observation sampling
    type_indices = {}
    for i, t in enumerate(terrain_types):
        type_indices.setdefault(int(t), []).append(i)

    def sampler(n_tasks: int, rng: np.random.Generator, n_robots: int = 4):
        """Each task gives each robot a DIFFERENT terrain sample from the same type.

        This simulates private observations: robots observe different patches
        of the same terrain region, producing heterogeneous predictions.
        """
        indices = rng.choice(n_samples, size=n_tasks, replace=True)
        tasks = []
        for j, idx in enumerate(indices):
            # True cost comes from this sample
            true_cost = float(costs_normalized[idx])
            true_var = float(per_sample_var_ale[idx])
            t_type = int(terrain_types[idx])

            # Each robot gets a different sample from the same terrain type
            same_type = type_indices.get(t_type, list(range(n_samples)))
            robot_indices = rng.choice(same_type, size=n_robots, replace=True)

            robot_obs = {}
            for robot_id in range(n_robots):
                r_idx = robot_indices[robot_id]
                robot_obs[robot_id] = (all_mus[:, r_idx], all_log_vars[:, r_idx])

            task = PrivateObsTask(
                task_id=j,
                true_mu=true_cost,
                true_var_ale=true_var,
                robot_observations=robot_obs,
                terrain_type=t_type,
            )
            tasks.append(task)
        return tasks

    return sampler
