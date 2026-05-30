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


class EnsembleRobot(HonestRobot):
    """Robot that uses real ensemble predictions instead of simulated ones."""

    def _get_ensemble_predictions(self, task):
        if isinstance(task, RealTask):
            return task.ensemble_mus.copy(), task.ensemble_log_vars.copy()
        # Fallback to simulated predictions for non-real tasks
        return super()._get_ensemble_predictions(task)


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
        mus_denorm = mus.cpu().numpy() * cost_std + cost_mean
        log_vars_denorm = log_vars.cpu().numpy() + 2 * np.log(cost_std)
        all_mus.append(mus_denorm)
        all_log_vars.append(log_vars_denorm)

    all_mus = np.concatenate(all_mus, axis=1)  # (K, N)
    all_log_vars = np.concatenate(all_log_vars, axis=1)  # (K, N)
    costs_np = np.array(costs)
    heightmaps_np = heightmaps.squeeze(1).numpy()
    scalars_np = scalars.numpy()

    n_samples = len(costs_np)
    print(f"Real task sampler ready: {n_samples} samples")

    # Compute per-sample aleatoric variance (for true_var_ale in SimTask)
    per_sample_var_ale = np.exp(all_log_vars).mean(axis=0)  # (N,)

    def sampler(n_tasks: int, rng: np.random.Generator) -> List[RealTask]:
        indices = rng.choice(n_samples, size=n_tasks, replace=True)
        tasks = []
        for j, idx in enumerate(indices):
            task = RealTask(
                task_id=j,
                true_mu=float(costs_np[idx]),  # actual realized cost as "true mean"
                true_var_ale=float(per_sample_var_ale[idx]),
                heightmap=heightmaps_np[idx],
                scalars=scalars_np[idx],
                ensemble_mus=all_mus[:, idx],
                ensemble_log_vars=all_log_vars[:, idx],
                terrain_type=int(terrain_types[idx]),
            )
            tasks.append(task)
        return tasks

    return sampler
