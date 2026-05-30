"""Held-out terrain: collection script for laptop + proxy for Mac testing.

For laptop (Isaac Gym): collects rollouts on terrain levels 15-19 only,
which are underrepresented in training data → high epistemic variance.

For Mac (proxy): creates synthetic held-out tasks with inflated epistemic
variance to simulate the effect of novel terrain on the mechanism.
"""
import numpy as np
from typing import List
from terrain_bidding.adversaries import SimTask
from terrain_bidding.mechanism import Bid


class HeldOutTask(SimTask):
    """Task with artificially high epistemic variance (simulates novel terrain)."""

    def __init__(self, task_id: int, true_mu: float, true_var_ale: float,
                 ensemble_mus: np.ndarray, ensemble_log_vars: np.ndarray,
                 terrain_type: int = 5):
        super().__init__(task_id=task_id, true_mu=true_mu, true_var_ale=true_var_ale)
        self.ensemble_mus = ensemble_mus
        self.ensemble_log_vars = ensemble_log_vars
        self.terrain_type = terrain_type
        self.heightmap = np.random.randn(16, 16).astype(np.float32) * 0.1
        self.scalars = np.zeros(5, dtype=np.float32)


def make_held_out_proxy_sampler(real_sampler=None, epi_inflation: float = 5.0):
    """Create a held-out terrain proxy sampler.

    Takes real tasks and inflates epistemic variance by spreading ensemble means.
    This simulates what happens when the ensemble encounters novel terrain:
    the networks disagree more (high σ²_epi) while aleatoric stays the same.

    Args:
        real_sampler: the real data sampler (if available)
        epi_inflation: factor by which to inflate epistemic spread
    """
    def sampler(n_tasks: int, rng: np.random.Generator) -> List[HeldOutTask]:
        tasks = []
        if real_sampler is not None:
            # Get real tasks and inflate their epistemic variance
            real_tasks = real_sampler(n_tasks, rng)
            for j, rt in enumerate(real_tasks):
                mus = rt.ensemble_mus.copy()
                log_vars = rt.ensemble_log_vars.copy()
                # Inflate epistemic spread: push means apart
                mu_hat = mus.mean()
                mus = mu_hat + (mus - mu_hat) * epi_inflation
                # Add noise to simulate model uncertainty on novel terrain
                mus += rng.normal(0, abs(mu_hat) * 0.1, len(mus))
                tasks.append(HeldOutTask(
                    task_id=j,
                    true_mu=rt.true_mu * (1 + rng.normal(0, 0.2)),  # novel terrain has shifted costs
                    true_var_ale=float(np.exp(log_vars).mean()),
                    ensemble_mus=mus,
                    ensemble_log_vars=log_vars,
                ))
        else:
            # Pure synthetic held-out
            for j in range(n_tasks):
                mu = rng.normal(2500, 800)
                var_ale = rng.uniform(100, 400) ** 2
                K = 5
                # High epistemic spread
                mus = mu + rng.normal(0, abs(mu) * 0.15, K)
                log_vars = np.full(K, np.log(var_ale)) + rng.normal(0, 0.3, K)
                tasks.append(HeldOutTask(
                    task_id=j, true_mu=mu, true_var_ale=var_ale,
                    ensemble_mus=mus, ensemble_log_vars=log_vars,
                ))
        return tasks
    return sampler


def make_mixed_sampler(real_sampler, held_out_fraction: float = 0.3,
                       epi_inflation: float = 5.0):
    """Sampler that mixes in-distribution and held-out tasks.

    Args:
        real_sampler: real data sampler
        held_out_fraction: fraction of tasks that are held-out
        epi_inflation: epistemic inflation factor for held-out tasks
    """
    held_out_sampler = make_held_out_proxy_sampler(real_sampler, epi_inflation)

    def sampler(n_tasks: int, rng: np.random.Generator):
        n_held_out = max(1, int(n_tasks * held_out_fraction))
        n_in_dist = n_tasks - n_held_out
        in_dist = real_sampler(n_in_dist, rng) if n_in_dist > 0 else []
        held_out = held_out_sampler(n_held_out, rng)
        all_tasks = list(in_dist) + list(held_out)
        # Re-index
        for j, t in enumerate(all_tasks):
            t.task_id = j
        rng.shuffle(all_tasks)
        return all_tasks

    return sampler


# ─── Laptop collection script for real held-out data ──────────────────────────

HELD_OUT_COLLECTION_SCRIPT = """
# Run this on the laptop with Isaac Gym to collect held-out terrain data.
# This collects rollouts ONLY on terrain levels 15-19 (hardest, underrepresented in training).

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Modify collection to force high difficulty levels only
from terrain_bidding.collection import collect
from terrain_bidding.configs import CollectionConfig

# Override: collect fewer samples on held-out terrain
cfg = CollectionConfig()
cfg.num_rollouts = 2000

# The collection script needs terrain levels 15-19 only.
# Set max_init_terrain_level=19 and min level=15 in the env config.
# This requires modifying the env_cfg in collection before env creation.

print("Collecting held-out terrain data (levels 15-19)...")
print("Run: python -m terrain_bidding.collection")
print("Then manually filter for high terrain levels, or modify collection to force them.")
"""
