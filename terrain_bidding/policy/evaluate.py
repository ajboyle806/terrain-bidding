"""Evaluation gate for locomotion policy.

Pass criteria:
- Failure rate < 5% on all terrain types
- Cost CV > 0.15 on at least 3 terrain types
- No single failure mode > 50% of failures on any terrain
"""
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from terrain_bidding.configs import PolicyConfig, CostWeights
from terrain_bidding.policy import ActorCritic
from terrain_bidding.envs import TERRAIN_NAMES

try:
    from terrain_bidding.envs import TerrainBiddingEnv
    HAS_ISAAC = True
except Exception:
    HAS_ISAAC = False


@dataclass
class EvalResult:
    terrain: str
    episodes: int
    failure_rate: float
    fall_count: int
    timeout_count: int
    mean_cost: float
    std_cost: float
    cost_cv: float
    dominant_failure_frac: float  # fraction of failures from most common mode


def evaluate_policy(checkpoint_path: str, episodes_per_terrain: int = 1000,
                    cost_weights: CostWeights = CostWeights()) -> list[EvalResult]:
    """Run evaluation on each terrain type."""
    if not HAS_ISAAC:
        raise RuntimeError("Isaac Gym required. Run on GPU machine.")

    cfg = PolicyConfig()
    policy = ActorCritic(cfg).to("cuda")
    policy.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
    policy.eval()

    results = []
    for terrain_idx, terrain_name in enumerate(TERRAIN_NAMES):
        costs, falls, timeouts = [], 0, 0
        for _ in range(episodes_per_terrain):
            outcome = _run_episode(policy, terrain_idx, cost_weights)
            if outcome["success"]:
                costs.append(outcome["cost"])
            elif outcome["fall"]:
                falls += 1
            else:
                timeouts += 1

        total_failures = falls + timeouts
        failure_rate = total_failures / episodes_per_terrain
        dominant = max(falls, timeouts) / max(total_failures, 1)
        costs_arr = np.array(costs) if costs else np.array([0.0])

        results.append(EvalResult(
            terrain=terrain_name,
            episodes=episodes_per_terrain,
            failure_rate=failure_rate,
            fall_count=falls,
            timeout_count=timeouts,
            mean_cost=costs_arr.mean(),
            std_cost=costs_arr.std(),
            cost_cv=costs_arr.std() / max(costs_arr.mean(), 1e-8),
            dominant_failure_frac=dominant,
        ))
    return results


def check_gate(results: list[EvalResult]) -> tuple[bool, list[str]]:
    """Check if policy passes evaluation gate. Returns (passed, issues)."""
    issues = []

    # Failure rate < 5% on all terrain
    for r in results:
        if r.failure_rate > 0.05:
            issues.append(f"{r.terrain}: failure rate {r.failure_rate:.1%} > 5%")

    # Cost CV > 0.15 on at least 3 terrain types
    high_cv = [r for r in results if r.cost_cv > 0.15]
    if len(high_cv) < 3:
        issues.append(f"Only {len(high_cv)} terrains with cost CV > 0.15 (need 3)")

    # No single failure mode > 50%
    for r in results:
        if r.failure_rate > 0 and r.dominant_failure_frac > 0.5:
            mode = "falls" if r.fall_count > r.timeout_count else "timeouts"
            issues.append(f"{r.terrain}: {mode} account for {r.dominant_failure_frac:.0%} of failures")

    return len(issues) == 0, issues


def _run_episode(policy, terrain_idx, cost_weights):
    """Run single episode on specified terrain. Returns outcome dict.

    Requires Isaac Gym runtime.
    """
    if not HAS_ISAAC:
        raise RuntimeError("Isaac Gym required. Run on GPU machine.")

    from terrain_bidding.collection import _get_env_singleton, _compute_cost
    import torch

    env = _get_env_singleton()
    dt = 1.0 / 50.0
    max_steps = int(15.0 / dt)

    # Reset env to specified terrain type
    env.reset()
    start_pos = env.root_states[0, :3].clone()

    torques_list, joint_vels_list = [], []
    for step in range(max_steps):
        obs = env.obs_buf[0:1]
        with torch.no_grad():
            action = policy.act(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]

        env.step(action)
        torques_list.append(env.torques[0].cpu().numpy().copy())
        joint_vels_list.append(env.dof_vel[0].cpu().numpy().copy())

        # Check fall
        base_height = env.root_states[0, 2].item()
        if base_height < 0.15:
            return {"success": False, "fall": True, "timeout": False}

        grav = env.projected_gravity[0].cpu().numpy()
        pitch = abs(np.arctan2(grav[0], grav[2]))
        roll = abs(np.arctan2(grav[1], grav[2]))
        if pitch > 1.047 or roll > 1.047:
            return {"success": False, "fall": True, "timeout": False}

        # Check goal reached (8m default goal distance)
        pos = env.root_states[0, :3]
        if torch.norm(pos[:2] - start_pos[:2]).item() >= 7.5:
            trajectory = {
                "torques": np.array(torques_list),
                "joint_vels": np.array(joint_vels_list),
                "dt": dt,
                "duration": (step + 1) * dt,
            }
            cost = _compute_cost(trajectory, cost_weights)
            return {"success": True, "fall": False, "timeout": False, "cost": cost}

    return {"success": False, "fall": False, "timeout": True}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/policy_best.pt"
    results = evaluate_policy(path)
    print("\n=== Evaluation Results ===")
    for r in results:
        status = "✓" if r.failure_rate < 0.05 else "✗"
        print(f"{status} {r.terrain:20s} | fail={r.failure_rate:.1%} | "
              f"cost={r.mean_cost:.2f}±{r.std_cost:.2f} | CV={r.cost_cv:.3f}")
    passed, issues = check_gate(results)
    print(f"\n{'PASSED' if passed else 'FAILED'}")
    for issue in issues:
        print(f"  ⚠ {issue}")
