"""PPO training loop for locomotion policy."""
import os
import torch
import numpy as np
from pathlib import Path
from terrain_bidding.configs import PolicyConfig
from terrain_bidding.policy import ActorCritic

try:
    from terrain_bidding.envs import TerrainBiddingEnv, TerrainBiddingEnvCfg
    HAS_ISAAC = True
except Exception:
    HAS_ISAAC = False


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self, num_envs, num_steps, obs_dim, act_dim, device):
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, act_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.step = 0
        self.num_steps = num_steps

    def add(self, obs, actions, log_probs, rewards, values, dones):
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = rewards
        self.values[self.step] = values
        self.dones[self.step] = dones
        self.step += 1

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        advantages = torch.zeros_like(self.rewards)
        last_adv = 0
        for t in reversed(range(self.num_steps)):
            next_val = last_value if t == self.num_steps - 1 else self.values[t + 1]
            next_done = 0 if t == self.num_steps - 1 else self.dones[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            advantages[t] = last_adv = delta + gamma * lam * (1 - next_done) * last_adv
        returns = advantages + self.values
        return advantages, returns

    def reset(self):
        self.step = 0


def train(cfg: PolicyConfig = PolicyConfig(), device="cuda", save_dir="checkpoints"):
    """Main PPO training loop."""
    if not HAS_ISAAC:
        raise RuntimeError("Isaac Gym required for training. Run on GPU machine.")

    os.makedirs(save_dir, exist_ok=True)
    env = _make_env(cfg)
    policy = ActorCritic(cfg).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    num_steps_per_batch = 24  # steps per env before update
    buffer = RolloutBuffer(cfg.num_envs, num_steps_per_batch, cfg.obs_dim, cfg.act_dim, device)

    total_steps = 0
    best_reward = -float("inf")

    while total_steps < cfg.max_steps:
        # Collect rollouts
        obs = env.obs_buf.clone()
        buffer.reset()
        for _ in range(num_steps_per_batch):
            with torch.no_grad():
                actions, log_probs = policy.act(obs)
                _, values = policy(obs)
                values = values.squeeze(-1)

            env.step(actions)
            buffer.add(obs, actions, log_probs, env.rew_buf, values, env.reset_buf.float())
            obs = env.obs_buf.clone()
            total_steps += cfg.num_envs

        # Compute GAE
        with torch.no_grad():
            _, last_value = policy(obs)
            last_value = last_value.squeeze(-1)
        advantages, returns = buffer.compute_gae(last_value, lam=cfg.gae_lambda)

        # PPO update
        _ppo_update(policy, optimizer, buffer, advantages, returns, cfg)

        # LR decay
        frac = 1.0 - total_steps / cfg.max_steps
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr * frac

        # Logging + checkpointing
        mean_reward = buffer.rewards.sum(0).mean().item()
        if total_steps % (cfg.num_envs * 100) == 0:
            print(f"Steps: {total_steps:,} | Mean reward: {mean_reward:.3f}")
            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(policy.state_dict(), f"{save_dir}/policy_best.pt")

    torch.save(policy.state_dict(), f"{save_dir}/policy_final.pt")
    return policy


def _ppo_update(policy, optimizer, buffer, advantages, returns, cfg):
    """PPO clipped objective with entropy bonus."""
    # Flatten batch
    b_obs = buffer.obs.reshape(-1, cfg.obs_dim)
    b_actions = buffer.actions.reshape(-1, cfg.act_dim)
    b_log_probs = buffer.log_probs.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)

    # Normalize advantages
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    batch_size = b_obs.shape[0]
    for _ in range(cfg.epochs_per_batch):
        indices = torch.randperm(batch_size, device=b_obs.device)
        for start in range(0, batch_size, cfg.minibatch_size):
            end = start + cfg.minibatch_size
            idx = indices[start:end]

            log_prob, entropy, values = policy.evaluate(b_obs[idx], b_actions[idx])
            ratio = (log_prob - b_log_probs[idx]).exp()

            # Clipped surrogate
            adv = b_advantages[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * ((values - b_returns[idx]) ** 2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_loss + cfg.entropy_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()


def _make_env(cfg):
    """Create Isaac Gym environment."""
    from isaacgym import gymapi
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 200.0  # 200Hz sim
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.num_threads = 4
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    env_cfg = TerrainBiddingEnvCfg()
    env_cfg.env.num_envs = cfg.num_envs
    return TerrainBiddingEnv(env_cfg, sim_params, gymapi.SIM_PHYSX, "cuda:0", headless=True)


if __name__ == "__main__":
    train()
