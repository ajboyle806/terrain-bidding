"""Actor-critic policy for legged locomotion."""
import torch
import torch.nn as nn
from terrain_bidding.configs import PolicyConfig


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs with shared observation encoding."""

    def __init__(self, cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        dims = cfg.hidden_dims  # [512, 256, 128]

        # Actor: obs -> Gaussian params over joint offsets
        self.actor = nn.Sequential(
            nn.Linear(cfg.obs_dim, dims[0]), nn.ELU(),
            nn.Linear(dims[0], dims[1]), nn.ELU(),
            nn.Linear(dims[1], dims[2]), nn.ELU(),
            nn.Linear(dims[2], cfg.act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(cfg.act_dim))

        # Critic: obs -> scalar value
        self.critic = nn.Sequential(
            nn.Linear(cfg.obs_dim, dims[0]), nn.ELU(),
            nn.Linear(dims[0], dims[1]), nn.ELU(),
            nn.Linear(dims[1], dims[2]), nn.ELU(),
            nn.Linear(dims[2], 1),
        )
        self.action_clip = cfg.action_clip

    def forward(self, obs):
        mean = self.actor(obs)
        value = self.critic(obs)
        return mean, value

    def act(self, obs, deterministic=False):
        mean = self.actor(obs)
        if deterministic:
            return torch.clamp(mean, -self.action_clip, self.action_clip)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -self.action_clip, self.action_clip)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, obs, actions):
        mean = self.actor(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value
