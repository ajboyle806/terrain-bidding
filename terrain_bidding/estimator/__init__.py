"""Phase 3: Ensemble cost estimator with uncertainty decomposition.

K=5 independent networks, each predicting (μ, log σ²) for heteroscedastic NLL.
Epistemic uncertainty from ensemble disagreement, aleatoric from predicted variance.
"""
import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass
from terrain_bidding.configs import EnsembleConfig


class CostNetwork(nn.Module):
    """Single cost estimator: CNN terrain encoder + scalar features → (μ, log σ²)."""

    def __init__(self, cfg: EnsembleConfig = EnsembleConfig(), large: bool = False):
        super().__init__()
        if large:
            embed_dim = cfg.large_terrain_embed_dim
            mlp_h = cfg.large_mlp_hidden
            self.terrain_encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(128 * 16, embed_dim), nn.ReLU(),
            )
        else:
            embed_dim = cfg.terrain_embed_dim
            mlp_h = cfg.mlp_hidden
            self.terrain_encoder = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(32 * 16, embed_dim), nn.ReLU(),
            )

        scalar_dim = 5  # slope, roughness, friction, distance, elevation_change
        self.head = nn.Sequential(
            nn.Linear(embed_dim + scalar_dim, mlp_h), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(mlp_h, mlp_h), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(mlp_h, 2),  # (μ, log σ²)
        )
        self.log_var_clip = cfg.log_var_clip

    def forward(self, heightmap, scalars):
        """
        Args:
            heightmap: (B, 1, 16, 16)
            scalars: (B, 5) — slope, roughness, friction, distance, elevation
        Returns:
            mu: (B,), log_var: (B,)
        """
        terrain_feat = self.terrain_encoder(heightmap)
        x = torch.cat([terrain_feat, scalars], dim=-1)
        out = self.head(x)
        mu, log_var = out[:, 0], out[:, 1]
        log_var = torch.clamp(log_var, *self.log_var_clip)
        return mu, log_var


class Ensemble:
    """K independent CostNetworks with uncertainty decomposition."""

    def __init__(self, cfg: EnsembleConfig = EnsembleConfig(), large: bool = False,
                 device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.networks = [CostNetwork(cfg, large).to(device) for _ in range(cfg.K)]
        self.temperature = 1.0  # calibration temperature

    def predict(self, heightmap, scalars):
        """Ensemble prediction with uncertainty decomposition.

        Returns:
            mu_hat: ensemble mean prediction
            var_ale: mean predicted aleatoric variance
            var_epi: variance of predicted means (epistemic)
        """
        mus, log_vars = [], []
        for net in self.networks:
            net.eval()
            with torch.no_grad():
                mu, log_var = net(heightmap, scalars)
                mus.append(mu)
                log_vars.append(log_var + np.log(self.temperature))
        mus = torch.stack(mus)          # (K, B)
        log_vars = torch.stack(log_vars)  # (K, B)

        mu_hat = mus.mean(dim=0)                          # (B,)
        var_ale = log_vars.exp().mean(dim=0)              # (B,)
        var_epi = ((mus - mu_hat.unsqueeze(0)) ** 2).mean(dim=0)  # (B,)
        return mu_hat, var_ale, var_epi

    def predict_raw(self, heightmap, scalars):
        """Return raw per-network predictions (for submission to planner)."""
        mus, log_vars = [], []
        for net in self.networks:
            net.eval()
            with torch.no_grad():
                mu, log_var = net(heightmap, scalars)
                mus.append(mu)
                log_vars.append(log_var)
        return torch.stack(mus), torch.stack(log_vars)  # (K, B), (K, B)

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        for i, net in enumerate(self.networks):
            torch.save(net.state_dict(), f"{path}/net_{i}.pt")
        torch.save({"temperature": self.temperature}, f"{path}/meta.pt")

    def load(self, path: str):
        for i, net in enumerate(self.networks):
            net.load_state_dict(torch.load(f"{path}/net_{i}.pt", map_location=self.device))
        meta = torch.load(f"{path}/meta.pt", map_location=self.device)
        self.temperature = meta["temperature"]


def heteroscedastic_nll(mu, log_var, target):
    """Negative log-likelihood loss for heteroscedastic Gaussian."""
    var = log_var.exp()
    return 0.5 * ((target - mu) ** 2 / var + log_var).mean()


class RolloutDataset(torch.utils.data.Dataset):
    """Load HDF5 rollout data for ensemble training."""

    def __init__(self, path: str):
        with h5py.File(path, "r") as f:
            self.heightmaps = torch.tensor(f["heightmap"][:], dtype=torch.float32)
            self.scalars = torch.tensor(np.stack([
                f["slope"][:], f["roughness"][:], f["friction"][:],
                f["distance"][:], f["elevation_change"][:]
            ], axis=1), dtype=torch.float32)
            self.costs = torch.tensor(f["cost"][:], dtype=torch.float32)
        # Add channel dim to heightmaps
        self.heightmaps = self.heightmaps.unsqueeze(1)  # (N, 1, 16, 16)

    def __len__(self):
        return len(self.costs)

    def __getitem__(self, idx):
        return self.heightmaps[idx], self.scalars[idx], self.costs[idx]


def train_ensemble(cfg: EnsembleConfig = EnsembleConfig(), data_dir: str = "data",
                   save_dir: str = "checkpoints/ensemble", large: bool = False,
                   device: str = "cuda"):
    """Train K independent networks with different seeds."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    train_ds = RolloutDataset(f"{data_dir}/train.hdf5")
    val_ds = RolloutDataset(f"{data_dir}/val.hdf5")

    ensemble = Ensemble(cfg, large=large, device=device)

    for k in range(cfg.K):
        print(f"\n=== Training network {k+1}/{cfg.K} ===")
        torch.manual_seed(k * 1337)
        net = ensemble.networks[k]
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.max_epochs)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False)

        best_val_loss, patience_counter = float("inf"), 0
        for epoch in range(cfg.max_epochs):
            # Train
            net.train()
            train_loss = 0
            for hm, sc, cost in train_loader:
                hm, sc, cost = hm.to(device), sc.to(device), cost.to(device)
                mu, log_var = net(hm, sc)
                loss = heteroscedastic_nll(mu, log_var, cost)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            # Validate
            net.eval()
            val_loss = 0
            with torch.no_grad():
                for hm, sc, cost in val_loader:
                    hm, sc, cost = hm.to(device), sc.to(device), cost.to(device)
                    mu, log_var = net(hm, sc)
                    val_loss += heteroscedastic_nll(mu, log_var, cost).item()

            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(net.state_dict(), f"{save_dir}/net_{k}.pt")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: val_nll={val_loss:.4f}")

        # Reload best
        net.load_state_dict(torch.load(f"{save_dir}/net_{k}.pt", map_location=device))

    # Calibrate
    temperature = calibrate_temperature(ensemble, val_ds, device)
    ensemble.temperature = temperature
    ensemble.save(save_dir)
    print(f"\nEnsemble saved. Temperature: {temperature:.4f}")
    return ensemble


def calibrate_temperature(ensemble: Ensemble, val_ds: RolloutDataset,
                          device: str = "cuda") -> float:
    """Fit scalar temperature T to minimize calibration error on validation set."""
    loader = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False)
    all_mu, all_var, all_cost = [], [], []

    for hm, sc, cost in loader:
        hm, sc, cost = hm.to(device), sc.to(device), cost.to(device)
        mu_hat, var_ale, var_epi = ensemble.predict(hm, sc)
        all_mu.append(mu_hat.cpu())
        all_var.append((var_ale + var_epi).cpu())
        all_cost.append(cost.cpu())

    mu = torch.cat(all_mu)
    var = torch.cat(all_var)
    cost = torch.cat(all_cost)

    # Grid search for temperature that minimizes ECE
    best_t, best_ece = 1.0, float("inf")
    for log_t in np.linspace(-1, 1, 50):
        t = np.exp(log_t)
        ece = _compute_ece(mu, var * t, cost)
        if ece < best_ece:
            best_ece = ece
            best_t = t
    return best_t


def _compute_ece(mu, var, cost, n_bins: int = 10) -> float:
    """Expected calibration error for Gaussian predictions."""
    std = var.sqrt()
    z = (cost - mu) / std
    # Check coverage at nominal levels
    coverages = [0.5, 0.7, 0.9]
    total_error = 0
    for nominal in coverages:
        from scipy.stats import norm
        z_thresh = norm.ppf(0.5 + nominal / 2)
        empirical = (z.abs() < z_thresh).float().mean().item()
        total_error += abs(empirical - nominal)
    return total_error / len(coverages)


if __name__ == "__main__":
    train_ensemble()
