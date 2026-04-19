from dataclasses import dataclass

import torch
import torch.nn as nn

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class ResidualNetConfig(ConfigABC):
    mlp_activation: str = "LeakyReLU"  # activation function for the MLP
    input_feature_dim: int = 4
    hidden_dims: int = 64  # number of hidden dimensions
    n_hidden_layers: int = 5  # number of hidden layers
    output_sdf_scale: float = 0.1  # scale the output SDF
    bound_min: list[float] = None
    bound_max: list[float] = None


class ResidualNet(nn.Module):
    def __init__(self, cfg: ResidualNetConfig):
        """
        Args:
            cfg: configuration of the network
        """
        super().__init__()
        self.cfg = cfg
        self.bound_min = cfg.bound_min
        self.bound_max = cfg.bound_max

        activation = getattr(nn, cfg.mlp_activation)

        layers = []
        in_dim = cfg.input_feature_dim + 1
        for _ in range(cfg.n_hidden_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dims))
            layers.append(activation())
            in_dim = cfg.hidden_dims
        layers.append(nn.Linear(in_dim, 1))
        self.residual_net = nn.Sequential(*layers)

    def get_sdf(self, residual_features: torch.Tensor):
        sdf = self.residual_net(residual_features) * self.cfg.output_sdf_scale
        return sdf

    def forward(self, residual_features: torch.Tensor):
        sdf = self.get_sdf(residual_features)
        return sdf
