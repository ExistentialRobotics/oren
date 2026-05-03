from dataclasses import dataclass, field

import torch
import torch.nn as nn

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class ResidualNetConfig(ConfigABC):
    mlp_activation: str = "LeakyReLU"  # activation function for the MLP
    input_feature_dim: int = 4
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64, 64])
    output_sdf_scale: float = 0.1  # scale the output SDF


class ResidualNet(nn.Module):
    def __init__(self, cfg: ResidualNetConfig):
        """
        Args:
            cfg: configuration of the network
        """
        super().__init__()
        self.cfg = cfg

        activation = getattr(nn, cfg.mlp_activation)

        layers = []
        in_dim = cfg.input_feature_dim + 1  # +1 for the prior sdf value
        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.residual_net = nn.Sequential(*layers)

    def get_sdf(self, residual_features: torch.Tensor):
        sdf = self.residual_net(residual_features) * self.cfg.output_sdf_scale
        return sdf

    def forward(self, residual_features: torch.Tensor):
        sdf = self.get_sdf(residual_features)
        return sdf
