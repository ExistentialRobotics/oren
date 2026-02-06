from dataclasses import dataclass

import tinycudann as tcnn
import torch
import torch.nn as nn

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class ResidualNetConfig(ConfigABC):
    mlp_activation: str = "LeakyReLU"  # activation function for the MLP
    hidden_dims: int = 64  # number of hidden dimensions
    n_hidden_layers: int = 5  # number of hidden layers
    output_sdf_scale: float = 0.1  # scale the output SDF
    residual_feature_dim: int = 4


class ResidualNet(nn.Module):
    def __init__(self, cfg: ResidualNetConfig):
        """
        Args:
            cfg: configuration of the network
        """
        super().__init__()
        self.cfg = cfg

        self.residual_net = tcnn.Network(
            n_input_dims=cfg.residual_feature_dim + 3,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": cfg.mlp_activation,
                "output_activation": "None",
                "n_neurons": cfg.hidden_dims,
                "n_hidden_layers": cfg.n_hidden_layers,
            },
        )

    def get_sdf(self, residual_features: torch.Tensor):
        sdf = self.residual_net(residual_features) * self.cfg.output_sdf_scale
        return sdf

    def forward(self, residual_features: torch.Tensor):
        sdf = self.get_sdf(residual_features)
        return sdf
