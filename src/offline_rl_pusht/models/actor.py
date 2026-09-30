from __future__ import annotations
import torch
from torch import nn


class MLPActor(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | tuple[int, ...] = (128, 128),
        output_dim: int = 2,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
