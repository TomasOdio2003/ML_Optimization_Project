import torch
import torch.nn as nn

class MLP16(nn.Module):
    """MLP: d -> 16 -> 16 -> 1"""
    def __init__(self, d: int, width: int = 16, activation: str = "tanh"):
        super().__init__()
        act = nn.Tanh() if activation == "tanh" else nn.ReLU()

        self.net = nn.Sequential(
            nn.Linear(d, width),
            act,
            nn.Linear(width, width),
            act,
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
