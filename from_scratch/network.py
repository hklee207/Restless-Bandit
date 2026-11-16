"""f_theta: state (D, B) -> estimated Whittle index. Sec. 5.1: 2 -> 16 -> 32 -> 1, ReLU.
Input is raw integers (D, B), no scaling (SPEC.md (b) item 11)."""
import torch
import torch.nn as nn


class FTheta(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, s):                      # s: (..., 2) float tensor
        return self.net(s).squeeze(-1)
