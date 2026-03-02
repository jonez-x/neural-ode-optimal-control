"""ODE right-hand side for the 1-D block-with-friction system.

State vector z = (x, v):
    dx/dt = v
    dv/dt = u(t) - mu * v
"""

import torch
import torch.nn as nn


class BlockDynamics(nn.Module):
    """ODE right-hand side: dz/dt = f(t, z)."""

    def __init__(self, controller: nn.Module, mu: float = 0.5) -> None:
        super().__init__()
        self.controller = controller
        self.mu = mu

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        v = z[..., 1:2]
        u = self.controller(t.reshape(1).to(z.dtype)).reshape(1)

        dx_dt = v
        dv_dt = u - self.mu * v
        return torch.cat([dx_dt, dv_dt], dim=-1)
