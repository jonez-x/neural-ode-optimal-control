"""ODE right-hand side for a 2-D point mass with friction.

State vector z = (x, y, vx, vy):
    dx/dt  = vx
    dy/dt  = vy
    dvx/dt = Fx - mu * vx
    dvy/dt = Fy - mu * vy
"""

import torch
import torch.nn as nn


class PointMass2DDynamics(nn.Module):
    """ODE right-hand side: dz/dt = f(t, z) for a 2D point mass."""

    def __init__(self, controller: nn.Module, mu: float = 0.3) -> None:
        super().__init__()
        self.controller = controller
        self.mu = mu

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Evaluate dz/dt = [vx, vy, Fx - mu*vx, Fy - mu*vy]."""
        vx = z[..., 2:3]
        vy = z[..., 3:4]

        u = self.controller(t.reshape(1).to(z.dtype))  # (1, 2)
        fx = u[0, 0].reshape(1)
        fy = u[0, 1].reshape(1)

        dx_dt = vx
        dy_dt = vy
        dvx_dt = fx - self.mu * vx
        dvy_dt = fy - self.mu * vy
        return torch.cat([dx_dt, dy_dt, dvx_dt, dvy_dt], dim=-1)
