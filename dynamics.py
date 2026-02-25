"""ODE right-hand side for the 1-D block-with-friction system.

State vector z = (x, v):
    dx/dt = v
    dv/dt = u(t) - mu * v

The neural controller is evaluated at each solver time step to
provide the control input u(t).
"""

import torch
import torch.nn as nn


class BlockDynamics(nn.Module):
    """ODE right-hand side: dz/dt = f(t, z).

    Parameters
    ----------
    controller : nn.Module
        Maps a scalar time ``t`` to a scalar control ``u(t)``.
    mu : float
        Viscous friction coefficient.
    """

    def __init__(self, controller: nn.Module, mu: float = 0.5) -> None:
        super().__init__()
        self.controller = controller
        self.mu = mu

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Evaluate dz/dt = [v, u(t) - mu * v].

        Parameters
        ----------
        t : Tensor (scalar)
            Current time.
        z : Tensor, shape (..., 2)
            Current state ``[x, v]``.

        Returns
        -------
        Tensor, same shape as *z*
            Time derivative ``[dx/dt, dv/dt]``.
        """
        v = z[..., 1:2]
        u = self.controller(t.reshape(1).to(z.dtype)).reshape(1)

        dx_dt = v
        dv_dt = u - self.mu * v
        return torch.cat([dx_dt, dv_dt], dim=-1)
