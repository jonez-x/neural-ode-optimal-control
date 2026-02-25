"""
dynamics.py
-----------
Systemdynamik des 1D-Block-mit-Reibung Problems.

Zustandsvektor z = (x, v):
  dx/dt = v
  dv/dt = u(t) - mu * v

Der Controller wird als Argument übergeben und vom ODE-Solver
bei jedem Zeitschritt ausgewertet.
"""

import torch
import torch.nn as nn


class BlockDynamics(nn.Module):
    """ODE-Rechte-Seite für das Block-Reibungs-System.

    Parameters
    ----------
    controller : nn.Module
        Neuronales Netz, das t -> u(t) abbildet.
    mu : float
        Reibungskoeffizient (Standard: 0.5).
    """

    def __init__(self, controller: nn.Module, mu: float = 0.5):
        super().__init__()
        self.controller = controller
        self.mu = mu

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Berechnet dz/dt = f(t, z).

        Parameters
        ----------
        t : torch.Tensor
            Aktueller Zeitpunkt, Skalar.
        z : torch.Tensor
            Zustandsvektor der Form (batch, 2) oder (2,).

        Returns
        -------
        torch.Tensor
            Zeitableitung dz/dt, gleiche Form wie z.
        """
        # Zustand aufteilen
        x = z[..., 0:1]  # Position  (... x 1)
        v = z[..., 1:2]  # Geschwindigkeit (... x 1)

        # Steuerung: u = u(t; theta)
        # t muss als (1,) Tensor vorliegen damit der Controller ihn verarbeiten kann
        t_in = t.reshape(1).to(z.dtype)
        u = self.controller(t_in)          # (1,) oder (1, 1)
        u = u.reshape(1)                   # sicherstellen: (1,)

        # ODE
        dx_dt = v
        dv_dt = u - self.mu * v

        return torch.cat([dx_dt, dv_dt], dim=-1)
