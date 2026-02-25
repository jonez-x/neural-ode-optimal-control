"""
controller.py
-------------
Neuronales Netz, das die optimale Steuerung u(t; theta) parameterisiert.

Architektur:
  Input:  t  (1D, normiert auf [0,1])
  Hidden: 4 x 32 Neuronen mit ELU-Aktivierung
  Output: u  (1D, unbeschränkt)
"""

import torch
import torch.nn as nn


class NeuralController(nn.Module):
    """MLP-Controller: t -> u(t).

    Parameters
    ----------
    hidden_dim : int
        Anzahl Neuronen pro Hidden-Layer (Standard: 32).
    n_layers : int
        Anzahl Hidden-Layer (Standard: 4).
    T : float
        Zeithorizont, wird zur Normierung von t genutzt.
    """

    def __init__(self, hidden_dim: int = 32, n_layers: int = 4, T: float = 2.0):
        super().__init__()
        self.T = T

        layers: list[nn.Module] = [nn.Linear(1, hidden_dim), nn.ELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

        # Gewichte mit kleinen Werten initialisieren → sanfter Start
        self._init_weights()

        # float64 für numerische Stabilität (vgl. Spezifikation)
        self.double()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Berechnet u(t).

        Parameters
        ----------
        t : torch.Tensor
            Zeitpunkt(e), Form beliebig.

        Returns
        -------
        torch.Tensor
            Steuerung u, gleiche Lead-Dimension wie t, letzte Dim = 1.
        """
        t_norm = (t / self.T).reshape(-1, 1)   # Normierung auf [0,1]
        return self.net(t_norm).squeeze(-1)     # -> (N,)

    @torch.no_grad()
    def get_control_trajectory(self, t_span: torch.Tensor) -> torch.Tensor:
        """Gibt u(t) für alle Zeitpunkte in t_span zurück (kein Gradient).

        Parameters
        ----------
        t_span : torch.Tensor
            1-D Tensor mit N Zeitpunkten.

        Returns
        -------
        torch.Tensor
            Steuerungstrajektorie, Form (N,).
        """
        self.eval()
        u = self.forward(t_span)
        self.train()
        return u
