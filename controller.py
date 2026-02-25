"""Neural network controller that maps time to a control signal.

Architecture
    Input:  t  (scalar, normalised to [0, 1] via t/T)
    Hidden: configurable depth and width, ELU activations
    Output: u  (scalar, unbounded)
"""

import torch
import torch.nn as nn


class NeuralController(nn.Module):
    """MLP that parameterises the open-loop control law u(t; theta).

    Parameters
    ----------
    hidden_dim : int
        Number of neurons per hidden layer.
    n_layers : int
        Number of hidden layers.
    T : float
        Time horizon, used to normalise the input to [0, 1].
    """

    def __init__(self, hidden_dim: int = 32, n_layers: int = 4, T: float = 2.0) -> None:
        super().__init__()
        self.T = T

        layers: list[nn.Module] = [nn.Linear(1, hidden_dim), nn.ELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()
        self.double()

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation with reduced gain for a gentle start."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute u(t) for one or many time points.

        Parameters
        ----------
        t : Tensor, arbitrary shape
            Time point(s) in [0, T].

        Returns
        -------
        Tensor, shape ``t.shape``
            Control value(s).
        """
        t_normalised = (t / self.T).reshape(-1, 1)
        return self.net(t_normalised).squeeze(-1)

    @torch.no_grad()
    def get_control_trajectory(self, t_span: torch.Tensor) -> torch.Tensor:
        """Evaluate u(t) over a time grid without tracking gradients.

        Parameters
        ----------
        t_span : Tensor, shape (N,)
            Time points at which to evaluate the controller.

        Returns
        -------
        Tensor, shape (N,)
            Control values.
        """
        self.eval()
        u = self.forward(t_span)
        self.train()
        return u
