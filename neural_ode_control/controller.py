"""Neural network controller with configurable input/output dimensions.

Architecture
    Input:  ``input_dim`` (default 1: normalised time t/T)
    Hidden: configurable depth and width, ELU activations
    Output: ``output_dim`` (1 for scalar, 2+ for vector control)
"""

import torch
import torch.nn as nn


class NeuralController(nn.Module):
    """MLP that parameterises the control law u(t; theta).

    Parameters
    ----------
    input_dim : int
        Number of input features (1 for time-only).
    output_dim : int
        Dimension of the control vector.
    hidden_dim : int
        Neurons per hidden layer.
    n_layers : int
        Number of hidden layers.
    T : float
        Time horizon (used to normalise the time input to [0, 1]).
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 32,
        n_layers: int = 4,
        T: float = 2.0,
    ) -> None:
        super().__init__()
        self.T = T
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()
        self.double()

    def _init_weights(self) -> None:
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute u(t) for one or many time points.

        Parameters
        ----------
        t : Tensor, arbitrary shape
            Time point(s) in ``[0, T]``.

        Returns
        -------
        Tensor
            Shape ``(N,)`` when ``output_dim == 1``,
            shape ``(N, output_dim)`` otherwise.
        """
        t_normalised = (t / self.T).reshape(-1, 1)
        out = self.net(t_normalised)
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out

    @torch.no_grad()
    def get_control_trajectory(self, t_span: torch.Tensor) -> torch.Tensor:
        """Evaluate u(t) over a time grid without tracking gradients."""
        self.eval()
        u = self.forward(t_span)
        self.train()
        return u
