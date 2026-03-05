import torch.nn as nn
from .system import A, B


class NeuralController(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, t):
        return self.net(t.reshape(1))


def make_dynamics(controller):
    def dynamics(t, x):
        u = controller(t)
        return A @ x + (B @ u).squeeze()
    return dynamics
