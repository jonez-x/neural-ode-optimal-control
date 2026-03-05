import copy
import torch
from torchdiffeq import odeint
from .system import x0, x_star, T, N_NODES
from .controller import NeuralController, make_dynamics


def train(n_epochs=5_000, lr=0.02, n_timesteps=40, snapshots=(50, 150, 300)):
    controller   = NeuralController(hidden_size=6)
    optimizer    = torch.optim.Adam(controller.parameters(), lr=lr)
    t_span       = torch.linspace(0, T, n_timesteps)
    loss_history = []

    snapshot_controllers = []
    snapshot_epochs      = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        x_traj  = odeint(make_dynamics(controller), x0, t_span, method='rk4')
        x_final = x_traj[-1]
        loss    = (1 / N_NODES) * torch.sum((x_final - x_star) ** 2)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (epoch + 1) in snapshots:
            snapshot_controllers.append(copy.deepcopy(controller))
            snapshot_epochs.append(epoch + 1)
            print(f"  Snapshot saved at epoch {epoch + 1}")

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch + 1:6d}  |  loss = {loss.item():.6f}")

    snapshot_controllers.append(copy.deepcopy(controller))
    snapshot_epochs.append(n_epochs)

    return controller, t_span, loss_history, snapshot_controllers, snapshot_epochs
