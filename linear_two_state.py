"""
AI Pontryagin – Linear Two-State System
========================================
Reproduces the two-node example from the paper (Fig. 2):

  A = [[1, 0],      B = [[1],
       [1, 0]]           [0]]

  x(0) = (1, 0.5),  x* = (0, 0),  T = 1

The neural network learns u(t) by minimising only the terminal MSE loss:
  J = (1/N) * ||x(T) - x*||^2

No energy term in the loss. The energy regularisation is implicit.

Run:
  python linear_two_state.py
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves to file instead of opening a window
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import numpy as np


# ---------------------------------------------------------------------------
# System definition  (paper Eq. 8)
# ---------------------------------------------------------------------------

A = torch.tensor([[1.0, 0.0],
                  [1.0, 0.0]])

B = torch.tensor([[1.0],
                  [0.0]])

x0     = torch.tensor([1.0, 0.5])   # initial state
x_star = torch.tensor([0.0, 0.0])   # target state
T      = 1.0                         # control horizon
N_NODES = 2


# ---------------------------------------------------------------------------
# Neural network controller:  t (scalar) --> u(t) (scalar)
# Paper uses 1 hidden layer with 6 ELU units (Methods section)
# ---------------------------------------------------------------------------

class NeuralController(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),   # single control input
        )
        # Kaiming uniform init (paper Methods)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, t):
        # t is a scalar tensor; reshape to [1] for the linear layer
        return self.net(t.reshape(1))


# ---------------------------------------------------------------------------
# Coupled ODE:  dx/dt = Ax + Bu(t)
# torchdiffeq expects f(t, x) with x shape [N]
# ---------------------------------------------------------------------------

def make_dynamics(controller):
    def dynamics(t, x):
        u = controller(t)                 # shape [1]
        return A @ x + (B @ u).squeeze()  # shape [2]
    return dynamics


# ---------------------------------------------------------------------------
# Optimal control baseline  (paper Eq. 6-7)
# u*(t) = B^T e^{A^T(T-t)} W(T)^{-1} v(T)
# W(T) = integral_0^T e^{At} B B^T e^{A^T t} dt   (Simpson's rule)
# v(T) = x* - e^{AT} x0
# ---------------------------------------------------------------------------

def matrix_exp(M, t):
    """Compute matrix exponential e^{Mt} via torch.linalg.matrix_exp."""
    return torch.linalg.matrix_exp(M * t)


def compute_gramian(n_steps=1000):
    """Numerically integrate W(T) using Simpson's rule."""
    ts = torch.linspace(0, T, n_steps)
    dt = ts[1] - ts[0]
    W = torch.zeros(N_NODES, N_NODES)
    for i, t in enumerate(ts):
        eAt  = matrix_exp(A, t)
        eBBt = eAt @ B @ B.T @ eAt.T
        # Simpson weights: 1, 4, 2, 4, ..., 1
        if i == 0 or i == n_steps - 1:
            w = 1
        elif i % 2 == 1:
            w = 4
        else:
            w = 2
        W = W + w * eBBt
    return W * dt / 3.0


def optimal_control(t_tensor):
    """Returns u*(t) for each t in t_tensor, shape [len(t_tensor)]."""
    W     = compute_gramian()
    eAT   = matrix_exp(A, T)
    v     = x_star - eAT @ x0          # v(T) = x* - e^{AT} x0
    W_inv = torch.linalg.inv(W)

    us = []
    for t in t_tensor:
        eAt_adj = matrix_exp(A.T, T - t)   # e^{A^T (T-t)}
        u = B.T @ eAt_adj @ W_inv @ v      # shape [1, 1]
        us.append(u.item())
    return torch.tensor(us)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_epochs=30_000, lr=0.02, n_timesteps=40):
    controller = NeuralController(hidden_size=6)
    optimizer  = torch.optim.Adam(controller.parameters(), lr=lr)
    t_span     = torch.linspace(0, T, n_timesteps)

    loss_history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Integrate the controlled ODE
        x_traj = odeint(make_dynamics(controller), x0, t_span,
                        method='rk4')   # shape [n_timesteps, 2]

        x_final = x_traj[-1]              # x(T)

        # Terminal MSE loss  (paper Eq. 5) — NO energy term
        loss = (1 / N_NODES) * torch.sum((x_final - x_star) ** 2)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 5000 == 0:
            print(f"Epoch {epoch+1:6d}  |  loss = {loss.item():.6f}")

    return controller, t_span, loss_history


# ---------------------------------------------------------------------------
# Plotting  (reproduces Fig. 2a and 2b)
# ---------------------------------------------------------------------------

def plot_results(controller, t_span, loss_history, snapshot_epochs,
                 snapshot_controllers):
    t_span_np = t_span.detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- Fig 2a: phase portrait ---
    ax = axes[0]
    ax.set_title("Phase portrait  (Fig. 2a)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    colors = ["tab:blue", "tab:purple", "tab:red", "tab:orange"]
    labels = [f"{e} epochs" for e in snapshot_epochs]

    for ctrl, color, label in zip(snapshot_controllers, colors, labels):
        with torch.no_grad():
            traj = odeint(make_dynamics(ctrl), x0, t_span, method='rk4')
        traj_np = traj.detach().numpy()
        ax.plot(traj_np[:, 0], traj_np[:, 1], color=color, label=label)

    # Optimal control trajectory
    with torch.no_grad():
        def oc_dynamics(t, x):
            u_val = optimal_control(t.unsqueeze(0))
            return A @ x + (B * u_val).squeeze()

        traj_oc = odeint(oc_dynamics, x0, t_span, method='rk4')
    ax.plot(traj_oc[:, 0].numpy(), traj_oc[:, 1].numpy(),
            'k--', label='Optimal control', linewidth=2)

    ax.scatter(*x0.numpy(),    color='green', zorder=5, label='x(0)')
    ax.scatter(*x_star.numpy(), color='red',   zorder=5, label='x*', marker='*', s=150)
    ax.legend(fontsize=7)

    # --- Fig 2b: control energy over time ---
    ax = axes[1]
    ax.set_title("Control energy  (Fig. 2b)")
    ax.set_xlabel("t")
    ax.set_ylabel("Eₜ[u]")

    with torch.no_grad():
        u_final = torch.stack([controller(t) for t in t_span]).squeeze()
        # cumulative energy: integral of u^2 (trapezoidal)
        energy_nn = torch.cumsum(u_final ** 2, dim=0) * (T / len(t_span))

    u_oc = optimal_control(t_span)
    energy_oc = torch.cumsum(u_oc ** 2, dim=0) * (T / len(t_span))

    ax.plot(t_span_np, energy_nn.numpy(), color='tab:orange', label='AI Pontryagin')
    ax.plot(t_span_np, energy_oc.numpy(), 'k--', label='Optimal control')
    ax.legend()

    # --- Training loss ---
    ax = axes[2]
    ax.set_title("Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.semilogy(loss_history)

    plt.tight_layout()
    plt.savefig("results_linear_two_state.png", dpi=150)
    print("Saved: results_linear_two_state.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_EPOCHS   = 5_000
    SNAPSHOTS  = [50, 150, 300]   # epochs at which to save a trajectory snapshot

    snapshot_controllers = []
    snapshot_epochs_done = []

    print("Training AI Pontryagin on the linear two-state system...")
    print(f"  x(0) = {x0.tolist()},  x* = {x_star.tolist()},  T = {T}")
    print(f"  {N_EPOCHS} epochs, lr = 0.02\n")

    controller = NeuralController(hidden_size=6)
    optimizer  = torch.optim.Adam(controller.parameters(), lr=0.02)
    t_span     = torch.linspace(0, T, 40)
    loss_history = []

    import copy

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        x_traj = odeint(make_dynamics(controller), x0, t_span, method='rk4')
        x_final = x_traj[-1]
        loss = (1 / N_NODES) * torch.sum((x_final - x_star) ** 2)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (epoch + 1) in SNAPSHOTS:
            snapshot_controllers.append(copy.deepcopy(controller))
            snapshot_epochs_done.append(epoch + 1)
            print(f"  Snapshot saved at epoch {epoch+1}")

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:6d}  |  loss = {loss.item():.6f}")

    # Add final controller as last snapshot
    snapshot_controllers.append(copy.deepcopy(controller))
    snapshot_epochs_done.append(N_EPOCHS)

    print(f"\nFinal loss: {loss_history[-1]:.6f}")
    print("Plotting...")

    plot_results(controller, t_span, loss_history,
                 snapshot_epochs_done, snapshot_controllers)
