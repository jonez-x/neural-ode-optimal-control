"""Training loop for the Neural ODE optimal control problem.

The total loss combines a terminal state penalty with a running
energy penalty on the control signal:

    L = w_terminal * ||z(T) - z*||²  +  w_energy * ∫₀ᵀ u(t)² dt

Gradients are obtained by back-propagating through the ODE solver
(direct mode, not adjoint).
"""

import time
from typing import Any

import torch
import torch.optim as optim
from torchdiffeq import odeint

from controller import NeuralController
from dynamics import BlockDynamics


def compute_loss(
    z_traj: torch.Tensor,
    t_eval: torch.Tensor,
    controller: NeuralController,
    z_target: torch.Tensor,
    w_terminal: float,
    w_energy: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the weighted optimal-control loss and its components.

    The terminal cost is the unweighted squared Euclidean distance
    ``||z(T) - z*||²``.  The energy cost is the trapezoidal-rule
    approximation of ``∫ u² dt`` evaluated at *t_eval*.

    Parameters
    ----------
    z_traj : Tensor, shape (N, 2)
        State trajectory returned by the ODE solver.
    t_eval : Tensor, shape (N,)
        Time grid at which *z_traj* was evaluated.
    controller : NeuralController
        The neural controller whose output is penalised.
    z_target : Tensor, shape (2,)
        Desired terminal state.
    w_terminal : float
        Weight applied to the terminal cost.
    w_energy : float
        Weight applied to the energy cost.

    Returns
    -------
    total_loss : Tensor (scalar)
        ``w_terminal * terminal_cost + w_energy * energy_cost``.
    terminal_cost : Tensor (scalar)
        Unweighted ``||z(T) - z*||²``.
    energy_cost : Tensor (scalar)
        Unweighted ``∫ u² dt`` (trapezoidal approximation).
    """
    terminal_cost = torch.sum((z_traj[-1] - z_target) ** 2)

    u_vals = controller(t_eval)
    u_squared = u_vals ** 2
    dt = t_eval[1:] - t_eval[:-1]
    energy_cost = torch.sum(0.5 * (u_squared[:-1] + u_squared[1:]) * dt)

    total_loss = w_terminal * terminal_cost + w_energy * energy_cost
    return total_loss, terminal_cost, energy_cost


def train(
    controller: NeuralController,
    dynamics: BlockDynamics,
    config: dict[str, Any],
) -> tuple[NeuralController, dict[str, list[float]]]:
    """Run the training loop.

    Early stopping triggers when the Euclidean terminal error
    ``||z(T) - z*||`` drops below ``config["early_stop"]``.  When the
    criterion is met the gradient step is **skipped** so that the
    returned controller corresponds exactly to the state that satisfied
    the criterion.

    Parameters
    ----------
    controller : NeuralController
        Neural network to be trained (modified in-place).
    dynamics : BlockDynamics
        ODE right-hand side that references *controller*.
    config : dict
        Must contain keys: ``z0``, ``z_target``, ``T``, ``n_eval``,
        ``n_epochs``, ``lr``, ``w_terminal``, ``w_energy``,
        ``early_stop``, ``solver``, ``rtol``, ``atol``.

    Returns
    -------
    controller : NeuralController
        The trained controller.
    history : dict[str, list[float]]
        Training curves with keys ``total``, ``terminal``, ``energy``,
        ``epoch``.
    """
    dtype = torch.float64
    device = next(controller.parameters()).device

    t_eval = torch.linspace(0.0, config["T"], config["n_eval"], dtype=dtype, device=device)
    z0 = torch.tensor(config["z0"], dtype=dtype, device=device)
    z_target = torch.tensor(config["z_target"], dtype=dtype, device=device)

    optimizer = optim.Adam(controller.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-6,
    )

    history: dict[str, list[float]] = {
        "total": [], "terminal": [], "energy": [], "epoch": [],
    }

    solver_kwargs: dict[str, Any] = {"method": config["solver"]}
    if config["solver"] == "dopri5":
        solver_kwargs["rtol"] = config["rtol"]
        solver_kwargs["atol"] = config["atol"]

    t_start = time.time()
    print(f"{'Epoch':>8}  {'Total':>12}  {'Terminal':>12}  {'Energy':>12}  {'LR':>10}")
    print("-" * 62)

    for epoch in range(1, config["n_epochs"] + 1):
        optimizer.zero_grad()

        z_traj = odeint(dynamics, z0, t_eval, **solver_kwargs)

        total_loss, terminal_cost, energy_cost = compute_loss(
            z_traj, t_eval, controller, z_target,
            config["w_terminal"], config["w_energy"],
        )

        history["total"].append(total_loss.detach().item())
        history["terminal"].append(terminal_cost.detach().item())
        history["energy"].append(energy_cost.detach().item())
        history["epoch"].append(epoch)

        euclidean_error = torch.sqrt(terminal_cost).item()
        if euclidean_error < config["early_stop"]:
            print(f"\nEarly stopping at epoch {epoch}: ||z(T)-z*|| = {euclidean_error:.2e}")
            break

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step(total_loss.detach())

        if epoch % 100 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            print(
                f"{epoch:>8}  {total_loss.item():>12.6f}  "
                f"{terminal_cost.item():>12.6f}  {energy_cost.item():>12.6f}  "
                f"{lr_now:>10.2e}  [{elapsed:.1f}s]"
            )

    print(f"\nTraining finished. Total time: {time.time() - t_start:.1f}s")
    return controller, history
