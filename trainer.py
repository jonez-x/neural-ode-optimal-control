"""
trainer.py
----------
Training-Loop für das Neural-ODE-Optimal-Control-Problem.

Loss:
  L = w_terminal * ||z(T) - z_target||^2
    + w_energy   * integral_0^T u(t)^2 dt

Das Integral wird mit der Trapezregel über die Solver-Zeitpunkte approximiert.
Gradienten fließen durch den ODE-Solver (backprop-through-solver, kein Adjoint).
"""

import time
from typing import Any

import torch
import torch.optim as optim
from torchdiffeq import odeint

from dynamics import BlockDynamics
from controller import NeuralController


def compute_loss(
    z_traj: torch.Tensor,
    t_eval: torch.Tensor,
    controller: NeuralController,
    z_target: torch.Tensor,
    w_terminal: float,
    w_energy: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Berechnet den Gesamt-Loss und seine Komponenten.

    Parameters
    ----------
    z_traj : torch.Tensor
        Zustandstrajektorie aus odeint, Form (N, 2).
    t_eval : torch.Tensor
        Auswertungszeitpunkte, Form (N,).
    controller : NeuralController
        Trainierter Controller.
    z_target : torch.Tensor
        Zielzustand (2,).
    w_terminal : float
        Gewicht Terminal-Cost.
    w_energy : float
        Gewicht Running-Cost.

    Returns
    -------
    (total_loss, terminal_loss, energy_loss) : Tuple von Skalartensoren
    """
    # --- Terminal Cost ---
    z_T = z_traj[-1]                          # (2,)
    terminal_loss = torch.sum((z_T - z_target) ** 2)

    # --- Running / Energy Cost ---
    # u(t) für alle Solver-Zeitpunkte berechnen (mit Gradient)
    u_traj = controller(t_eval)               # (N,)
    energy_integrand = u_traj ** 2            # (N,)

    # Trapezregel: integral ≈ sum_i 0.5*(f_i + f_{i+1})*(t_{i+1} - t_i)
    dt = t_eval[1:] - t_eval[:-1]            # (N-1,)
    energy_loss = torch.sum(
        0.5 * (energy_integrand[:-1] + energy_integrand[1:]) * dt
    )

    total_loss = w_terminal * terminal_loss + w_energy * energy_loss
    return total_loss, terminal_loss, energy_loss


def train(
    controller: NeuralController,
    dynamics: BlockDynamics,
    config: dict[str, Any],
) -> tuple[NeuralController, dict[str, list[float]]]:
    """Führt den Training-Loop aus.

    Parameters
    ----------
    controller : NeuralController
        Neuronales Netz (wird in-place trainiert).
    dynamics : BlockDynamics
        ODE-Rechte-Seite, enthält controller als Referenz.
    config : dict
        Konfigurationsparameter:
          - z0           : Anfangszustand [x0, v0]
          - z_target     : Zielzustand [xT, vT]
          - T            : Zeithorizont
          - n_eval       : Anzahl Auswertungspunkte
          - n_epochs     : Maximale Epochenzahl
          - lr           : Lernrate
          - w_terminal   : Gewicht Terminal-Cost
          - w_energy     : Gewicht Energie-Cost
          - early_stop   : Terminal-Loss Schwelle für Early-Stopping
          - solver       : ODE-Solver ('dopri5' oder 'rk4')
          - rtol, atol   : Toleranzen für adaptiven Solver

    Returns
    -------
    controller : NeuralController
        Trainierter Controller.
    history : dict
        Loss-Verläufe: 'total', 'terminal', 'energy', 'epoch'.
    """
    dtype = torch.float64
    device = next(controller.parameters()).device

    # Zeitpunkte für ODE-Integration
    t_eval = torch.linspace(0.0, config["T"], config["n_eval"], dtype=dtype, device=device)

    # Anfangs- und Zielzustand
    z0 = torch.tensor(config["z0"], dtype=dtype, device=device)
    z_target = torch.tensor(config["z_target"], dtype=dtype, device=device)

    # Optimizer + LR-Scheduler
    optimizer = optim.Adam(controller.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-6
    )

    history: dict[str, list[float]] = {"total": [], "terminal": [], "energy": [], "epoch": []}

    solver_kwargs: dict[str, Any] = {"method": config["solver"]}
    if config["solver"] == "dopri5":
        solver_kwargs["rtol"] = config["rtol"]
        solver_kwargs["atol"] = config["atol"]

    t0 = time.time()
    print(f"{'Epoche':>8}  {'Total':>12}  {'Terminal':>12}  {'Energie':>12}  {'LR':>10}")
    print("-" * 62)

    for epoch in range(1, config["n_epochs"] + 1):
        optimizer.zero_grad()

        # ODE integrieren – Gradienten fließen durch Solver
        z_traj = odeint(dynamics, z0, t_eval, **solver_kwargs)  # (N, 2)

        # Loss berechnen
        total_loss, terminal_loss, energy_loss = compute_loss(
            z_traj, t_eval, controller, z_target,
            config["w_terminal"], config["w_energy"]
        )

        # Backprop + Update
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step(total_loss.detach())

        # Logging
        history["total"].append(total_loss.item())
        history["terminal"].append(terminal_loss.item())
        history["energy"].append(energy_loss.item())
        history["epoch"].append(epoch)

        if epoch % 100 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"{epoch:>8}  {total_loss.item():>12.6f}  "
                f"{terminal_loss.item():>12.6f}  {energy_loss.item():>12.6f}  "
                f"{lr_now:>10.2e}  [{elapsed:.1f}s]"
            )

        # Early Stopping – Kriterium: euklidischer Fehler ||z(T) - z*|| < early_stop
        eucl_err = torch.sqrt(terminal_loss).item()
        if eucl_err < config["early_stop"]:
            print(f"\nEarly stopping bei Epoche {epoch}: ||z(T)-z*|| = {eucl_err:.2e}")
            break

    print(f"\nTraining abgeschlossen. Gesamtzeit: {time.time() - t0:.1f}s")
    return controller, history
