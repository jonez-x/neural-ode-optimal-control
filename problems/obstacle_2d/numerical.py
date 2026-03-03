"""Numerical reference solution for the 2D obstacle-avoidance problem.

Direct single shooting with exact gradients via torchdiffeq autograd.

The control is parameterised as a piecewise-linear ``ControlTable`` — a
classical direct shooting representation — using the same ODE solver and
cost function as the neural ODE training for a fair comparison.

Two-stage warm start:
  1. Unconstrained (w_obstacle=0): always reaches the target quickly.
  2. Full cost (obstacle penalty re-enabled): deviates around the obstacle,
     starting from the already-converged unconstrained solution.
"""

import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

from neural_ode_control.base import ReferenceResult
from problems.obstacle_2d.dynamics import PointMass2DDynamics
from problems.obstacle_2d.problem import _get_obstacles


class ControlTable(nn.Module):
    """Piecewise-linear control law over a uniform time grid.

    Drop-in replacement for NeuralController: same forward(t) interface,
    but parameterised as N trainable node values instead of MLP weights.
    """

    def __init__(
        self, n_nodes: int, control_dim: int, T: float, dtype: torch.dtype = torch.float64
    ) -> None:
        super().__init__()
        self.T = T
        self.control_dim = control_dim
        self.register_buffer("t_grid", torch.linspace(0.0, T, n_nodes, dtype=dtype))
        self.values = nn.Parameter(torch.zeros(n_nodes, control_dim, dtype=dtype))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate control at time(s) t via piecewise-linear interpolation.

        Parameters
        ----------
        t : Tensor, arbitrary shape
        Returns : Tensor (M, control_dim) where M = t.numel()
        """
        t_flat = t.reshape(-1)
        idx = torch.searchsorted(self.t_grid, t_flat).clamp(1, len(self.t_grid) - 1)
        t0 = self.t_grid[idx - 1]
        t1 = self.t_grid[idx]
        alpha = ((t_flat - t0) / (t1 - t0).clamp(min=1e-12)).clamp(0.0, 1.0)
        v0 = self.values[idx - 1]
        v1 = self.values[idx]
        return (1.0 - alpha.unsqueeze(-1)) * v0 + alpha.unsqueeze(-1) * v1

    @torch.no_grad()
    def get_control_trajectory(self, t_span: torch.Tensor) -> torch.Tensor:
        return self.forward(t_span)


def _make_cost(config: dict, w_obstacle_override: Optional[float] = None):
    """Return a cost function matching Obstacle2DProblem.compute_loss."""
    z_target = torch.tensor(config["z_target"], dtype=torch.float64)
    w_terminal = float(config["w_terminal"])
    w_energy = float(config["w_energy"])
    w_obs = float(config["w_obstacle"]) if w_obstacle_override is None else w_obstacle_override
    obstacles = _get_obstacles(config)

    def cost(z_traj, t_eval, controller):
        terminal_cost = torch.sum((z_traj[-1] - z_target) ** 2)

        u_vals = controller(t_eval)  # (N, 2)
        u_sq = torch.sum(u_vals ** 2, dim=-1)
        dt = t_eval[1:] - t_eval[:-1]
        energy_cost = torch.sum(0.5 * (u_sq[:-1] + u_sq[1:]) * dt)

        positions = z_traj[:, :2]
        obstacle_cost = torch.zeros(1, dtype=z_traj.dtype, device=z_traj.device).squeeze()
        for obs in obstacles:
            c = torch.tensor(obs["center"], dtype=z_traj.dtype, device=z_traj.device)
            r = float(obs["radius"])
            dist = torch.norm(positions - c, dim=-1)
            penetration = torch.clamp(r - dist, min=0.0) ** 2
            obstacle_cost = obstacle_cost + torch.sum(0.5 * (penetration[:-1] + penetration[1:]) * dt)

        total = w_terminal * terminal_cost + w_energy * energy_cost + w_obs * obstacle_cost
        return total, {"terminal": terminal_cost, "energy": energy_cost, "obstacle": obstacle_cost}

    return cost


def _run_stage(
    table: ControlTable,
    dynamics: nn.Module,
    z0: torch.Tensor,
    t_eval: torch.Tensor,
    cost_fn,
    n_epochs: int,
    lr: float,
    early_stop: float,
    solver_kwargs: dict,
    label: str,
    obstacle_stop: Optional[float] = None,
) -> float:
    """Train the ControlTable for one stage.

    Early stopping fires when terminal error < early_stop.
    If obstacle_stop is set, the obstacle cost must also be below that
    threshold before stopping — preventing premature exit in Stage 2.
    """
    optimizer = optim.Adam(table.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-6,
    )

    t0 = time.time()
    print(f"  {label} ...")
    print(f"  {'Epoch':>8}  {'Loss':>12}  {'TermErr':>10}  {'ObsCost':>10}  {'LR':>8}")
    print(f"  {'-' * 58}")

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        z_traj = odeint(dynamics, z0, t_eval, **solver_kwargs)
        loss, components = cost_fn(z_traj, t_eval, table)

        term_err = torch.sqrt(components["terminal"]).item()
        obs_cost = components["obstacle"].item()

        terminal_ok = term_err < early_stop
        obstacle_ok = obstacle_stop is None or obs_cost < obstacle_stop
        if terminal_ok and obstacle_ok:
            print(
                f"\n  Early stop at epoch {epoch}: "
                f"terminal error = {term_err:.2e}, obstacle cost = {obs_cost:.2e}"
            )
            break

        loss.backward()
        nn.utils.clip_grad_norm_(table.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step(loss.detach())

        if epoch % 200 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  {epoch:>8}  {loss.item():>12.6f}  {term_err:>10.2e}"
                f"  {obs_cost:>10.2e}  {lr_now:>8.1e}  [{elapsed:.1f}s]"
            )

    elapsed = time.time() - t0
    print(f"  done  [{elapsed:.1f}s]")
    return elapsed


def solve_reference(config: dict[str, Any], n_nodes: int = 50) -> ReferenceResult:
    """Solve the obstacle-avoidance OCP by direct single shooting with autodiff."""
    dtype = torch.float64
    device = torch.device("cpu")

    z0 = torch.tensor(config["z0"], dtype=dtype, device=device)
    T = float(config["T"])
    t_eval = torch.linspace(0.0, T, config["n_eval"], dtype=dtype, device=device)

    solver_kwargs: dict = {"method": config["solver"]}
    if config["solver"] == "dopri5":
        solver_kwargs["rtol"] = config["rtol"]
        solver_kwargs["atol"] = config["atol"]

    table = ControlTable(n_nodes=n_nodes, control_dim=2, T=T, dtype=dtype).to(device)
    dynamics = PointMass2DDynamics(controller=table, mu=config["mu"])

    print("\n--- Reference (direct single shooting) ---")
    print(f"  Control nodes: {n_nodes}  |  Parameters: {n_nodes * 2}")

    # ── Stage 1: unconstrained warm start ─────────────────────────────────
    t_warm = _run_stage(
        table, dynamics, z0, t_eval,
        cost_fn=_make_cost(config, w_obstacle_override=0.0),
        n_epochs=3000, lr=1e-2, early_stop=1e-3,
        solver_kwargs=solver_kwargs,
        label="Stage 1/2: warm start (unconstrained)",
    )

    # ── Symmetry-breaking nudge ────────────────────────────────────────────
    # When the unconstrained trajectory passes through an obstacle centre the
    # penalty gradient is (pos - centre)/dist = 0/0 → zero.  A small Fx bias
    # deflects the path slightly so Stage 2 has a well-defined gradient.
    with torch.no_grad():
        table.values[:, 0] += 0.05

    # ── Stage 2: full cost with obstacle penalty ───────────────────────────
    t_full = _run_stage(
        table, dynamics, z0, t_eval,
        cost_fn=_make_cost(config),
        n_epochs=5000, lr=5e-3, early_stop=config["early_stop"],
        solver_kwargs=solver_kwargs,
        label="Stage 2/2: full solve (with obstacle penalty)",
        obstacle_stop=1e-4,
    )

    print(f"  Total reference time: {t_warm + t_full:.1f}s")

    # ── Extract final trajectory ───────────────────────────────────────────
    table.eval()
    with torch.no_grad():
        z_traj = odeint(dynamics, z0, t_eval, **solver_kwargs)
        u_traj = table.get_control_trajectory(t_eval)

    z_np = z_traj.numpy()
    u_np = u_traj.numpy()
    t_np = t_eval.numpy()
    z_target_np = np.array(config["z_target"])
    obs_c_np = np.array(config["obstacle_center"])
    obs_r = float(config["obstacle_radius"])
    dt = t_np[1:] - t_np[:-1]

    terminal = float(np.sum((z_np[-1] - z_target_np) ** 2))
    u_sq = np.sum(u_np ** 2, axis=1)
    energy = float(np.sum(0.5 * (u_sq[:-1] + u_sq[1:]) * dt))
    dist = np.linalg.norm(z_np[:, :2] - obs_c_np, axis=1)
    penetration = np.maximum(0.0, obs_r - dist) ** 2
    obstacle_val = float(np.sum(0.5 * (penetration[:-1] + penetration[1:]) * dt))
    total_cost = (
        config["w_terminal"] * terminal
        + config["w_energy"] * energy
        + config["w_obstacle"] * obstacle_val
    )

    return ReferenceResult(
        t=t_np,
        z=z_np,
        u=u_np,
        state_labels=["x", "y", "vx", "vy"],
        control_labels=["Fx", "Fy"],
        cost=total_cost,
        converged=True,
    )
