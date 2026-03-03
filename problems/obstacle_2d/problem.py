"""2D Point Mass with Obstacle Avoidance – problem definition."""

from typing import Any, Optional

import matplotlib.figure
import torch
import torch.nn as nn

from neural_ode_control.base import ProblemDefinition, ReferenceResult, TrajectoryResult
from neural_ode_control.registry import register_problem

from problems.obstacle_2d.dynamics import PointMass2DDynamics


def _get_obstacles(config: dict) -> list[dict]:
    """Return a list of obstacle dicts from config.

    Supports both the legacy single-obstacle format
    (``obstacle_center`` / ``obstacle_radius``) and the new multi-obstacle
    format (``obstacles``: list of ``{"center": [...], "radius": float}``).
    """
    if config.get("obstacles"):
        return config["obstacles"]
    return [{"center": config["obstacle_center"], "radius": config["obstacle_radius"]}]
from problems.obstacle_2d.numerical import solve_reference as _solve_reference
from problems.obstacle_2d.visualization import plot_results as _plot_results


@register_problem
class Obstacle2DProblem(ProblemDefinition):

    @property
    def name(self) -> str:
        return "obstacle_2d"

    @property
    def display_name(self) -> str:
        return "2D Point Mass with Obstacle"

    @property
    def description(self) -> str:
        return (
            r"2D point mass with friction: "
            r"$\dot{x}=v_x,\; \dot{v}_x=F_x - \mu v_x$ (same for $y$). "
            r"Steer from $z_0$ to $z^*$ while avoiding a circular obstacle."
        )

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def control_dim(self) -> int:
        return 2

    def default_config(self) -> dict[str, Any]:
        return {
            "z0": [0.0, 0.0, 0.0, 0.0],
            "z_target": [2.0, 2.0, 0.0, 0.0],
            "T": 3.0,
            "mu": 0.3,
            "obstacle_center": [1.0, 1.0],
            "obstacle_radius": 0.3,
            "w_terminal": 10.0,
            "w_energy": 0.001,
            "w_obstacle": 10000.0,
            "hidden_dim": 64,
            "n_layers": 5,
            "lr": 1e-3,
            "n_epochs": 5000,
            "early_stop": 5e-3,
            "solver": "dopri5",
            "n_eval": 300,
            "rtol": 1e-7,
            "atol": 1e-9,
            "seed": 42,
        }

    def create_dynamics(self, controller: nn.Module, config: dict) -> nn.Module:
        return PointMass2DDynamics(controller=controller, mu=config["mu"])

    def compute_loss(
        self,
        z_traj: torch.Tensor,
        t_eval: torch.Tensor,
        controller: nn.Module,
        config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z_target = torch.tensor(
            config["z_target"], dtype=z_traj.dtype, device=z_traj.device,
        )

        # Terminal cost
        terminal_cost = torch.sum((z_traj[-1] - z_target) ** 2)

        # Energy cost (trapezoidal integration of ||u||^2)
        u_vals = controller(t_eval)  # (N, 2)
        u_squared = torch.sum(u_vals ** 2, dim=-1)  # (N,)
        dt = t_eval[1:] - t_eval[:-1]
        energy_cost = torch.sum(0.5 * (u_squared[:-1] + u_squared[1:]) * dt)

        # Obstacle penalty: sum over all obstacles
        positions = z_traj[:, :2]  # (N, 2)
        obstacle_cost = torch.zeros(1, dtype=z_traj.dtype, device=z_traj.device).squeeze()
        for obs in _get_obstacles(config):
            obs_c = torch.tensor(obs["center"], dtype=z_traj.dtype, device=z_traj.device)
            obs_r = float(obs["radius"])
            dist = torch.norm(positions - obs_c, dim=-1)
            penetration = torch.clamp(obs_r - dist, min=0.0) ** 2
            obstacle_cost = obstacle_cost + torch.sum(0.5 * (penetration[:-1] + penetration[1:]) * dt)

        total = (
            config["w_terminal"] * terminal_cost
            + config["w_energy"] * energy_cost
            + config["w_obstacle"] * obstacle_cost
        )

        return total, {
            "terminal": terminal_cost,
            "energy": energy_cost,
            "obstacle": obstacle_cost,
        }

    def has_reference(self) -> bool:
        return True

    def solve_reference(self, config: dict) -> ReferenceResult:
        return _solve_reference(config)

    def plot_results(
        self,
        trajectory: TrajectoryResult,
        history: dict[str, list[float]],
        config: dict,
        reference: Optional[ReferenceResult] = None,
    ) -> matplotlib.figure.Figure:
        return _plot_results(trajectory, history, config, reference)
