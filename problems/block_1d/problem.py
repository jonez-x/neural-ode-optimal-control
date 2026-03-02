"""1D Block with Friction – problem definition."""

from typing import Any, Optional

import matplotlib.figure
import torch
import torch.nn as nn

from neural_ode_control.base import (
    ProblemDefinition,
    ReferenceResult,
    TrajectoryResult,
)
from neural_ode_control.registry import register_problem

from problems.block_1d.analytical import solve_reference as _solve_reference
from problems.block_1d.dynamics import BlockDynamics
from problems.block_1d.visualization import plot_results as _plot_results


@register_problem
class Block1DProblem(ProblemDefinition):

    @property
    def name(self) -> str:
        return "block_1d"

    @property
    def display_name(self) -> str:
        return "1D Block with Friction"

    @property
    def description(self) -> str:
        return (
            r"Linear system with viscous friction: "
            r"$\dot{x}=v,\; \dot{v}=u - \mu v$. "
            r"Find the control $u(t)$ that drives the block from $z_0$ to $z^*$ "
            r"while minimising a terminal + energy cost."
        )

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def control_dim(self) -> int:
        return 1

    def default_config(self) -> dict[str, Any]:
        return {
            "z0": [0.0, 1.0],
            "z_target": [1.0, 1.0],
            "T": 2.0,
            "mu": 0.5,
            "hidden_dim": 32,
            "n_layers": 4,
            "w_terminal": 100.0,
            "w_energy": 0.01,
            "lr": 1e-3,
            "n_epochs": 3000,
            "early_stop": 1e-3,
            "solver": "dopri5",
            "n_eval": 200,
            "rtol": 1e-7,
            "atol": 1e-9,
            "seed": 42,
        }

    def create_dynamics(self, controller: nn.Module, config: dict) -> nn.Module:
        return BlockDynamics(controller=controller, mu=config["mu"])

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
        terminal_cost = torch.sum((z_traj[-1] - z_target) ** 2)

        u_vals = controller(t_eval)
        u_squared = u_vals ** 2
        dt = t_eval[1:] - t_eval[:-1]
        energy_cost = torch.sum(0.5 * (u_squared[:-1] + u_squared[1:]) * dt)

        total = config["w_terminal"] * terminal_cost + config["w_energy"] * energy_cost
        return total, {"terminal": terminal_cost, "energy": energy_cost}

    def has_reference(self) -> bool:
        return True

    def solve_reference(self, config: dict) -> ReferenceResult:
        return _solve_reference(config, n_nodes=config.get("n_eval", 200))

    def plot_results(
        self,
        trajectory: TrajectoryResult,
        history: dict[str, list[float]],
        config: dict,
        reference: Optional[ReferenceResult] = None,
    ) -> matplotlib.figure.Figure:
        return _plot_results(trajectory, history, config, reference)
