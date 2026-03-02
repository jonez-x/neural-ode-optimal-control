"""Background training session with thread-safe live updates."""

import threading
from typing import Any

import numpy as np
import torch
from torchdiffeq import odeint

from neural_ode_control.base import ProblemDefinition, TrajectoryResult
from neural_ode_control.controller import NeuralController
from neural_ode_control.trainer import train


class TrainingSession:
    """Manages a training run in a background thread with live data access."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.live_data: dict[str, Any] = {}
        self.is_running = False
        self.is_done = False
        self.result_controller: NeuralController | None = None
        self.result_history: dict[str, list[float]] | None = None
        self.error: str | None = None

    def start(
        self,
        problem: ProblemDefinition,
        controller: NeuralController,
        config: dict[str, Any],
        callback_every: int = 50,
    ) -> None:
        """Launch training in a daemon thread."""
        self.is_running = True
        self.is_done = False
        self.error = None

        def _callback(
            epoch: int,
            total_loss: float,
            components: dict[str, float],
            history: dict[str, list[float]],
        ) -> None:
            with self.lock:
                self.live_data = {
                    "epoch": epoch,
                    "total_loss": total_loss,
                    "components": dict(components),
                    "history": {k: list(v) for k, v in history.items()},
                }

        def _run() -> None:
            try:
                ctrl, hist = train(
                    problem, controller, config,
                    callback=_callback,
                    callback_every=callback_every,
                )
                self.result_controller = ctrl
                self.result_history = hist
            except Exception as e:
                self.error = str(e)
            finally:
                self.is_running = False
                self.is_done = True

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def get_live_data(self) -> dict[str, Any]:
        """Return a snapshot of the latest training data (thread-safe)."""
        with self.lock:
            return dict(self.live_data)


def evaluate_trajectory(
    problem: ProblemDefinition,
    controller: NeuralController,
    config: dict[str, Any],
) -> TrajectoryResult:
    """Run a forward pass and return the trajectory as a TrajectoryResult."""
    dtype = torch.float64
    device = next(controller.parameters()).device

    t_eval = torch.linspace(0.0, config["T"], config["n_eval"], dtype=dtype, device=device)
    z0 = torch.tensor(config["z0"], dtype=dtype, device=device)

    dynamics = problem.create_dynamics(controller, config)

    controller.eval()
    with torch.no_grad():
        z_traj = odeint(
            dynamics, z0, t_eval,
            method=config["solver"], rtol=config["rtol"], atol=config["atol"],
        )
        u_traj = controller.get_control_trajectory(t_eval)

    state_labels = {2: ["x", "v"], 4: ["x", "y", "vx", "vy"]}.get(
        problem.state_dim, [f"z{i}" for i in range(problem.state_dim)],
    )
    control_labels = {1: ["u"], 2: ["Fx", "Fy"]}.get(
        problem.control_dim, [f"u{i}" for i in range(problem.control_dim)],
    )

    return TrajectoryResult(
        t=t_eval.numpy(),
        z=z_traj.numpy(),
        u=u_traj.numpy(),
        state_labels=state_labels,
        control_labels=control_labels,
    )
