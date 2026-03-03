"""Quick-preview experiments for the 2D obstacle problem.

Runs several configurations with reduced epochs for fast visual comparison.
Results are saved as results_exp_<name>.png — no reference solution is
computed to keep each run short.

Usage:
    python run_experiments.py
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
from torchdiffeq import odeint

import problems  # noqa: F401

from neural_ode_control.base import TrajectoryResult
from neural_ode_control.controller import NeuralController
from neural_ode_control.registry import get_problem
from neural_ode_control.trainer import train

# ── Experiment definitions ────────────────────────────────────────────────────
# Each entry overrides fields of the default obstacle_2d config.
# "obstacles" (list) activates multi-obstacle mode.
# Absent keys fall back to the problem default.

EXPERIMENTS: dict[str, dict] = {
    # Single obstacle – vary radius
    "small_r15": {
        "obstacles": [{"center": [1.0, 1.0], "radius": 0.15}],
    },
    "large_r45": {
        "obstacles": [{"center": [1.0, 1.0], "radius": 0.45}],
    },
    # Single obstacle – off the diagonal
    "below_path": {
        "obstacles": [{"center": [1.3, 0.7], "radius": 0.3}],
    },
    "above_path": {
        "obstacles": [{"center": [0.7, 1.3], "radius": 0.3}],
    },
    # Two obstacles in a line along the diagonal
    "two_in_line": {
        "obstacles": [
            {"center": [0.75, 0.75], "radius": 0.2},
            {"center": [1.25, 1.25], "radius": 0.2},
        ],
    },
    # Two obstacles flanking the path (narrow corridor)
    "corridor": {
        "obstacles": [
            {"center": [1.3, 0.7], "radius": 0.25},
            {"center": [0.7, 1.3], "radius": 0.25},
        ],
    },
    # Three obstacles – triangle
    "triangle": {
        "obstacles": [
            {"center": [1.0, 0.7], "radius": 0.2},
            {"center": [0.7, 1.2], "radius": 0.2},
            {"center": [1.3, 1.2], "radius": 0.2},
        ],
    },
}

# Quick training overrides (faster previews)
QUICK = {
    "n_epochs": 2000,
    "early_stop": 0.05,
}


def run_experiment(name: str, overrides: dict) -> None:
    print(f"\n{'=' * 62}")
    print(f"  Experiment: {name}")
    print(f"{'=' * 62}")

    problem = get_problem("obstacle_2d")
    config = problem.default_config()
    config.update(QUICK)
    config.update(overrides)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_default_dtype(torch.float64)

    dtype = torch.float64
    device = torch.device("cpu")

    controller = NeuralController(
        input_dim=1,
        output_dim=problem.control_dim,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        T=config["T"],
    ).to(dtype=dtype, device=device)

    t_eval = torch.linspace(0.0, config["T"], config["n_eval"], dtype=dtype, device=device)
    z0 = torch.tensor(config["z0"], dtype=dtype, device=device)
    dynamics = problem.create_dynamics(controller, config)

    solver_kwargs = {"method": config["solver"], "rtol": config["rtol"], "atol": config["atol"]}

    # Callback: capture the first trajectory that gets close to the target
    first_close_z = [None]

    def _capture_first_close(epoch, total_loss, components, hist):
        if first_close_z[0] is None and components.get("terminal", float("inf")) ** 0.5 < 0.3:
            controller.eval()
            with torch.no_grad():
                z_snap = odeint(dynamics, z0, t_eval, **solver_kwargs)
            controller.train()
            first_close_z[0] = z_snap.numpy()

    controller, history = train(
        problem, controller, config,
        callback=_capture_first_close, callback_every=50,
    )
    if first_close_z[0] is not None:
        history["first_close_z"] = first_close_z[0]

    controller.eval()
    with torch.no_grad():
        z_traj = odeint(
            dynamics, z0, t_eval,
            method=config["solver"], rtol=config["rtol"], atol=config["atol"],
        )
        u_traj = controller.get_control_trajectory(t_eval)

    trajectory = TrajectoryResult(
        t=t_eval.numpy(),
        z=z_traj.numpy(),
        u=u_traj.numpy(),
        state_labels=["x", "y", "vx", "vy"],
        control_labels=["Fx", "Fy"],
    )

    fig = problem.plot_results(trajectory, history, config, reference=None)
    save_path = f"results_exp_{name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {save_path}")

    import matplotlib.pyplot as plt
    plt.close(fig)

    z_final = z_traj[-1].numpy()
    target = np.array(config["z_target"])
    error = np.linalg.norm(z_final - target)
    print(f"  Final error: {error:.4e}  |  Final loss: {history['total'][-1]:.4f}")


if __name__ == "__main__":
    names = list(EXPERIMENTS.keys())
    print(f"Running {len(names)} experiments: {names}")
    for name, overrides in EXPERIMENTS.items():
        try:
            run_experiment(name, overrides)
        except Exception as e:
            print(f"\n  ERROR in '{name}': {e}")
    print("\n=== All experiments done ===")
