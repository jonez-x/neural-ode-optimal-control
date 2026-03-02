"""CLI entry point for the Neural ODE optimal control platform.

Uses the new modular package structure while keeping the original
CLI workflow intact.  Defaults to the 1D block-with-friction problem.

Usage
-----
    python main.py                  # run block_1d (default)
    python main.py obstacle_2d     # run 2D obstacle problem
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
from torchdiffeq import odeint

# Trigger problem registration
import problems  # noqa: F401

from neural_ode_control.controller import NeuralController
from neural_ode_control.registry import get_problem, list_problems
from neural_ode_control.trainer import train


def main(problem_name: str = "block_1d") -> None:
    """Train a neural controller for the given problem and plot results."""
    problem = get_problem(problem_name)
    config = problem.default_config()

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_default_dtype(torch.float64)

    dtype = torch.float64
    device = torch.device("cpu")

    print("=" * 62)
    print(f"  {problem.display_name}")
    print("=" * 62)
    print(f"  z0 = {config['z0']}")
    print(f"  z* = {config['z_target']}")
    print(f"  T  = {config['T']}")
    print("=" * 62)

    controller = NeuralController(
        input_dim=1,
        output_dim=problem.control_dim,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        T=config["T"],
    ).to(dtype=dtype, device=device)

    n_params = sum(p.numel() for p in controller.parameters())
    print(f"\nController parameters: {n_params}")

    print("\n--- Training ---")
    controller, history = train(problem, controller, config)

    # Evaluate final trajectory
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

    z_final = z_traj[-1].numpy()
    target = np.array(config["z_target"])
    error = np.linalg.norm(z_final - target)

    print(f"\n--- Result ---")
    print(f"  Final state:     {z_final}")
    print(f"  Target:          {target}")
    print(f"  Euclidean error: {error:.4e}")
    print(f"  Final loss:      {history['total'][-1]:.6f}")

    # Reference
    reference = None
    if problem.has_reference():
        print("\n--- Reference ---")
        reference = problem.solve_reference(config)
        ref_error = np.linalg.norm(reference.z[-1] - target)
        print(f"  Ref error:       {ref_error:.4e}")
        print(f"  Ref cost:        {reference.cost:.6f}")

    # Plot
    from neural_ode_control.base import TrajectoryResult

    state_labels = {2: ["x", "v"], 4: ["x", "y", "vx", "vy"]}.get(
        problem.state_dim, [f"z{i}" for i in range(problem.state_dim)],
    )
    control_labels = {1: ["u"], 2: ["Fx", "Fy"]}.get(
        problem.control_dim, [f"u{i}" for i in range(problem.control_dim)],
    )

    trajectory = TrajectoryResult(
        t=t_eval.numpy(),
        z=z_traj.numpy(),
        u=u_traj.numpy(),
        state_labels=state_labels,
        control_labels=control_labels,
    )

    fig = problem.plot_results(trajectory, history, config, reference)
    save_path = f"results_{problem_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {save_path}")

    import matplotlib.pyplot as plt
    plt.close(fig)

    print("\n=== Done ===")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "block_1d"
    available = [n for n, _ in list_problems()]
    if name not in available:
        print(f"Unknown problem '{name}'. Available: {available}")
        sys.exit(1)
    main(name)
