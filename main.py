"""Entry point for training a Neural ODE optimal controller.

Solves the linear block-with-friction optimal control problem:
    dx/dt = v
    dv/dt = u(t) - mu * v

by parameterising the control u(t) as a neural network and
differentiating through an ODE solver (backprop-through-solver).

A closed-form LQR reference solution is computed for comparison.

Usage
-----
    pip install -r requirements.txt
    python main.py
"""

import numpy as np
import torch
from torchdiffeq import odeint

from analytical import solve_reference
from controller import NeuralController
from dynamics import BlockDynamics
from trainer import train
from visualization import plot_results

CONFIG: dict = {
    # System
    "z0": [0.0, 1.0],
    "z_target": [1.0, 1.0],
    "T": 2.0,
    "mu": 0.5,
    # Network
    "hidden_dim": 32,
    "n_layers": 4,
    # Loss weights
    "w_terminal": 100.0,
    "w_energy": 0.01,
    # Training
    "lr": 1e-3,
    "n_epochs": 3000,
    "early_stop": 1e-3,
    # ODE solver
    "solver": "dopri5",
    "n_eval": 200,
    "rtol": 1e-7,
    "atol": 1e-9,
    # Misc
    "seed": 42,
    "save_path": "results.png",
}


def main() -> None:
    """Train the neural controller, compute the LQR reference, and plot."""
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.set_default_dtype(torch.float64)

    device = torch.device("cpu")
    dtype = torch.float64

    _print_header(CONFIG)

    controller = NeuralController(
        hidden_dim=CONFIG["hidden_dim"],
        n_layers=CONFIG["n_layers"],
        T=CONFIG["T"],
    ).to(dtype=dtype, device=device)

    dynamics = BlockDynamics(controller=controller, mu=CONFIG["mu"])

    n_params = sum(p.numel() for p in controller.parameters())
    print(f"\nController parameters: {n_params}")

    print("\n--- Training ---")
    controller, history = train(controller, dynamics, CONFIG)

    t_eval = torch.linspace(0.0, CONFIG["T"], CONFIG["n_eval"], dtype=dtype, device=device)
    z0_tensor = torch.tensor(CONFIG["z0"], dtype=dtype, device=device)

    controller.eval()
    with torch.no_grad():
        z_traj = odeint(
            dynamics,
            z0_tensor,
            t_eval,
            method=CONFIG["solver"],
            rtol=CONFIG["rtol"],
            atol=CONFIG["atol"],
        )
        u_traj = controller.get_control_trajectory(t_eval)

    t_np = t_eval.numpy()
    z_np = z_traj.numpy()
    u_np = u_traj.numpy()

    z_final = z_np[-1]
    target = np.array(CONFIG["z_target"])
    error = np.linalg.norm(z_final - target)

    print("\n--- Neural ODE result ---")
    print(f"  Final state:       x(T) = {z_final[0]:.6f}, v(T) = {z_final[1]:.6f}")
    print(f"  Target:            x*   = {target[0]}, v* = {target[1]}")
    print(f"  Euclidean error:   {error:.4e}")
    print(f"  Final total loss:  {history['total'][-1]:.6f}")

    print("\n--- Reference solution ---")
    ref = solve_reference(CONFIG, n_nodes=CONFIG["n_eval"])
    ref_final = np.array([ref["x"][-1], ref["v"][-1]])
    ref_error = np.linalg.norm(ref_final - target)
    print(f"  Euclidean error:   {ref_error:.4e}")

    print("\n--- Plot ---")
    plot_results(
        t_nn=t_np,
        z_nn=z_np,
        u_nn=u_np,
        ref=ref,
        history=history,
        config=CONFIG,
        save_path=CONFIG["save_path"],
    )
    print("\n=== Done ===")


def _print_header(config: dict) -> None:
    """Print a summary of the run configuration."""
    print("=" * 62)
    print("  Neural ODE Optimal Control – Block with Friction")
    print("=" * 62)
    print(f"  Initial state:  z0 = {config['z0']}")
    print(f"  Target state:   z* = {config['z_target']}")
    print(f"  Time horizon:   T  = {config['T']}")
    print(f"  Friction:       mu = {config['mu']}")
    print(f"  Epochs:         {config['n_epochs']}  (early stop: ||z(T)-z*|| < {config['early_stop']:.0e})")
    print("=" * 62)


if __name__ == "__main__":
    main()
