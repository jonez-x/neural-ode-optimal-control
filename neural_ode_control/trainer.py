"""Problem-agnostic training loop for Neural ODE optimal control.

Delegates loss computation and early-stopping criteria to the
``ProblemDefinition`` instance so that the same loop works for any
registered problem.
"""

import time
from typing import Any, Callable, Optional

import torch
import torch.optim as optim
from torchdiffeq import odeint

from neural_ode_control.base import ProblemDefinition

TrainCallback = Callable[
    [int, float, dict[str, float], dict[str, list[float]]], None
]


def train(
    problem: ProblemDefinition,
    controller: torch.nn.Module,
    config: dict[str, Any],
    callback: Optional[TrainCallback] = None,
    callback_every: int = 50,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    """Run the training loop.

    Parameters
    ----------
    problem : ProblemDefinition
        Provides dynamics, loss, and early-stopping logic.
    controller : nn.Module
        Neural controller to train (modified in-place).
    config : dict
        Full configuration dict for the problem.
    callback : callable, optional
        Called every *callback_every* epochs with
        ``(epoch, total_loss, components_dict, history)``.
    callback_every : int
        How often to invoke the callback (in epochs).

    Returns
    -------
    controller, history
    """
    dtype = torch.float64
    device = next(controller.parameters()).device

    dynamics = problem.create_dynamics(controller, config)

    t_eval = torch.linspace(
        0.0, config["T"], config["n_eval"], dtype=dtype, device=device,
    )
    z0 = torch.tensor(config["z0"], dtype=dtype, device=device)

    optimizer = optim.Adam(controller.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-6,
    )

    history: dict[str, list[float]] = {"total": [], "epoch": []}

    solver_kwargs: dict[str, Any] = {"method": config["solver"]}
    if config["solver"] == "dopri5":
        solver_kwargs["rtol"] = config["rtol"]
        solver_kwargs["atol"] = config["atol"]

    t_start = time.time()
    print(f"{'Epoch':>8}  {'Total':>12}  {'Metric':>12}  {'LR':>10}")
    print("-" * 50)

    for epoch in range(1, config["n_epochs"] + 1):
        optimizer.zero_grad()

        z_traj = odeint(dynamics, z0, t_eval, **solver_kwargs)

        total_loss, components = problem.compute_loss(
            z_traj, t_eval, controller, config,
        )

        # Record history
        history["total"].append(total_loss.detach().item())
        history["epoch"].append(epoch)
        for name, val in components.items():
            history.setdefault(name, []).append(val.detach().item())

        # Early stopping
        metric = problem.early_stop_metric(components)
        if metric < config["early_stop"]:
            print(f"\nEarly stopping at epoch {epoch}: metric = {metric:.2e}")
            if callback:
                callback(
                    epoch,
                    total_loss.item(),
                    {k: v.item() for k, v in components.items()},
                    history,
                )
            break

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step(total_loss.detach())

        # Logging
        if epoch % 100 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            print(
                f"{epoch:>8}  {total_loss.item():>12.6f}  "
                f"{metric:>12.2e}  {lr_now:>10.2e}  [{elapsed:.1f}s]"
            )

        # Live callback
        if callback and epoch % callback_every == 0:
            callback(
                epoch,
                total_loss.item(),
                {k: v.item() for k, v in components.items()},
                history,
            )

    print(f"\nTraining finished. Total time: {time.time() - t_start:.1f}s")
    return controller, history
