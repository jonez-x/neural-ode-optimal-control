"""Visualisation for the 2D obstacle-avoidance problem.

Three-panel figure:
    (1) 2D trajectory (x, y) with obstacle and start/target markers
    (2) Control magnitude ||u(t)|| and components over time
    (3) Training loss curves (log scale) with all components
"""

from typing import Any, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from neural_ode_control.base import ReferenceResult, TrajectoryResult
from problems.obstacle_2d.problem import _get_obstacles

COLORS = {
    "traj": "#1f77b4",
    "obstacle": "#d62728",
    "total": "#d62728",
    "terminal": "#9467bd",
    "energy": "#8c564b",
    "obstacle_loss": "#ff7f0e",
    "fx": "#2ca02c",
    "fy": "#17becf",
}


def plot_results(
    trajectory: TrajectoryResult,
    history: dict[str, list[float]],
    config: dict[str, Any],
    reference: Optional[ReferenceResult] = None,
) -> plt.Figure:
    """Create and return the result figure."""
    _apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Neural ODE Optimal Control \u2013 2D Point Mass with Obstacle",
        fontsize=13, y=1.02,
    )

    first_close_z = history.get("first_close_z")
    _plot_trajectory_2d(axes[0], trajectory, config, reference, first_close_z=first_close_z)
    _plot_control(axes[1], trajectory, reference)
    _plot_loss(axes[2], history)

    plt.tight_layout()
    return fig


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.8,
        "legend.framealpha": 0.8,
        "legend.fontsize": 9,
    })


def _plot_trajectory_2d(
    ax, traj: TrajectoryResult, config: dict,
    reference: Optional[ReferenceResult] = None,
    first_close_z: Optional[np.ndarray] = None,
) -> None:
    z0 = config["z0"]
    z_target = config["z_target"]

    # Obstacles (single or multiple)
    for obs in _get_obstacles(config):
        obs_c = obs["center"]
        obs_r = obs["radius"]
        ax.add_patch(plt.Circle(obs_c, obs_r, color=COLORS["obstacle"], alpha=0.3, zorder=2))
        ax.add_patch(plt.Circle(obs_c, obs_r, fill=False, color=COLORS["obstacle"], lw=2, zorder=3))

    # First trajectory that reached the target region
    if first_close_z is not None:
        ax.plot(first_close_z[:, 0], first_close_z[:, 1], color="orange",
                lw=1.4, ls="--", alpha=0.75, zorder=4, label="First close trajectory")

    # Reference trajectory
    if reference is not None:
        ax.plot(reference.z[:, 0], reference.z[:, 1], color=COLORS["traj"],
                lw=1.5, ls="--", alpha=0.7, zorder=4, label="Reference (numerical)")

    # Neural ODE trajectory
    ax.plot(traj.z[:, 0], traj.z[:, 1], color=COLORS["traj"], lw=2, zorder=5, label="Neural ODE")

    # Direction arrows along trajectory
    n = len(traj.z)
    for frac in [0.25, 0.5, 0.75]:
        idx = int(frac * n)
        if idx + 1 < n:
            ax.annotate(
                "", xy=(traj.z[idx + 1, 0], traj.z[idx + 1, 1]),
                xytext=(traj.z[idx - 1, 0], traj.z[idx - 1, 1]),
                arrowprops=dict(arrowstyle="->", color=COLORS["traj"], lw=1.5),
            )

    # Start / Target
    ax.scatter(z0[0], z0[1], s=120, color="green", zorder=6, marker="o",
               label=f"Start ({z0[0]}, {z0[1]})")
    ax.scatter(z_target[0], z_target[1], s=120, color="red", zorder=6, marker="*",
               label=f"Target ({z_target[0]}, {z_target[1]})")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("2D Trajectory")

    # Fix axis limits based on trained trajectory + start/target (ignore initial trajectory)
    xs = np.concatenate([[z0[0], z_target[0]], traj.z[:, 0]])
    ys = np.concatenate([[z0[1], z_target[1]], traj.z[:, 1]])
    margin = 0.3
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)

    ax.set_aspect("equal")
    ax.legend(loc="best")


def _plot_control(
    ax, traj: TrajectoryResult,
    reference: Optional[ReferenceResult] = None,
) -> None:
    t = traj.t
    u = traj.u
    if u.ndim == 1:
        ax.plot(t, u, color=COLORS["fx"], label="u")
    else:
        ax.plot(t, u[:, 0], color=COLORS["fx"], label="$F_x$ Neural ODE")
        ax.plot(t, u[:, 1], color=COLORS["fy"], label="$F_y$ Neural ODE")
        magnitude = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
        ax.plot(t, magnitude, color="gray", ls=":", alpha=0.7, label="$||u||$ Neural ODE")
    if reference is not None:
        u_ref = reference.u
        ax.plot(reference.t, u_ref[:, 0], color=COLORS["fx"], ls="--", alpha=0.7, label="$F_x$ Ref")
        ax.plot(reference.t, u_ref[:, 1], color=COLORS["fy"], ls="--", alpha=0.7, label="$F_y$ Ref")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Control")
    ax.set_title("Control Trajectory")
    ax.legend()


def _plot_loss(ax, history: dict[str, list[float]]) -> None:
    epochs = history["epoch"]
    ax.semilogy(epochs, history["total"], color=COLORS["total"], label="Total loss")
    if "terminal" in history:
        ax.semilogy(epochs, history["terminal"], color=COLORS["terminal"],
                     label="Terminal cost", ls="--")
    if "energy" in history:
        ax.semilogy(epochs, history["energy"], color=COLORS["energy"],
                     label="Energy cost", ls=":")
    if "obstacle" in history:
        ax.semilogy(epochs, history["obstacle"], color=COLORS["obstacle_loss"],
                     label="Obstacle penalty", ls="-.")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Convergence")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
