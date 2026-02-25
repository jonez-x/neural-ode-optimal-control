"""Result visualisation: 2x2 subplot figure.

Panels
    (1) State trajectories  x(t), v(t)  — Neural ODE vs. reference
    (2) Control trajectory  u(t)        — Neural ODE vs. reference
    (3) Phase portrait      (x, v)      — with start / target markers
    (4) Training loss curve             — log scale
"""

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

COLORS = {
    "nn_x": "#1f77b4",
    "nn_v": "#ff7f0e",
    "nn_u": "#2ca02c",
    "total": "#d62728",
    "terminal": "#9467bd",
    "energy": "#8c564b",
}


def plot_results(
    t_nn: np.ndarray,
    z_nn: np.ndarray,
    u_nn: np.ndarray,
    ref: dict[str, Any],
    history: dict[str, list[float]],
    config: dict[str, Any],
    save_path: str = "results.png",
) -> None:
    """Create and save the four-panel result figure.

    Parameters
    ----------
    t_nn : ndarray, shape (N,)
        Time grid of the Neural ODE solution.
    z_nn : ndarray, shape (N, 2)
        State trajectory ``[x, v]`` of the Neural ODE solution.
    u_nn : ndarray, shape (N,)
        Control trajectory of the Neural ODE solution.
    ref : dict
        Reference solution with keys ``t``, ``x``, ``v``, ``u``.
    history : dict
        Training curves with keys ``total``, ``terminal``, ``energy``,
        ``epoch``.
    config : dict
        Run configuration (used for ``z0``, ``z_target``, ``T``, ``mu``).
    save_path : str
        File path for the saved figure.
    """
    z0 = config["z0"]
    z_target = config["z_target"]
    mu = config["mu"]
    T = config["T"]

    _apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Neural ODE Optimal Control – Block with Friction\n"
        rf"$\dot{{x}}=v,\quad \dot{{v}}=u-\mu v,\quad \mu={mu},\quad T={T}$",
        fontsize=13,
        y=1.01,
    )

    _plot_states(axes[0, 0], t_nn, z_nn, ref, z_target, T)
    _plot_control(axes[0, 1], t_nn, u_nn, ref)
    _plot_phase(axes[1, 0], z_nn, ref, z0, z_target)
    _plot_loss(axes[1, 1], history)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.close(fig)


def _apply_style() -> None:
    """Set matplotlib RC parameters for a clean look."""
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


def _plot_states(
    ax: plt.Axes,
    t_nn: np.ndarray,
    z_nn: np.ndarray,
    ref: dict[str, Any],
    z_target: list[float],
    T: float,
) -> None:
    """Panel 1: state trajectories x(t) and v(t)."""
    ax.plot(t_nn, z_nn[:, 0], color=COLORS["nn_x"], label=r"$x(t)$ Neural ODE")
    ax.plot(t_nn, z_nn[:, 1], color=COLORS["nn_v"], label=r"$v(t)$ Neural ODE")
    ax.plot(ref["t"], ref["x"], color=COLORS["nn_x"], ls="--", alpha=0.7, label=r"$x(t)$ Reference")
    ax.plot(ref["t"], ref["v"], color=COLORS["nn_v"], ls="--", alpha=0.7, label=r"$v(t)$ Reference")

    ax.axhline(z_target[0], color=COLORS["nn_x"], ls=":", lw=1.0, alpha=0.5)
    ax.axhline(z_target[1], color=COLORS["nn_v"], ls=":", lw=1.0, alpha=0.5)

    ax.scatter([T], [z_nn[-1, 0]], color=COLORS["nn_x"], zorder=5, s=50)
    ax.scatter([T], [z_nn[-1, 1]], color=COLORS["nn_v"], zorder=5, s=50)

    ax.axvline(0.0, color="green", lw=0.8, ls=":", alpha=0.6)
    ax.axvline(T, color="red", lw=0.8, ls=":", alpha=0.6)

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("State")
    ax.set_title("State Trajectories")
    ax.legend(ncol=2, loc="lower right")


def _plot_control(
    ax: plt.Axes,
    t_nn: np.ndarray,
    u_nn: np.ndarray,
    ref: dict[str, Any],
) -> None:
    """Panel 2: control signal u(t)."""
    ax.plot(t_nn, u_nn, color=COLORS["nn_u"], label="Neural ODE")
    ax.plot(ref["t"], ref["u"], color=COLORS["nn_u"], ls="--", alpha=0.7, label="Reference")
    ax.axhline(0, color="gray", lw=0.8)

    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"Control $u(t)$")
    ax.set_title("Control Trajectory")
    ax.legend()


def _plot_phase(
    ax: plt.Axes,
    z_nn: np.ndarray,
    ref: dict[str, Any],
    z0: list[float],
    z_target: list[float],
) -> None:
    """Panel 3: phase portrait (x, v)."""
    ax.plot(z_nn[:, 0], z_nn[:, 1], color=COLORS["nn_x"], label="Neural ODE")
    ax.plot(ref["x"], ref["v"], color=COLORS["nn_x"], ls="--", alpha=0.7, label="Reference")

    ax.scatter(*z0, s=100, color="green", zorder=6, marker="o", label=f"Start {tuple(z0)}")
    ax.scatter(*z_target, s=100, color="red", zorder=6, marker="*", label=f"Target {tuple(z_target)}")

    mid = len(z_nn) // 2
    ax.annotate(
        "",
        xy=(z_nn[mid + 1, 0], z_nn[mid + 1, 1]),
        xytext=(z_nn[mid - 1, 0], z_nn[mid - 1, 1]),
        arrowprops=dict(arrowstyle="->", color=COLORS["nn_x"], lw=1.5),
    )

    ax.set_xlabel(r"Position $x$")
    ax.set_ylabel(r"Velocity $v$")
    ax.set_title("Phase Portrait")
    ax.legend(loc="best")


def _plot_loss(ax: plt.Axes, history: dict[str, list[float]]) -> None:
    """Panel 4: training loss curves on log scale."""
    epochs = history["epoch"]
    ax.semilogy(epochs, history["total"], color=COLORS["total"], label="Total loss")
    ax.semilogy(epochs, history["terminal"], color=COLORS["terminal"], label="Terminal cost", ls="--")
    ax.semilogy(epochs, history["energy"], color=COLORS["energy"], label="Energy cost", ls=":")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Convergence")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
