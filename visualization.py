"""
visualization.py
----------------
Erstellt einen 2x2-Subplot-Figure mit:
  1) Zustandstrajektorien  x(t), v(t)  – Neural ODE vs. Referenz
  2) Steuertrajektorie     u(t)         – Neural ODE vs. Referenz
  3) Phasenportrait        (x, v)       mit Start- / Zielpunkt
  4) Loss-Kurve                          (log-Skala, Training)

Gespeichert als 'results.png' mit dpi=150.
"""

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch


def plot_results(
    t_nn: np.ndarray,
    z_nn: np.ndarray,
    u_nn: np.ndarray,
    ref: dict[str, Any],
    history: dict[str, list[float]],
    config: dict[str, Any],
    save_path: str = "results.png",
) -> None:
    """Erstellt und speichert den Ergebnis-Plot.

    Parameters
    ----------
    t_nn : np.ndarray  (N,)
        Zeitpunkte der Neural-ODE-Lösung.
    z_nn : np.ndarray  (N, 2)
        Zustandstrajektorie [x, v] der Neural-ODE-Lösung.
    u_nn : np.ndarray  (N,)
        Steuerungstrajektorie der Neural-ODE-Lösung.
    ref : dict
        Referenzlösung: Schlüssel 't', 'x', 'v', 'u'.
    history : dict
        Trainings-Verlauf: Schlüssel 'total', 'terminal', 'energy', 'epoch'.
    config : dict
        Konfiguration (z0, z_target, T, ...).
    save_path : str
        Pfad für das gespeicherte Bild.
    """
    z0 = config["z0"]
    z_target = config["z_target"]

    # ------------------------------------------------------------------ #
    #  Stil                                                                #
    # ------------------------------------------------------------------ #
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Neural ODE Optimal Control – Block mit Reibung\n"
        r"$\dot{x}=v,\quad \dot{v}=u-\mu v,\quad \mu=0.5,\quad T=2$",
        fontsize=13,
        y=1.01,
    )

    colors = {
        "nn_x": "#1f77b4",   # blau
        "nn_v": "#ff7f0e",   # orange
        "ref_x": "#1f77b4",  # blau gestrichelt
        "ref_v": "#ff7f0e",  # orange gestrichelt
        "nn_u": "#2ca02c",   # grün
        "ref_u": "#2ca02c",  # grün gestrichelt
        "total": "#d62728",  # rot
        "terminal": "#9467bd",
        "energy": "#8c564b",
    }

    # ------------------------------------------------------------------ #
    # (1) Oben links: Zustandstrajektorien                                #
    # ------------------------------------------------------------------ #
    ax1 = axes[0, 0]
    ax1.plot(t_nn, z_nn[:, 0], color=colors["nn_x"], label=r"$x(t)$ Neural ODE")
    ax1.plot(t_nn, z_nn[:, 1], color=colors["nn_v"], label=r"$v(t)$ Neural ODE")
    ax1.plot(ref["t"], ref["x"], color=colors["ref_x"], ls="--", alpha=0.7, label=r"$x(t)$ Referenz")
    ax1.plot(ref["t"], ref["v"], color=colors["ref_v"], ls="--", alpha=0.7, label=r"$v(t)$ Referenz")

    # Ziellinie
    ax1.axhline(z_target[0], color=colors["nn_x"], ls=":", lw=1.0, alpha=0.5)
    ax1.axhline(z_target[1], color=colors["nn_v"], ls=":", lw=1.0, alpha=0.5)

    # Endpunkt-Marker
    ax1.scatter([z_target[0]] * 0 + [config["T"]], [z_nn[-1, 0]],
                color=colors["nn_x"], zorder=5, s=50)
    ax1.scatter([config["T"]], [z_nn[-1, 1]],
                color=colors["nn_v"], zorder=5, s=50)

    ax1.set_xlabel("Zeit $t$")
    ax1.set_ylabel("Zustand")
    ax1.set_title("Zustandstrajektorien")
    ax1.legend(ncol=2, loc="lower right")

    _annotate_endpoints(ax1, config)

    # ------------------------------------------------------------------ #
    # (2) Oben rechts: Steuerung u(t)                                     #
    # ------------------------------------------------------------------ #
    ax2 = axes[0, 1]
    ax2.plot(t_nn, u_nn, color=colors["nn_u"], label="Neural ODE")
    ax2.plot(ref["t"], ref["u"], color=colors["ref_u"], ls="--", alpha=0.7, label="Referenz")
    ax2.axhline(0, color="gray", lw=0.8, ls="-")

    ax2.set_xlabel("Zeit $t$")
    ax2.set_ylabel(r"Kraft $u(t)$")
    ax2.set_title("Steuerungstrajektorie")
    ax2.legend()

    # ------------------------------------------------------------------ #
    # (3) Unten links: Phasenportrait (x, v)                              #
    # ------------------------------------------------------------------ #
    ax3 = axes[1, 0]
    ax3.plot(z_nn[:, 0], z_nn[:, 1], color=colors["nn_x"], label="Neural ODE")
    ax3.plot(ref["x"], ref["v"], color=colors["ref_x"], ls="--", alpha=0.7, label="Referenz")

    # Start- und Zielpunkt
    ax3.scatter(*z0, s=100, color="green", zorder=6, marker="o", label=f"Start {tuple(z0)}")
    ax3.scatter(*z_target, s=100, color="red", zorder=6, marker="*",
                label=f"Ziel {tuple(z_target)}")

    # Pfeil entlang Neural-ODE-Trajektorie
    mid = len(z_nn) // 2
    ax3.annotate(
        "", xy=(z_nn[mid + 1, 0], z_nn[mid + 1, 1]),
        xytext=(z_nn[mid - 1, 0], z_nn[mid - 1, 1]),
        arrowprops=dict(arrowstyle="->", color=colors["nn_x"], lw=1.5),
    )

    ax3.set_xlabel(r"Position $x$")
    ax3.set_ylabel(r"Geschwindigkeit $v$")
    ax3.set_title("Phasenportrait")
    ax3.legend(loc="best")

    # ------------------------------------------------------------------ #
    # (4) Unten rechts: Loss-Kurve                                        #
    # ------------------------------------------------------------------ #
    ax4 = axes[1, 1]
    epochs = history["epoch"]
    ax4.semilogy(epochs, history["total"], color=colors["total"], label="Gesamt-Loss")
    ax4.semilogy(epochs, history["terminal"], color=colors["terminal"],
                 label="Terminal-Cost", ls="--")
    ax4.semilogy(epochs, history["energy"], color=colors["energy"],
                 label="Energie-Cost", ls=":")

    ax4.set_xlabel("Epoche")
    ax4.set_ylabel("Loss (log-Skala)")
    ax4.set_title("Trainings-Konvergenz")
    ax4.legend()
    ax4.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    # ------------------------------------------------------------------ #
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {save_path}")
    plt.close(fig)


def _annotate_endpoints(ax: plt.Axes, config: dict[str, Any]) -> None:
    """Markiert Start- und Endzeit mit vertikalen Linien."""
    ax.axvline(0.0, color="green", lw=0.8, ls=":", alpha=0.6)
    ax.axvline(config["T"], color="red", lw=0.8, ls=":", alpha=0.6)
