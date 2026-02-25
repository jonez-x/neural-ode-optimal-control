"""
analytical.py
-------------
Analytische Referenzlösung des Optimal Control Problems via LQR
(Linear Quadratic Regulator / Pontryagin Minimumprinzip).

System:    ż = Az + Bu,   z = (x, v)
           A = [[0, 1], [0, -mu]],  B = [[0], [1]]
Cost:      J = w_T * ||z(T) - z*||^2  +  w_e * integral_0^T u(t)^2 dt

Die optimale Steuerung (Pontryagin):
   u*(t) = -B^T P(t) z(t) + B^T r(t)

wobei P(t) ∈ R^{2x2} die Riccati-ODE rückwärts löst und r(t) ∈ R^2
die Kostate für den Zielpunkt ist:

  dP/dt = -A^T P - P A + P B B^T P     (backward, P(T) = w_T * I)
  dr/dt  = (P B B^T - A^T) r            (backward, r(T) = w_T * z*)

Numerisch: Zeitumkehr τ = T - t, Integration von τ=0 bis τ=T (vorwärts).
"""

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


def solve_reference(config: dict[str, Any], n_nodes: int = 200) -> dict[str, np.ndarray]:
    """LQR-Referenzlösung für das lineare Block-Reibungs-OCP.

    Parameters
    ----------
    config : dict
        Muss folgende Schlüssel enthalten:
        - 'z0'        : Anfangszustand [x0, v0]
        - 'z_target'  : Zielzustand [xT, vT]
        - 'T'         : Zeithorizont
        - 'mu'        : Reibungskoeffizient
        - 'w_energy'  : Gewicht Energie-Cost (= R in u^T R u)
        - 'w_terminal': Gewicht Terminal-Cost
    n_nodes : int
        Anzahl Auswertungspunkte (Standard: 200).

    Returns
    -------
    dict mit:
        - 't'        : Zeitpunkte (N,)
        - 'x'        : Positionstrajektorie (N,)
        - 'v'        : Geschwindigkeitstrajektorie (N,)
        - 'u'        : Optimale Steuerungstrajektorie (N,)
        - 'cost'     : Skalarer Gesamtkostenterm
        - 'converged': True (LQR ist immer konvergiert)
    """
    z0 = np.array(config["z0"], dtype=float)
    z_target = np.array(config["z_target"], dtype=float)
    T = float(config["T"])
    mu = float(config["mu"])
    w_e = float(config["w_energy"])   # Eingangsgewicht R = w_e  → u^T (w_e) u
    w_T = float(config["w_terminal"])  # Terminalgewicht

    print("Berechne Referenzlösung (LQR, Pontryagin Minimumprinzip) ...")

    t_nodes = np.linspace(0.0, T, n_nodes)

    # ------------------------------------------------------------------
    # 1. Riccati + Kostate rückwärts integrieren (τ = T - t, vorwärts)
    # ------------------------------------------------------------------
    # System: dP/dτ = A^T P + P A - (1/w_e) P B B^T P   (mit R = w_e)
    # Mit Bᵀ = [0, 1], BBᵀ = [[0,0],[0,1]]:
    #
    #   dp00/dτ = -(1/w_e) p01^2
    #   dp01/dτ =  p00 - mu*p01 - (1/w_e)*p01*p11
    #   dp11/dτ =  2*p01 - 2*mu*p11 - (1/w_e)*p11^2
    #
    # Kostate:   dr/dτ = (A^T - (1/w_e) P B B^T) r
    #   dr0/dτ = -(1/w_e)*p01 * r1
    #   dr1/dτ =  r0 - (mu + (1/w_e)*p11) * r1
    #
    # Anfangsbedingungen (bei τ=0 entspricht t=T):
    #   P(T) = w_T * I   →   p00=w_T, p01=0, p11=w_T
    #   r(T) = w_T * z*

    inv_we = 1.0 / w_e

    def riccati_rhs(tau: float, y: np.ndarray) -> list[float]:
        p00, p01, p11, r0, r1 = y
        dp00 = -inv_we * p01 ** 2
        dp01 = p00 - mu * p01 - inv_we * p01 * p11
        dp11 = 2.0 * p01 - 2.0 * mu * p11 - inv_we * p11 ** 2
        dr0 = -inv_we * p01 * r1
        dr1 = r0 - (mu + inv_we * p11) * r1
        return [dp00, dp01, dp11, dr0, dr1]

    y0_back = [w_T, 0.0, w_T, w_T * z_target[0], w_T * z_target[1]]

    # τ ∈ [0, T] entspricht t ∈ [T, 0] – nach Umkehr in derselben Reihenfolge
    # wie t_nodes verfügbar durch [::-1]
    sol_back = solve_ivp(
        riccati_rhs,
        [0.0, T],
        y0_back,
        t_eval=t_nodes,          # τ = t_nodes[i]  ↔  t = T - t_nodes[i]
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )

    # Werte umkehren: Index i entspricht τ=t_nodes[n-1-i] → t=t_nodes[i]
    p01_t = sol_back.y[1, ::-1]  # p01 als Funktion von t ∈ [0,T]
    p11_t = sol_back.y[2, ::-1]
    r1_t  = sol_back.y[4, ::-1]

    # ------------------------------------------------------------------
    # 2. Vorwärtssimulation mit optimalem Feedback
    # ------------------------------------------------------------------
    # u*(t) = -(1/w_e) * (p01(t)*z[0] + p11(t)*z[1]) + (1/w_e)*r1(t)
    # ż = A z + B u*(t)  →  dx/dt = v,  dv/dt = u*(t) - mu*v

    def dynamics_opt(t: float, z: np.ndarray) -> np.ndarray:
        p01 = float(np.interp(t, t_nodes, p01_t))
        p11 = float(np.interp(t, t_nodes, p11_t))
        r1  = float(np.interp(t, t_nodes, r1_t))
        u = inv_we * (-(p01 * z[0] + p11 * z[1]) + r1)
        return np.array([z[1], u - mu * z[1]])

    sol_fwd = solve_ivp(
        dynamics_opt,
        [0.0, T],
        z0,
        t_eval=t_nodes,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )

    z_traj = sol_fwd.y.T  # (N, 2)

    # Steuertrajektorie aus dem Feedback-Gesetz
    u_opt = inv_we * (
        -(p01_t * z_traj[:, 0] + p11_t * z_traj[:, 1]) + r1_t
    )

    # ------------------------------------------------------------------
    # 3. Kosten und Ausgabe
    # ------------------------------------------------------------------
    z_T = z_traj[-1]
    terminal_err = np.linalg.norm(z_T - z_target)
    energy = float(np.trapz(u_opt ** 2, t_nodes))
    cost = w_e * energy + w_T * float(np.sum((z_T - z_target) ** 2))

    print(f"  Finaler Zustand: x={z_T[0]:.6f}, v={z_T[1]:.6f}  (Fehler: {terminal_err:.2e})")
    print(f"  Optimalkosten:   {cost:.6f}")

    return {
        "t": t_nodes,
        "x": z_traj[:, 0],
        "v": z_traj[:, 1],
        "u": u_opt,
        "cost": cost,
        "converged": True,
    }
