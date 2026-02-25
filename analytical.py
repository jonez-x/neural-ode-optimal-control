"""Closed-form LQR reference solution via the Pontryagin minimum principle.

System
    dz/dt = A z + B u,    z = (x, v)
    A = [[0, 1], [0, -mu]],   B = [[0], [1]]

Cost
    J = w_T ||z(T) - z*||²  +  w_e ∫₀ᵀ u² dt

The optimal control is a linear state feedback derived from a
matrix Riccati ODE and an affine costate term:

    u*(t) = -(1/w_e) Bᵀ P(t) z(t)  +  (1/w_e) Bᵀ r(t)

where P(t) and r(t) satisfy backward ODEs with terminal conditions
P(T) = w_T I and r(T) = w_T z*.  The backward integration is
performed via time reversal tau = T - t.
"""

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


def solve_reference(
    config: dict[str, Any],
    n_nodes: int = 200,
) -> dict[str, np.ndarray]:
    """Compute the LQR-optimal trajectory for the block-friction system.

    Parameters
    ----------
    config : dict
        Must contain: ``z0``, ``z_target``, ``T``, ``mu``,
        ``w_energy``, ``w_terminal``.
    n_nodes : int
        Number of uniformly spaced evaluation points in [0, T].

    Returns
    -------
    dict
        Keys: ``t`` (N,), ``x`` (N,), ``v`` (N,), ``u`` (N,),
        ``cost`` (float), ``converged`` (bool, always True).
    """
    z0 = np.array(config["z0"], dtype=float)
    z_target = np.array(config["z_target"], dtype=float)
    T = float(config["T"])
    mu = float(config["mu"])
    w_energy = float(config["w_energy"])
    w_terminal = float(config["w_terminal"])

    print("Computing LQR reference solution (Pontryagin) ...")

    t_grid = np.linspace(0.0, T, n_nodes)
    inv_we = 1.0 / w_energy

    # ------------------------------------------------------------------
    # Step 1: Integrate Riccati + costate backward (tau = T - t, forward)
    # ------------------------------------------------------------------
    # P is symmetric 2x2 with independent entries p00, p01, p11.
    # r = (r0, r1) is the affine costate tracking the target.
    #
    # dP/dtau =  A^T P + P A - (1/w_e) P B B^T P
    # dr/dtau = (A^T - (1/w_e) P B B^T) r
    #
    # Terminal (tau=0 <=> t=T): P(T) = w_T I,  r(T) = w_T z*

    def riccati_rhs(_tau: float, y: np.ndarray) -> list[float]:
        p00, p01, p11, r0, r1 = y
        dp00 = -inv_we * p01 ** 2
        dp01 = p00 - mu * p01 - inv_we * p01 * p11
        dp11 = 2.0 * p01 - 2.0 * mu * p11 - inv_we * p11 ** 2
        dr0 = -inv_we * p01 * r1
        dr1 = r0 - (mu + inv_we * p11) * r1
        return [dp00, dp01, dp11, dr0, dr1]

    y0_backward = [
        w_terminal, 0.0, w_terminal,
        w_terminal * z_target[0], w_terminal * z_target[1],
    ]

    sol_backward = solve_ivp(
        riccati_rhs,
        [0.0, T],
        y0_backward,
        t_eval=t_grid,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )

    # Reverse to convert from tau-order to t-order
    p01_of_t = sol_backward.y[1, ::-1]
    p11_of_t = sol_backward.y[2, ::-1]
    r1_of_t = sol_backward.y[4, ::-1]

    # ------------------------------------------------------------------
    # Step 2: Forward simulation with optimal feedback
    # ------------------------------------------------------------------
    # u*(t) = (1/w_e) * (- p01(t) z[0] - p11(t) z[1] + r1(t))

    def optimal_dynamics(t: float, z: np.ndarray) -> np.ndarray:
        p01 = float(np.interp(t, t_grid, p01_of_t))
        p11 = float(np.interp(t, t_grid, p11_of_t))
        r1 = float(np.interp(t, t_grid, r1_of_t))
        u = inv_we * (-(p01 * z[0] + p11 * z[1]) + r1)
        return np.array([z[1], u - mu * z[1]])

    sol_forward = solve_ivp(
        optimal_dynamics,
        [0.0, T],
        z0,
        t_eval=t_grid,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )

    z_traj = sol_forward.y.T
    u_optimal = inv_we * (
        -(p01_of_t * z_traj[:, 0] + p11_of_t * z_traj[:, 1]) + r1_of_t
    )

    # ------------------------------------------------------------------
    # Step 3: Evaluate cost
    # ------------------------------------------------------------------
    z_final = z_traj[-1]
    terminal_error = np.linalg.norm(z_final - z_target)
    _trapz = getattr(np, "trapezoid", np.trapz)
    energy = float(_trapz(u_optimal ** 2, t_grid))
    cost = w_energy * energy + w_terminal * float(np.sum((z_final - z_target) ** 2))

    print(f"  Final state: x={z_final[0]:.6f}, v={z_final[1]:.6f}  (error: {terminal_error:.2e})")
    print(f"  Optimal cost: {cost:.6f}")

    return {
        "t": t_grid,
        "x": z_traj[:, 0],
        "v": z_traj[:, 1],
        "u": u_optimal,
        "cost": cost,
        "converged": True,
    }
