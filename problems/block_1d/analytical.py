"""Closed-form LQR reference solution via the Pontryagin minimum principle.

System
    dz/dt = A z + B u,    z = (x, v)
    A = [[0, 1], [0, -mu]],   B = [[0], [1]]

Cost
    J = w_T ||z(T) - z*||^2  +  w_e integral_0^T u^2 dt
"""

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from neural_ode_control.base import ReferenceResult


def solve_reference(config: dict[str, Any], n_nodes: int = 200) -> ReferenceResult:
    """Compute the LQR-optimal trajectory for the block-friction system."""
    z0 = np.array(config["z0"], dtype=float)
    z_target = np.array(config["z_target"], dtype=float)
    T = float(config["T"])
    mu = float(config["mu"])
    w_energy = float(config["w_energy"])
    w_terminal = float(config["w_terminal"])

    t_grid = np.linspace(0.0, T, n_nodes)
    inv_we = 1.0 / w_energy

    # Riccati + costate backward (tau = T - t)
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
        riccati_rhs, [0.0, T], y0_backward,
        t_eval=t_grid, method="RK45", rtol=1e-9, atol=1e-11,
    )

    p01_of_t = sol_backward.y[1, ::-1]
    p11_of_t = sol_backward.y[2, ::-1]
    r1_of_t = sol_backward.y[4, ::-1]

    # Forward simulation with optimal feedback
    def optimal_dynamics(t: float, z: np.ndarray) -> np.ndarray:
        p01 = float(np.interp(t, t_grid, p01_of_t))
        p11 = float(np.interp(t, t_grid, p11_of_t))
        r1 = float(np.interp(t, t_grid, r1_of_t))
        u = inv_we * (-(p01 * z[0] + p11 * z[1]) + r1)
        return np.array([z[1], u - mu * z[1]])

    sol_forward = solve_ivp(
        optimal_dynamics, [0.0, T], z0,
        t_eval=t_grid, method="RK45", rtol=1e-9, atol=1e-11,
    )

    z_traj = sol_forward.y.T
    u_optimal = inv_we * (
        -(p01_of_t * z_traj[:, 0] + p11_of_t * z_traj[:, 1]) + r1_of_t
    )

    z_final = z_traj[-1]
    _trapz = getattr(np, "trapezoid", None)
    if _trapz is None:
        _trapz = np.trapz
    energy = float(_trapz(u_optimal ** 2, t_grid))
    cost = w_energy * energy + w_terminal * float(np.sum((z_final - z_target) ** 2))

    return ReferenceResult(
        t=t_grid,
        z=z_traj,
        u=u_optimal,
        state_labels=["x", "v"],
        control_labels=["u"],
        cost=cost,
        converged=True,
    )
