import torch
from .system import A, B, x0, x_star, T, N_NODES


def matrix_exp(M, t):
    return torch.linalg.matrix_exp(M * t)


def compute_gramian(n_steps=1000):
    """Numerically integrate W(T) = ∫₀ᵀ e^{At} B Bᵀ e^{Aᵀt} dt via Simpson's rule."""
    ts = torch.linspace(0, T, n_steps)
    dt = ts[1] - ts[0]
    W = torch.zeros(N_NODES, N_NODES)
    for i, t in enumerate(ts):
        eAt  = matrix_exp(A, t)
        eBBt = eAt @ B @ B.T @ eAt.T
        if i == 0 or i == n_steps - 1:
            w = 1
        elif i % 2 == 1:
            w = 4
        else:
            w = 2
        W = W + w * eBBt
    return W * dt / 3.0


def optimal_control(t_tensor):
    """Returns u*(t) for each t in t_tensor using the controllability Gramian."""
    W     = compute_gramian()
    eAT   = matrix_exp(A, T)
    v     = x_star - eAT @ x0
    W_inv = torch.linalg.inv(W)

    us = []
    for t in t_tensor:
        eAt_adj = matrix_exp(A.T, T - t)
        u = B.T @ eAt_adj @ W_inv @ v
        us.append(u.item())
    return torch.tensor(us)
