import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from .system import A, B, x0, x_star, T
from .controller import make_dynamics
from .optimal_control import optimal_control


def plot_results(controller, t_span, loss_history, snapshot_epochs,
                 snapshot_controllers, out_path="results_linear_two_state.png"):
    t_span_np = t_span.detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- Phase portrait ---
    ax = axes[0]
    ax.set_title("Phase portrait  (Fig. 2a)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    colors = ["tab:blue", "tab:purple", "tab:red", "tab:brown", "tab:orange"]
    for ctrl, color, epoch in zip(snapshot_controllers, colors, snapshot_epochs):
        with torch.no_grad():
            traj = odeint(make_dynamics(ctrl), x0, t_span, method='rk4')
        traj_np = traj.detach().numpy()
        ax.plot(traj_np[:, 0], traj_np[:, 1], color=color, label=f"{epoch} epochs")

    with torch.no_grad():
        def oc_dynamics(t, x):
            u_val = optimal_control(t.unsqueeze(0))
            return A @ x + (B * u_val).squeeze()
        traj_oc = odeint(oc_dynamics, x0, t_span, method='rk4')
    ax.plot(traj_oc[:, 0].numpy(), traj_oc[:, 1].numpy(),
            'k--', label='Optimal control', linewidth=2)

    ax.scatter(*x0.numpy(),     color='green', zorder=5, label='x(0)')
    ax.scatter(*x_star.numpy(), color='red',   zorder=5, label='x*', marker='*', s=150)
    ax.legend(fontsize=7)

    # --- Control energy ---
    ax = axes[1]
    ax.set_title("Control energy  (Fig. 2b)")
    ax.set_xlabel("t")
    ax.set_ylabel("Eₜ[u]")

    with torch.no_grad():
        u_final   = torch.stack([controller(t) for t in t_span]).squeeze()
        energy_nn = torch.cumsum(u_final ** 2, dim=0) * (T / len(t_span))

    u_oc      = optimal_control(t_span)
    energy_oc = torch.cumsum(u_oc ** 2, dim=0) * (T / len(t_span))

    ax.plot(t_span_np, energy_nn.numpy(), color='tab:orange', label='AI Pontryagin')
    ax.plot(t_span_np, energy_oc.numpy(), 'k--',              label='Optimal control')
    ax.legend()

    # --- Training loss ---
    ax = axes[2]
    ax.set_title("Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.semilogy(loss_history)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
