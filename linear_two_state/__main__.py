import torch
from .system import x0, x_star, T
from .train import train
from .visualization import plot_results

N_EPOCHS  = 5_000
SNAPSHOTS = (50, 150, 300, 1000)

print("Training AI Pontryagin on the linear two-state system...")
print(f"  Seed: {torch.initial_seed()}")
print(f"  x(0) = {x0.tolist()},  x* = {x_star.tolist()},  T = {T}")
print(f"  {N_EPOCHS} epochs, lr = 0.02\n")

controller, t_span, loss_history, snapshot_controllers, snapshot_epochs = train(
    n_epochs=N_EPOCHS,
    snapshots=SNAPSHOTS,
)

print(f"\nFinal loss: {loss_history[-1]:.6f}")
print("Plotting...")

plot_results(controller, t_span, loss_history, snapshot_epochs, snapshot_controllers)
