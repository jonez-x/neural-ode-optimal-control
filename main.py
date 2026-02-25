"""
main.py
-------
Hauptskript: Training des Neural-ODE-Controllers, Berechnung der
Referenzlösung und Erstellung des Ergebnis-Plots.

Ausführen:
  cd neural_ode_control
  pip install -r requirements.txt
  python main.py
"""

import sys
import torch
import numpy as np
from torchdiffeq import odeint

# Eigene Module
from controller import NeuralController
from dynamics import BlockDynamics
from trainer import train
from analytical import solve_reference
from visualization import plot_results


# ===========================================================================
# Konfiguration
# ===========================================================================
CONFIG: dict = {
    # --- System ---
    "z0":        [0.0, 1.0],    # Anfangszustand (x0, v0)
    "z_target":  [1.0, 1.0],   # Zielzustand    (xT, vT)
    "T":         2.0,           # Zeithorizont
    "mu":        0.5,           # Reibungskoeffizient

    # --- Netz ---
    "hidden_dim": 32,
    "n_layers":   4,

    # --- Loss ---
    "w_terminal": 100.0,
    "w_energy":   0.01,

    # --- Training ---
    "lr":         1e-3,
    "n_epochs":   3000,
    "early_stop": 1e-3,         # Euklidischer Fehler ||z(T)-z*|| < early_stop

    # --- ODE-Solver ---
    "solver":     "dopri5",
    "n_eval":     200,          # Auswertungspunkte in [0, T]
    "rtol":       1e-7,
    "atol":       1e-9,

    # --- Sonstiges ---
    "seed":       42,
    "save_path":  "results.png",
}


def main() -> None:
    # -----------------------------------------------------------------------
    # Reproduzierbarkeit
    # -----------------------------------------------------------------------
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # float64 global als Standard-Dtype
    torch.set_default_dtype(torch.float64)

    device = torch.device("cpu")
    dtype = torch.float64

    print("=" * 62)
    print("  Neural ODE Optimal Control – Block mit Reibung")
    print("=" * 62)
    print(f"  Anfangszustand:  z0 = {CONFIG['z0']}")
    print(f"  Zielzustand:     z* = {CONFIG['z_target']}")
    print(f"  Zeithorizont:    T  = {CONFIG['T']}")
    print(f"  Reibung:         mu = {CONFIG['mu']}")
    print(f"  Epochen:         {CONFIG['n_epochs']}  (Early-Stop: ||z(T)-z*|| < {CONFIG['early_stop']:.0e})")
    print("=" * 62)

    # -----------------------------------------------------------------------
    # Modell initialisieren
    # -----------------------------------------------------------------------
    controller = NeuralController(
        hidden_dim=CONFIG["hidden_dim"],
        n_layers=CONFIG["n_layers"],
        T=CONFIG["T"],
    ).to(dtype=dtype, device=device)

    dynamics = BlockDynamics(controller=controller, mu=CONFIG["mu"])

    n_params = sum(p.numel() for p in controller.parameters())
    print(f"\nController-Parameter: {n_params}")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    print("\n--- Training ---")
    controller, history = train(controller, dynamics, CONFIG)

    # -----------------------------------------------------------------------
    # Finale Trajektorie auswerten
    # -----------------------------------------------------------------------
    t_eval = torch.linspace(0.0, CONFIG["T"], CONFIG["n_eval"], dtype=dtype, device=device)
    z0_t = torch.tensor(CONFIG["z0"], dtype=dtype, device=device)

    controller.eval()
    with torch.no_grad():
        z_traj = odeint(
            dynamics, z0_t, t_eval,
            method=CONFIG["solver"],
            rtol=CONFIG["rtol"],
            atol=CONFIG["atol"],
        )
        u_traj = controller.get_control_trajectory(t_eval)

    t_np   = t_eval.numpy()
    z_np   = z_traj.numpy()     # (N, 2)
    u_np   = u_traj.numpy()     # (N,)

    z_T    = z_np[-1]
    err    = np.linalg.norm(z_T - np.array(CONFIG["z_target"]))

    print("\n--- Ergebnis Neural ODE ---")
    print(f"  Erreichter Endzustand:  x(T) = {z_T[0]:.6f}, v(T) = {z_T[1]:.6f}")
    print(f"  Zielzustand:            x*   = {CONFIG['z_target'][0]}, v* = {CONFIG['z_target'][1]}")
    print(f"  Euklidischer Fehler:    ||z(T) - z*|| = {err:.4e}")
    print(f"  Finaler Gesamt-Loss:    {history['total'][-1]:.6f}")
    print(f"  Finaler Terminal-Loss:  {history['terminal'][-1]:.6f}")
    print(f"  Finaler Energie-Loss:   {history['energy'][-1]:.6f}")

    # -----------------------------------------------------------------------
    # Referenzlösung (scipy Direct Shooting)
    # -----------------------------------------------------------------------
    print("\n--- Referenzlösung ---")
    ref = solve_reference(CONFIG, n_nodes=CONFIG["n_eval"])

    ref_T = np.array([ref["x"][-1], ref["v"][-1]])
    ref_err = np.linalg.norm(ref_T - np.array(CONFIG["z_target"]))
    print(f"  Euklidischer Fehler Referenz: {ref_err:.4e}")

    # -----------------------------------------------------------------------
    # Visualisierung
    # -----------------------------------------------------------------------
    print(f"\n--- Plot ---")
    plot_results(
        t_nn=t_np,
        z_nn=z_np,
        u_nn=u_np,
        ref=ref,
        history=history,
        config=CONFIG,
        save_path=CONFIG["save_path"],
    )

    print("\n=== Fertig! ===")


if __name__ == "__main__":
    main()
