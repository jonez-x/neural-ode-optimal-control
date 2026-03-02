"""Training tab: start button, live updates, progress tracking."""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from neural_ode_control.base import ProblemDefinition
from neural_ode_control.controller import NeuralController
from ui.live_training import TrainingSession, evaluate_trajectory


def render_training_view(problem: ProblemDefinition, config: dict[str, Any]) -> None:
    """Render the training controls and live/post-training display."""

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        live_enabled = st.toggle("Live updates", value=True)
    with col2:
        if live_enabled:
            callback_every = st.number_input(
                "Update every N epochs", min_value=5, max_value=500,
                value=50, step=10,
            )
        else:
            callback_every = 9999
    with col3:
        train_clicked = st.button("Train", type="primary", use_container_width=True)

    if train_clicked:
        _run_training(problem, config, live_enabled, callback_every)

    # Show results if training is done
    if "result_history" in st.session_state and st.session_state["result_history"]:
        _show_post_training(problem, config)


def _run_training(
    problem: ProblemDefinition,
    config: dict[str, Any],
    live_enabled: bool,
    callback_every: int,
) -> None:
    """Execute training, optionally with live updates."""
    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))
    torch.set_default_dtype(torch.float64)

    controller = NeuralController(
        input_dim=1,
        output_dim=problem.control_dim,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        T=config["T"],
    ).to(dtype=torch.float64)

    session = TrainingSession()

    if live_enabled:
        # Live mode: run in background thread, poll for updates
        progress_bar = st.progress(0, text="Starting training...")
        metrics_container = st.empty()
        plot_container = st.empty()

        session.start(problem, controller, config, callback_every)

        while session.is_running:
            data = session.get_live_data()
            if data and "epoch" in data:
                epoch = data["epoch"]
                n_epochs = config["n_epochs"]
                progress = min(epoch / n_epochs, 1.0)
                progress_bar.progress(progress, text=f"Epoch {epoch}/{n_epochs}")

                col_a, col_b, col_c = metrics_container.columns(3)
                col_a.metric("Total Loss", f"{data['total_loss']:.6f}")
                comps = data.get("components", {})
                col_b.metric("Terminal", f"{comps.get('terminal', 0):.2e}")
                col_c.metric("Energy", f"{comps.get('energy', 0):.4f}")

                hist = data.get("history", {})
                if hist and len(hist.get("epoch", [])) > 1:
                    fig = _make_live_loss_plot(hist)
                    plot_container.pyplot(fig)
                    plt.close(fig)

            time.sleep(0.3)

        progress_bar.progress(1.0, text="Training complete!")

    else:
        # Blocking mode: run in foreground with a spinner
        with st.spinner("Training..."):
            from neural_ode_control.trainer import train
            session.result_controller, session.result_history = train(
                problem, controller, config,
            )
            session.is_done = True

    # Store results in session state
    if session.error:
        st.error(f"Training failed: {session.error}")
    elif session.is_done:
        st.session_state["result_controller"] = session.result_controller
        st.session_state["result_history"] = session.result_history
        st.session_state["result_config"] = dict(config)
        st.session_state["result_problem_name"] = problem.name
        n_epochs = len(session.result_history["epoch"])
        final_loss = session.result_history["total"][-1]
        st.success(f"Training complete after {n_epochs} epochs. Final loss: {final_loss:.6f}")


def _show_post_training(problem: ProblemDefinition, config: dict[str, Any]) -> None:
    """Display summary metrics after training."""
    history = st.session_state["result_history"]
    st.divider()
    st.subheader("Training Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Epochs", len(history["epoch"]))
    col2.metric("Final Loss", f"{history['total'][-1]:.6f}")
    if "terminal" in history:
        final_err = np.sqrt(history["terminal"][-1])
        col3.metric("Final Error", f"{final_err:.2e}")


def _make_live_loss_plot(history: dict[str, list[float]]) -> plt.Figure:
    """Quick loss plot for live updates."""
    fig, ax = plt.subplots(figsize=(8, 3))
    epochs = history["epoch"]
    ax.semilogy(epochs, history["total"], color="#d62728", label="Total", lw=1.5)
    if "terminal" in history:
        ax.semilogy(epochs, history["terminal"], color="#9467bd", label="Terminal", ls="--", lw=1)
    if "energy" in history:
        ax.semilogy(epochs, history["energy"], color="#8c564b", label="Energy", ls=":", lw=1)
    if "obstacle" in history:
        ax.semilogy(epochs, history["obstacle"], color="#ff7f0e", label="Obstacle", ls="-.", lw=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
