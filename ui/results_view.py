"""Results tab: post-training trajectory plots and reference comparison."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from neural_ode_control.base import ProblemDefinition
from ui.live_training import evaluate_trajectory


def render_results_view(problem: ProblemDefinition, config: dict[str, Any]) -> None:
    """Render the post-training results panel."""
    controller = st.session_state.get("result_controller")
    history = st.session_state.get("result_history")

    if controller is None or history is None:
        st.info("Train a model first to see results here.")
        return

    # Check if results match current problem
    if st.session_state.get("result_problem_name") != problem.name:
        st.warning("Results are from a different problem. Train again to update.")
        return

    result_config = st.session_state.get("result_config", config)

    # Evaluate trajectory
    with st.spinner("Evaluating trajectory..."):
        trajectory = evaluate_trajectory(problem, controller, result_config)

    # Final state info
    st.subheader("Final State")
    z_final = trajectory.z[-1]
    z_target = np.array(result_config["z_target"])
    error = np.linalg.norm(z_final - z_target)

    cols = st.columns(len(trajectory.state_labels) + 1)
    for i, label in enumerate(trajectory.state_labels):
        cols[i].metric(
            f"{label}(T)",
            f"{z_final[i]:.6f}",
            delta=f"target: {z_target[i]}",
            delta_color="off",
        )
    cols[-1].metric("Euclidean Error", f"{error:.2e}")

    # Reference solution (if available)
    reference = None
    if problem.has_reference():
        with st.spinner("Computing reference solution..."):
            reference = problem.solve_reference(result_config)
        ref_final = reference.z[-1]
        ref_error = np.linalg.norm(ref_final - z_target)
        st.caption(f"Reference error: {ref_error:.2e} | Reference cost: {reference.cost:.6f}")

    # Problem-specific result plot
    st.subheader("Result Plots")
    fig = problem.plot_results(trajectory, history, result_config, reference)
    st.pyplot(fig)
    plt.close(fig)
