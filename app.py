"""Streamlit entry point for the Neural ODE Optimal Control platform.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st

# Import problems to trigger registration
import problems  # noqa: F401

from neural_ode_control.registry import get_problem, list_problems
from ui.config_editor import render_config_editor
from ui.results_view import render_results_view
from ui.training_view import render_training_view

st.set_page_config(
    page_title="Neural ODE Control",
    page_icon="\u2699\ufe0f",
    layout="wide",
)

st.title("Neural ODE Optimal Control")

# ── Sidebar: problem selection + config ──
with st.sidebar:
    st.header("Problem")
    available = list_problems()
    choice = st.selectbox(
        "Select problem",
        options=[name for name, _ in available],
        format_func=lambda n: dict(available)[n],
    )

    problem = get_problem(choice)
    st.caption(problem.description)
    st.divider()

    st.header("Configuration")
    config = render_config_editor(problem)

# ── Main area: tabs ──
tab_train, tab_results = st.tabs(["Training", "Results"])

with tab_train:
    render_training_view(problem, config)

with tab_results:
    render_results_view(problem, config)
