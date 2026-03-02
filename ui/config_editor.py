"""Dynamic configuration editor that builds a form from a problem's default_config."""

import json
from typing import Any

import streamlit as st

from neural_ode_control.base import ProblemDefinition

# Keys grouped by category for a cleaner sidebar layout
_GROUPS = {
    "System": ["z0", "z_target", "T", "mu", "obstacle_center", "obstacle_radius"],
    "Weights": ["w_terminal", "w_energy", "w_obstacle"],
    "Network": ["hidden_dim", "n_layers"],
    "Training": ["lr", "n_epochs", "early_stop", "seed"],
    "Solver": ["solver", "n_eval", "rtol", "atol"],
}


def render_config_editor(problem: ProblemDefinition) -> dict[str, Any]:
    """Render the configuration sidebar and return the edited config dict."""
    defaults = problem.default_config()
    config: dict[str, Any] = {}

    for group_name, keys in _GROUPS.items():
        present = [k for k in keys if k in defaults]
        if not present:
            continue
        with st.expander(group_name, expanded=(group_name == "System")):
            for key in present:
                config[key] = _render_field(key, defaults[key])

    # Catch any remaining keys not in groups
    grouped_keys = {k for keys in _GROUPS.values() for k in keys}
    for key, val in defaults.items():
        if key not in grouped_keys:
            config[key] = _render_field(key, val)

    return config


def _render_field(key: str, default: Any) -> Any:
    """Render a single config field and return its value."""
    label = key.replace("_", " ").title()

    if isinstance(default, bool):
        return st.checkbox(label, value=default, key=f"cfg_{key}")
    elif isinstance(default, int):
        return st.number_input(label, value=default, step=1, key=f"cfg_{key}")
    elif isinstance(default, float):
        return st.number_input(
            label, value=default, format="%g", step=None, key=f"cfg_{key}",
        )
    elif isinstance(default, list):
        text = st.text_input(label, value=json.dumps(default), key=f"cfg_{key}")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return default
    elif isinstance(default, str):
        return st.text_input(label, value=default, key=f"cfg_{key}")
    else:
        st.text(f"{label}: {default}")
        return default
