"""Abstract base class for problem definitions and shared data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib.figure
import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrajectoryResult:
    """Standardised output from a forward simulation."""

    t: np.ndarray
    z: np.ndarray
    u: np.ndarray
    state_labels: list[str]
    control_labels: list[str]


@dataclass
class ReferenceResult(TrajectoryResult):
    """Reference / analytical solution, if available."""

    cost: float = 0.0
    converged: bool = True


class ProblemDefinition(ABC):
    """Interface that every problem plugin must implement."""

    # ── Metadata ──

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier, e.g. 'block_1d'."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for the UI."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Markdown description shown in the sidebar."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector z."""

    @property
    @abstractmethod
    def control_dim(self) -> int:
        """Dimension of the control vector u."""

    # ── Configuration ──

    @abstractmethod
    def default_config(self) -> dict[str, Any]:
        """Return the default CONFIG dict for this problem."""

    # ── Dynamics ──

    @abstractmethod
    def create_dynamics(self, controller: nn.Module, config: dict) -> nn.Module:
        """Return an nn.Module with forward(t, z) → dz/dt."""

    # ── Loss ──

    @abstractmethod
    def compute_loss(
        self,
        z_traj: torch.Tensor,
        t_eval: torch.Tensor,
        controller: nn.Module,
        config: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total loss and named components.

        Returns
        -------
        total_loss : scalar Tensor (differentiable)
        components : dict  –  must include ``'terminal'`` key
        """

    # ── Early stopping ──

    def early_stop_metric(self, components: dict[str, torch.Tensor]) -> float:
        """Scalar compared against ``config['early_stop']``.

        Default: euclidean terminal error ``sqrt(components['terminal'])``.
        """
        return torch.sqrt(components["terminal"]).item()

    # ── Reference solution (optional) ──

    def has_reference(self) -> bool:
        return False

    def solve_reference(self, config: dict) -> ReferenceResult:
        raise NotImplementedError

    # ── Visualisation ──

    @abstractmethod
    def plot_results(
        self,
        trajectory: TrajectoryResult,
        history: dict[str, list[float]],
        config: dict,
        reference: Optional[ReferenceResult] = None,
    ) -> matplotlib.figure.Figure:
        """Create the result figure.  Return the Figure (do NOT close it)."""
