"""Problem registry with decorator-based registration."""

from typing import Type

from neural_ode_control.base import ProblemDefinition

_REGISTRY: dict[str, Type[ProblemDefinition]] = {}


def register_problem(cls: Type[ProblemDefinition]) -> Type[ProblemDefinition]:
    """Class decorator that registers a problem definition."""
    instance = cls()
    key = instance.name
    if key in _REGISTRY:
        raise ValueError(f"Duplicate problem registration: '{key}'")
    _REGISTRY[key] = cls
    return cls


def get_problem(name: str) -> ProblemDefinition:
    """Instantiate and return a registered problem by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown problem: '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def list_problems() -> list[tuple[str, str]]:
    """Return ``[(name, display_name), ...]`` for all registered problems."""
    return [(name, cls().display_name) for name, cls in _REGISTRY.items()]
