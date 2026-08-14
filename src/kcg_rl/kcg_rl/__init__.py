"""Reinforcement-learning adapters for the KCG robot simulation.

Keep this package initializer free of ROS imports.  Isaac Sim uses its own
Python runtime, where the pure RL modules are usable but ROS 2 Humble's
``rclpy`` extension for Python 3.10 is not.  The legacy cylinder symbols stay
available through lazy attribute loading for backwards compatibility.
"""

from __future__ import annotations

from typing import Any


__all__ = ["CylinderEnvConfig", "KcgCylinderEnv"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import cylinder_env

        return getattr(cylinder_env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
