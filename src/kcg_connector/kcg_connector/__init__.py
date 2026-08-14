"""KUKA/KCG electrical-connector task geometry and success logic."""

from .geometry import (
    axis_angle_error,
    helical_travel,
    relative_pose,
    split_axial_error,
    unwrap_angle,
)
from .export_isaac_urdf import sanitize_urdf
from .task_logic import (
    ConnectorMetrics,
    ConnectorPhase,
    ConnectorTaskConfig,
    TaskEvaluation,
    evaluate_connector_task,
)

__all__ = [
    "ConnectorMetrics",
    "ConnectorPhase",
    "ConnectorTaskConfig",
    "TaskEvaluation",
    "axis_angle_error",
    "evaluate_connector_task",
    "helical_travel",
    "relative_pose",
    "sanitize_urdf",
    "split_axial_error",
    "unwrap_angle",
]
