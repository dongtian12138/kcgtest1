"""Strict 30-D residual observation with compensated wrist F/T.

The first 24 float32 values are copied byte-for-byte from the frozen residual
v0 encoder.  Six compensated connector-task-frame wrench axes are appended in
``Fx, Fy, Fz, Tx, Ty, Tz`` order.  Disabled or incomplete calibration blocks
configuration loading; unhealthy, stale or privileged samples are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .residual_rl import (
    RESIDUAL_ACTION_SIZE,
    RESIDUAL_OBSERVATION_SIZE,
    ConnectorResidualConfig,
    ConnectorResidualState,
    residual_observation,
)


INTERFACE_VERSION = "kcg_connector_twist_residual_wrist_ft_v1"
BASE_INTERFACE_VERSION = "kcg_connector_twist_residual_v0"
ACTION_SIZE = 4
OBSERVATION_SIZE = 30
WRENCH_NAMES = (
    "wrist_force_x",
    "wrist_force_y",
    "wrist_force_z",
    "wrist_torque_x",
    "wrist_torque_y",
    "wrist_torque_z",
)
TASK_FRAME = "connector_task_frame"


class WristFtV1BlockedError(RuntimeError):
    """The v1 consumer is disabled or lacks required calibration values."""


@dataclass(frozen=True)
class WristFtV1Config:
    """Validated normalization and sample-age contract."""

    lateral_force_n: float
    axial_force_n: float
    bending_torque_nm: float
    tightening_torque_nm: float
    stale_timeout_s: float

    @property
    def axis_scales(self) -> np.ndarray:
        return np.asarray(
            (
                self.lateral_force_n,
                self.lateral_force_n,
                self.axial_force_n,
                self.bending_torque_nm,
                self.bending_torque_nm,
                self.tightening_torque_nm,
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CompensatedTaskWrench:
    """One health-bound compensated sample in connector task coordinates."""

    values: tuple[float, float, float, float, float, float]
    timestamp_s: float
    frame_id: str
    health: str
    baseline_ready: bool
    compensation_ready: bool
    simulation_ground_truth_control_authority_count: int = 0
    privileged_contact_wrench_control_authority_count: int = 0


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WristFtV1BlockedError(f"{name} must be a mapping")
    return value


def _positive(value: Any, name: str, blockers: list[str]) -> float | None:
    if value is None:
        blockers.append(f"{name} is null")
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        blockers.append(f"{name} must be a finite positive number")
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        blockers.append(f"{name} must be a finite positive number")
        return None
    return result


def default_wrist_ft_v1_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / (
        "wrist_ft_v1_contract.yaml"
    )


def load_wrist_ft_v1_config(
    config_path: str | Path | None = None,
) -> WristFtV1Config:
    """Load required v1 scales, collecting every fail-closed blocker."""
    path = Path(config_path or default_wrist_ft_v1_config_path()).resolve()
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise WristFtV1BlockedError(
            f"cannot load wrist F/T v1 contract: {path}: {error}"
        ) from error
    root = _mapping(document, "wrist F/T contract")
    residual = _mapping(root.get("residual_v1"), "residual_v1")
    scales = _mapping(
        residual.get("normalization_scales"),
        "residual_v1.normalization_scales",
    )
    safety = _mapping(root.get("safety_limits"), "safety_limits")
    blockers: list[str] = []
    if root.get("enabled") is not True:
        blockers.append("enabled is not true")
    expected = {
        "interface_version": INTERFACE_VERSION,
        "base_interface_version": BASE_INTERFACE_VERSION,
        "action_size": ACTION_SIZE,
        "observation_size": OBSERVATION_SIZE,
        "observation_source": "compensated_connector_task_frame_wrench",
    }
    for name, expected_value in expected.items():
        if residual.get(name) != expected_value:
            blockers.append(
                f"residual_v1.{name} must be {expected_value!r}"
            )
    if tuple(residual.get("appended_observation_names", ())) != WRENCH_NAMES:
        blockers.append("residual_v1 appended six-axis order changed")
    if residual.get("normalized_clip_range") != [-1.0, 1.0]:
        blockers.append("residual_v1 normalized clip range must be [-1, 1]")
    if RESIDUAL_ACTION_SIZE != ACTION_SIZE:
        blockers.append("frozen v0 action size changed")
    if RESIDUAL_OBSERVATION_SIZE != 24:
        blockers.append("frozen v0 observation size changed")

    lateral = _positive(
        scales.get("lateral_force_n"),
        "normalization_scales.lateral_force_n",
        blockers,
    )
    axial = _positive(
        scales.get("axial_force_n"),
        "normalization_scales.axial_force_n",
        blockers,
    )
    bending = _positive(
        scales.get("bending_torque_nm"),
        "normalization_scales.bending_torque_nm",
        blockers,
    )
    tightening = _positive(
        scales.get("tightening_torque_nm"),
        "normalization_scales.tightening_torque_nm",
        blockers,
    )
    stale = _positive(
        safety.get("stale_timeout_s"),
        "safety_limits.stale_timeout_s",
        blockers,
    )
    if blockers:
        raise WristFtV1BlockedError(
            "wrist F/T residual v1 is BLOCKED:\n- "
            + "\n- ".join(blockers)
        )
    assert lateral is not None
    assert axial is not None
    assert bending is not None
    assert tightening is not None
    assert stale is not None
    return WristFtV1Config(
        lateral_force_n=lateral,
        axial_force_n=axial,
        bending_torque_nm=bending,
        tightening_torque_nm=tightening,
        stale_timeout_s=stale,
    )


def _strict_zero_counter(value: Any, name: str) -> None:
    if type(value) is not int or value != 0:
        raise ValueError(f"{name} must be the integer zero")


def _validated_wrench(
    sample: CompensatedTaskWrench,
    config: WristFtV1Config,
    *,
    now_s: float,
) -> np.ndarray:
    if not isinstance(sample, CompensatedTaskWrench):
        raise TypeError("wrench sample must be CompensatedTaskWrench")
    if sample.frame_id != TASK_FRAME:
        raise ValueError("wrench sample must use connector_task_frame")
    if sample.health != "OK":
        raise ValueError("wrench sample health must be OK")
    if sample.baseline_ready is not True:
        raise ValueError("wrench compensation baseline is not ready")
    if sample.compensation_ready is not True:
        raise ValueError("wrench compensation is not ready")
    _strict_zero_counter(
        sample.simulation_ground_truth_control_authority_count,
        "simulation ground-truth control-authority counter",
    )
    _strict_zero_counter(
        sample.privileged_contact_wrench_control_authority_count,
        "privileged contact-wrench control-authority counter",
    )
    if isinstance(now_s, bool) or not isinstance(now_s, Real):
        raise ValueError("now_s must be finite")
    now = float(now_s)
    timestamp = float(sample.timestamp_s)
    if not math.isfinite(now) or not math.isfinite(timestamp):
        raise ValueError("wrench timestamps must be finite")
    age = now - timestamp
    if age < 0.0:
        raise ValueError("wrench timestamp cannot be in the future")
    if age > config.stale_timeout_s:
        raise ValueError("wrench sample is stale")
    try:
        values = np.asarray(sample.values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("wrench values must be six finite numbers") from error
    if values.shape != (6,):
        raise ValueError("wrench values must have shape (6,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("wrench values must be finite")
    scales = config.axis_scales
    if scales.shape != (6,) or not np.all(np.isfinite(scales)):
        raise ValueError("wrench scales must contain six finite values")
    if np.any(scales <= 0.0):
        raise ValueError("wrench scales must be positive")
    return np.clip(values / scales, -1.0, 1.0).astype(np.float32)


def residual_wrist_ft_observation(
    state: ConnectorResidualState,
    residual_config: ConnectorResidualConfig,
    wrench: CompensatedTaskWrench,
    wrist_ft_config: WristFtV1Config,
    *,
    now_s: float,
) -> np.ndarray:
    """Encode 24-D v0 bytes followed by normalized six-axis task wrench."""
    if not isinstance(wrist_ft_config, WristFtV1Config):
        raise TypeError("wrist_ft_config must be WristFtV1Config")
    base = residual_observation(state, residual_config)
    if base.shape != (24,) or base.dtype != np.dtype(np.float32):
        raise RuntimeError("frozen v0 encoder contract changed")
    appended = _validated_wrench(wrench, wrist_ft_config, now_s=now_s)
    observation = np.empty(OBSERVATION_SIZE, dtype=np.float32)
    observation[:24] = base
    observation[24:] = appended
    prefix_equal = np.array_equal(
        observation[:24].view(np.uint8), base.view(np.uint8)
    )
    if not prefix_equal:
        raise RuntimeError("v0 observation prefix bytes changed")
    return observation


__all__ = [
    "ACTION_SIZE",
    "BASE_INTERFACE_VERSION",
    "INTERFACE_VERSION",
    "OBSERVATION_SIZE",
    "TASK_FRAME",
    "WRENCH_NAMES",
    "CompensatedTaskWrench",
    "WristFtV1BlockedError",
    "WristFtV1Config",
    "default_wrist_ft_v1_config_path",
    "load_wrist_ft_v1_config",
    "residual_wrist_ft_observation",
]
