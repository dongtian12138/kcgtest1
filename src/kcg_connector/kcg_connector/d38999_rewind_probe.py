"""Pure fail-closed contract for a D38999 q7 rewind and regrasp probe."""

from __future__ import annotations

import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import yaml

from kcg_connector.d38999_twist_probe import (
    load_d38999_twist_probe_contract,
)


SCHEMA_VERSION = "kcg_d38999_q7_rewind_probe_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_q7_rewind_probe_v1.yaml"
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} keys are not exact")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_d38999_rewind_probe_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repository: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load and cross-check the stage120 release/rewind/regrasp contract."""

    config_path = Path(path).resolve()
    root = (
        Path(repository).resolve()
        if repository is not None
        else config_path.parents[3]
    )
    document = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "document"
    )
    _exact(
        document,
        {
            "schema_version",
            "enabled",
            "status",
            "inputs",
            "control",
            "interstroke_self_lock_brake_proxy",
            "sensing",
            "acceptance",
            "boundaries",
        },
        "document",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported D38999 rewind probe schema")
    if document["enabled"] is not True:
        raise ValueError("D38999 rewind probe must be explicitly enabled")
    if document["status"] != "stage120_release_rewind_regrasp_proxy_probe":
        raise ValueError("D38999 rewind probe status is invalid")

    inputs = _mapping(document["inputs"], "inputs")
    _exact(
        inputs,
        {
            "stage120_twist_contract",
            "nut_regrasp_physx",
            "runner_source",
            "rewind_contract_source",
        },
        "inputs",
    )
    resolved: dict[str, Path] = {}
    for name, raw in inputs.items():
        item = _mapping(raw, f"inputs.{name}")
        _exact(item, {"path", "sha256"}, f"inputs.{name}")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"inputs.{name}.path must be repository-relative")
        target = (root / relative).resolve()
        if not target.is_file() or root not in target.parents:
            raise ValueError(f"inputs.{name} is missing or outside repository")
        if _sha256(target) != item["sha256"]:
            raise ValueError(f"inputs.{name} SHA-256 mismatch")
        resolved[name] = target

    twist, twist_inputs = load_d38999_twist_probe_contract(
        resolved["stage120_twist_contract"], repository=root
    )
    if twist["probe_id"] != "stage120":
        raise ValueError("rewind probe requires the stage120 twist contract")
    if twist_inputs["nut_regrasp_physx"] != resolved["nut_regrasp_physx"]:
        raise ValueError("rewind and stage120 contracts bind different scenes")
    if twist_inputs["runner_source"] != resolved["runner_source"]:
        raise ValueError(
            "rewind and stage120 contracts bind different runners"
        )

    control = _mapping(document["control"], "control")
    _exact(
        control,
        {
            "physics_rate_hz",
            "release_s",
            "open_settle_s",
            "rewind_delta_rad",
            "rewind_speed_rad_s",
            "rewind_duration_s",
            "post_rewind_settle_s",
            "second_open_tare_s",
            "reclosure_s",
            "preload_s",
            "final_hold_s",
        },
        "control",
    )
    values = {
        key: _positive(control[key], f"control.{key}")
        for key in control
    }
    if values["physics_rate_hz"] != 240.0:
        raise ValueError("rewind physics rate changed")
    if not math.isclose(
        values["rewind_delta_rad"], math.radians(120.0), abs_tol=1e-15
    ):
        raise ValueError("rewind must return q7 by exactly 120 degrees")
    if not math.isclose(
        values["rewind_duration_s"],
        values["rewind_delta_rad"] / values["rewind_speed_rad_s"],
        abs_tol=1e-15,
    ):
        raise ValueError("rewind speed and duration are inconsistent")
    expected_times = {
        "release_s": 2.5,
        "open_settle_s": 0.5,
        "rewind_duration_s": 24.0,
        "post_rewind_settle_s": 0.5,
        "second_open_tare_s": 0.5,
        "reclosure_s": 3.5,
        "preload_s": 0.5,
        "final_hold_s": 1.0,
    }
    for key, expected in expected_times.items():
        if values[key] != expected or values[key] * 240 != round(
            values[key] * 240
        ):
            raise ValueError("rewind phase timing changed")

    brake = _mapping(
        document["interstroke_self_lock_brake_proxy"],
        "interstroke_self_lock_brake_proxy",
    )
    _exact(
        brake,
        {
            "enabled",
            "drive_type",
            "stiffness",
            "damping",
            "target_mode",
            "target_velocity_degrees_per_second",
            "maximum_force_nm",
            "applied_after_twist",
            "removed_after_regrasp_preload",
        },
        "interstroke_self_lock_brake_proxy",
    )
    if (
        brake["enabled"] is not True
        or brake["drive_type"] != "force"
        or _positive(brake["stiffness"], "brake stiffness") != 1.0
        or _positive(brake["damping"], "brake damping") != 0.01
        or brake["target_mode"] != "measured_relative_angle_after_twist"
        or _finite(
            brake["target_velocity_degrees_per_second"],
            "brake target velocity",
        )
        != 0.0
        or _positive(
            brake["maximum_force_nm"], "brake maximum force"
        )
        != 0.05
        or brake["applied_after_twist"] is not True
        or brake["removed_after_regrasp_preload"] is not True
    ):
        raise ValueError("interstroke self-lock brake proxy changed")

    sensing = _mapping(document["sensing"], "sensing")
    _exact(
        sensing,
        {
            "torque_joint_names",
            "loaded_torque_threshold_nm",
            "minimum_loaded_channels",
            "operational_torque_target_nm",
            "hard_stop_nm",
        },
        "sensing",
    )
    if (
        sensing["torque_joint_names"] != ["f1j2", "f2j1", "f3j2"]
        or _positive(sensing["loaded_torque_threshold_nm"], "loaded")
        != 0.020
        or sensing["minimum_loaded_channels"] != 3
        or sensing["operational_torque_target_nm"] != 1.8
        or sensing["hard_stop_nm"] != 2.0
    ):
        raise ValueError("rewind sensing contract changed")

    acceptance = _mapping(document["acceptance"], "acceptance")
    _exact(
        acceptance,
        {
            "maximum_q7_rewind_tracking_error_rad",
            "maximum_released_nut_drift_rad",
            "maximum_released_body_axial_drift_m",
            "maximum_body_lateral_drift_m",
            "maximum_body_axis_error_rad",
            "maximum_fixed_translation_drift_m",
            "maximum_fixed_rotation_drift_rad",
            "maximum_settle_q7_observable_speed_rad_s",
            "maximum_settle_nut_observable_speed_rad_s",
            "maximum_final_nut_progress_loss_rad",
        },
        "acceptance",
    )
    expected_acceptance = {
        "maximum_q7_rewind_tracking_error_rad": math.radians(1.0),
        "maximum_released_nut_drift_rad": math.radians(0.5),
        "maximum_released_body_axial_drift_m": 0.00005,
        "maximum_body_lateral_drift_m": 0.00015,
        "maximum_body_axis_error_rad": math.radians(0.5),
        "maximum_fixed_translation_drift_m": 0.000001,
        "maximum_fixed_rotation_drift_rad": 0.00001,
        "maximum_settle_q7_observable_speed_rad_s": math.radians(0.5),
        "maximum_settle_nut_observable_speed_rad_s": math.radians(0.5),
        "maximum_final_nut_progress_loss_rad": math.radians(2.0),
    }
    for key, expected in expected_acceptance.items():
        if not math.isclose(
            _positive(acceptance[key], f"acceptance.{key}"),
            expected,
            abs_tol=1e-15,
        ):
            raise ValueError("rewind acceptance changed")

    boundaries = _mapping(document["boundaries"], "boundaries")
    _exact(
        boundaries,
        {
            "attachment_allowed",
            "object_pose_drive_allowed",
            "object_pose_writes_after_start_allowed",
            "physical_insertion_included",
            "full_rotation_included",
            "assembly_success_claimed",
            "continuous_collision_verified",
            "real_thread_self_lock_verified",
            "interstroke_brake_proxy_used",
        },
        "boundaries",
    )
    if (
        any(
            boundaries[name] is not False
            for name in (
                "attachment_allowed",
                "object_pose_drive_allowed",
                "object_pose_writes_after_start_allowed",
                "physical_insertion_included",
                "full_rotation_included",
                "assembly_success_claimed",
                "continuous_collision_verified",
                "real_thread_self_lock_verified",
            )
        )
        or boundaries["interstroke_brake_proxy_used"] is not True
    ):
        raise ValueError("rewind boundaries are unsafe")
    return dict(document), resolved


__all__ = (
    "DEFAULT_CONFIG_PATH",
    "SCHEMA_VERSION",
    "load_d38999_rewind_probe_contract",
)
