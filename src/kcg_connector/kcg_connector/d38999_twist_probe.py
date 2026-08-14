"""Pure fail-closed contract for the first D38999 q7 twist probe."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import yaml

from kcg_connector.d38999_assembly_baseline import (
    load_d38999_assembly_baseline,
)


SCHEMA_VERSION = "kcg_d38999_q7_twist_probe_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_q7_twist_probe_v1.yaml"
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


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_last_metrics(path: Path) -> Mapping[str, Any]:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("{"):
            result = json.loads(line)
            if not isinstance(result, Mapping):
                break
            return result
    raise ValueError(f"{path} has no metrics JSON")


def load_d38999_twist_probe_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repository: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load, content-address and cross-check the 20-degree probe."""

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
            "probe_id",
            "inputs",
            "runtime_thread",
            "probe",
            "sensing",
            "acceptance",
            "boundaries",
        },
        "document",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported D38999 twist probe schema")
    if document["enabled"] is not True:
        raise ValueError("D38999 twist probe must be explicitly enabled")
    if document["status"] != "prepared_engage_nut_only_proxy_probe":
        raise ValueError("D38999 twist probe status is invalid")
    probe_id = document["probe_id"]
    if probe_id not in ("stage20", "stage120"):
        raise ValueError("D38999 twist probe id is unsupported")

    inputs = _mapping(document["inputs"], "inputs")
    _exact(
        inputs,
        {
            "assembly_baseline",
            "nut_regrasp_physx",
            "passed_regrasp_run_1",
            "passed_regrasp_run_2",
            "runner_source",
            "twist_contract_source",
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

    baseline = load_d38999_assembly_baseline(resolved["assembly_baseline"])
    for name in ("passed_regrasp_run_1", "passed_regrasp_run_2"):
        metrics = _load_last_metrics(resolved[name])
        if (
            metrics.get("passed") is not True
            or metrics.get("final_all_fingers_nut_contact") is not True
            or metrics.get("final_zero_finger_body_contact") is not True
            or metrics.get("zero_forbidden_contacts") is not True
        ):
            raise ValueError(f"{name} does not prove nut-only regrasp")

    thread = _mapping(document["runtime_thread"], "runtime_thread")
    _exact(
        thread,
        {
            "root_prim_path",
            "prismatic_prim_path",
            "rack_prim_path",
            "prismatic_axis",
            "lower_limit_m",
            "upper_limit_m",
            "rack_ratio_degrees_per_meter",
            "filtered_pair_mode",
            "expected_nut_segment_count",
            "expected_body_mating_segment_count",
            "expected_fixed_entry_segment_count",
            "expected_filtered_pair_count",
        },
        "runtime_thread",
    )
    if (
        thread["prismatic_axis"] != "Z"
        or thread["filtered_pair_mode"] != "proxy_false_contacts_only"
    ):
        raise ValueError("runtime thread topology is unsupported")
    lower = _finite(thread["lower_limit_m"], "lower_limit_m")
    upper = _finite(thread["upper_limit_m"], "upper_limit_m")
    ratio = _finite(
        thread["rack_ratio_degrees_per_meter"], "rack ratio"
    )
    nut_count = _positive_integer(
        thread["expected_nut_segment_count"], "nut segment count"
    )
    body_count = _positive_integer(
        thread["expected_body_mating_segment_count"],
        "body mating segment count",
    )
    fixed_count = _positive_integer(
        thread["expected_fixed_entry_segment_count"],
        "fixed entry segment count",
    )
    pair_count = _positive_integer(
        thread["expected_filtered_pair_count"], "filtered pair count"
    )
    if lower != -0.0031 or upper != 0.0001 or lower >= upper:
        raise ValueError("one-way prismatic limits changed")
    if (
        ratio != 120000.0
        or body_count != fixed_count
        or pair_count != nut_count * fixed_count + body_count
    ):
        raise ValueError("rack/filter geometry changed")

    probe = _mapping(document["probe"], "probe")
    _exact(
        probe,
        {
            "q7_joint_name",
            "q7_delta_rad",
            "q7_speed_rad_s",
            "motion_duration_s",
            "hold_settle_duration_s",
            "hold_evaluation_duration_s",
            "total_hold_duration_s",
            "expected_nut_delta_rad",
            "expected_axial_travel_m",
            "lead_m_per_revolution",
        },
        "probe",
    )
    q7_delta = _finite(probe["q7_delta_rad"], "q7 delta")
    speed = _positive(probe["q7_speed_rad_s"], "q7 speed")
    duration = _positive(probe["motion_duration_s"], "motion duration")
    hold_settle = _positive(
        probe["hold_settle_duration_s"], "hold settle duration"
    )
    hold_evaluation = _positive(
        probe["hold_evaluation_duration_s"], "hold evaluation duration"
    )
    total_hold = _positive(
        probe["total_hold_duration_s"], "total hold duration"
    )
    nut_delta = _positive(
        probe["expected_nut_delta_rad"], "expected nut delta"
    )
    axial = _finite(
        probe["expected_axial_travel_m"], "expected axial travel"
    )
    lead = _positive(probe["lead_m_per_revolution"], "lead")
    expected_by_probe = {
        "stage20": {
            "q7_delta": -math.radians(20.0),
            "nut_delta": math.radians(20.0),
            "duration": 4.0,
            "axial": -1.0 / 6000.0,
            "hold_settle": 0.25,
            "hold_evaluation": 0.5,
            "total_hold": 0.75,
        },
        "stage120": {
            "q7_delta": -math.radians(120.0),
            "nut_delta": math.radians(120.0),
            "duration": 24.0,
            "axial": -0.001,
            "hold_settle": 0.5,
            "hold_evaluation": 2.0,
            "total_hold": 2.5,
        },
    }[probe_id]
    if (
        probe["q7_joint_name"] != baseline.q7_direction.joint_name
        or not math.isclose(
            q7_delta, expected_by_probe["q7_delta"], abs_tol=1e-15
        )
        or not math.isclose(
            nut_delta, expected_by_probe["nut_delta"], abs_tol=1e-15
        )
        or not math.isclose(
            duration, expected_by_probe["duration"], abs_tol=1e-15
        )
        or not math.isclose(
            axial, expected_by_probe["axial"], abs_tol=1e-15
        )
        or not math.isclose(
            hold_settle,
            expected_by_probe["hold_settle"],
            abs_tol=1e-15,
        )
        or not math.isclose(
            hold_evaluation,
            expected_by_probe["hold_evaluation"],
            abs_tol=1e-15,
        )
        or not math.isclose(
            total_hold, expected_by_probe["total_hold"], abs_tol=1e-15
        )
        or not math.isclose(q7_delta, -nut_delta, abs_tol=1e-15)
        or not math.isclose(duration, abs(q7_delta) / speed, abs_tol=1e-15)
        or not math.isclose(lead, baseline.thread_proxy.lead_m_per_revolution)
        or not math.isclose(
            axial, -lead * nut_delta / math.tau, abs_tol=1e-15
        )
        or not math.isclose(
            total_hold, hold_settle + hold_evaluation, abs_tol=1e-15
        )
        or hold_settle * 240 != round(hold_settle * 240)
        or hold_evaluation * 240 != round(hold_evaluation * 240)
        or total_hold * 240 != round(total_hold * 240)
        or duration * 240 != round(duration * 240)
    ):
        raise ValueError("probe arithmetic differs from assembly baseline")

    sensing = _mapping(document["sensing"], "sensing")
    if (
        sensing.get("torque_joint_names")
        != list(baseline.sensing.torque_joint_names)
        or sensing.get("operational_torque_target_nm") != 1.8
        or sensing.get("hard_stop_nm") != 2.0
        or sensing.get("minimum_loaded_channels") != 3
    ):
        raise ValueError("probe sensing contract is inconsistent")
    boundaries = _mapping(document["boundaries"], "boundaries")
    if (
        boundaries.get("real_thread_pitch_claimed") is not False
        or boundaries.get("thread_teeth_collision_modeled") is not False
        or boundaries.get("exact_proxy_collision_filter_required") is not True
        or boundaries.get("assembly_success_claimed") is not False
        or any(
            boundaries.get(name) is not False
            for name in (
                "attachment_allowed",
                "object_pose_drive_allowed",
                "object_pose_writes_after_start_allowed",
                "physical_insertion_included",
                "full_rotation_included",
            )
        )
    ):
        raise ValueError("probe boundaries are unsafe")
    acceptance = _mapping(document["acceptance"], "acceptance")
    if _positive_integer(
        acceptance.get("hold_axial_observable_window_steps"),
        "hold axial observable window steps",
    ) != 5:
        raise ValueError("hold axial observable window must remain five steps")
    if _positive(
        acceptance.get("maximum_hold_axial_speed_m_s"),
        "maximum hold axial speed",
    ) != 0.00005:
        raise ValueError("hold axial speed gate changed")
    stage_acceptance = {
        "stage20": (math.radians(15.0), math.radians(25.0), 0.75),
        "stage120": (math.radians(115.0), math.radians(125.0), 0.90),
    }[probe_id]
    if (
        _positive(
            acceptance.get("minimum_nut_progress_rad"),
            "minimum nut progress",
        )
        != stage_acceptance[0]
        or _positive(
            acceptance.get("maximum_nut_progress_rad"),
            "maximum nut progress",
        )
        != stage_acceptance[1]
        or _positive(
            acceptance.get("minimum_axial_progress_fraction"),
            "minimum axial progress fraction",
        )
        != stage_acceptance[2]
    ):
        raise ValueError("probe acceptance differs from its stage")
    return dict(document), resolved


__all__ = (
    "DEFAULT_CONFIG_PATH",
    "SCHEMA_VERSION",
    "load_d38999_twist_probe_contract",
)
