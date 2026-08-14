"""Pure fail-closed contract and state machine for tactile proxy engage.

This module intentionally imports no Isaac, USD, ROS, or GPU package.  It is
an opt-in experiment seam between the visual-XY 12 mm preinsert probe and the
existing physical insertion/regrasp/twist code.  In particular, the virtual
wrist wrench remains an uncalibrated monitor; all numeric ceilings are only
synthetic-scene abort guards and never authorize hardware motion.

The v2 entry helper is a CPU validation seam, not a runner integration.  A
future runtime must construct fresh ``EntryConfirmationEvidence`` and call
``entry_confirmation_candidate`` immediately before passing its result into
``decide_engage_transition``.  The state machine's bare ``entry_confirmed``
boolean is intentionally testable, but a literal ``True`` is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_d38999_tactile_engage_probe_v2"
LEGACY_SCHEMA_VERSION = "kcg_d38999_tactile_engage_probe_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_tactile_engage_probe_v2.yaml"
)
LEGACY_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_tactile_engage_probe_v1.yaml"
)
CENTERED_NO_LIP_ENTRY_MODE = "centered_no_lip_contact_v1"
LEGACY_LIP_BREAKTHROUGH_MODE = "lip_contact_breakthrough_v1"
CPU_VALIDATION_NODES = (
    "contract_hash_and_boundary_tests",
    "signed_contact_moment_math_tests",
    "centered_no_lip_entry_evidence_tests",
    "fail_closed_state_machine_tests",
    "bounded_spiral_geometry_tests",
)
LEGACY_CPU_VALIDATION_NODES = tuple(
    node
    for node in CPU_VALIDATION_NODES
    if node != "centered_no_lip_entry_evidence_tests"
)
GPU_VALIDATION_NODES = (
    "visual_preinsert_same_world_pass",
    "signed_lip_contact_characterization_plus_x_minus_x_plus_y_minus_y",
    "one_offset_tactile_engage_without_twist",
    "three_visual_offset_tactile_engage_ab",
    "repeated_tactile_engage_regression",
    "chain_existing_regrasp_twist_and_home",
)


class EngageState(str, Enum):
    """States of the bounded, explicit-opt-in tactile experiment."""

    WAIT_PREINSERT_PASS = "WAIT_PREINSERT_PASS"
    LOCAL_REFERENCE = "LOCAL_REFERENCE"
    GUARDED_APPROACH = "GUARDED_APPROACH"
    RETRACT_UNLOAD = "RETRACT_UNLOAD"
    CENTER_CORRECTION = "CENTER_CORRECTION"
    SPIRAL_FALLBACK = "SPIRAL_FALLBACK"
    ENTRY_CONFIRM = "ENTRY_CONFIRM"
    COMPLIANT_INSERT = "COMPLIANT_INSERT"
    ENGAGE_HOLD = "ENGAGE_HOLD"
    SIM_TRUTH_AUDIT = "SIM_TRUTH_AUDIT"
    READY_FOR_EXISTING_PROXY_TWIST = "READY_FOR_EXISTING_PROXY_TWIST"
    ABORT_RETRACT = "ABORT_RETRACT"
    TERMINAL_ABORT = "TERMINAL_ABORT"


@dataclass(frozen=True)
class InputArtifact:
    path: str
    sha256: str


@dataclass(frozen=True)
class EligibilityPolicy:
    required_preinsert_status: str
    require_same_world_and_capture_id: bool
    require_payload_baseline_ready: bool
    require_zero_loose_fixed_contact_at_start: bool
    require_three_finger_body_contact: bool
    require_no_object_pose_writes_after_start: bool
    allowed_orientation_source: str
    axial_progress_source: str
    provider_xy_error_bound_m: float
    observed_fixed_xy_error_m: float
    search_radius_m: float
    provider_error_bound_fully_covered: bool


@dataclass(frozen=True)
class SensorPolicy:
    source: str
    controller_wrench_mode: str
    control_rate_hz: int
    maximum_sample_age_s: float
    local_reference_samples: int
    contact_debounce_samples: int
    release_debounce_samples: int
    entry_confirmation_samples: int
    local_reference_is_safety_tare: bool
    requires_lip_contact_direction_calibration: bool
    requires_quasistatic_motion: bool


@dataclass(frozen=True)
class MotionPolicy:
    preinsert_gap_m: float
    entry_gap_m: float
    entry_confirmation_gap_m: float
    engage_gap_m: float
    guarded_approach_speed_m_s: float
    compliant_insert_speed_m_s: float
    unload_retract_speed_m_s: float
    unload_retract_distance_m: float
    moment_guided_xy_step_m: float
    maximum_xy_speed_m_s: float
    maximum_search_radius_m: float
    spiral_radial_pitch_m: float
    spiral_arc_step_m: float
    maximum_contact_attempts: int
    maximum_search_duration_s: float
    minimum_contact_lever_radius_m: float
    maximum_contact_lever_radius_m: float
    compressive_axial_force_sign_candidate: int


@dataclass(frozen=True)
class ContactDetection:
    contact_on_compressive_axial_force_n: float
    contact_off_compressive_axial_force_n: float
    contact_on_bending_torque_nm: float
    contact_off_bending_torque_nm: float
    breakthrough_axial_travel_m: float


@dataclass(frozen=True)
class EntryConfirmationPolicy:
    """Fail-closed contract for confirming centered, contact-free entry."""

    default_mode: str
    compatible_modes: tuple[str, ...]
    required_task_frame_id: str
    required_task_rotation_world: tuple[tuple[float, float, float], ...]
    command_gap_source: str
    measured_gap_source: str
    minimum_registered_preentry_gap_m: float
    maximum_confirmation_gap_m: float
    minimum_axial_travel_m: float
    required_consecutive_samples: int
    maximum_absolute_axial_force_n: float
    maximum_lateral_force_n: float
    maximum_bending_torque_nm: float
    maximum_tightening_torque_nm: float
    require_zero_loose_fixed_contact: bool
    require_release_candidate_each_sample: bool
    require_task_frame_provenance: bool
    require_same_world_and_capture_id: bool
    require_no_object_pose_writes_after_start: bool
    truth_control_allowed: bool
    first_contact_gap_required: bool
    legacy_lip_contact_mode_requires_explicit_call: bool


@dataclass(frozen=True)
class AbortEnvelope:
    maximum_absolute_axial_force_n: float
    maximum_lateral_force_n: float
    maximum_bending_torque_nm: float
    maximum_tightening_torque_nm: float
    maximum_finger_base_torque_nm: float
    calibrated_hardware_safety_limit: bool


@dataclass(frozen=True)
class TactileEngageContract:
    schema_version: str
    enabled_by_default: bool
    status: str
    inputs: dict[str, InputArtifact]
    input_paths: dict[str, Path]
    eligibility: EligibilityPolicy
    sensor: SensorPolicy
    motion: MotionPolicy
    contact: ContactDetection
    entry_confirmation: EntryConfirmationPolicy
    abort: AbortEnvelope
    states: tuple[EngageState, ...]
    cpu_nodes: tuple[str, ...]
    gpu_nodes: tuple[str, ...]
    proxy_boundaries: dict[str, Any]
    boundaries: dict[str, bool]
    config_sha256: str


@dataclass(frozen=True)
class EngageObservation:
    """One controller-tick observation in connector task coordinates."""

    sample_age_s: float
    axial_force_n: float
    lateral_force_xy_n: tuple[float, float]
    bending_torque_xy_nm: tuple[float, float]
    tightening_torque_nm: float
    finger_base_torques_nm: tuple[float, float, float]
    estimated_gap_m: float
    search_offset_xy_m: tuple[float, float]
    contact_attempts: int
    elapsed_search_s: float
    finite: bool = True
    three_finger_body_contact: bool = True
    forbidden_contact: bool = False


@dataclass(frozen=True)
class EntryConfirmationEvidence:
    """One exact consecutive window for centered, no-lip entry.

    The registered pre-entry reference is deliberately separate from the
    24-tick confirmation window.  Neither field represents a first contact:
    this mode proves progress through the entry datum while contact remains
    absent and the stopped-preinsert-referenced wrench stays released.
    """

    mode: str
    registered_preentry_command_fk_gap_m: float
    registered_preentry_measured_gap_m: float
    tick_indices: tuple[int, ...]
    command_fk_gap_samples_m: tuple[float, ...]
    measured_gap_samples_m: tuple[float, ...]
    loose_fixed_contact_records: tuple[int, ...]
    observations: tuple[EngageObservation, ...]
    preinsert_capture_id: str
    current_capture_id: str
    upstream_trial_id: str
    current_trial_id: str
    task_frame_id: str
    task_rotation_world: tuple[tuple[float, float, float], ...]
    command_gap_source: str
    measured_gap_source: str
    runner_source_sha256: str
    engage_config_sha256: str
    preinsert_plan_sha256: str
    registered_pose_sha256: str
    same_world_and_capture_id: bool
    object_pose_writes_after_start: int
    truth_used_for_entry_control: bool


@dataclass(frozen=True)
class EngageDecision:
    next_state: EngageState
    reason: str
    command_delta_xy_m: tuple[float, float] = (0.0, 0.0)
    command_delta_z_m: float = 0.0
    requires_abort_retract: bool = False


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], expected, label: str) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise ValueError(
            f"{label} keys differ: missing={sorted(wanted - actual)}, "
            f"extra={sorted(actual - wanted)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty unpadded text")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} must be finite{' and positive' if positive else ''}")
    return result


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _texts(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicates")
    return result


def _rotation_matrix(
    value: Any, label: str
) -> tuple[tuple[float, float, float], ...]:
    """Parse one finite right-handed orthonormal 3x3 rotation."""

    try:
        rotation = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a numeric 3x3 matrix") from error
    if (
        rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        > 1.0e-9
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-9
    ):
        raise ValueError(
            f"{label} must be finite, orthonormal, and right-handed"
        )
    return tuple(
        tuple(float(component) for component in row) for row in rotation
    )


def _resolve_artifact(value: Any, label: str, root: Path) -> tuple[InputArtifact, Path]:
    document = _mapping(value, label)
    _exact(document, {"path", "sha256"}, label)
    relative = Path(_text(document["path"], f"{label}.path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}.path must be repository-relative")
    expected = _text(document["sha256"], f"{label}.sha256")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"{label}.sha256 must be lowercase SHA-256")
    resolved = (root / relative).resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"{label}.path is missing")
    actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} hash mismatch")
    return InputArtifact(relative.as_posix(), expected), resolved


def load_tactile_engage_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repository: str | Path | None = None,
) -> TactileEngageContract:
    """Load, hash-check, and reject any silent scope or safety upgrade."""

    config_path = Path(path).expanduser().resolve()
    root = (
        Path(repository).expanduser().resolve()
        if repository is not None
        else config_path.parents[3]
    )
    document = _mapping(yaml.safe_load(config_path.read_text()), "root")
    schema_version = document.get("schema_version")
    if schema_version not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ValueError("unsupported tactile engage schema")
    expected_root = {
        "schema_version", "enabled_by_default", "status", "inputs",
        "eligibility", "sensor_policy", "motion_policy",
        "contact_detection", "experimental_abort_envelope",
        "state_machine", "validation_sequence", "proxy_boundaries",
        "boundaries",
    }
    if schema_version == SCHEMA_VERSION:
        expected_root.add("entry_confirmation_policy")
    _exact(
        document,
        expected_root,
        "root",
    )
    if document["enabled_by_default"] is not False:
        raise ValueError("tactile engage must remain disabled by default")
    if document["status"] != "cpu_contract_ready_gpu_contact_characterization_required":
        raise ValueError("tactile engage status overclaims validation")

    inputs_raw = _mapping(document["inputs"], "inputs")
    expected_inputs = {
        "visual_preinsert_contract", "physical_insertion_contract",
        "virtual_wrist_ft_monitor", "visual_pick_evidence",
        "wrist_ft_repeatability_evidence",
    }
    _exact(inputs_raw, expected_inputs, "inputs")
    inputs: dict[str, InputArtifact] = {}
    input_paths: dict[str, Path] = {}
    for name in sorted(expected_inputs):
        inputs[name], input_paths[name] = _resolve_artifact(
            inputs_raw[name], f"inputs.{name}", root
        )
    # Check the semantic boundaries behind the hashes too.  A matching file
    # is necessary but not sufficient if this loader is later pointed at an
    # intentionally regenerated dependency set.
    preinsert_document = _mapping(
        yaml.safe_load(
            input_paths["visual_preinsert_contract"].read_text(
                encoding="utf-8"
            )
        ),
        "visual_preinsert_contract",
    )
    insertion_document = _mapping(
        yaml.safe_load(
            input_paths["physical_insertion_contract"].read_text(
                encoding="utf-8"
            )
        ),
        "physical_insertion_contract",
    )
    monitor_document = _mapping(
        yaml.safe_load(
            input_paths["virtual_wrist_ft_monitor"].read_text(
                encoding="utf-8"
            )
        ),
        "virtual_wrist_ft_monitor",
    )
    visual_evidence = _mapping(
        json.loads(
            input_paths["visual_pick_evidence"].read_text(encoding="utf-8")
        ),
        "visual_pick_evidence",
    )
    ft_evidence = _mapping(
        json.loads(
            input_paths["wrist_ft_repeatability_evidence"].read_text(
                encoding="utf-8"
            )
        ),
        "wrist_ft_repeatability_evidence",
    )
    if (
        preinsert_document.get("enabled_by_default") is not False
        or preinsert_document.get("boundaries", {}).get("engage_executed")
        is not False
        or insertion_document.get("boundaries", {}).get("real_keying_modeled")
        is not False
        or insertion_document.get("boundaries", {}).get("thread_teeth_modeled")
        is not False
    ):
        raise ValueError("input motion contracts exceed tactile probe scope")
    monitor_limits = _mapping(
        monitor_document.get("safety_limits"), "monitor safety_limits"
    )
    monitor_boundaries = _mapping(
        monitor_document.get("boundaries"), "monitor boundaries"
    )
    if (
        any(value is not None for value in monitor_limits.values())
        or monitor_boundaries.get("monitor_only") is not True
        or monitor_boundaries.get("safety_gate_claimed") is not False
        or monitor_boundaries.get("calibrated_thresholds_claimed") is not False
        or ft_evidence.get("calibrated_safety_limits") is not None
        or ft_evidence.get("claims", {}).get("calibration_claimed") is not False
        or ft_evidence.get("claims", {}).get("safety_gate_enabled") is not False
    ):
        raise ValueError("virtual FT inputs cannot authorize a safety gate")
    if (
        visual_evidence.get("passed") is not True
        or visual_evidence.get("truth_xy_used_for_target") is not False
        or visual_evidence.get("production_control_authorized") is not False
    ):
        raise ValueError("visual pick evidence is not eligible")

    eligibility_raw = _mapping(document["eligibility"], "eligibility")
    _exact(eligibility_raw, EligibilityPolicy.__dataclass_fields__, "eligibility")
    eligibility = EligibilityPolicy(
        required_preinsert_status=_text(eligibility_raw["required_preinsert_status"], "required_preinsert_status"),
        require_same_world_and_capture_id=_bool(eligibility_raw["require_same_world_and_capture_id"], "require_same_world_and_capture_id"),
        require_payload_baseline_ready=_bool(eligibility_raw["require_payload_baseline_ready"], "require_payload_baseline_ready"),
        require_zero_loose_fixed_contact_at_start=_bool(eligibility_raw["require_zero_loose_fixed_contact_at_start"], "require_zero_loose_fixed_contact_at_start"),
        require_three_finger_body_contact=_bool(eligibility_raw["require_three_finger_body_contact"], "require_three_finger_body_contact"),
        require_no_object_pose_writes_after_start=_bool(eligibility_raw["require_no_object_pose_writes_after_start"], "require_no_object_pose_writes_after_start"),
        allowed_orientation_source=_text(eligibility_raw["allowed_orientation_source"], "allowed_orientation_source"),
        axial_progress_source=_text(eligibility_raw["axial_progress_source"], "axial_progress_source"),
        provider_xy_error_bound_m=_number(eligibility_raw["provider_xy_error_bound_m"], "provider_xy_error_bound_m", positive=True),
        observed_fixed_xy_error_m=_number(eligibility_raw["observed_fixed_xy_error_m"], "observed_fixed_xy_error_m", positive=True),
        search_radius_m=_number(eligibility_raw["search_radius_m"], "search_radius_m", positive=True),
        provider_error_bound_fully_covered=_bool(eligibility_raw["provider_error_bound_fully_covered"], "provider_error_bound_fully_covered"),
    )
    if (
        eligibility.required_preinsert_status != "PASSED_AT_PREINSERT_OUTSIDE_ENTRY"
        or eligibility.allowed_orientation_source != "registered_nominal_fk"
        or eligibility.axial_progress_source
        != "measured_robot_fk_against_registered_fixed_z"
        or not all((
            eligibility.require_same_world_and_capture_id,
            eligibility.require_payload_baseline_ready,
            eligibility.require_zero_loose_fixed_contact_at_start,
            eligibility.require_three_finger_body_contact,
            eligibility.require_no_object_pose_writes_after_start,
        ))
        or eligibility.provider_error_bound_fully_covered
        or eligibility.search_radius_m >= eligibility.provider_xy_error_bound_m
    ):
        raise ValueError("tactile eligibility boundary changed or overclaims coverage")
    fixed_truth_error = _number(
        visual_evidence.get("truth_evaluation", {}).get("fixed_xy_error_m"),
        "visual fixed_xy_error_m",
        positive=True,
    )
    fixed_error_bound = _number(
        visual_evidence.get("pose_provider", {})
        .get("diagnostics", {})
        .get("endpoints", {})
        .get("fixed_receptacle", {})
        .get("xy_error_bound_m"),
        "visual fixed xy_error_bound_m",
        positive=True,
    )
    if (
        abs(fixed_truth_error - eligibility.observed_fixed_xy_error_m)
        > 1.0e-6
        or fixed_error_bound != eligibility.provider_xy_error_bound_m
    ):
        raise ValueError("visual error evidence does not match eligibility")

    sensor_raw = _mapping(document["sensor_policy"], "sensor_policy")
    _exact(sensor_raw, SensorPolicy.__dataclass_fields__, "sensor_policy")
    sensor = SensorPolicy(
        source=_text(sensor_raw["source"], "sensor.source"),
        controller_wrench_mode=_text(
            sensor_raw["controller_wrench_mode"],
            "sensor.controller_wrench_mode",
        ),
        control_rate_hz=_positive_int(sensor_raw["control_rate_hz"], "sensor.control_rate_hz"),
        maximum_sample_age_s=_number(sensor_raw["maximum_sample_age_s"], "maximum_sample_age_s", positive=True),
        local_reference_samples=_positive_int(sensor_raw["local_reference_samples"], "local_reference_samples"),
        contact_debounce_samples=_positive_int(sensor_raw["contact_debounce_samples"], "contact_debounce_samples"),
        release_debounce_samples=_positive_int(sensor_raw["release_debounce_samples"], "release_debounce_samples"),
        entry_confirmation_samples=_positive_int(sensor_raw["entry_confirmation_samples"], "entry_confirmation_samples"),
        local_reference_is_safety_tare=_bool(sensor_raw["local_reference_is_safety_tare"], "local_reference_is_safety_tare"),
        requires_lip_contact_direction_calibration=_bool(sensor_raw["requires_lip_contact_direction_calibration"], "requires_lip_contact_direction_calibration"),
        requires_quasistatic_motion=_bool(sensor_raw["requires_quasistatic_motion"], "requires_quasistatic_motion"),
    )
    if (
        sensor.source != "virtual_wrist_ft_compensated_connector_task_frame"
        or sensor.controller_wrench_mode
        != "subtract_stopped_preinsert_local_reference"
        or sensor.control_rate_hz != 240
        or sensor.maximum_sample_age_s > 2.0 / sensor.control_rate_hz
        or sensor.local_reference_is_safety_tare
        or not sensor.requires_lip_contact_direction_calibration
        or not sensor.requires_quasistatic_motion
    ):
        raise ValueError("tactile sensor policy changed")

    motion_raw = _mapping(document["motion_policy"], "motion_policy")
    _exact(motion_raw, MotionPolicy.__dataclass_fields__, "motion_policy")
    integer_motion = {"maximum_contact_attempts"}
    motion_kwargs = {}
    for name in MotionPolicy.__dataclass_fields__:
        if name in integer_motion:
            motion_kwargs[name] = _positive_int(motion_raw[name], f"motion.{name}")
        elif name == "compressive_axial_force_sign_candidate":
            value = motion_raw[name]
            if type(value) is not int or value not in (-1, 1):
                raise ValueError("compressive force sign candidate must be -1 or +1")
            motion_kwargs[name] = value
        else:
            motion_kwargs[name] = _number(motion_raw[name], f"motion.{name}", positive=True)
    motion = MotionPolicy(**motion_kwargs)
    preinsert_axial = _mapping(
        preinsert_document.get("axial_scope"), "preinsert axial_scope"
    )
    insertion_filter = _mapping(
        insertion_document.get("proxy_collision_filter"),
        "insertion proxy_collision_filter",
    )
    if not (
        motion.preinsert_gap_m > motion.entry_gap_m
        > motion.entry_confirmation_gap_m > motion.engage_gap_m
        and motion.maximum_search_radius_m == eligibility.search_radius_m
        and motion.moment_guided_xy_step_m <= motion.spiral_radial_pitch_m
        and motion.spiral_arc_step_m <= motion.maximum_search_radius_m
        and motion.minimum_contact_lever_radius_m
        < motion.maximum_contact_lever_radius_m
        and (
            motion.maximum_contact_attempts * motion.spiral_arc_step_m
            / motion.maximum_xy_speed_m_s
        )
        <= motion.maximum_search_duration_s
        and motion.preinsert_gap_m == preinsert_axial.get("preinsert_gap_m")
        and motion.entry_gap_m == preinsert_axial.get("entry_gap_m")
        and insertion_filter.get("expected_filtered_pair_count") == 500
    ):
        raise ValueError("tactile motion geometry is inconsistent")

    contact_raw = _mapping(document["contact_detection"], "contact_detection")
    _exact(contact_raw, ContactDetection.__dataclass_fields__, "contact_detection")
    contact = ContactDetection(**{
        name: _number(contact_raw[name], f"contact.{name}", positive=True)
        for name in ContactDetection.__dataclass_fields__
    })
    if (
        contact.contact_off_compressive_axial_force_n
        >= contact.contact_on_compressive_axial_force_n
        or contact.contact_off_bending_torque_nm
        >= contact.contact_on_bending_torque_nm
    ):
        raise ValueError("contact hysteresis is invalid")

    if schema_version == SCHEMA_VERSION:
        entry_raw = _mapping(
            document["entry_confirmation_policy"],
            "entry_confirmation_policy",
        )
        _exact(
            entry_raw,
            EntryConfirmationPolicy.__dataclass_fields__,
            "entry_confirmation_policy",
        )
        entry_confirmation = EntryConfirmationPolicy(
            default_mode=_text(
                entry_raw["default_mode"], "entry_confirmation.default_mode"
            ),
            compatible_modes=_texts(
                entry_raw["compatible_modes"],
                "entry_confirmation.compatible_modes",
            ),
            required_task_frame_id=_text(
                entry_raw["required_task_frame_id"],
                "entry_confirmation.required_task_frame_id",
            ),
            required_task_rotation_world=_rotation_matrix(
                entry_raw["required_task_rotation_world"],
                "entry_confirmation.required_task_rotation_world",
            ),
            command_gap_source=_text(
                entry_raw["command_gap_source"],
                "entry_confirmation.command_gap_source",
            ),
            measured_gap_source=_text(
                entry_raw["measured_gap_source"],
                "entry_confirmation.measured_gap_source",
            ),
            minimum_registered_preentry_gap_m=_number(
                entry_raw["minimum_registered_preentry_gap_m"],
                "entry_confirmation.minimum_registered_preentry_gap_m",
                positive=True,
            ),
            maximum_confirmation_gap_m=_number(
                entry_raw["maximum_confirmation_gap_m"],
                "entry_confirmation.maximum_confirmation_gap_m",
                positive=True,
            ),
            minimum_axial_travel_m=_number(
                entry_raw["minimum_axial_travel_m"],
                "entry_confirmation.minimum_axial_travel_m",
                positive=True,
            ),
            required_consecutive_samples=_positive_int(
                entry_raw["required_consecutive_samples"],
                "entry_confirmation.required_consecutive_samples",
            ),
            maximum_absolute_axial_force_n=_number(
                entry_raw["maximum_absolute_axial_force_n"],
                "entry_confirmation.maximum_absolute_axial_force_n",
                positive=True,
            ),
            maximum_lateral_force_n=_number(
                entry_raw["maximum_lateral_force_n"],
                "entry_confirmation.maximum_lateral_force_n",
                positive=True,
            ),
            maximum_bending_torque_nm=_number(
                entry_raw["maximum_bending_torque_nm"],
                "entry_confirmation.maximum_bending_torque_nm",
                positive=True,
            ),
            maximum_tightening_torque_nm=_number(
                entry_raw["maximum_tightening_torque_nm"],
                "entry_confirmation.maximum_tightening_torque_nm",
                positive=True,
            ),
            require_zero_loose_fixed_contact=_bool(
                entry_raw["require_zero_loose_fixed_contact"],
                "entry_confirmation.require_zero_loose_fixed_contact",
            ),
            require_release_candidate_each_sample=_bool(
                entry_raw["require_release_candidate_each_sample"],
                "entry_confirmation.require_release_candidate_each_sample",
            ),
            require_task_frame_provenance=_bool(
                entry_raw["require_task_frame_provenance"],
                "entry_confirmation.require_task_frame_provenance",
            ),
            require_same_world_and_capture_id=_bool(
                entry_raw["require_same_world_and_capture_id"],
                "entry_confirmation.require_same_world_and_capture_id",
            ),
            require_no_object_pose_writes_after_start=_bool(
                entry_raw["require_no_object_pose_writes_after_start"],
                "entry_confirmation.require_no_object_pose_writes_after_start",
            ),
            truth_control_allowed=_bool(
                entry_raw["truth_control_allowed"],
                "entry_confirmation.truth_control_allowed",
            ),
            first_contact_gap_required=_bool(
                entry_raw["first_contact_gap_required"],
                "entry_confirmation.first_contact_gap_required",
            ),
            legacy_lip_contact_mode_requires_explicit_call=_bool(
                entry_raw[
                    "legacy_lip_contact_mode_requires_explicit_call"
                ],
                "entry_confirmation.legacy_mode_requires_explicit_call",
            ),
        )
        if not (
            entry_confirmation.default_mode == CENTERED_NO_LIP_ENTRY_MODE
            and entry_confirmation.compatible_modes
            == (
                CENTERED_NO_LIP_ENTRY_MODE,
                LEGACY_LIP_BREAKTHROUGH_MODE,
            )
            and entry_confirmation.required_task_frame_id
            == "connector_task_frame"
            and entry_confirmation.required_task_rotation_world
            == (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
            and entry_confirmation.command_gap_source
            == "commanded_fixed_q7_fk_against_registered_fixed_z"
            and entry_confirmation.measured_gap_source
            == "measured_tcp_prim_against_registered_fixed_z"
            and entry_confirmation.minimum_registered_preentry_gap_m
            == motion.entry_gap_m
            and entry_confirmation.maximum_confirmation_gap_m
            == motion.entry_confirmation_gap_m
            and entry_confirmation.minimum_axial_travel_m
            == contact.breakthrough_axial_travel_m
            and entry_confirmation.required_consecutive_samples
            == sensor.entry_confirmation_samples
            and entry_confirmation.maximum_absolute_axial_force_n
            <= contact.contact_off_compressive_axial_force_n
            and entry_confirmation.maximum_lateral_force_n
            <= contact.contact_off_compressive_axial_force_n
            and entry_confirmation.maximum_bending_torque_nm
            <= contact.contact_off_bending_torque_nm
            and entry_confirmation.maximum_tightening_torque_nm
            <= contact.contact_off_bending_torque_nm
            and entry_confirmation.require_zero_loose_fixed_contact
            and entry_confirmation.require_release_candidate_each_sample
            and entry_confirmation.require_task_frame_provenance
            and entry_confirmation.require_same_world_and_capture_id
            and entry_confirmation.require_no_object_pose_writes_after_start
            and not entry_confirmation.truth_control_allowed
            and not entry_confirmation.first_contact_gap_required
            and entry_confirmation.legacy_lip_contact_mode_requires_explicit_call
        ):
            raise ValueError("centered entry confirmation policy changed")
    else:
        # The frozen v1 contract remains loadable for old reports and the
        # untouched Isaac runner.  Its contact-gap semantics are available
        # only through the explicitly named legacy helper below.
        entry_confirmation = EntryConfirmationPolicy(
            default_mode=LEGACY_LIP_BREAKTHROUGH_MODE,
            compatible_modes=(LEGACY_LIP_BREAKTHROUGH_MODE,),
            required_task_frame_id="connector_task_frame",
            required_task_rotation_world=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            command_gap_source="legacy_not_recorded",
            measured_gap_source=eligibility.axial_progress_source,
            minimum_registered_preentry_gap_m=motion.entry_gap_m,
            maximum_confirmation_gap_m=motion.entry_confirmation_gap_m,
            minimum_axial_travel_m=contact.breakthrough_axial_travel_m,
            required_consecutive_samples=sensor.entry_confirmation_samples,
            maximum_absolute_axial_force_n=(
                contact.contact_off_compressive_axial_force_n
            ),
            maximum_lateral_force_n=(
                contact.contact_off_compressive_axial_force_n
            ),
            maximum_bending_torque_nm=(
                contact.contact_off_bending_torque_nm
            ),
            maximum_tightening_torque_nm=(
                contact.contact_off_bending_torque_nm
            ),
            require_zero_loose_fixed_contact=False,
            require_release_candidate_each_sample=True,
            require_task_frame_provenance=False,
            require_same_world_and_capture_id=False,
            require_no_object_pose_writes_after_start=False,
            truth_control_allowed=False,
            first_contact_gap_required=True,
            legacy_lip_contact_mode_requires_explicit_call=True,
        )

    abort_raw = _mapping(document["experimental_abort_envelope"], "experimental_abort_envelope")
    _exact(abort_raw, AbortEnvelope.__dataclass_fields__, "experimental_abort_envelope")
    abort = AbortEnvelope(
        **{
            name: (
                _bool(abort_raw[name], f"abort.{name}")
                if name == "calibrated_hardware_safety_limit"
                else _number(abort_raw[name], f"abort.{name}", positive=True)
            )
            for name in AbortEnvelope.__dataclass_fields__
        }
    )
    if abort.calibrated_hardware_safety_limit or abort.maximum_finger_base_torque_nm != 2.0:
        raise ValueError("experimental abort envelope cannot claim hardware calibration")
    observed_peaks = _mapping(
        ft_evidence.get("observed_statistics", {}).get(
            "protected_phase_absolute_peaks"
        ),
        "FT observed peaks",
    )
    observed_maxima = {
        name: max(
            _number(phase[name]["maximum_observed"], f"FT {phase_name}.{name}")
            for phase_name, phase in observed_peaks.items()
        )
        for name in (
            "axial_force_n",
            "lateral_force_n",
            "bending_torque_nm",
            "tightening_torque_nm",
        )
    }
    if not (
        abort.maximum_absolute_axial_force_n
        > observed_maxima["axial_force_n"]
        and abort.maximum_lateral_force_n
        > observed_maxima["lateral_force_n"]
        and abort.maximum_bending_torque_nm
        > observed_maxima["bending_torque_nm"]
        and abort.maximum_tightening_torque_nm
        > observed_maxima["tightening_torque_nm"]
    ):
        raise ValueError("experimental abort envelope clips prior nominal run")

    machine = _mapping(document["state_machine"], "state_machine")
    _exact(machine, {"initial_state", "terminal_ready_state", "terminal_abort_state", "states"}, "state_machine")
    states = tuple(EngageState(value) for value in _texts(machine["states"], "states"))
    if (
        states != tuple(EngageState)
        or machine["initial_state"] != EngageState.WAIT_PREINSERT_PASS.value
        or machine["terminal_ready_state"] != EngageState.READY_FOR_EXISTING_PROXY_TWIST.value
        or machine["terminal_abort_state"] != EngageState.TERMINAL_ABORT.value
    ):
        raise ValueError("tactile state set changed")

    validation = _mapping(document["validation_sequence"], "validation_sequence")
    _exact(validation, {"cpu_nodes", "gpu_nodes"}, "validation_sequence")
    cpu_nodes = _texts(validation["cpu_nodes"], "cpu_nodes")
    gpu_nodes = _texts(validation["gpu_nodes"], "gpu_nodes")
    expected_cpu_nodes = (
        CPU_VALIDATION_NODES
        if schema_version == SCHEMA_VERSION
        else LEGACY_CPU_VALIDATION_NODES
    )
    if cpu_nodes != expected_cpu_nodes or gpu_nodes != GPU_VALIDATION_NODES:
        raise ValueError("validation node order changed")

    proxy = dict(_mapping(document["proxy_boundaries"], "proxy_boundaries"))
    expected_proxy = {
        "segmented_box_entry_shell": True,
        "exact_insert_geometry_available": False,
        "entry_chamfer_modeled": False,
        "keying_modeled": False,
        "thread_teeth_modeled": False,
        "filtered_proxy_collision_pair_count": 500,
        "contact_face_collision_mode": "visual_only",
        "tactile_result_transferable_to_hardware": False,
    }
    if proxy != expected_proxy:
        raise ValueError("proxy limitations changed")

    boundaries = dict(_mapping(document["boundaries"], "boundaries"))
    expected_boundary_keys = {
        "explicit_opt_in_required", "existing_e2e_modified", "existing_runner_modified",
        "robot_or_connector_asset_modified", "frozen_baseline_modified",
        "foundationpose_required_for_this_proxy_probe", "yaw_estimated_from_rgbd",
        "full_6d_claimed", "arbitrary_visual_error_coverage_claimed",
        "virtual_ft_is_calibrated_safety_gate", "truth_used_for_engage_search_control",
        "sim_truth_used_for_posthoc_acceptance_only", "engage_executed",
        "twist_executed", "home_return_executed", "gpu_or_physx_validated",
        "production_control_authorized", "real_connector_assembly_claimed",
    }
    if schema_version == SCHEMA_VERSION:
        expected_boundary_keys.update(
            {
                "entry_confirmation_runtime_integrated",
                "bare_entry_confirmed_proves_entry_evidence",
            }
        )
    if set(boundaries) != expected_boundary_keys:
        raise ValueError("boundary keys changed")
    # FoundationPose is explicitly *not* required; all other positive claims
    # except opt-in and post-hoc audit are forbidden.
    if boundaries["foundationpose_required_for_this_proxy_probe"] is not False:
        raise ValueError("FoundationPose must remain optional for this probe")
    if (
        boundaries["explicit_opt_in_required"] is not True
        or boundaries["sim_truth_used_for_posthoc_acceptance_only"]
        is not True
    ):
        raise ValueError("required fail-closed boundaries changed")
    if schema_version == SCHEMA_VERSION and (
        boundaries["entry_confirmation_runtime_integrated"] is not False
        or boundaries["bare_entry_confirmed_proves_entry_evidence"]
        is not False
    ):
        raise ValueError("CPU entry helper integration boundary changed")
    allowed_positive_boundaries = {
        "explicit_opt_in_required",
        "sim_truth_used_for_posthoc_acceptance_only",
    }
    if any(
        value
        for key, value in boundaries.items()
        if key not in allowed_positive_boundaries
    ):
        raise ValueError("tactile contract overclaims execution or safety")

    return TactileEngageContract(
        schema_version=schema_version,
        enabled_by_default=False,
        status=document["status"],
        inputs=inputs,
        input_paths=input_paths,
        eligibility=eligibility,
        sensor=sensor,
        motion=motion,
        contact=contact,
        entry_confirmation=entry_confirmation,
        abort=abort,
        states=states,
        cpu_nodes=cpu_nodes,
        gpu_nodes=gpu_nodes,
        proxy_boundaries=proxy,
        boundaries=boundaries,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
    )


def spiral_offset(attempt: int, policy: MotionPolicy) -> tuple[float, float]:
    """Return a deterministic bounded Archimedean-spiral sample."""

    if type(attempt) is not int or attempt < 0:
        raise ValueError("attempt must be a non-negative integer")
    if attempt == 0:
        return (0.0, 0.0)
    # For r=b*theta, invert the closed-form arc length by bisection.  This
    # keeps successive waypoints close to ``spiral_arc_step_m`` instead of
    # becoming very sparse at the outer radius.
    b = policy.spiral_radial_pitch_m / (2.0 * math.pi)
    maximum_angle = policy.maximum_search_radius_m / b

    def arc_length(angle: float) -> float:
        return 0.5 * b * (
            angle * math.sqrt(1.0 + angle * angle)
            + math.asinh(angle)
        )

    desired_length = min(
        attempt * policy.spiral_arc_step_m,
        arc_length(maximum_angle),
    )
    lower, upper = 0.0, maximum_angle
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        if arc_length(middle) < desired_length:
            lower = middle
        else:
            upper = middle
    angle = 0.5 * (lower + upper)
    radius = b * angle
    return (radius * math.cos(angle), radius * math.sin(angle))


def moment_guided_center_step(
    bending_torque_xy_nm: Sequence[Real],
    axial_force_n: Real,
    policy: MotionPolicy,
) -> tuple[float, float]:
    """Map a lip-contact bending moment to one bounded XY correction.

    For task-frame ``F=(0,0,Fz)`` at radial contact ``r=(rx,ry,0)``,
    ``M=r x F=(ry*Fz,-rx*Fz,0)``.  Thus the inferred radial contact vector is
    ``(-My/Fz, Mx/Fz)``.  Moving opposite that vector recenters the plug.  A
    four-direction GPU calibration is mandatory because proxy contact may not
    realize this ideal resultant at the wrist boundary.
    """

    torque = np.asarray(tuple(bending_torque_xy_nm), dtype=np.float64)
    force = _number(axial_force_n, "axial_force_n")
    if torque.shape != (2,) or not np.all(np.isfinite(torque)):
        raise ValueError("bending_torque_xy_nm must be a finite XY vector")
    if abs(force) < 1.0e-9:
        raise ValueError("axial force is too small for moment direction")
    radial = np.asarray((-torque[1] / force, torque[0] / force))
    radius = float(np.linalg.norm(radial))
    if not policy.minimum_contact_lever_radius_m <= radius <= policy.maximum_contact_lever_radius_m:
        raise ValueError("inferred lip-contact lever radius is implausible")
    direction = -radial / radius
    step = policy.moment_guided_xy_step_m * direction
    return (float(step[0]), float(step[1]))


def contact_candidate(
    observation: EngageObservation, contract: TactileEngageContract
) -> bool:
    """Return one un-debounced lip-contact sample decision.

    The wrench fields are deltas from the stopped-preinsert local reference;
    the underlying monitor payload tare remains untouched and monitor-only.
    """

    compression = (
        contract.motion.compressive_axial_force_sign_candidate
        * float(observation.axial_force_n)
    )
    bending = math.hypot(*observation.bending_torque_xy_nm)
    return bool(
        compression >= contract.contact.contact_on_compressive_axial_force_n
        or bending >= contract.contact.contact_on_bending_torque_nm
    )


def contact_release_candidate(
    observation: EngageObservation, contract: TactileEngageContract
) -> bool:
    """Return one un-debounced contact-release sample decision."""

    compression = (
        contract.motion.compressive_axial_force_sign_candidate
        * float(observation.axial_force_n)
    )
    bending = math.hypot(*observation.bending_torque_xy_nm)
    return bool(
        compression <= contract.contact.contact_off_compressive_axial_force_n
        and bending <= contract.contact.contact_off_bending_torque_nm
    )


def entry_confirmation_candidate(
    evidence: EntryConfirmationEvidence,
    contract: TactileEngageContract,
    *,
    expected_runner_source_sha256: str,
    expected_preinsert_plan_sha256: str,
    expected_registered_pose_sha256: str,
) -> bool:
    """Validate 24 centered, contact-free ticks below the entry datum.

    This default v2 path never accepts a first-contact gap.  Both commanded
    fixed-q7 FK and measured TCP gap must move from a registered reference at
    or outside 10 mm to the 9.5 mm confirmation region.  Every consecutive
    tick must remain contact-free, fresh, finite and in the low-load release
    band, with explicit task-frame and hash provenance.
    """

    if not isinstance(evidence, EntryConfirmationEvidence):
        return False
    policy = contract.entry_confirmation
    if (
        contract.schema_version != SCHEMA_VERSION
        or policy.default_mode != CENTERED_NO_LIP_ENTRY_MODE
        or evidence.mode != CENTERED_NO_LIP_ENTRY_MODE
    ):
        return False

    # These strings and digests are not mere report decoration: incomplete or
    # cross-config evidence cannot satisfy the controller gate.
    try:
        preinsert_capture_id = _text(
            evidence.preinsert_capture_id,
            "entry.preinsert_capture_id",
        )
        current_capture_id = _text(
            evidence.current_capture_id,
            "entry.current_capture_id",
        )
        upstream_trial_id = _text(
            evidence.upstream_trial_id,
            "entry.upstream_trial_id",
        )
        current_trial_id = _text(
            evidence.current_trial_id,
            "entry.current_trial_id",
        )
        task_frame_id = _text(
            evidence.task_frame_id, "entry.task_frame_id"
        )
        command_source = _text(
            evidence.command_gap_source, "entry.command_gap_source"
        )
        measured_source = _text(
            evidence.measured_gap_source, "entry.measured_gap_source"
        )
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        preinsert_capture_id != current_capture_id
        or upstream_trial_id != current_trial_id
        or task_frame_id != policy.required_task_frame_id
        or command_source != policy.command_gap_source
        or measured_source != policy.measured_gap_source
        or evidence.engage_config_sha256 != contract.config_sha256
        or type(evidence.same_world_and_capture_id) is not bool
        or evidence.same_world_and_capture_id is not True
        or type(evidence.object_pose_writes_after_start) is not int
        or evidence.object_pose_writes_after_start != 0
        or type(evidence.truth_used_for_entry_control) is not bool
        or evidence.truth_used_for_entry_control is not False
    ):
        return False
    expected_digests = (
        expected_runner_source_sha256,
        contract.config_sha256,
        expected_preinsert_plan_sha256,
        expected_registered_pose_sha256,
    )
    evidence_digests = (
        evidence.runner_source_sha256,
        evidence.engage_config_sha256,
        evidence.preinsert_plan_sha256,
        evidence.registered_pose_sha256,
    )
    for digest in (*expected_digests, *evidence_digests):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest == "0" * 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return False
    if evidence_digests != expected_digests:
        return False

    try:
        rotation = _rotation_matrix(
            evidence.task_rotation_world,
            "entry.task_rotation_world",
        )
    except (TypeError, ValueError, OverflowError):
        return False
    if rotation != policy.required_task_rotation_world:
        return False

    count = policy.required_consecutive_samples
    sequences = (
        evidence.tick_indices,
        evidence.command_fk_gap_samples_m,
        evidence.measured_gap_samples_m,
        evidence.loose_fixed_contact_records,
        evidence.observations,
    )
    if any(
        not isinstance(value, tuple) or len(value) != count
        for value in sequences
    ):
        return False
    if any(type(tick) is not int for tick in evidence.tick_indices):
        return False
    if any(tick < 0 for tick in evidence.tick_indices):
        return False
    if any(
        second != first + 1
        for first, second in zip(
            evidence.tick_indices, evidence.tick_indices[1:]
        )
    ):
        return False
    if any(
        type(records) is not int or records != 0
        for records in evidence.loose_fixed_contact_records
    ):
        return False

    try:
        command_start = _number(
            evidence.registered_preentry_command_fk_gap_m,
            "entry.registered_preentry_command_fk_gap_m",
        )
        measured_start = _number(
            evidence.registered_preentry_measured_gap_m,
            "entry.registered_preentry_measured_gap_m",
        )
        command_gaps = tuple(
            _number(value, "entry.command_fk_gap_samples_m[]")
            for value in evidence.command_fk_gap_samples_m
        )
        measured_gaps = tuple(
            _number(value, "entry.measured_gap_samples_m[]")
            for value in evidence.measured_gap_samples_m
        )
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        min(command_start, measured_start)
        < policy.minimum_registered_preentry_gap_m
        or min(command_gaps) < 0.0
        or min(measured_gaps) < 0.0
        or max(command_gaps) > policy.maximum_confirmation_gap_m
        or max(measured_gaps) > policy.maximum_confirmation_gap_m
        or command_start - command_gaps[-1]
        < policy.minimum_axial_travel_m
        or measured_start - measured_gaps[-1]
        < policy.minimum_axial_travel_m
    ):
        return False
    for gaps in (command_gaps, measured_gaps):
        if any(
            following > previous + 1.0e-12
            for previous, following in zip(gaps, gaps[1:])
        ):
            return False

    for measured_gap, observation in zip(
        measured_gaps, evidence.observations
    ):
        if not isinstance(observation, EngageObservation):
            return False
        try:
            invalid = (
                _abort_reason(observation, contract) is not None
                or not math.isclose(
                    float(observation.estimated_gap_m),
                    measured_gap,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not contact_release_candidate(observation, contract)
                or abs(float(observation.axial_force_n))
                > policy.maximum_absolute_axial_force_n
                or math.hypot(*observation.lateral_force_xy_n)
                > policy.maximum_lateral_force_n
                or math.hypot(*observation.bending_torque_xy_nm)
                > policy.maximum_bending_torque_nm
                or abs(float(observation.tightening_torque_nm))
                > policy.maximum_tightening_torque_nm
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if invalid:
            return False
    return True


def lip_contact_entry_confirmation_candidate(
    first_contact_gap_m: Real,
    observation: EngageObservation,
    contract: TactileEngageContract,
    *,
    mode: str,
) -> bool:
    """Compatibility-only first-contact breakthrough predicate.

    Callers must name the legacy mode explicitly.  Keeping this separate makes
    it impossible for a missing first-contact observation to be fabricated in
    the centered no-lip path.
    """

    if mode != LEGACY_LIP_BREAKTHROUGH_MODE:
        raise ValueError("legacy lip entry mode must be explicit")
    if mode not in contract.entry_confirmation.compatible_modes:
        return False
    contact_gap = _number(first_contact_gap_m, "first_contact_gap_m")
    axial_travel = contact_gap - float(observation.estimated_gap_m)
    return bool(
        observation.estimated_gap_m
        <= contract.motion.entry_confirmation_gap_m
        and axial_travel >= contract.contact.breakthrough_axial_travel_m
        and contact_release_candidate(observation, contract)
    )


def _abort_reason(
    observation: EngageObservation,
    contract: TactileEngageContract,
) -> str | None:
    """Return a stable fail-closed reason for arbitrary observation input."""

    if not isinstance(observation, EngageObservation):
        return "malformed_observation"
    try:
        if (
            type(observation.finite) is not bool
            or type(observation.three_finger_body_contact) is not bool
            or type(observation.forbidden_contact) is not bool
            or len(observation.lateral_force_xy_n) != 2
            or len(observation.bending_torque_xy_nm) != 2
            or len(observation.finger_base_torques_nm) != 3
            or len(observation.search_offset_xy_m) != 2
            or type(observation.contact_attempts) is not int
            or observation.contact_attempts < 0
        ):
            return "malformed_observation"
        values = (
            observation.sample_age_s,
            observation.axial_force_n,
            *observation.lateral_force_xy_n,
            *observation.bending_torque_xy_nm,
            observation.tightening_torque_nm,
            *observation.finger_base_torques_nm,
            observation.estimated_gap_m,
            *observation.search_offset_xy_m,
            observation.elapsed_search_s,
        )
        numeric = tuple(float(value) for value in values)
        if not observation.finite or not all(
            math.isfinite(value) for value in numeric
        ):
            return "nonfinite_observation"
        if (
            float(observation.sample_age_s) < 0.0
            or float(observation.estimated_gap_m) < 0.0
            or float(observation.elapsed_search_s) < 0.0
        ):
            return "invalid_observation_range"
        if (
            float(observation.sample_age_s)
            > contract.sensor.maximum_sample_age_s
        ):
            return "stale_wrench"
        if observation.forbidden_contact:
            return "forbidden_contact"
        if not observation.three_finger_body_contact:
            return "grasp_contact_lost"
        if (
            abs(float(observation.axial_force_n))
            > contract.abort.maximum_absolute_axial_force_n
        ):
            return "experimental_axial_force_ceiling"
        if (
            math.hypot(
                *(float(value) for value in observation.lateral_force_xy_n)
            )
            > contract.abort.maximum_lateral_force_n
        ):
            return "experimental_lateral_force_ceiling"
        if (
            math.hypot(
                *(float(value) for value in observation.bending_torque_xy_nm)
            )
            > contract.abort.maximum_bending_torque_nm
        ):
            return "experimental_bending_torque_ceiling"
        if (
            abs(float(observation.tightening_torque_nm))
            > contract.abort.maximum_tightening_torque_nm
        ):
            return "experimental_tightening_torque_ceiling"
        if (
            max(
                abs(float(value))
                for value in observation.finger_base_torques_nm
            )
            > contract.abort.maximum_finger_base_torque_nm
        ):
            return "finger_base_torque_hard_stop"
        if (
            math.hypot(
                *(float(value) for value in observation.search_offset_xy_m)
            )
            > contract.motion.maximum_search_radius_m + 1.0e-12
        ):
            return "search_radius_exceeded"
        if (
            observation.contact_attempts
            > contract.motion.maximum_contact_attempts
        ):
            return "contact_attempt_budget_exceeded"
        if (
            float(observation.elapsed_search_s)
            > contract.motion.maximum_search_duration_s
        ):
            return "search_timeout"
    except (TypeError, ValueError, OverflowError):
        return "malformed_observation"
    return None


def decide_engage_transition(
    state: EngageState,
    observation: EngageObservation,
    contract: TactileEngageContract,
    *,
    prerequisite_passed: bool = True,
    reference_ready: bool = True,
    contact_debounced: bool = False,
    contact_released: bool = False,
    entry_confirmed: bool = False,
    entry_confirmation_mode: str | None = None,
    engage_hold_complete: bool = False,
    sim_truth_audit_passed: bool = False,
    moment_direction_calibrated: bool = False,
) -> EngageDecision:
    """Evaluate one state transition without simulator or hidden truth input.

    ``entry_confirmed`` is only a pure state-machine seam.  A future runtime
    must pass the immediately preceding result of
    :func:`entry_confirmation_candidate`; a caller-supplied literal ``True``
    is not provenance-bearing integration evidence.
    """

    if not isinstance(state, EngageState):
        state = EngageState(state)
    confirmation_mode = (
        contract.entry_confirmation.default_mode
        if entry_confirmation_mode is None
        else entry_confirmation_mode
    )
    if confirmation_mode not in contract.entry_confirmation.compatible_modes:
        return EngageDecision(
            EngageState.ABORT_RETRACT,
            "unsupported_entry_confirmation_mode",
            requires_abort_retract=True,
        )
    abort = _abort_reason(observation, contract)
    if abort is not None and state not in {EngageState.ABORT_RETRACT, EngageState.TERMINAL_ABORT}:
        return EngageDecision(EngageState.ABORT_RETRACT, abort, requires_abort_retract=True)
    if state is EngageState.ABORT_RETRACT:
        return EngageDecision(
            EngageState.TERMINAL_ABORT,
            "bounded_retract_complete",
            command_delta_z_m=contract.motion.unload_retract_distance_m,
        )
    if state is EngageState.TERMINAL_ABORT:
        return EngageDecision(state, "terminal_abort_latched")
    if state is EngageState.WAIT_PREINSERT_PASS:
        return EngageDecision(
            EngageState.LOCAL_REFERENCE if prerequisite_passed else state,
            "preinsert_prerequisite_passed" if prerequisite_passed else "waiting_for_preinsert_pass",
        )
    if state is EngageState.LOCAL_REFERENCE:
        return EngageDecision(
            EngageState.GUARDED_APPROACH if reference_ready else state,
            "local_reference_ready" if reference_ready else "collecting_local_reference",
        )
    if state is EngageState.GUARDED_APPROACH:
        if contact_debounced:
            return EngageDecision(EngageState.RETRACT_UNLOAD, "lip_contact_debounced")
        if observation.estimated_gap_m <= contract.motion.entry_confirmation_gap_m:
            reason = (
                "centered_no_lip_entry_window_candidate"
                if confirmation_mode == CENTERED_NO_LIP_ENTRY_MODE
                else "legacy_lip_breakthrough_candidate"
            )
            return EngageDecision(EngageState.ENTRY_CONFIRM, reason)
        return EngageDecision(
            state,
            "continue_guarded_approach",
            command_delta_z_m=-contract.motion.guarded_approach_speed_m_s / contract.sensor.control_rate_hz,
        )
    if state is EngageState.RETRACT_UNLOAD:
        return EngageDecision(
            EngageState.CENTER_CORRECTION if contact_released else state,
            "contact_unloaded" if contact_released else "continue_bounded_unload",
            command_delta_z_m=contract.motion.unload_retract_speed_m_s / contract.sensor.control_rate_hz,
        )
    if state is EngageState.CENTER_CORRECTION:
        if moment_direction_calibrated:
            try:
                step = moment_guided_center_step(
                    observation.bending_torque_xy_nm,
                    observation.axial_force_n,
                    contract.motion,
                )
            except ValueError:
                return EngageDecision(EngageState.SPIRAL_FALLBACK, "moment_resultant_unusable")
            target = np.asarray(observation.search_offset_xy_m) + np.asarray(step)
            if float(np.linalg.norm(target)) <= contract.motion.maximum_search_radius_m:
                return EngageDecision(EngageState.GUARDED_APPROACH, "moment_guided_center_step", step)
        return EngageDecision(EngageState.SPIRAL_FALLBACK, "moment_direction_not_calibrated")
    if state is EngageState.SPIRAL_FALLBACK:
        target = spiral_offset(observation.contact_attempts + 1, contract.motion)
        current = np.asarray(observation.search_offset_xy_m)
        delta = np.asarray(target) - current
        return EngageDecision(
            EngageState.GUARDED_APPROACH,
            "bounded_spiral_step",
            (float(delta[0]), float(delta[1])),
        )
    if state is EngageState.ENTRY_CONFIRM:
        if entry_confirmed:
            reason = (
                "centered_no_lip_entry_confirmed"
                if confirmation_mode == CENTERED_NO_LIP_ENTRY_MODE
                else "legacy_lip_breakthrough_confirmed"
            )
            return EngageDecision(EngageState.COMPLIANT_INSERT, reason)
        return EngageDecision(EngageState.GUARDED_APPROACH, "entry_confirmation_failed_retry")
    if state is EngageState.COMPLIANT_INSERT:
        if observation.estimated_gap_m <= contract.motion.engage_gap_m:
            return EngageDecision(EngageState.ENGAGE_HOLD, "engage_gap_reached")
        return EngageDecision(
            state,
            "continue_compliant_insert",
            command_delta_z_m=-contract.motion.compliant_insert_speed_m_s / contract.sensor.control_rate_hz,
        )
    if state is EngageState.ENGAGE_HOLD:
        return EngageDecision(
            EngageState.SIM_TRUTH_AUDIT if engage_hold_complete else state,
            "engage_hold_complete" if engage_hold_complete else "holding_engage",
        )
    if state is EngageState.SIM_TRUTH_AUDIT:
        return EngageDecision(
            EngageState.READY_FOR_EXISTING_PROXY_TWIST if sim_truth_audit_passed else EngageState.ABORT_RETRACT,
            "posthoc_sim_truth_audit_passed" if sim_truth_audit_passed else "posthoc_sim_truth_audit_failed",
            requires_abort_retract=not sim_truth_audit_passed,
        )
    return EngageDecision(state, "ready_state_latched")


__all__ = [
    "CENTERED_NO_LIP_ENTRY_MODE", "CPU_VALIDATION_NODES",
    "DEFAULT_CONFIG_PATH", "GPU_VALIDATION_NODES",
    "LEGACY_CONFIG_PATH", "LEGACY_LIP_BREAKTHROUGH_MODE",
    "LEGACY_SCHEMA_VERSION", "SCHEMA_VERSION", "AbortEnvelope",
    "ContactDetection", "EngageDecision", "EngageObservation", "EngageState",
    "EntryConfirmationEvidence", "EntryConfirmationPolicy", "MotionPolicy",
    "SensorPolicy", "TactileEngageContract",
    "contact_candidate", "contact_release_candidate",
    "decide_engage_transition", "entry_confirmation_candidate",
    "lip_contact_entry_confirmation_candidate",
    "load_tactile_engage_contract", "moment_guided_center_step",
    "spiral_offset",
]
