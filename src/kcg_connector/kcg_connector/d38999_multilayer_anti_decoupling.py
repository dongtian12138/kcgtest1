"""Offline anti-decoupling relation for the D38999 multilayer model.

The master model defines a 36-period directional resistance proxy, but it does
not define an absolute cam/follower phase origin.  This module therefore makes
the relative periodic relation reviewable while failing closed at the runtime
request boundary.  It never emits robot commands, writes poses, or accepts
contact/event truth as an input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "EIGHT-HOUR-E6-ANTI-DECOUPLING-RESISTANCE"
E5_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-E5-THREAD-AXIAL-FOLLOW/TASK_RESULT.json"
)
MOMENT_COMPONENT_LIMIT_NM = 0.30
HIGH_DETAIL_FORWARD_PROXY_PEAK_NM = 0.060021022609

FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml": (
        "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e"
    ),
    "src/kcg_connector/kcg_connector/d38999_multilayer_thread_axial_follow.py": (
        "1e849695228619462493ab01fbc48deac39590dac2d71cb4251e8d94f3fa1444"
    ),
    E5_RESULT_PATH: (
        "29da341b38738b9335357594f0d60d0fbf723fb0ed826dfef736ff02f1ab095e"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
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


@dataclass(frozen=True)
class AntiDecouplingContract:
    source_rows: tuple[tuple[str, str], ...]
    source_class: str
    cycle_count_per_revolution: int
    follower_count: int
    mean_radius_m: float
    forward_ramp_angle_deg: float
    reverse_face_angle_deg: float
    cam_radial_rise_m: float
    per_follower_stiffness_n_m: float
    per_follower_damping_n_s_m: float
    nominal_radial_preload_m: float
    maximum_radial_deflection_m: float
    pitch_rad: float
    forward_span_rad: float
    reverse_span_rad: float
    dwell_span_rad: float
    initial_forward_resistance_nm: float
    maximum_forward_resistance_nm: float
    initial_reverse_resistance_nm: float
    maximum_reverse_resistance_nm: float
    moment_component_limit_nm: float
    absolute_phase_origin_authorized: bool
    hardware_curve_claimed: bool
    current_e5_outcome: str
    current_e5_dynamic_thread_follow_passed: bool
    current_e5_evidence_sha256: str


@dataclass(frozen=True)
class AntiDecouplingReadiness:
    e5_evidence_path: str
    e5_evidence_sha256: str
    e5_dynamic_thread_follow_passed: bool
    physical_detent_runtime_ready: bool
    absolute_phase_origin_authorized: bool


def _directional_resistance_nm(
    *,
    follower_count: int,
    stiffness_n_m: float,
    deflection_m: float,
    face_angle_deg: float,
    mean_radius_m: float,
) -> float:
    return (
        follower_count
        * stiffness_n_m
        * deflection_m
        * math.tan(math.radians(face_angle_deg))
        * mean_radius_m
    )


def load_anti_decoupling_contract(
    repository_root: str | Path,
) -> AntiDecouplingContract:
    """Load and independently derive the frozen master detent relation."""

    root = Path(repository_root).resolve()
    rows: list[tuple[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"frozen E6 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen E6 source hash mismatch: {relative}")
        rows.append((relative, actual))

    master = _mapping(
        yaml.safe_load(
            (root / "src/kcg_connector/config/"
             "d38999_master_model_contract_v1.yaml").read_text(
                encoding="utf-8"
            )
        ),
        "master model contract",
    )
    raw = _mapping(master.get("anti_decoupling"), "master.anti_decoupling")
    cycles = raw.get("cycle_count_per_revolution")
    followers = raw.get("follower_count")
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        raise ValueError("cycle count must be a positive integer")
    if isinstance(followers, bool) or not isinstance(followers, int) or followers <= 0:
        raise ValueError("follower count must be a positive integer")

    radius = _positive(raw.get("mean_radius_m"), "mean radius")
    forward_angle = _positive(raw.get("forward_ramp_angle_deg"), "forward angle")
    reverse_angle = _positive(raw.get("reverse_face_angle_deg"), "reverse angle")
    rise = _positive(raw.get("cam_radial_rise_m"), "cam rise")
    stiffness = _positive(raw.get("per_follower_stiffness_n_m"), "stiffness")
    damping = _positive(raw.get("per_follower_damping_n_s_m"), "damping")
    preload = _positive(raw.get("nominal_radial_preload_m"), "preload")
    maximum_deflection = _positive(
        raw.get("maximum_radial_deflection_m"), "maximum deflection"
    )
    if not math.isclose(maximum_deflection, preload + rise, abs_tol=1e-15):
        raise ValueError("maximum detent deflection must equal preload plus cam rise")
    if (
        raw.get("source_class") != "equivalent_assumption"
        or raw.get("assembly_control_representation")
        != "bounded_36_period_directional_resistance"
        or raw.get("magnetic_mechanism_allowed") is not False
        or raw.get("hardware_curve_claimed") is not False
    ):
        raise ValueError("master anti-decoupling provenance or boundary changed")

    pitch = 2.0 * math.pi / cycles
    forward_span = rise / (radius * math.tan(math.radians(forward_angle)))
    reverse_span = rise / (radius * math.tan(math.radians(reverse_angle)))
    dwell_span = pitch - forward_span - reverse_span
    if dwell_span <= 0.0:
        raise ValueError("detent profile has no nonnegative dwell")

    initial_forward = _directional_resistance_nm(
        follower_count=followers,
        stiffness_n_m=stiffness,
        deflection_m=preload,
        face_angle_deg=forward_angle,
        mean_radius_m=radius,
    )
    maximum_forward = _directional_resistance_nm(
        follower_count=followers,
        stiffness_n_m=stiffness,
        deflection_m=maximum_deflection,
        face_angle_deg=forward_angle,
        mean_radius_m=radius,
    )
    initial_reverse = _directional_resistance_nm(
        follower_count=followers,
        stiffness_n_m=stiffness,
        deflection_m=preload,
        face_angle_deg=reverse_angle,
        mean_radius_m=radius,
    )
    maximum_reverse = _directional_resistance_nm(
        follower_count=followers,
        stiffness_n_m=stiffness,
        deflection_m=maximum_deflection,
        face_angle_deg=reverse_angle,
        mean_radius_m=radius,
    )

    high_detail = _mapping(
        yaml.safe_load(
            (root / "src/kcg_connector/config/"
             "d38999_keyed_v3_physical_model_contract_r12_v1.yaml").read_text(
                encoding="utf-8"
            )
        ),
        "high-detail model contract",
    )
    force_parameters = _mapping(
        _mapping(high_detail.get("physical_proxy_boundaries"), "physical proxy")
        .get("force_parameters"),
        "force parameters",
    )
    proxy = _mapping(
        force_parameters.get("anti_decoupling_detent"),
        "high-detail anti-decoupling proxy",
    )
    if (
        proxy.get("complete_pair_public_torque_limits_are_not_detent_limits")
        is not True
        or proxy.get("directional_hysteresis_claim")
        != "physical_geometry_proxy_not_hardware_curve"
        or not math.isclose(
            _finite(proxy.get("accepted_probe_measured_peak_nm"), "proxy peak"),
            HIGH_DETAIL_FORWARD_PROXY_PEAK_NM,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("high-detail proxy provenance changed")

    e5 = _mapping(
        json.loads((root / E5_RESULT_PATH).read_text(encoding="utf-8")),
        "E5 task result",
    )
    if (
        e5.get("task_id") != "EIGHT-HOUR-E5-THREAD-AXIAL-FOLLOW"
        or e5.get("outcome") != "OFFLINE_PASS"
        or type(e5.get("dynamic_thread_follow_pass_claimed")) is not bool
        or e5.get("software_pose_write_requested") is not False
    ):
        raise ValueError("E5 evidence does not support E6")

    return AntiDecouplingContract(
        source_rows=tuple(rows),
        source_class="equivalent_assumption",
        cycle_count_per_revolution=cycles,
        follower_count=followers,
        mean_radius_m=radius,
        forward_ramp_angle_deg=forward_angle,
        reverse_face_angle_deg=reverse_angle,
        cam_radial_rise_m=rise,
        per_follower_stiffness_n_m=stiffness,
        per_follower_damping_n_s_m=damping,
        nominal_radial_preload_m=preload,
        maximum_radial_deflection_m=maximum_deflection,
        pitch_rad=pitch,
        forward_span_rad=forward_span,
        reverse_span_rad=reverse_span,
        dwell_span_rad=dwell_span,
        initial_forward_resistance_nm=initial_forward,
        maximum_forward_resistance_nm=maximum_forward,
        initial_reverse_resistance_nm=initial_reverse,
        maximum_reverse_resistance_nm=maximum_reverse,
        moment_component_limit_nm=MOMENT_COMPONENT_LIMIT_NM,
        absolute_phase_origin_authorized=False,
        hardware_curve_claimed=False,
        current_e5_outcome=str(e5.get("outcome")),
        current_e5_dynamic_thread_follow_passed=e5[
            "dynamic_thread_follow_pass_claimed"
        ],
        current_e5_evidence_sha256=FROZEN_SOURCES[E5_RESULT_PATH],
    )


def derive_relative_periodic_profile(
    contract: AntiDecouplingContract,
    relative_progress_rad: float,
) -> dict[str, Any]:
    """Return one relative tooth profile without asserting absolute phase."""

    progress = _finite(relative_progress_rad, "relative progress") % contract.pitch_rad
    ascent_start = contract.dwell_span_rad
    drop_start = ascent_start + contract.forward_span_rad
    if progress < ascent_start:
        branch = "base_dwell"
        fraction = 0.0
        deflection = contract.nominal_radial_preload_m
    elif progress < drop_start:
        branch = "shallow_ascent"
        fraction = (progress - ascent_start) / contract.forward_span_rad
        deflection = (
            contract.nominal_radial_preload_m
            + fraction * contract.cam_radial_rise_m
        )
    else:
        branch = "steep_reverse_drop"
        fraction = (progress - drop_start) / contract.reverse_span_rad
        deflection = (
            contract.maximum_radial_deflection_m
            - fraction * contract.cam_radial_rise_m
        )
    return {
        "relative_progress_rad": progress,
        "relative_progress_deg": math.degrees(progress),
        "profile_branch": branch,
        "branch_fraction": fraction,
        "radial_deflection_m": deflection,
        "absolute_phase_used": False,
        "hardware_curve_claimed": False,
    }


def derive_directional_resistance(
    contract: AntiDecouplingContract,
    *,
    direction: str,
    radial_deflection_m: float,
) -> dict[str, Any]:
    """Derive a static directional bound from master parameters only."""

    deflection = _finite(radial_deflection_m, "radial deflection")
    if not (
        contract.nominal_radial_preload_m
        <= deflection
        <= contract.maximum_radial_deflection_m
    ):
        raise ValueError("radial deflection outside the master-contract interval")
    if direction == "positive_coupling":
        face_angle = contract.forward_ramp_angle_deg
        sign = -1.0
    elif direction == "reverse_decoupling":
        face_angle = contract.reverse_face_angle_deg
        sign = 1.0
    else:
        raise ValueError("direction must be positive_coupling or reverse_decoupling")
    magnitude = _directional_resistance_nm(
        follower_count=contract.follower_count,
        stiffness_n_m=contract.per_follower_stiffness_n_m,
        deflection_m=deflection,
        face_angle_deg=face_angle,
        mean_radius_m=contract.mean_radius_m,
    )
    return {
        "direction": direction,
        "resistance_moment_nm": sign * magnitude,
        "resistance_magnitude_nm": magnitude,
        "within_authorized_moment_component": (
            magnitude <= contract.moment_component_limit_nm
        ),
        "clipped_to_limit": False,
        "hardware_curve_claimed": False,
    }


def evaluate_anti_decoupling_gate(
    contract: AntiDecouplingContract,
    readiness: AntiDecouplingReadiness,
) -> str | None:
    if (
        readiness.e5_evidence_path != E5_RESULT_PATH
        or readiness.e5_evidence_sha256 != contract.current_e5_evidence_sha256
    ):
        return "E5_EVIDENCE_ID_MISMATCH"
    if (
        contract.current_e5_dynamic_thread_follow_passed is not True
        or readiness.e5_dynamic_thread_follow_passed is not True
    ):
        return "E5_THREAD_AXIAL_FOLLOW_NOT_DYNAMIC"
    if readiness.physical_detent_runtime_ready is not True:
        return "PHYSICAL_DETENT_RUNTIME_NOT_READY"
    if (
        contract.absolute_phase_origin_authorized is not True
        or readiness.absolute_phase_origin_authorized is not True
    ):
        return "DETENT_ABSOLUTE_PHASE_ORIGIN_UNAUTHORIZED"
    return None


def build_anti_decoupling_request(
    contract: AntiDecouplingContract,
    readiness: AntiDecouplingReadiness,
    *,
    relative_progress_rad: float,
    direction: str,
) -> dict[str, Any]:
    """Build a non-commanding physical-model request after all gates pass."""

    progress = _finite(relative_progress_rad, "relative progress")
    rejection = evaluate_anti_decoupling_gate(contract, readiness)
    if rejection is not None:
        return {
            "schema_version": 1,
            "task_id": TASK_ID,
            "request_ready": False,
            "rejection_code": rejection,
            "relative_profile": None,
            "resistance": None,
            "physical_model_internal_constraint_requested": False,
            "software_pose_write_requested": False,
            "force_or_moment_command_requested": False,
            "robot_commands_emitted": 0,
            "control_authorized": False,
            "dynamic_anti_decoupling_pass_claimed": False,
        }
    profile = derive_relative_periodic_profile(contract, progress)
    resistance = derive_directional_resistance(
        contract,
        direction=direction,
        radial_deflection_m=profile["radial_deflection_m"],
    )
    if not resistance["within_authorized_moment_component"]:
        return {
            "schema_version": 1,
            "task_id": TASK_ID,
            "request_ready": False,
            "rejection_code": "DETENT_RESISTANCE_EXCEEDS_AUTHORIZED_MOMENT",
            "relative_profile": profile,
            "resistance": resistance,
            "physical_model_internal_constraint_requested": False,
            "software_pose_write_requested": False,
            "force_or_moment_command_requested": False,
            "robot_commands_emitted": 0,
            "control_authorized": False,
            "dynamic_anti_decoupling_pass_claimed": False,
        }
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "request_ready": True,
        "rejection_code": None,
        "relative_profile": profile,
        "resistance": resistance,
        "physical_model_internal_constraint_requested": True,
        "software_pose_write_requested": False,
        "force_or_moment_command_requested": False,
        "robot_commands_emitted": 0,
        "control_authorized": False,
        "dynamic_anti_decoupling_pass_claimed": False,
    }


def current_readiness(contract: AntiDecouplingContract) -> AntiDecouplingReadiness:
    return AntiDecouplingReadiness(
        e5_evidence_path=E5_RESULT_PATH,
        e5_evidence_sha256=contract.current_e5_evidence_sha256,
        e5_dynamic_thread_follow_passed=(
            contract.current_e5_dynamic_thread_follow_passed
        ),
        physical_detent_runtime_ready=False,
        absolute_phase_origin_authorized=False,
    )


__all__ = [
    "E5_RESULT_PATH",
    "HIGH_DETAIL_FORWARD_PROXY_PEAK_NM",
    "AntiDecouplingContract",
    "AntiDecouplingReadiness",
    "build_anti_decoupling_request",
    "current_readiness",
    "derive_directional_resistance",
    "derive_relative_periodic_profile",
    "evaluate_anti_decoupling_gate",
    "load_anti_decoupling_contract",
]
