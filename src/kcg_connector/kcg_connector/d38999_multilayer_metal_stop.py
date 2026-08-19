"""Evidence-bound physical metal-stop interface for the multilayer model.

The stop is a collision pair in the physics model.  Nominal separation is
audit metadata, never a controller-side contact boolean.  PhysX pair evidence
may be inspected post hoc, but it cannot flow back into task control.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "EIGHT-HOUR-E7-METAL-STOP"
E6_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-E6-ANTI-DECOUPLING-RESISTANCE/TASK_RESULT.json"
)
MAPPING_PATH = "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json"
ASSEMBLY_ASSET_PATH = (
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
    "D38999_ASSEMBLY_CONTROL_V1.usda"
)
FIXED_COLLISION_ROLE = "continuous_real_metal_stop_fixed"
PLUG_COLLISION_ROLE = "continuous_real_metal_stop_plug"

FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml": (
        "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e"
    ),
    MAPPING_PATH: (
        "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783"
    ),
    ASSEMBLY_ASSET_PATH: (
        "26c44d86372fa9db64acd6503499f7335ddbabb14b8dd82c7ec7e31c6dc37cec"
    ),
    E6_RESULT_PATH: (
        "f86c369e3988126304870d41e42cfcfc3fb6a4604c44fb6e168ea286ab9e8579"
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


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class MetalStopContract:
    source_rows: tuple[tuple[str, str], ...]
    source_class: str
    definition: str
    fixed_stop_path: str
    plug_stop_path: str
    fixed_collision_role: str
    plug_collision_role: str
    nominal_bottoming_separation_m: float
    calibration_range_m: tuple[float, float]
    calibration_range_is_public_bottoming_depth_claim: bool
    fixed_cap_radius_m: float
    fixed_cap_axial_thickness_m: float
    plug_stop_distribution_radius_m: float
    event_name: str
    event_ordinal: int
    event_position_tolerance_m: float
    maximum_hard_penetration_m: float
    determined_by_physical_collision_not_pose_or_boolean: bool
    continuous_real_collision_required: bool
    current_e6_outcome: str
    current_e6_dynamic_anti_decoupling_passed: bool
    current_e6_evidence_sha256: str


@dataclass(frozen=True)
class MetalStopReadiness:
    e6_evidence_path: str
    e6_evidence_sha256: str
    e6_dynamic_anti_decoupling_passed: bool
    physical_collision_runtime_ready: bool
    posthoc_contact_audit_channel_ready: bool
    controller_consumes_contact_truth: bool
    controller_uses_pose_boolean: bool


@dataclass(frozen=True)
class PosthocMetalStopEvidence:
    evidence_kind: str
    run_id: str
    fixed_stop_path: str
    plug_stop_path: str
    physical_contact_active: bool
    normal_impulse_n_s: float
    measured_separation_m: float
    maximum_hard_penetration_m: float
    solver_error_count: int
    post_run_pose_write_count: int
    controller_consumed_contact_truth: bool
    controller_used_pose_boolean: bool
    offline_fixture_only: bool


def load_metal_stop_contract(repository_root: str | Path) -> MetalStopContract:
    """Bind master, mapping, authored asset, high-detail source, and E6 evidence."""

    root = Path(repository_root).resolve()
    rows: list[tuple[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"frozen E7 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen E7 source hash mismatch: {relative}")
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
    stop = _mapping(master.get("metal_stop"), "master.metal_stop")
    collision = _mapping(
        stop.get("assembly_control_collision"),
        "master.metal_stop.assembly_control_collision",
    )
    events = _mapping(master.get("assembly_events"), "master.assembly_events")
    ordered = events.get("ordered")
    if not isinstance(ordered, list) or len(ordered) != 7:
        raise ValueError("master assembly event list must contain seven rows")
    event = _mapping(ordered[-1], "seventh assembly event")
    limits = _mapping(master.get("acceptance_limits"), "master.acceptance_limits")

    calibration = stop.get("calibration_range_m")
    if not isinstance(calibration, list) or len(calibration) != 2:
        raise ValueError("metal-stop calibration range must contain two values")
    low = _finite(calibration[0], "calibration low")
    high = _finite(calibration[1], "calibration high")
    nominal = _finite(stop.get("nominal_bottoming_separation_m"), "nominal stop")
    if not low <= nominal <= high:
        raise ValueError("nominal metal stop lies outside calibration range")
    if (
        stop.get("source_class") != "equivalent_assumption"
        or stop.get("definition")
        != "physical_receptacle_engaging_shell_to_plug_internal_engaging_shell_contact"
        or stop.get("determined_by_physical_collision_not_pose_or_boolean") is not True
        or stop.get("calibration_range_is_public_bottoming_depth_claim") is not False
        or collision.get("type") != "continuous_real_collision"
        or event.get("ordinal") != 7
        or event.get("name") != "shell_to_shell_metal_bottoming"
        or not math.isclose(
            _finite(event.get("nominal_separation_m"), "event separation"),
            nominal,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("master physical metal-stop boundary changed")

    mapping = _mapping(
        json.loads((root / MAPPING_PATH).read_text(encoding="utf-8")),
        "model mapping",
    )
    mapped_stop = _mapping(mapping.get("metal_stop"), "mapping.metal_stop")
    representations = _mapping(mapping.get("representations"), "representations")
    assembly = _mapping(
        representations.get("D38999_ASSEMBLY_CONTROL_V1"),
        "assembly representation",
    )
    logical_paths = _mapping(assembly.get("logical_paths"), "logical paths")
    fixed_path = str(logical_paths.get("metal_stop_fixed"))
    plug_path = str(logical_paths.get("metal_stop_plug"))
    if (
        mapped_stop != stop
        or "metal_stop" not in assembly.get("continuous_real_collision_roles", [])
        or not fixed_path.startswith(str(assembly.get("pair_root")))
        or not plug_path.startswith(str(assembly.get("pair_root")))
    ):
        raise ValueError("mapping does not preserve the master metal stop")

    asset_text = (root / ASSEMBLY_ASSET_PATH).read_text(encoding="utf-8")
    required_asset_tokens = (
        'custom string kcg:collisionRole = "continuous_real_metal_stop_fixed"',
        'custom string kcg:collisionRole = "continuous_real_metal_stop_plug"',
        'bool physics:collisionEnabled = 1',
        'custom string kcg:eventName = "shell_to_shell_metal_bottoming"',
    )
    if any(token not in asset_text for token in required_asset_tokens):
        raise ValueError("assembly asset lacks authored physical metal-stop metadata")

    high_detail = _mapping(
        yaml.safe_load(
            (root / "src/kcg_connector/config/"
             "d38999_keyed_v3_physical_model_contract_r12_v1.yaml").read_text(
                encoding="utf-8"
            )
        ),
        "high-detail contract",
    )
    public_geometry = _mapping(
        high_detail.get("public_geometry"), "public geometry"
    )
    axial_interface = _mapping(
        public_geometry.get("axial_interface"), "public axial interface"
    )
    public_stop = _mapping(
        axial_interface.get("metal_bottoming"), "public metal bottoming source"
    )
    blueprint = _mapping(
        high_detail.get("a2_collision_authoring_blueprint"),
        "A2 collision authoring blueprint",
    )
    high_stop = _mapping(
        blueprint.get("metal_bottoming"), "metal bottoming proxy"
    )
    if (
        public_stop.get("definition") != stop.get("definition")
        or public_stop.get(
            "public_spec_confirms_surface_pair_but_not_their_datum_assignment"
        ) is not True
        or not math.isclose(
            _finite(
                public_stop.get("full_mate_datum_B_separation_mm"),
                "public-source full-mate proxy",
            ) / 1000.0,
            nominal,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or public_stop.get(
            "calibration_range_is_not_a_public_bottoming_depth_claim"
        ) is not True
        or high_stop.get("response_role") != "hard_metal_bottoming"
        or high_stop.get("old_segmented_collider_count") != 0
        or high_stop.get("only_named_stop_group_to_stop_group_pairs_enabled") is not True
        or not math.isclose(
            _finite(high_stop.get("nominal_bottoming_separation_m"), "proxy nominal"),
            nominal,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("high-detail metal-stop provenance changed")

    e6 = _mapping(
        json.loads((root / E6_RESULT_PATH).read_text(encoding="utf-8")),
        "E6 task result",
    )
    if (
        e6.get("task_id") != "EIGHT-HOUR-E6-ANTI-DECOUPLING-RESISTANCE"
        or e6.get("outcome") != "OFFLINE_PASS"
        or type(e6.get("dynamic_anti_decoupling_pass_claimed")) is not bool
        or e6.get("software_pose_write_requested") is not False
    ):
        raise ValueError("E6 evidence does not support E7")

    return MetalStopContract(
        source_rows=tuple(rows),
        source_class=str(stop.get("source_class")),
        definition=str(stop.get("definition")),
        fixed_stop_path=fixed_path,
        plug_stop_path=plug_path,
        fixed_collision_role=FIXED_COLLISION_ROLE,
        plug_collision_role=PLUG_COLLISION_ROLE,
        nominal_bottoming_separation_m=nominal,
        calibration_range_m=(low, high),
        calibration_range_is_public_bottoming_depth_claim=False,
        fixed_cap_radius_m=_finite(collision.get("fixed_cap_radius_m"), "cap radius"),
        fixed_cap_axial_thickness_m=_finite(
            collision.get("fixed_cap_axial_thickness_m"), "cap thickness"
        ),
        plug_stop_distribution_radius_m=_finite(
            collision.get("plug_stop_distribution_radius_m"), "stop radius"
        ),
        event_name=str(event.get("name")),
        event_ordinal=int(event.get("ordinal")),
        event_position_tolerance_m=_finite(
            limits.get("event_position_tolerance_m"), "event tolerance"
        ),
        maximum_hard_penetration_m=_finite(
            limits.get("maximum_noncompliant_hard_penetration_m"),
            "hard penetration limit",
        ),
        determined_by_physical_collision_not_pose_or_boolean=True,
        continuous_real_collision_required=True,
        current_e6_outcome=str(e6.get("outcome")),
        current_e6_dynamic_anti_decoupling_passed=e6[
            "dynamic_anti_decoupling_pass_claimed"
        ],
        current_e6_evidence_sha256=FROZEN_SOURCES[E6_RESULT_PATH],
    )


def evaluate_metal_stop_gate(
    contract: MetalStopContract,
    readiness: MetalStopReadiness,
) -> str | None:
    if (
        readiness.e6_evidence_path != E6_RESULT_PATH
        or readiness.e6_evidence_sha256 != contract.current_e6_evidence_sha256
    ):
        return "E6_EVIDENCE_ID_MISMATCH"
    if (
        contract.current_e6_dynamic_anti_decoupling_passed is not True
        or readiness.e6_dynamic_anti_decoupling_passed is not True
    ):
        return "E6_ANTI_DECOUPLING_NOT_DYNAMIC"
    if readiness.physical_collision_runtime_ready is not True:
        return "PHYSICAL_METAL_STOP_RUNTIME_NOT_READY"
    if readiness.posthoc_contact_audit_channel_ready is not True:
        return "POSTHOC_CONTACT_AUDIT_CHANNEL_NOT_READY"
    if readiness.controller_consumes_contact_truth is not False:
        return "CONTROLLER_CONTACT_TRUTH_FORBIDDEN"
    if readiness.controller_uses_pose_boolean is not False:
        return "POSE_BOOLEAN_METAL_STOP_FORBIDDEN"
    return None


def build_metal_stop_request(
    contract: MetalStopContract,
    readiness: MetalStopReadiness,
) -> dict[str, Any]:
    """Return a non-commanding collision request; never claim contact occurred."""

    rejection = evaluate_metal_stop_gate(contract, readiness)
    ready = rejection is None
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "request_ready": ready,
        "rejection_code": rejection,
        "fixed_stop_path": contract.fixed_stop_path if ready else None,
        "plug_stop_path": contract.plug_stop_path if ready else None,
        "physical_collision_pair_requested": ready,
        "nominal_separation_is_audit_metadata_only": True,
        "position_boolean_requested": False,
        "contact_truth_routed_to_controller": False,
        "software_pose_write_requested": False,
        "force_or_moment_command_requested": False,
        "robot_commands_emitted": 0,
        "metal_stop_event_claimed": False,
        "dynamic_metal_stop_pass_claimed": False,
        "control_authorized": False,
    }


def audit_posthoc_metal_stop_evidence(
    contract: MetalStopContract,
    evidence: PosthocMetalStopEvidence,
) -> dict[str, Any]:
    """Audit completed-run evidence without exposing it to the controller."""

    impulse = _finite(evidence.normal_impulse_n_s, "normal impulse")
    separation = _finite(evidence.measured_separation_m, "measured separation")
    penetration = _finite(
        evidence.maximum_hard_penetration_m, "maximum hard penetration"
    )
    solver_errors = _nonnegative_integer(
        evidence.solver_error_count, "solver error count"
    )
    pose_writes = _nonnegative_integer(
        evidence.post_run_pose_write_count, "post-run pose write count"
    )
    reasons: list[str] = []
    if evidence.evidence_kind != "posthoc_physx_contact_audit":
        reasons.append("POSTHOC_EVIDENCE_KIND_INVALID")
    if not evidence.run_id:
        reasons.append("RUN_ID_MISSING")
    if (
        evidence.fixed_stop_path != contract.fixed_stop_path
        or evidence.plug_stop_path != contract.plug_stop_path
    ):
        reasons.append("METAL_STOP_PAIR_ID_MISMATCH")
    if evidence.offline_fixture_only:
        reasons.append("OFFLINE_FIXTURE_NOT_DYNAMIC_EVIDENCE")
    if evidence.physical_contact_active is not True or impulse <= 0.0:
        reasons.append("PHYSICAL_CONTACT_NOT_PROVEN")
    if not contract.calibration_range_m[0] <= separation <= contract.calibration_range_m[1]:
        reasons.append("BOTTOMING_SEPARATION_OUTSIDE_CALIBRATION_RANGE")
    if penetration < 0.0 or penetration > contract.maximum_hard_penetration_m:
        reasons.append("HARD_PENETRATION_LIMIT_EXCEEDED")
    if solver_errors != 0:
        reasons.append("SOLVER_ERROR_PRESENT")
    if pose_writes != 0:
        reasons.append("POST_RUN_POSE_WRITE_PRESENT")
    if evidence.controller_consumed_contact_truth is not False:
        reasons.append("CONTROLLER_CONTACT_TRUTH_FORBIDDEN")
    if evidence.controller_used_pose_boolean is not False:
        reasons.append("POSE_BOOLEAN_METAL_STOP_FORBIDDEN")
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "posthoc_only": True,
        "accepted": not reasons,
        "rejection_codes": reasons,
        "metal_stop_event_proven": not reasons,
        "controller_input_allowed": False,
        "control_authorized": False,
    }


def current_readiness(contract: MetalStopContract) -> MetalStopReadiness:
    return MetalStopReadiness(
        e6_evidence_path=E6_RESULT_PATH,
        e6_evidence_sha256=contract.current_e6_evidence_sha256,
        e6_dynamic_anti_decoupling_passed=(
            contract.current_e6_dynamic_anti_decoupling_passed
        ),
        physical_collision_runtime_ready=False,
        posthoc_contact_audit_channel_ready=False,
        controller_consumes_contact_truth=False,
        controller_uses_pose_boolean=False,
    )


__all__ = [
    "E6_RESULT_PATH",
    "MetalStopContract",
    "MetalStopReadiness",
    "PosthocMetalStopEvidence",
    "audit_posthoc_metal_stop_evidence",
    "build_metal_stop_request",
    "current_readiness",
    "evaluate_metal_stop_gate",
    "load_metal_stop_contract",
]
