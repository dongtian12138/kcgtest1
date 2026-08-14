"""Pure planning boundary for an independent visual-XY D38999 pick probe.

The runtime capture adapter below copies only ray-plane estimates and
visibility/depth evidence into a partial :class:`PoseProviderSample`.  Scene
truth is deliberately absent from that function and is accepted only by the
separate post-hoc evaluation helper.  Local fixed-q7 IK shifts the already
validated nominal pregrasp/grasp targets; it is not a global IK or collision
planner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from kcg_connector.d38999_physical_insertion import (
    solve_fixed_q7_tcp_pose,
)
from kcg_connector.d38999_tabletop_pick import (
    D38999TabletopPickConfig,
    iiwa14_grasp_tcp_transform,
    load_d38999_tabletop_pick_config,
)
from kcg_connector.d38999_tabletop_scene import (
    D38999TabletopScene,
    load_d38999_tabletop_scene,
)
from kcg_connector.d38999_visual_xy_control_adapter import (
    DIAGNOSTICS_SCHEMA_VERSION,
    VisualXyAdaptationResult,
    VisualXyControlAdapterContract,
    adapt_visual_xy_to_world_targets,
    load_visual_xy_control_adapter_contract,
)
from kcg_connector.pose_provider import (
    POSE_PROVIDER_SAMPLE_SCHEMA_VERSION,
    PoseProviderPurpose,
    PoseProviderSample,
)
from kcg_connector.rgbd_pose_bootstrap import (
    D38999RgbdBootstrap,
    load_rgbd_bootstrap,
)


SCHEMA_VERSION = "kcg_d38999_visual_xy_pick_probe_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_visual_xy_pick_probe_v1.yaml"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLES = ("loose_plug", "fixed_receptacle")

# CLI selection is intentionally limited to these content-addressed files.
# Adding or changing a trial requires a reviewed source change as well as a new
# config digest; coordinates can never be supplied as free-form CLI values.
APPROVED_PROBE_VARIANTS = {
    "d38999_visual_xy_pick_probe_v1.yaml": {
        "sha256": (
            "4bab601ac0dd27425f29da7e585dc2c10c0877bfd88729f31f5af28ac0ed8d19"
        ),
        "trial_id": "loose_plus_10mm_xy_fixed_nominal",
        "loose_xy_m": (0.530, -0.200),
    },
    "d38999_visual_xy_pick_px20_y0_v1.yaml": {
        "sha256": (
            "7e761f87ecf967a6bc06c8b3397558ea347c21c1d41b5f6e9d02784e048c31f1"
        ),
        "trial_id": "loose_plus_20mm_x_fixed_nominal",
        "loose_xy_m": (0.540, -0.210),
    },
    "d38999_visual_xy_pick_mx20_y0_v1.yaml": {
        "sha256": (
            "b5cec2e61846b85d04123c5e48db857578176ec3024fbb04a862e79fd9997f65"
        ),
        "trial_id": "loose_minus_20mm_x_fixed_nominal",
        "loose_xy_m": (0.500, -0.210),
    },
    "d38999_visual_xy_pick_x0_my20_v1.yaml": {
        "sha256": (
            "f600c167cfecb379dc386259f21e5c5eb18c8dd68bed08e9bf8013f664b0d929"
        ),
        "trial_id": "loose_minus_20mm_y_fixed_nominal",
        "loose_xy_m": (0.520, -0.230),
    },
}


@dataclass(frozen=True)
class LocalIkPolicy:
    maximum_iterations: int
    damping: float
    maximum_fk_position_error_m: float
    maximum_fk_orientation_error_rad: float
    maximum_abs_joint_delta_from_nominal_rad: float


@dataclass(frozen=True)
class VisualXyPickProbeContract:
    schema_version: str
    enabled_by_default: bool
    status: str
    input_paths: dict[str, Path]
    adapter: VisualXyControlAdapterContract
    pick: D38999TabletopPickConfig
    tabletop: D38999TabletopScene
    rgbd: D38999RgbdBootstrap
    trial_id: str
    authored_loose_xy_m: tuple[float, float]
    authored_fixed_xy_m: tuple[float, float]
    orientation_source: str
    loose_yaw_rad: float
    fixed_yaw_rad: float
    confidence_kind: str
    per_capture_xy_error_bound_m: float
    local_ik: LocalIkPolicy
    output_directory: str
    cpu_plan_filename: str
    report_filename: str
    boundaries: dict[str, bool]


@dataclass(frozen=True)
class VisualXyPickPlan:
    """CPU-verifiable target/IK plan consumed by the independent smoke."""

    trial_id: str
    capture_id: str
    adapter_result: VisualXyAdaptationResult
    arm_targets_rad: dict[str, tuple[float, ...]]
    tcp_targets_world_m: dict[str, tuple[float, float, float]]
    fk_position_errors_m: dict[str, float]
    fk_orientation_errors_rad: dict[str, float]
    maximum_abs_joint_delta_from_nominal_rad: float
    planned_peak_joint_speed_rad_s: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "CPU_PLAN_READY_FOR_INDEPENDENT_ISAAC_PROBE",
            "trial_id": self.trial_id,
            "capture_id": self.capture_id,
            "adapter": self.adapter_result.to_mapping(),
            "arm_targets_rad": {
                name: list(values)
                for name, values in sorted(self.arm_targets_rad.items())
            },
            "tcp_targets_world_m": {
                name: list(values)
                for name, values in sorted(
                    self.tcp_targets_world_m.items()
                )
            },
            "fk_position_errors_m": dict(
                sorted(self.fk_position_errors_m.items())
            ),
            "fk_orientation_errors_rad": dict(
                sorted(self.fk_orientation_errors_rad.items())
            ),
            "maximum_abs_joint_delta_from_nominal_rad": (
                self.maximum_abs_joint_delta_from_nominal_rad
            ),
            "planned_peak_joint_speed_rad_s": (
                self.planned_peak_joint_speed_rad_s
            ),
            "uses_truth_xy_for_target": False,
            "orientation_source": self.adapter_result.orientation_source,
            "full_6d": False,
            "collision_planned": False,
            "production_control_authorized": False,
            "gpu_or_physx_validated": False,
        }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty unpadded text")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
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
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain {length} finite numbers")
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError(
            f"{label} must contain {length} finite numbers"
        ) from error
    if len(result) != length:
        raise ValueError(f"{label} must contain {length} finite numbers")
    return tuple(
        _finite(item, f"{label}[{index}]")
        for index, item in enumerate(result)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_inputs(value: Any, repository: Path) -> dict[str, Path]:
    inputs = _mapping(value, "inputs")
    names = {
        "visual_xy_adapter",
        "nominal_pick",
        "tabletop_scene",
        "rgbd_config",
    }
    _exact(inputs, names, "inputs")
    resolved = {}
    for name in sorted(names):
        label = f"inputs.{name}"
        item = _mapping(inputs[name], label)
        _exact(item, {"path", "sha256"}, label)
        relative = Path(_text(item["path"], f"{label}.path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label}.path must be repository-relative")
        expected = _text(item["sha256"], f"{label}.sha256")
        if not _SHA256.fullmatch(expected):
            raise ValueError(f"{label}.sha256 must be lowercase SHA-256")
        path = (repository / relative).resolve()
        if not path.is_file() or repository not in path.parents:
            raise ValueError(f"{label} is missing or outside repository")
        if _sha256(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
        resolved[name] = path
    return resolved


def _approved_probe_variant(
    config_path: Path, root: Path
) -> Mapping[str, Any]:
    """Resolve one repository-owned config to its immutable manifest entry."""

    approved = APPROVED_PROBE_VARIANTS.get(config_path.name)
    approved_path = (
        root / "src/kcg_connector/config" / config_path.name
    ).resolve()
    if approved is None or config_path != approved_path:
        raise ValueError(
            "visual XY pick config is not an approved repository variant"
        )
    if _sha256(config_path) != approved["sha256"]:
        raise ValueError("visual XY pick approved config SHA-256 mismatch")
    return approved


def load_visual_xy_pick_probe_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repository: str | Path | None = None,
) -> VisualXyPickProbeContract:
    """Load the independent probe and cross-check every reused contract."""

    config_path = Path(path).expanduser().resolve()
    root = (
        Path(repository).expanduser().resolve()
        if repository is not None
        else config_path.parents[3]
    )
    approved = _approved_probe_variant(config_path, root)
    document = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "document"
    )
    _exact(
        document,
        {
            "schema_version",
            "enabled_by_default",
            "status",
            "inputs",
            "trial",
            "rgbd_observation",
            "local_fixed_q7_ik",
            "execution",
            "output",
            "boundaries",
        },
        "document",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("visual XY pick probe schema is unsupported")
    if document["enabled_by_default"] is not False:
        raise ValueError(
            "visual XY pick probe must remain disabled by default"
        )
    status = _text(document["status"], "status")
    if status != "prepared_independent_visual_xy_pick_probe":
        raise ValueError("visual XY pick probe status is unsupported")
    inputs = _resolve_inputs(document["inputs"], root)
    adapter = load_visual_xy_control_adapter_contract(
        inputs["visual_xy_adapter"], repository=root
    )
    pick = load_d38999_tabletop_pick_config(inputs["nominal_pick"])
    tabletop = load_d38999_tabletop_scene(inputs["tabletop_scene"])
    rgbd = load_rgbd_bootstrap(inputs["rgbd_config"])
    if (
        adapter.input_paths["nominal_pick"] != inputs["nominal_pick"]
        or adapter.input_paths["rgbd_config"] != inputs["rgbd_config"]
        or adapter.vision_contract.input_paths["tabletop_scene"]
        != inputs["tabletop_scene"]
        or Path(rgbd.tabletop_config).name != inputs["tabletop_scene"].name
    ):
        raise ValueError("visual pick probe dependency frames do not agree")

    trial = _mapping(document["trial"], "trial")
    _exact(
        trial,
        {
            "trial_id",
            "author_before_physics",
            "loose_plug_xy_m",
            "fixed_receptacle_xy_m",
            "orientation_source",
            "loose_yaw_rad",
            "fixed_yaw_rad",
        },
        "trial",
    )
    loose_xy = _vector(trial["loose_plug_xy_m"], 2, "loose_plug_xy_m")
    fixed_xy = _vector(
        trial["fixed_receptacle_xy_m"], 2, "fixed_receptacle_xy_m"
    )
    if (
        trial["author_before_physics"] is not True
        or trial["trial_id"] != approved["trial_id"]
        or loose_xy != approved["loose_xy_m"]
        or fixed_xy != adapter.nominal_fixed_xy_m
        or trial["orientation_source"] != "registered_nominal"
        or _finite(trial["loose_yaw_rad"], "loose_yaw_rad") != 0.0
        or _finite(trial["fixed_yaw_rad"], "fixed_yaw_rad") != 0.0
    ):
        raise ValueError(
            "visual pick trial must match its approved enumerated variant"
        )
    if not (
        adapter.vision_contract.loose_plug.x_m.contains(loose_xy[0])
        and adapter.vision_contract.loose_plug.y_m.contains(loose_xy[1])
        and adapter.vision_contract.fixed_receptacle.x_m.contains(fixed_xy[0])
        and adapter.vision_contract.fixed_receptacle.y_m.contains(fixed_xy[1])
    ):
        raise ValueError("visual pick authored XY leaves bounded domain")

    observation = _mapping(document["rgbd_observation"], "rgbd_observation")
    _exact(
        observation,
        {
            "estimator_kind",
            "confidence_kind",
            "per_capture_xy_error_bound_source",
            "per_capture_xy_error_bound_m",
            "truth_xy_keys_forbidden_in_provider_adapter_path",
        },
        "rgbd_observation",
    )
    if (
        observation["estimator_kind"] != adapter.estimator_kind
        or observation["confidence_kind"]
        != "minimum_normalized_visibility_depth_gate_score"
        or observation["per_capture_xy_error_bound_source"]
        != "five_anchor_empirical_10mm_contract"
        or observation["per_capture_xy_error_bound_m"]
        != adapter.gates.maximum_xy_error_bound_m
        or observation["truth_xy_keys_forbidden_in_provider_adapter_path"]
        is not True
    ):
        raise ValueError("visual pick RGB-D observation scope changed")

    ik = _mapping(document["local_fixed_q7_ik"], "local_fixed_q7_ik")
    _exact(
        ik,
        {
            "maximum_iterations",
            "damping",
            "maximum_fk_position_error_m",
            "maximum_fk_orientation_error_rad",
            "maximum_abs_joint_delta_from_nominal_rad",
            "q7_preserved",
            "arbitrary_pose_ik_claimed",
        },
        "local_fixed_q7_ik",
    )
    ik_policy = LocalIkPolicy(
        maximum_iterations=_positive_integer(
            ik["maximum_iterations"], "maximum_iterations"
        ),
        damping=_positive(ik["damping"], "damping"),
        maximum_fk_position_error_m=_positive(
            ik["maximum_fk_position_error_m"],
            "maximum_fk_position_error_m",
        ),
        maximum_fk_orientation_error_rad=_positive(
            ik["maximum_fk_orientation_error_rad"],
            "maximum_fk_orientation_error_rad",
        ),
        maximum_abs_joint_delta_from_nominal_rad=_positive(
            ik["maximum_abs_joint_delta_from_nominal_rad"],
            "maximum_abs_joint_delta_from_nominal_rad",
        ),
    )
    if (
        ik_policy.maximum_iterations != 50
        or ik_policy.damping != 1.0e-6
        or ik_policy.maximum_fk_position_error_m != 1.0e-7
        or ik_policy.maximum_fk_orientation_error_rad != 1.0e-7
        or ik_policy.maximum_abs_joint_delta_from_nominal_rad != 0.150
        or ik["q7_preserved"] is not True
        or ik["arbitrary_pose_ik_claimed"] is not False
    ):
        raise ValueError("visual pick local IK policy changed")

    execution = _mapping(document["execution"], "execution")
    _exact(
        execution,
        {
            "interpolation",
            "reuse_nominal_home_and_safe_mid_segments",
            "reuse_nominal_hand_targets_and_durations",
            "execute_phases",
            "reuse_pick_finite_contact_torque_lift_slip_gates",
            "operational_torque_target_nm",
            "hard_stop_nm",
        },
        "execution",
    )
    if (
        execution["interpolation"] != "minimum_jerk"
        or execution["reuse_nominal_home_and_safe_mid_segments"] is not True
        or execution["reuse_nominal_hand_targets_and_durations"] is not True
        or tuple(execution["execute_phases"])
        != ("HOME", "VISUAL_PREFLIGHT", "PREGRASP", "GRASP", "LIFT", "HOLD")
        or execution["reuse_pick_finite_contact_torque_lift_slip_gates"]
        is not True
        or execution["operational_torque_target_nm"]
        != pick.sensing.operational_torque_target_nm
        or execution["hard_stop_nm"]
        != pick.sensing.maximum_absolute_torque_delta_nm
    ):
        raise ValueError("visual pick execution policy changed")

    output = _mapping(document["output"], "output")
    _exact(
        output,
        {"directory", "cpu_plan_filename", "report_filename"},
        "output",
    )
    output_directory = _text(output["directory"], "output.directory")
    output_relative = Path(output_directory)
    if output_relative.is_absolute() or ".." in output_relative.parts:
        raise ValueError("output.directory must be repository-relative")

    boundaries = dict(_mapping(document["boundaries"], "boundaries"))
    expected_boundaries = {
        "explicit_opt_in_required": True,
        "existing_e2e_modified": False,
        "existing_regrasp_modified": False,
        "robot_asset_modified": False,
        "frozen_baseline_modified": False,
        "truth_xy_used_for_target": False,
        "truth_xy_evaluation_only": True,
        "orientation_estimated_from_rgbd": False,
        "full_6d_claimed": False,
        "production_control_authorized": False,
        "collision_planned": False,
        "arbitrary_pose_reachability_claimed": False,
        "object_pose_writes_after_physics_allowed": False,
        "real_assembly_success_claimed": False,
    }
    if boundaries != expected_boundaries:
        raise ValueError("visual pick probe boundaries changed")

    return VisualXyPickProbeContract(
        schema_version=SCHEMA_VERSION,
        enabled_by_default=False,
        status=status,
        input_paths=inputs,
        adapter=adapter,
        pick=pick,
        tabletop=tabletop,
        rgbd=rgbd,
        trial_id=_text(trial["trial_id"], "trial_id"),
        authored_loose_xy_m=loose_xy,
        authored_fixed_xy_m=fixed_xy,
        orientation_source=trial["orientation_source"],
        loose_yaw_rad=0.0,
        fixed_yaw_rad=0.0,
        confidence_kind=observation["confidence_kind"],
        per_capture_xy_error_bound_m=observation[
            "per_capture_xy_error_bound_m"
        ],
        local_ik=ik_policy,
        output_directory=output_directory,
        cpu_plan_filename=_text(
            output["cpu_plan_filename"], "cpu_plan_filename"
        ),
        report_filename=_text(output["report_filename"], "report_filename"),
        boundaries=boundaries,
    )


def pose_provider_sample_from_rgbd_metrics(
    contract: VisualXyPickProbeContract,
    capture_metrics: Mapping[str, Any],
    *,
    timestamp_s: Real,
    capture_id: str,
) -> PoseProviderSample:
    """Copy visual estimates only; scene-truth fields are never consulted."""

    metrics = _mapping(capture_metrics, "capture_metrics")
    timestamp = _finite(timestamp_s, "timestamp_s")
    diagnostics_endpoints = {}
    acceptance = contract.rgbd.acceptance
    visibility = contract.adapter.vision_contract.visibility
    semantic_ids = _mapping(
        metrics.get("endpoint_semantic_ids"), "endpoint_semantic_ids"
    )
    for role in _ROLES:
        endpoint = _mapping(metrics.get(role), f"capture_metrics.{role}")
        estimate = _vector(
            endpoint.get(
                "ray_plane_registered_model_height_world_xyz_m"
            ),
            3,
            f"{role}.ray_plane_estimate",
        )
        depth = _mapping(endpoint.get("mask_depth"), f"{role}.mask_depth")
        pixel_count = _positive(depth.get("pixel_count"), "pixel_count")
        valid_count = _positive(
            depth.get("valid_depth_count"), "valid_depth_count"
        )
        visible_fraction = _positive(
            depth.get("visible_fraction"), "visible_fraction"
        )
        valid_fraction = min(1.0, valid_count / pixel_count)
        center = _mapping(
            endpoint.get("semantic_mask_center"),
            f"{role}.semantic_mask_center",
        )
        role_semantics = semantic_ids.get(role)
        semantic_gate = bool(
            isinstance(role_semantics, list)
            and role_semantics
            and all(
                isinstance(value, Integral) and int(value) not in (0, 1)
                for value in role_semantics
            )
        )
        depth_range_gate = bool(
            _finite(depth.get("minimum_depth_m"), "minimum_depth_m")
            >= acceptance.minimum_valid_depth_m
            and _finite(depth.get("maximum_depth_m"), "maximum_depth_m")
            <= acceptance.maximum_valid_depth_m
        )
        confidence = min(
            valid_fraction,
            min(1.0, pixel_count / acceptance.minimum_pixels_per_endpoint),
            min(
                1.0,
                visible_fraction
                / acceptance.minimum_visible_fraction_per_endpoint,
            ),
            1.0 if center.get("in_frame") is True else 0.0,
            1.0 if semantic_gate else 0.0,
            1.0 if depth_range_gate else 0.0,
        )
        if valid_fraction < visibility.minimum_valid_depth_fraction_in_mask:
            confidence = 0.0
        diagnostics_endpoints[role] = {
            "estimated_world_xy_m": list(estimate[:2]),
            "timestamp_s": timestamp,
            "confidence": float(confidence),
            "xy_error_bound_m": contract.per_capture_xy_error_bound_m,
        }
    return PoseProviderSample(
        schema_version=POSE_PROVIDER_SAMPLE_SCHEMA_VERSION,
        purpose=PoseProviderPurpose.PREFLIGHT,
        provider_id="d38999.masked_rgbd.ray_plane_xy_gate_score",
        provider_version="v1",
        capture_id=_text(capture_id, "capture_id"),
        clock_domain=contract.adapter.required_clock_domain,
        control_frame=contract.adapter.required_control_frame,
        calibration_sha256=(
            contract.adapter.required_calibration_sha256
        ),
        pair=None,
        reference_truth_pair=None,
        full_6d=False,
        keyed_orientation_observed=False,
        uses_truth_position=False,
        uses_truth_orientation=False,
        control_authorized=False,
        preflight_passed=False,
        diagnostics={
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "estimator": contract.adapter.estimator_kind,
            "endpoints": diagnostics_endpoints,
        },
    )


def _rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0))
    return math.acos(cosine)


def _minimum_jerk_peak_speed(
    first: Sequence[float], second: Sequence[float], duration_s: float
) -> float:
    # derivative peak of 10t^3 - 15t^4 + 6t^5 is 1.875 at t=0.5
    return 1.875 * max(
        abs(float(right) - float(left))
        for left, right in zip(first, second)
    ) / duration_s


def build_visual_xy_pick_plan(
    contract: VisualXyPickProbeContract,
    sample: PoseProviderSample,
    *,
    now_s: Real,
    explicit_probe_opt_in: bool,
) -> VisualXyPickPlan:
    """Adapt visual XY and solve nearby pregrasp/grasp fixed-q7 IK."""

    adapter_result = adapt_visual_xy_to_world_targets(
        contract.adapter,
        sample,
        now_s=now_s,
        explicit_independent_probe_opt_in=explicit_probe_opt_in,
        orientation_source=contract.orientation_source,
    )
    if not adapter_result.eligible_for_independent_probe:
        raise ValueError(
            "visual XY adapter rejected probe target: "
            + ",".join(adapter_result.rejection_reasons)
        )
    tcp_targets = {
        name: adapter_result.world_targets[name]
        for name in (
            "pregrasp_tcp",
            "closure_clearance_tcp",
            "grasp_tcp",
        )
    }
    nominal_arms = {
        "pregrasp": contract.pick.motion.approach_segments[-1].target_arm_rad,
        "closure_clearance": contract.pick.motion.closure_clearance_arm_rad,
        "grasp": contract.pick.motion.grasp_arm_rad,
    }
    target_names = {
        "pregrasp": "pregrasp_tcp",
        "closure_clearance": "closure_clearance_tcp",
        "grasp": "grasp_tcp",
    }
    solved = {}
    position_errors = {}
    orientation_errors = {}
    maximum_delta = 0.0
    for name, nominal in nominal_arms.items():
        target_name = target_names[name]
        nominal_transform = np.asarray(
            iiwa14_grasp_tcp_transform(nominal), dtype=np.float64
        )
        arm = solve_fixed_q7_tcp_pose(
            nominal,
            tcp_targets[target_name],
            target_rotation=nominal_transform[:3, :3],
            maximum_iterations=contract.local_ik.maximum_iterations,
            damping=contract.local_ik.damping,
        )
        transform = np.asarray(
            iiwa14_grasp_tcp_transform(arm), dtype=np.float64
        )
        position_error = float(
            np.linalg.norm(
                transform[:3, 3]
                - np.asarray(tcp_targets[target_name], dtype=np.float64)
            )
        )
        orientation_error = _rotation_error(
            transform[:3, :3], nominal_transform[:3, :3]
        )
        delta = max(abs(value - seed) for value, seed in zip(arm, nominal))
        if arm[6] != nominal[6]:
            raise ValueError(f"{name} local IK changed q7")
        if (
            position_error > contract.local_ik.maximum_fk_position_error_m
            or orientation_error
            > contract.local_ik.maximum_fk_orientation_error_rad
            or delta
            > contract.local_ik.maximum_abs_joint_delta_from_nominal_rad
        ):
            raise ValueError(f"{name} local IK leaves bounded acceptance")
        solved[name] = arm
        position_errors[name] = position_error
        orientation_errors[name] = orientation_error
        maximum_delta = max(maximum_delta, delta)

    safe_mid = contract.pick.motion.approach_segments[-2]
    final_approach = contract.pick.motion.approach_segments[-1]
    speed_candidates = (
        _minimum_jerk_peak_speed(
            safe_mid.target_arm_rad,
            solved["pregrasp"],
            final_approach.duration_s,
        ),
        _minimum_jerk_peak_speed(
            solved["pregrasp"],
            solved["closure_clearance"],
            contract.pick.motion.descent_duration_s,
        ),
        _minimum_jerk_peak_speed(
            solved["closure_clearance"],
            solved["grasp"],
            contract.pick.motion.closed_seating_duration_s,
        ),
        _minimum_jerk_peak_speed(
            solved["grasp"],
            solved["pregrasp"],
            contract.pick.motion.lift_duration_s,
        ),
    )
    peak_speed = max(speed_candidates)
    if peak_speed > (
        contract.pick.acceptance.maximum_observed_joint_speed_rad_s
    ):
        raise ValueError("visual XY CPU plan exceeds nominal joint-speed gate")
    return VisualXyPickPlan(
        trial_id=contract.trial_id,
        capture_id=sample.capture_id,
        adapter_result=adapter_result,
        arm_targets_rad=solved,
        tcp_targets_world_m=tcp_targets,
        fk_position_errors_m=position_errors,
        fk_orientation_errors_rad=orientation_errors,
        maximum_abs_joint_delta_from_nominal_rad=maximum_delta,
        planned_peak_joint_speed_rad_s=peak_speed,
    )


def evaluate_visual_xy_truth_only(
    plan: VisualXyPickPlan,
    *,
    loose_truth_xy_m: Sequence[Real],
    fixed_truth_xy_m: Sequence[Real],
) -> dict[str, Any]:
    """Compute post-hoc XY errors without feeding truth back into targets."""

    loose_truth = _vector(loose_truth_xy_m, 2, "loose_truth_xy_m")
    fixed_truth = _vector(fixed_truth_xy_m, 2, "fixed_truth_xy_m")
    # Reuse the immutable adapter outputs rather than reconstructing them from
    # authored scene truth or duplicated nominal endpoint constants.
    loose_estimate = plan.tcp_targets_world_m["grasp_tcp"][:2]
    fixed_estimate = plan.adapter_result.world_targets["engage_tcp"][:2]
    return {
        "scope": "post_hoc_truth_evaluation_not_target_input",
        "loose_xy_error_m": math.dist(loose_estimate, loose_truth),
        "fixed_xy_error_m": math.dist(fixed_estimate, fixed_truth),
        "truth_xy_used_for_target": False,
    }


__all__ = [
    "APPROVED_PROBE_VARIANTS",
    "DEFAULT_CONFIG_PATH",
    "SCHEMA_VERSION",
    "LocalIkPolicy",
    "VisualXyPickPlan",
    "VisualXyPickProbeContract",
    "build_visual_xy_pick_plan",
    "evaluate_visual_xy_truth_only",
    "load_visual_xy_pick_probe_contract",
    "pose_provider_sample_from_rgbd_metrics",
]
