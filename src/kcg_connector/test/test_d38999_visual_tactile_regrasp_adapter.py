"""Pure tests for the visual+tactile nut-regrasp target seam."""

import ast
import copy
from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_tactile_engage_probe import EngageState
from kcg_connector.d38999_visual_tactile_regrasp_adapter import (
    DEFAULT_CONFIG_PATH,
    POST_READY_EXECUTION_ORDER,
    READY_INPUT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TARGET_ORDER,
    build_cpu_sample_artifact,
    build_visual_tactile_regrasp_plan,
    load_visual_tactile_regrasp_adapter_contract,
    parse_tactile_ready_center,
    validate_cpu_sample_artifact,
)
from kcg_connector.d38999_visual_xy_control_adapter import (
    VisualXyAdaptationResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "kcg_connector/d38999_visual_tactile_regrasp_adapter.py"
)
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts/kcg_connector/"
    "d38999_visual_tactile_regrasp_adapter_v1/cpu_sample_plan.json"
)


def _contract():
    return load_visual_tactile_regrasp_adapter_contract(
        repository=PROJECT_ROOT
    )


def _visual(contract, delta=(0.0010, -0.0015), capture_id="capture-a"):
    world_targets = {}
    for name, nominal in contract.visual.nominal_targets.items():
        role_delta = (
            delta
            if contract.visual.target_roles[name] == "fixed_receptacle"
            else (0.0, 0.0)
        )
        world_targets[name] = (
            nominal[0] + role_delta[0],
            nominal[1] + role_delta[1],
            nominal[2],
        )
    return VisualXyAdaptationResult(
        status="ELIGIBLE_FOR_INDEPENDENT_VISUAL_XY_PROBE",
        eligible_for_independent_probe=True,
        rejection_reasons=(),
        capture_id=capture_id,
        translation_source=(
            "vision_semantic_mask_ray_plane_registered_height_xy"
        ),
        orientation_source="registered_nominal",
        uses_truth_orientation=False,
        loose_translation_xy_m=(0.0, 0.0),
        fixed_translation_xy_m=delta,
        world_targets=world_targets,
        validation_maximum_observed_xy_error_m=0.006183,
    )


def _ready(
    *,
    offset=(0.0004, 0.0002),
    state="READY_FOR_EXISTING_PROXY_TWIST",
    capture_id="capture-a",
    truth_search=False,
    truth_target=False,
    scope="cpu_contract_fixture_not_runtime_evidence",
    task_frame_id="connector_task_frame",
    task_rotation=((1.0, 0.0, 0.0),
                   (0.0, 1.0, 0.0),
                   (0.0, 0.0, 1.0)),
):
    return parse_tactile_ready_center(
        {
            "schema_version": READY_INPUT_SCHEMA_VERSION,
            "state": state,
            "capture_id": capture_id,
            "task_frame_id": task_frame_id,
            "task_rotation_world": [list(row) for row in task_rotation],
            "search_offset_task_xy_m": list(offset),
            "offset_source": "accepted_force_moment_search_offset_xy",
            "truth_used_for_tactile_search": truth_search,
            "posthoc_truth_used_for_target": truth_target,
            "source_scope": scope,
        }
    )


def test_contract_is_pure_disabled_and_hash_bound():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {"isaacsim", "omni", "pxr", "rclpy", "torch", "cv2", "open3d"}
    )

    contract = _contract()
    assert contract.schema_version == SCHEMA_VERSION
    assert contract.enabled_by_default is False
    assert contract.status.endswith("not_gpu_validated")
    assert tuple(contract.target_seeds) == TARGET_ORDER
    assert contract.required_ready_state is (
        EngageState.READY_FOR_EXISTING_PROXY_TWIST
    )
    assert contract.required_task_frame_id == "connector_task_frame"
    assert contract.expected_task_rotation_world == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert contract.boundaries["existing_runner_modified"] is False
    assert contract.boundaries["active_config_modified"] is False
    assert contract.boundaries["gpu_or_physx_validated"] is False
    assert contract.boundaries["production_control_authorized"] is False


def test_config_schema_is_exact_and_fail_closed(tmp_path):
    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    document["unexpected"] = True
    path = tmp_path / "extra.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_visual_tactile_regrasp_adapter_contract(
            path, repository=PROJECT_ROOT
        )

    document.pop("unexpected")
    document["enabled_by_default"] = True
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="remain disabled"):
        load_visual_tactile_regrasp_adapter_contract(
            path, repository=PROJECT_ROOT
        )


def test_ready_input_schema_rejects_nonready_and_extra_fields():
    waiting = _ready(state="ENGAGE_HOLD")
    assert waiting.state is EngageState.ENGAGE_HOLD
    with pytest.raises(ValueError, match="keys differ"):
        parse_tactile_ready_center(
            {
                "schema_version": READY_INPUT_SCHEMA_VERSION,
                "state": "READY_FOR_EXISTING_PROXY_TWIST",
                "capture_id": "capture-a",
                "task_frame_id": "connector_task_frame",
                "task_rotation_world": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "search_offset_task_xy_m": [0.0, 0.0],
                "offset_source": "accepted_force_moment_search_offset_xy",
                "truth_used_for_tactile_search": False,
                "posthoc_truth_used_for_target": False,
                "source_scope": "runtime_tactile_engage_evidence",
                "truth_pose": [99.0, 99.0, 99.0],
            }
        )


def test_accepted_center_shifts_regrasp_retreat_and_assembly_targets():
    contract = _contract()
    visual = _visual(contract)
    ready = _ready()
    plan = build_visual_tactile_regrasp_plan(
        contract,
        visual,
        ready,
        explicit_adapter_opt_in=True,
    )

    assert plan.visual_fixed_delta_xy_m == pytest.approx((0.001, -0.0015))
    assert plan.tactile_search_offset_task_xy_m == pytest.approx(
        (0.0004, 0.0002)
    )
    assert plan.tactile_search_offset_world_m == pytest.approx(
        (0.0004, 0.0002, 0.0)
    )
    assert plan.accepted_center_delta_xy_m == pytest.approx(
        (0.0014, -0.0013)
    )
    assert plan.accepted_center_world_xy_m == pytest.approx(
        (0.5514, 0.1837)
    )
    assert tuple(plan.tcp_targets_world_m) == TARGET_ORDER
    assert plan.tcp_targets_world_m["engage_hold"] == pytest.approx(
        (0.5514, 0.1837, 0.31298)
    )
    assert plan.tcp_targets_world_m["safe_retreat"] == pytest.approx(
        (0.5514, 0.1837, 0.3435)
    )
    nominal_regrasp = contract.target_seeds[
        "nut_only_regrasp"
    ].tcp_position_world_m
    assert plan.tcp_targets_world_m["nut_only_regrasp"] == pytest.approx(
        (
            nominal_regrasp[0] + 0.0014,
            nominal_regrasp[1] - 0.0013,
            nominal_regrasp[2],
        )
    )
    for name, solved in plan.arm_targets_rad.items():
        assert solved[6] == pytest.approx(
            contract.target_seeds[name].arm_rad[6], abs=1.0e-12
        )
    assert max(plan.fk_position_errors_m.values()) < 1.0e-9
    assert max(plan.fk_orientation_errors_rad.values()) == 0.0
    assert max(plan.target_joint_deltas_from_nominal_rad.values()) < 0.02

    report = plan.to_mapping()
    assert report["target_order"] == list(TARGET_ORDER)
    assert report["post_ready_execution_order"] == list(
        POST_READY_EXECUTION_ORDER
    )
    assert report["truth_position_used_for_target"] is False
    assert report["sim_truth_audit_used_for_target"] is False
    assert report["task_xy_assumed_equal_to_world_xy"] is False
    assert report["gpu_or_physx_validated"] is False
    json.dumps(report, allow_nan=False)


def test_task_xy_is_explicitly_rotated_into_world_xy():
    rotation = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    contract = replace(
        _contract(), expected_task_rotation_world=rotation
    )
    plan = build_visual_tactile_regrasp_plan(
        contract,
        _visual(contract, delta=(0.001, -0.0015)),
        _ready(offset=(0.0004, 0.0002), task_rotation=rotation),
        explicit_adapter_opt_in=True,
    )
    assert plan.tactile_search_offset_task_xy_m == pytest.approx(
        (0.0004, 0.0002)
    )
    assert plan.tactile_search_offset_world_m == pytest.approx(
        (-0.0002, 0.0004, 0.0)
    )
    assert plan.accepted_center_delta_xy_m == pytest.approx(
        (0.0008, -0.0011)
    )


@pytest.mark.parametrize(
    ("ready", "visual_mutator", "opt_in", "message"),
    (
        (_ready(state="ENGAGE_HOLD"), None, True, "READY state"),
        (_ready(capture_id="other"), None, True, "capture IDs"),
        (_ready(truth_search=True), None, True, "truth-contaminated"),
        (_ready(truth_target=True), None, True, "truth-contaminated"),
        (_ready(offset=(0.0031, 0.0)), None, True, "bounded radius"),
        (_ready(task_frame_id="world"), None, True, "frame ID"),
        (
            _ready(
                task_rotation=(
                    (0.0, -1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
            ),
            None,
            True,
            "hash-bound frame",
        ),
        (_ready(), "truth_orientation", True, "not eligible"),
        (_ready(), None, False, "opt-in"),
    ),
)
def test_builder_fails_closed_before_returning_targets(
    ready, visual_mutator, opt_in, message
):
    contract = _contract()
    visual = _visual(contract)
    if visual_mutator == "truth_orientation":
        visual = replace(visual, uses_truth_orientation=True)
    with pytest.raises(ValueError, match=message):
        build_visual_tactile_regrasp_plan(
            contract,
            visual,
            ready,
            explicit_adapter_opt_in=opt_in,
        )


@pytest.mark.parametrize(
    "rotation",
    (
        ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
        ((1.0, 0.0, 0.0), (0.0, float("nan"), 0.0), (0.0, 0.0, 1.0)),
    ),
)
def test_ready_parser_rejects_nonrotation_task_frames(rotation):
    with pytest.raises(ValueError, match="finite|orthogonal"):
        _ready(task_rotation=rotation)


def test_tampered_visual_target_provenance_is_rejected():
    contract = _contract()
    visual = _visual(contract)
    targets = dict(visual.world_targets)
    targets["engage_tcp"] = contract.visual.nominal_targets["engage_tcp"]
    contaminated = replace(visual, world_targets=targets)
    with pytest.raises(ValueError, match="provenance"):
        build_visual_tactile_regrasp_plan(
            contract,
            contaminated,
            _ready(),
            explicit_adapter_opt_in=True,
        )


@pytest.mark.parametrize(
    ("visual_delta", "tactile_delta"),
    (
        ((0.015, 0.015), (0.003, 0.0)),
        ((-0.015, -0.015), (-0.003, 0.0)),
        ((0.015, 0.015), (0.0, 0.003)),
        ((-0.015, -0.015), (0.0, -0.003)),
    ),
)
def test_bounded_visual_and_tactile_extremes_keep_fixed_q7_local_ik(
    visual_delta, tactile_delta
):
    contract = _contract()
    plan = build_visual_tactile_regrasp_plan(
        contract,
        _visual(contract, delta=visual_delta),
        _ready(offset=tactile_delta),
        explicit_adapter_opt_in=True,
    )
    assert max(plan.target_joint_deltas_from_nominal_rad.values()) < 0.2
    for name, arm in plan.arm_targets_rad.items():
        assert arm[6] == contract.target_seeds[name].arm_rad[6]


def test_builder_api_has_no_truth_pose_value_argument():
    signature = inspect.signature(build_visual_tactile_regrasp_plan)
    assert tuple(signature.parameters) == (
        "contract",
        "visual_result",
        "tactile_ready",
        "explicit_adapter_opt_in",
    )


def test_checked_in_cpu_sample_exactly_regenerates_and_hashes_all_inputs():
    contract = _contract()
    visual = _visual(
        contract, capture_id="cpu-contract-fixture-capture"
    )
    ready = _ready(capture_id="cpu-contract-fixture-capture")
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    regenerated = build_cpu_sample_artifact(contract, visual, ready)
    assert artifact == regenerated
    assert validate_cpu_sample_artifact(
        contract, visual, ready, artifact
    ) == regenerated
    assert artifact["artifact_schema_version"] == (
        "kcg_d38999_visual_tactile_regrasp_cpu_sample_v1"
    )
    assert artifact["artifact_scope"] == (
        "synthetic_cpu_contract_fixture_not_runtime_evidence"
    )
    plan = artifact["plan"]
    assert plan["schema_version"] == SCHEMA_VERSION
    assert plan["status"] == "CPU_TARGET_PLAN_BUILT_NOT_GPU_VALIDATED"
    assert plan["ready_source_scope"] == (
        "cpu_contract_fixture_not_runtime_evidence"
    )
    assert plan["gpu_or_physx_validated"] is False
    assert plan["production_control_authorized"] is False
    assert plan["assembly_success_claimed"] is False
    assert plan["target_order"] == list(TARGET_ORDER)
    assert set(artifact["input_dependencies"]) == set(contract.inputs)


@pytest.mark.parametrize("target_name", TARGET_ORDER)
@pytest.mark.parametrize("field", ("tcp_targets_world_m", "arm_targets_rad"))
def test_any_checked_in_sample_target_mutation_fails_exact_regeneration(
    target_name, field
):
    contract = _contract()
    visual = _visual(
        contract, capture_id="cpu-contract-fixture-capture"
    )
    ready = _ready(capture_id="cpu-contract-fixture-capture")
    artifact = build_cpu_sample_artifact(contract, visual, ready)
    mutated = copy.deepcopy(artifact)
    mutated["plan"][field][target_name][0] += 1.0e-6
    with pytest.raises(ValueError, match="exact regenerated"):
        validate_cpu_sample_artifact(contract, visual, ready, mutated)


@pytest.mark.parametrize(
    "mutation",
    ("extra", "missing", "config_hash", "module_hash", "input_hash"),
)
def test_sample_exact_schema_and_all_hash_edges_fail_closed(mutation):
    contract = _contract()
    visual = _visual(
        contract, capture_id="cpu-contract-fixture-capture"
    )
    ready = _ready(capture_id="cpu-contract-fixture-capture")
    mutated = build_cpu_sample_artifact(contract, visual, ready)
    if mutation == "extra":
        mutated["unexpected"] = True
    elif mutation == "missing":
        del mutated["plan"]
    elif mutation == "config_hash":
        mutated["config"]["sha256"] = "0" * 64
    elif mutation == "module_hash":
        mutated["module"]["sha256"] = "0" * 64
    else:
        first = sorted(mutated["input_dependencies"])[0]
        mutated["input_dependencies"][first]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="keys differ|exact regenerated"):
        validate_cpu_sample_artifact(contract, visual, ready, mutated)
