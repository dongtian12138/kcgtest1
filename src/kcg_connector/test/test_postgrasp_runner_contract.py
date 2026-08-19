import ast
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

PICK_SMOKE = Path(__file__).parents[1] / "isaac" / "d38999_tabletop_pick_smoke.py"
SHADOW_RUNTIME = (
    Path(__file__).parents[1]
    / "isaac"
    / "postgrasp_shadow_capture_runtime.py"
)
ESTIMATOR = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "postgrasp_shadow_estimator.py"
)
PLANNER = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "postgrasp_shadow_view_planner.py"
)
CONFIG = (
    Path(__file__).parents[1]
    / "config"
    / "d38999_postgrasp_shadow_v1.yaml"
)
KEYED_PICK_CONFIG = (
    Path(__file__).parents[1]
    / "config"
    / "d38999_keyed_v2_tabletop_pick_v1.yaml"
)


def test_default_runner_has_only_optin_shadow_hook():
    source = PICK_SMOKE.read_text(encoding="utf-8")
    assert "--postgrasp-shadow-capture" in source
    assert "postgrasp_shadow" in source
    assert "run_postgrasp_shadow_capture(" in source
    # Hook is gated by the opt-in flag; no shadow call is unconditional.
    hook_block = source.split("if arguments.postgrasp_shadow_capture:", 1)[1]
    hook_block = hook_block.split("if arguments.insertion_probe:", 1)[0]
    assert "passed" in hook_block
    assert "shadow_authorized" in hook_block
    # Frozen grasp contract must not be weakened by the hook.
    assert "set_world_pose" not in source
    assert "object_pose_write" not in hook_block


def test_formal_modules_have_no_semantic_identifier_binding():
    for path in (ESTIMATOR, PLANNER, SHADOW_RUNTIME):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "semantic" not in names, path
        assert "id_to_labels" not in names, path
        assert "registered_truth_xy" not in names, path


def test_config_candidate_thresholds_and_auth_frozen_false():
    doc = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert doc["threshold_label"] == "SIM_TUNING_ONLY_CANDIDATE"
    assert doc["covariance"]["coverage_calibrated"] is False
    assert doc["covariance"]["shadow_authorized"] is False
    assert doc["covariance"]["control_authorized"] is False
    assert "semantic" not in doc["formal_inputs"]["allowed_fields"]
    assert "semantic" in doc["formal_inputs"]["forbidden_fields"]


def test_live_palm_and_wrist_are_fixed_handbase_camera_children():
    source = PICK_SMOKE.read_text(encoding="utf-8")
    install_block = source.split("def install_live_view_cameras(", 1)[1]
    install_block = install_block.split("def main(", 1)[0]
    assert 'handbase_path + "/PalmCamera"' in install_block
    assert 'handbase_path + "/WristCamera"' in install_block
    assert 'handbase_path + "/PalmLiveViewCamera"' not in install_block
    assert 'handbase_path + "/WristLiveViewCamera"' not in install_block
    assert "SIM_VISUAL_MOUNT_CANDIDATE_FIXED_T_HC" in install_block
    assert "_calibrated_hand_camera_from_nominal_plug" not in install_block
    assert "nominal_hand_to_plug" not in install_block
    assert "focal_length_mm=24.0" in install_block
    assert "update_live_camera_aim" not in source
    assert source.index("live_view_cameras = install_live_view_cameras(") < (
        source.index("world.reset()")
    )
    assert source.index("attach_live_camera_streams(", source.index("def main")) > (
        source.index("world.reset()")
    )


def test_camera_rig_probe_is_bounded_and_exits_before_grasp_control():
    source = PICK_SMOKE.read_text(encoding="utf-8")
    assert '"--camera-rig-probe"' in source
    assert "SIM_MOUNT_RIGIDITY_PROBE_ONLY" in source
    helper = source.split("def run_camera_rig_probe(", 1)[1]
    helper = helper.split("def main(", 1)[0]
    assert '"control_authorized": False' in helper
    assert '"real_mount_calibrated": False' in helper
    assert '"viewpoint_human_review_required": True' in helper
    assert '"key_region_visibility_verified": False' in helper
    assert '"uses_object_or_contact_truth": False' in helper
    assert '"enters_grasp_or_insertion": False' in helper
    assert "for segment in approach_segments:" in helper
    assert "approach_segments[:2]" not in helper
    assert 'AnnotatorRegistry.get_annotator("rgb")' in helper
    assert "pause_timeline=True" in helper
    assert "delta_time=0.0" in helper
    assert ".get_rgba(" not in helper
    for forbidden in (
        "get_full_contact_report",
        "get_world_pose",
        "set_world_pose",
        "ClearXformOpOrder",
        "AddTransformOp",
    ):
        assert forbidden not in helper

    probe_call = source.index("probe_report = run_camera_rig_probe(")
    controller_definition = source.index("def observe_and_step(")
    assert probe_call < controller_definition
    early_branch = source[probe_call:controller_definition]
    assert "return process_exit_code" in early_branch
    assert "initialize_live_callbacks=not arguments.camera_rig_probe" in source
    assert "probe_replicator_annotator_only" in source
    assert "or arguments.camera_rig_probe" in source
    assert "or arguments.keyed_visual_control" in source


def test_camera_rig_probe_rejects_other_task_modes_and_existing_output():
    source = PICK_SMOKE.read_text(encoding="utf-8")
    validation = source.split("if arguments.camera_rig_probe:", 1)[1]
    validation = validation.split("if not arguments.no_live_telemetry:", 1)[0]
    assert "--camera-rig-probe requires --output-dir" in validation
    assert "probe_output.exists()" in validation
    assert '"insertion_probe": arguments.insertion_probe' in validation
    assert '"end_to_end_probe": arguments.end_to_end_probe' in validation
    assert '"formal_grasp": arguments.physical_grasp_method != "legacy"' in validation
    assert "arguments.no_live_telemetry = True" in validation


def test_visual_t_hp_is_fail_closed_before_insertion_planning():
    source = PICK_SMOKE.read_text(encoding="utf-8")
    insertion_block = source.split("visual_t_hp_requested =", 1)[1]
    insertion_block = insertion_block.split(
        'metrics["physical_insertion"] = insertion_report', 1
    )[0]
    assert "selected_t_hp_control_pose" in insertion_block
    assert "visual_t_hp_control_gate_rejected" in insertion_block
    assert '"c2_hypotheses"][0]' not in insertion_block
    assert "hp_estimate[2]" not in insertion_block
    assert "hp_estimate[5]" not in insertion_block
    rejection = insertion_block.split(
        "elif visual_t_hp_requested and visual_hp_estimate is None:", 1
    )[1].split("else:", 1)[0]
    assert "passed = False" in rejection
    assert "physical_insertion_included" in rejection


def test_key_branch_stage_is_explicitly_shadow_only_in_runtime():
    source = SHADOW_RUNTIME.read_text(encoding="utf-8")
    assert "KEYED_INSERTION_CONTROL_PROMOTION_ENABLED = False" in source
    assert '"KEYED_GEOMETRY_UNAVAILABLE"' in source
    assert 'current_model_id="d38999_shell25j_proxy_v1"' in source
    gate = source.split("def selected_t_hp_control_pose(", 1)[1]
    gate = gate.split("def _iiwa_tcp_fk(", 1)[0]
    assert "if not KEYED_INSERTION_CONTROL_PROMOTION_ENABLED:" in gate
    assert 'key_branch.get("shadow_selected_hypothesis_id")' in gate
    assert 'yaw_gate.get("status") != "PASSED_EVALUATION_ONLY"' in gate


def test_truth_driven_insertion_is_quarantined_before_isaac_start():
    source = PICK_SMOKE.read_text(encoding="utf-8")
    validation = source.index("truth_proxy_motion_requested = bool(")
    isaac_start = source.index("from isaacsim import SimulationApp")
    assert validation < isaac_start
    assert '"--sim-truth-proxy-regression"' in source
    assert "SIM_GROUND_TRUTH_PROXY_REGRESSION_ONLY" in source
    inline = source.split("inline_palm_t_hp = None", 1)[1]
    inline = inline.split("if arguments.postgrasp_shadow_capture:", 1)[0]
    assert "not arguments.sim_truth_proxy_regression" in inline


def test_keyed_visual_control_is_reachable_but_strictly_profile_bound():
    source = PICK_SMOKE.read_text(encoding="utf-8")
    assert '"--keyed-visual-control"' in source
    validation = source.split("if arguments.keyed_visual_control and (", 1)[1]
    validation = validation.split("# Keep the validated baseline", 1)[0]
    assert "not formal_grasp" in validation
    assert "single_finger_mode" in validation
    assert "arguments.insertion_probe" in validation
    assert "arguments.end_to_end_probe" in validation
    assert "arguments.postgrasp_snapshot_gate" in validation
    assert "arguments.postgrasp_shadow_capture" in validation
    assert "--keyed-visual-control requires --output-dir" in validation
    assert "d38999_keyed_v2_tabletop_pick_v1.yaml" in validation
    assert "d38999_keyed_v2_tabletop_physical_grasp_v1.yaml" in validation
    assert "checkpoint_a_acceptance_v2.json" in validation
    assert "KEYED_V2_CHECKPOINT_A2_PROMOTION_ENABLED = False" in source
    assert "if not KEYED_V2_CHECKPOINT_A2_PROMOTION_ENABLED:" in validation
    assert "physically model pin/socket engagement" in validation
    assert "coupling-thread " in validation
    assert '"interaction in a new immutable asset' in validation
    assert (
        "d38999_keyed_v2_tabletop_pick_xycomp_candidate_v1.yaml"
        not in validation
    )
    assert (
        "d38999_keyed_v2_tabletop_physical_grasp_xycomp_candidate_v1.yaml"
        not in validation
    )
    assert source.index("if arguments.keyed_visual_control and (") < source.index(
        "from isaacsim import SimulationApp"
    )


def test_model_first_gate_blocks_every_keyed_v2_episode_before_isaac():
    source = PICK_SMOKE.read_text(encoding="utf-8")
    parse = source.index("arguments = parser.parse_args()")
    contract_gate = source.index(
        "physical_model_contract = load_physical_model_contract()"
    )
    global_gate = source.index(
        "if keyed_v2_model_requested and not "
        "KEYED_V2_CHECKPOINT_A2_PROMOTION_ENABLED:"
    )
    isaac_start = source.index("from isaacsim import SimulationApp")
    assert parse < contract_gate < global_gate < isaac_start
    assert "dict(physical_model_contract.phase_gates)" in source[
        contract_gate:global_gate
    ]
    assert "== FINAL_PHASE_STATE" in source[contract_gate:global_gate]
    assert '"grasp_allowed"' in source[contract_gate:global_gate]
    assert "contacts are visual-only" in source[parse:global_gate]
    assert "coupling thread is unmodeled" in source[parse:global_gate]

    completed = subprocess.run(
        [
            sys.executable,
            str(PICK_SMOKE),
            "--no-live-telemetry",
            "--config",
            str(KEYED_PICK_CONFIG),
            "--physical-grasp-method",
            "sequential-compliant",
            "--formal-lift-mode",
            "staged",
            "--output-dir",
            "/tmp/keyed-v2-model-first-gate-contract",
        ],
        cwd=PICK_SMOKE.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "all keyed-v2 simulation episodes are blocked" in completed.stderr
    assert "finish and accept checkpoint A0-A5 physical modeling" in completed.stderr
    assert "isaacsim" not in completed.stderr.lower()


def test_keyed_visual_control_uses_revision_paths_without_digest_generation():
    source = PICK_SMOKE.read_text(encoding="utf-8")
    marker = (
        'if arguments.keyed_visual_control:\n'
        '                metrics["provenance"]'
    )
    keyed_provenance = source.split(marker, 1)[1].split(
        "            else:", 1
    )[0]
    assert "sha256" not in keyed_provenance.lower()
    assert "unique_revision_paths_and_strict_contracts" in keyed_provenance
    assert "checkpoint_a_acceptance_path" in keyed_provenance
    assert (
        'if not arguments.keyed_visual_control:\n'
        '                realized_randomization_report["payload_sha256"]'
        in source
    )


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    (
        (
            ("--keyed-visual-control", "--output-dir", "/tmp/keyed-visual"),
            "requires a non-single-finger formal grasp",
        ),
        (
            (
                "--keyed-visual-control",
                "--physical-grasp-method",
                "sequential-compliant",
                "--output-dir",
                "/tmp/keyed-visual",
            ),
            "checkpoint B is blocked: checkpoint A2",
        ),
        (
            (
                "--keyed-visual-control",
                "--physical-grasp-method",
                "sequential-compliant",
                "--insertion-probe",
                "--output-dir",
                "/tmp/keyed-visual",
            ),
            "insertion/end-to-end is quarantined",
        ),
    ),
)
def test_invalid_keyed_visual_combinations_stop_before_isaac(
    arguments, expected_error
):
    completed = subprocess.run(
        [
            sys.executable,
            str(PICK_SMOKE),
            "--no-live-telemetry",
            *arguments,
        ],
        cwd=PICK_SMOKE.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert expected_error in completed.stderr
    assert "isaacsim" not in completed.stderr.lower()


@pytest.mark.parametrize(
    "arguments,expected_error",
    (
        (
            ("--insertion-probe",),
            "insertion/end-to-end is quarantined",
        ),
        (
            ("--end-to-end-probe",),
            "insertion/end-to-end is quarantined",
        ),
        (
            ("--sim-truth-proxy-regression",),
            "requires --insertion-probe or --end-to-end-probe",
        ),
        (
            (
                "--end-to-end-probe",
                "--sim-truth-proxy-regression",
                "--physical-grasp-method",
                "sequential-compliant",
            ),
            "formal grasp is isolated from truth-driven insertion",
        ),
        (
            (
                "--end-to-end-probe",
                "--sim-truth-proxy-regression",
                "--visual-chain-report",
                "not_read.json",
            ),
            "cannot consume --visual-chain-report",
        ),
        (
            (
                "--end-to-end-probe",
                "--sim-truth-proxy-regression",
                "--pose-preflight",
                "masked-rgbd",
            ),
            "cannot combine with visual pose preflight",
        ),
    ),
)
def test_invalid_truth_proxy_combinations_fail_in_argument_parser(
    arguments, expected_error
):
    completed = subprocess.run(
        [
            sys.executable,
            str(PICK_SMOKE),
            "--no-live-telemetry",
            *arguments,
        ],
        cwd=PICK_SMOKE.parent,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert expected_error in completed.stderr
    assert "isaacsim" not in completed.stderr.lower()


def test_shadow_runtime_call_matches_keyword_only_signature():
    runtime_tree = ast.parse(SHADOW_RUNTIME.read_text(encoding="utf-8"))
    runner_tree = ast.parse(PICK_SMOKE.read_text(encoding="utf-8"))
    runtime_function = next(
        node
        for node in runtime_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_postgrasp_shadow_capture"
    )
    allowed = {argument.arg for argument in runtime_function.args.kwonlyargs}
    call = next(
        node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_postgrasp_shadow_capture"
    )
    supplied = {keyword.arg for keyword in call.keywords}
    assert supplied <= allowed
