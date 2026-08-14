"""Static fail-closed contract for the separated Body/Nut CPU search."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config/d38999_nut_regrasp_search_v1.yaml"
SOURCE_PATH = PACKAGE_ROOT / "src/d38999_nut_regrasp_search.cpp"
CMAKE_PATH = PACKAGE_ROOT / "CMakeLists.txt"


def _load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_is_disabled_and_all_inputs_are_content_addressed():
    config = _load_yaml(CONFIG_PATH)
    assert config["schema_version"] == "kcg_d38999_nut_regrasp_search_v1"
    assert config["enabled"] is False
    assert config["status"] == (
        "cpu_geometry_search_only_not_runtime_authorized"
    )
    assert set(config["inputs"]) == {
        "urdf_xacro",
        "candidate_srdf",
        "d38999_proxy",
        "d38999_scene",
        "d38999_assembly",
    }
    for item in config["inputs"].values():
        relative = Path(item["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        path = PROJECT_ROOT / relative
        assert path.is_file()
        assert _sha256(path) == item["sha256"]


def test_search_keeps_q7_fixed_and_anchors_are_ordered():
    config = _load_yaml(CONFIG_PATH)
    fixed_q7 = config["robot"]["fixed_q7_rad"]
    anchors = config["ik_family"]["anchors"]
    assert len(anchors) == 21
    z_values = [anchor["tcp_z_m"] for anchor in anchors]
    assert z_values == sorted(z_values)
    assert len(set(z_values)) == len(z_values)
    assert z_values[0] == config["search"]["tcp_z_lower_m"]
    assert z_values[-1] == config["search"]["tcp_z_upper_m"]
    assert all(len(anchor["arm_rad"]) == 7 for anchor in anchors)
    assert all(anchor["arm_rad"][6] == fixed_q7 for anchor in anchors)


def test_search_is_nut_only_and_preserves_all_clearance_gates():
    config = _load_yaml(CONFIG_PATH)
    acceptance = config["acceptance"]
    assert acceptance == {
        "nut_touch_signed_distance_max_m": 0.0,
        "body_signed_clearance_min_m": 0.001,
        "candidate_self_clearance_min_m": 0.001,
        "strict_self_clearance_min_m": 0.001,
        "robot_environment_clearance_min_m": 0.005,
        "require_all_three_fingers_touch_nut": True,
        "require_all_three_fingers_clear_body": True,
        "require_continuous_collision_verified": True,
    }
    boundaries = config["boundaries"]
    assert boundaries == {
        "physics_contact_verified": False,
        "force_closure_verified": False,
        "continuous_collision_verified": False,
        "thread_contact_modeled": False,
        "runtime_integration_allowed": False,
        "object_pose_drive_allowed": False,
        "assembly_success_claimed": False,
    }


def test_hand_search_can_change_outer_and_middle_fingers_independently():
    config = _load_yaml(CONFIG_PATH)
    search = config["search"]
    assert (
        search["coarse_outer_blend_lower"]
        < search["coarse_outer_blend_upper"]
    )
    assert (
        search["coarse_middle_blend_lower"]
        < search["coarse_middle_blend_upper"]
    )
    mapping = search["hand_command_mapping"]
    assert mapping == {
        "f1j1_fixed_rad": 1.0,
        "f1j2_from_outer_blend_scale": 0.9,
        "f2j1_from_middle_blend_scale": 0.7,
        "f3j2_from_outer_blend_scale": 0.9,
    }


def test_discrete_regrasp_path_is_240_hz_and_fail_closed():
    config = _load_yaml(CONFIG_PATH)
    path = config["discrete_regrasp_path"]
    assert path["rate_hz"] == 240
    assert path["interpolation"] == "minimum_jerk"
    assert path["carry_hand_rad"] == [1.0, 0.765, 0.595, 0.765]
    assert path["open_hand_rad"] == [1.0, 0.0, 0.0, 0.0]
    steps = sum(
        round(path[name] * path["rate_hz"])
        for name in (
            "release_duration_s",
            "open_reposition_duration_s",
            "nut_closure_duration_s",
        )
    )
    assert steps == path["expected_sample_count"] == 1920
    assert path["maximum_joint_step_rad"] == 0.0025
    assert path["open_reposition_endpoint_clearance_min_m"] == 0.001


def test_source_separates_body_and_nut_and_reenables_never_pairs():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert (
        'const std::string body_id = "d38999_regrasp_body_assembly";'
        in source
    )
    assert (
        'const std::string nut_id = "d38999_regrasp_coupling_nut";'
        in source
    )
    assert "strict_acm.setEntry(pair.link1_, pair.link2_, false);" in source
    assert "candidate.worst_nut_distance <= nut_touch_max" in source
    assert "candidate.minimum_body_distance >= body_clearance_min" in source
    assert "candidate.environment.distance >= environment_min" in source
    assert (
        "candidate.forbidden_endpoint.distance >= body_clearance_min"
        in source
    )
    assert (
        'report["continuous_collision_verified"] = '
        "continuous_collision_verified;"
    ) in source
    assert "const bool continuous_collision_verified = false;" in source
    assert (
        'report["status"] = "FAIL_CLOSED_DISCRETE_REGRASP_PATH";'
        in source
    )
    assert "path_audit.body_during_reapproach.sample.distance" in source
    assert "path_audit.nut_during_open_reposition.sample.distance" in source


def test_source_is_cpu_only_and_never_mutates_active_inputs():
    source = SOURCE_PATH.read_text(encoding="utf-8").lower()
    assert "cuda" not in source
    assert "isaac" not in source
    assert "gpu" not in source
    assert "copy_file" not in source
    assert "rename(" not in source
    assert "remove(" not in source
    assert "runtime_integration_allowed" not in source


def test_new_executable_is_built_and_installed_without_replacing_old_tools():
    cmake = CMAKE_PATH.read_text(encoding="utf-8")
    assert "add_executable(\n  d38999_nut_regrasp_search" in cmake
    assert "src/d38999_nut_regrasp_search.cpp" in cmake
    assert "d38999_closure_clearance_analyzer" in cmake
    assert "connector_pick_collision_audit" in cmake
