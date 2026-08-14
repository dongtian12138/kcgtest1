"""Static fail-closed contract for the standalone MoveIt collision audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_PATH = (
    PACKAGE_ROOT / "config/connector_pick_collision_audit_v1.yaml"
)
D38999_CONFIG_PATH = (
    PACKAGE_ROOT / "config/d38999_pick_collision_audit_v1.yaml"
)
SOURCE_PATH = PACKAGE_ROOT / "src/connector_pick_collision_audit.cpp"
ANALYZER_PATH = (
    PACKAGE_ROOT / "src/d38999_closure_clearance_analyzer.cpp"
)
EXPECTED_FINGER_LINKS = {
    "f1Link1",
    "f1Link2",
    "f1Link3",
    "f2Link1",
    "f2Link2",
    "f3Link1",
    "f3Link2",
    "f3Link3",
}


def _load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_inputs_are_project_relative_and_content_addressed():
    config = _load_yaml(CONFIG_PATH)
    assert config["schema_version"] == (
        "kcg_moveit_connector_pick_collision_audit_v1"
    )
    assert set(config["inputs"]) == {
        "urdf_xacro",
        "candidate_srdf",
        "home_to_pregrasp",
        "tabletop_pick",
        "tabletop_scene",
        "connector_task",
    }
    for item in config["inputs"].values():
        relative = Path(item["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        path = PROJECT_ROOT / relative
        assert path.is_file()
        assert _sha256(path) == item["sha256"]


def test_d38999_inputs_are_independent_and_content_addressed():
    config = _load_yaml(D38999_CONFIG_PATH)
    assert config["profile"] == "d38999_shell25j_v1"
    assert set(config["inputs"]) == {
        "urdf_xacro",
        "candidate_srdf",
        "d38999_pick",
        "d38999_scene",
        "d38999_proxy",
    }
    assert all(
        "connector_task.yaml" not in item["path"]
        for item in config["inputs"].values()
    )
    for item in config["inputs"].values():
        path = PROJECT_ROOT / item["path"]
        assert path.is_file()
        assert _sha256(path) == item["sha256"]


def test_candidate_is_independent_and_has_expected_reason_counts():
    config = _load_yaml(CONFIG_PATH)
    candidate = PROJECT_ROOT / config["inputs"]["candidate_srdf"]["path"]
    assert "artifacts/kcg_connector/self_collision_acm_candidate_v1" in str(
        candidate
    )
    root = ET.parse(candidate).getroot()
    reasons = [
        element.attrib["reason"]
        for element in root.findall("disable_collisions")
    ]
    assert reasons.count("Never") == 76
    assert reasons.count("Adjacent") == 16
    assert set(reasons) == {"Never", "Adjacent"}


def test_sampling_count_matches_the_exact_isaac_command_schedule():
    config = _load_yaml(CONFIG_PATH)
    home = _load_yaml(
        PROJECT_ROOT / config["inputs"]["home_to_pregrasp"]["path"]
    )
    pick = _load_yaml(
        PROJECT_ROOT / config["inputs"]["tabletop_pick"]["path"]
    )
    tabletop = _load_yaml(
        PROJECT_ROOT / config["inputs"]["tabletop_scene"]["path"]
    )
    rate = config["sampling"]["rate_hz"]
    durations = [
        tabletop["physics"]["settle_duration_s"],
        home["motion"]["hand_open_duration_s"],
        *(segment["duration_s"] for segment in home["motion"]["segments"]),
        home["motion"]["hold_duration_s"],
        pick["motion"]["descent_duration_s"],
        pick["motion"]["open_tare_duration_s"],
        pick["motion"]["closure_duration_s"],
        pick["motion"]["preload_duration_s"],
        pick["motion"]["lift_duration_s"],
        pick["motion"]["final_hold_duration_s"],
    ]
    steps = [round(duration * rate) for duration in durations]
    assert all(duration * rate == step for duration, step in zip(durations, steps))
    assert sum(steps) == 10656
    assert sum(steps) == config["sampling"]["expected_sample_count"]


def test_d38999_sampling_count_matches_its_own_isaac_schedule():
    config = _load_yaml(D38999_CONFIG_PATH)
    pick = _load_yaml(PROJECT_ROOT / config["inputs"]["d38999_pick"]["path"])
    tabletop = _load_yaml(
        PROJECT_ROOT / config["inputs"]["d38999_scene"]["path"]
    )
    rate = config["sampling"]["rate_hz"]
    motion = pick["motion"]
    durations = [
        tabletop["physics"]["settle_duration_s"],
        motion["hand_open_duration_s"],
        *(segment["duration_s"] for segment in motion["approach_segments"]),
        motion["pregrasp_hold_duration_s"],
        motion["descent_duration_s"],
        motion["open_tare_duration_s"],
        motion["closure_duration_s"],
        motion["closed_seating_duration_s"],
        motion["preload_duration_s"],
        motion["lift_duration_s"],
        motion["final_hold_duration_s"],
    ]
    steps = [round(duration * rate) for duration in durations]
    assert sum(steps) == 11376
    assert sum(steps) == config["sampling"]["expected_sample_count"]


def test_only_expected_finger_contacts_are_exempted_during_closure():
    config = _load_yaml(CONFIG_PATH)
    policy = config["collision_policy"]
    assert set(policy["intentional_touch_links"]) == EXPECTED_FINGER_LINKS
    assert policy["attachment_link"] == "grasp_tcp"
    assert policy["reenable_all_never_pairs_for_strict_audit"] is True


def test_continuous_collision_remains_explicitly_unverified_and_required():
    config = _load_yaml(CONFIG_PATH)
    source = SOURCE_PATH.read_text(encoding="utf-8")
    policy = config["collision_policy"]
    assert policy["require_continuous_collision_verified"] is True
    assert "const bool continuous_collision_verified = false;" in source
    assert "FAIL_CLOSED_CONTINUOUS_COLLISION_UNVERIFIED" in source
    assert "checkRobotCollision(collision_request, environment_collision" in source
    assert "distanceSelf(distance_request" in source
    assert "distanceRobot(distance_request" in source
    assert "state1" not in source
    assert "state2" not in source


def test_d38999_requires_five_millimetres_robot_table_margin():
    config = _load_yaml(D38999_CONFIG_PATH)
    synthetic = _load_yaml(CONFIG_PATH)
    assert config["collision_policy"][
        "minimum_robot_link_table_distance_m"
    ] == 0.005
    assert synthetic["collision_policy"][
        "minimum_robot_link_table_distance_m"
    ] == 0.0
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "candidate_robot_table_margin_clear" in source
    assert "strict_robot_table_margin_clear" in source
    assert 'report["final_object_contact_proxy"]' in source


def test_final_object_proxy_isolated_to_each_finger_group():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert 'isolated.setEntry(link, object_id, true);' in source
    assert 'isolated.setEntry(link, loose_world_id, false);' in source
    assert "final_touching_finger_count == 3U" in source
    assert "final finger-to-object proxy returned an unexpected pair" in source


def test_tool_is_cpu_only_and_does_not_install_or_replace_srdf():
    source = (
        SOURCE_PATH.read_text(encoding="utf-8")
        + ANALYZER_PATH.read_text(encoding="utf-8")
    ).lower()
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "cuda" not in source
    assert "isaac" not in source
    assert "gpu" not in source
    assert "copy_file" not in source
    assert "rename(" not in source
    assert "candidate_srdf" not in cmake
    assert "artifacts" not in cmake


def test_closure_analyzer_is_strict_and_keeps_continuous_unverified():
    source = ANALYZER_PATH.read_text(encoding="utf-8")
    assert "expected to restore exactly 76 reason=Never pairs" in source
    assert "const double requested_clearance = 0.001;" in source
    assert "const std::size_t dense_intervals = 20000U;" in source
    assert "const double b080_blend = 0.80;" in source
    assert "const double b085_blend = 0.85;" in source
    assert 'report["continuous_collision_verified"] = false;' in source
    assert 'report["safe_grasp_feasible_under_static_proxy"] = false;' in source
    assert "Negative finger-plug signed distance" in source
    assert "per_finger_effort_limit_nm" in source


def test_closure_analyzer_uses_unmasked_per_finger_plug_distance():
    source = ANALYZER_PATH.read_text(encoding="utf-8")
    assert 'isolated.setEntry(link, plug_id, true);' in source
    assert 'isolated.setEntry(link, plug_id, false);' in source
    assert '"finger_1", { "f1Link1", "f1Link2", "f1Link3" }' in source
    assert '"finger_2", { "f2Link1", "f2Link2" }' in source
    assert '"finger_3", { "f3Link1", "f3Link2", "f3Link3" }' in source
    assert "touching_finger_count == 3U" in source
