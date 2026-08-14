"""Static fail-closed contract for the prepared-engage PhysX A/B."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config/d38999_nut_regrasp_physx_v1.yaml"
SMOKE_PATH = PACKAGE_ROOT / "isaac/d38999_nut_regrasp_smoke.py"


def _load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_inputs_are_project_relative_and_content_addressed():
    config = _load_yaml(CONFIG_PATH)
    assert config["schema_version"] == "kcg_d38999_nut_regrasp_physx_v1"
    assert config["enabled"] is True
    assert set(config["inputs"]) == {
        "tabletop_pick",
        "tabletop_scene",
        "shell_proxy",
        "assembly_baseline",
        "cpu_search_config",
        "cpu_search_report",
        "cpu_command_compensation_config",
        "cpu_command_compensation_report",
        "uncompensated_physx_log",
        "robot_asset",
    }
    for item in config["inputs"].values():
        relative = Path(item["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        path = PROJECT_ROOT / relative
        assert path.is_file()
        assert _sha256(path) == item["sha256"]


def test_candidate_is_bound_to_the_cpu_search_report():
    config = _load_yaml(CONFIG_PATH)
    seed_report = _load_yaml(
        PROJECT_ROOT / config["inputs"]["cpu_search_report"]["path"]
    )
    report = _load_yaml(
        PROJECT_ROOT
        / config["inputs"]["cpu_command_compensation_report"]["path"]
    )
    for item in (seed_report, report):
        assert item["status"] == (
            "FAIL_CLOSED_CONTINUOUS_COLLISION_UNVERIFIED"
        )
        assert item["nut_only_command_candidate_found"] is True
        assert item["candidate_may_proceed_to_physx_static_ab"] is True
        assert item["discrete_regrasp_path"]["passed"] is True
    seed = config["uncompensated_seed"]
    seed_best = seed_report["best_feasible"]
    assert seed["tcp_position_world_m"][2] == seed_best["tcp_z_command_m"]
    assert seed["arm_rad"] == seed_best["arm_command_rad"]
    assert seed["hand_rad"] == seed_best["hand_command_rad"]
    best = report["best_feasible"]
    candidate = config["nut_only_candidate"]
    assert candidate["tcp_position_world_m"][2] == best["tcp_z_command_m"]
    assert candidate["arm_rad"] == best["arm_command_rad"]
    assert all(
        math.isclose(left, right, abs_tol=1.0e-14)
        for left, right in zip(
            candidate["hand_rad"], best["hand_command_rad"]
        )
    )
    assert math.isclose(
        candidate["cpu_minimum_acceptance_margin_m"],
        best["minimum_acceptance_margin_m"],
        abs_tol=1.0e-15,
    )
    assert math.isclose(
        candidate["cpu_minimum_body_clearance_m"],
        best["minimum_body_signed_distance_m"],
        abs_tol=1.0e-15,
    )
    assert math.isclose(
        candidate["cpu_worst_nut_signed_distance_m"],
        best["worst_nut_signed_distance_m"],
        abs_tol=1.0e-15,
    )
    assert config["tracking_compensation"]["fraction"] == 0.75
    assert candidate["desired_physical_tcp_position_world_m"] == (
        seed["tcp_position_world_m"]
    )


def test_user_confirmed_torque_limit_is_hard_and_operational_is_lower():
    sensing = _load_yaml(CONFIG_PATH)["sensing"]
    assert sensing["torque_joint_names"] == ["f1j2", "f2j1", "f3j2"]
    assert sensing["hard_stop_nm"] == 2.0
    assert sensing["operational_torque_target_nm"] == 1.8
    assert sensing["operational_torque_target_nm"] < sensing["hard_stop_nm"]
    assert sensing["fingertip_tactile_available"] is False


def test_acceptance_requires_nut_only_contact_after_regrasp():
    acceptance = _load_yaml(CONFIG_PATH)["acceptance"]
    for key in (
        "require_postclosure_all_fingers_nut_contact",
        "require_postclosure_zero_finger_body_contact",
        "require_final_all_fingers_nut_contact",
        "require_final_zero_finger_body_contact",
        "require_zero_nonfinger_endpoint_contacts",
    ):
        assert acceptance[key] is True
    assert acceptance["maximum_body_translation_drift_m"] == 0.000050
    assert acceptance["maximum_body_rotation_drift_rad"] == 0.0005


def test_prepared_constraint_is_explicitly_not_real_keying():
    config = _load_yaml(CONFIG_PATH)
    prepared = config["prepared_engage"]
    assert prepared["engage_gap_m"] == 0.003
    assert prepared["temporary_world_body_constraint"] == "fixed_joint"
    assert prepared["constraint_is_real_keying_claim"] is False
    boundaries = config["boundaries"]
    assert boundaries["physical_initial_mixed_closure_included"] is True
    assert boundaries["physical_insertion_included"] is False
    assert boundaries["real_keying_modeled"] is False
    assert boundaries["q7_twist_included"] is False
    assert boundaries["assembly_success_claimed"] is False


def test_smoke_is_lazy_and_never_drives_object_pose_after_start():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    before_main, after_main = source.split("def main():", 1)
    assert "isaacsim" not in before_main
    assert "omni." not in before_main
    assert "from pxr" not in before_main
    assert "set_world_pose(" not in source
    assert "object_pose_writes_after_start" in source
    assert "hard_stop_nm" in source
    assert "finger_body_group_records" in source
    assert "all_fingers_nut_contact" in source
    assert "zero_finger_body_contact" in source
    assert "assembly_success_claimed" in source
    assert '"assembly_success_claimed": False' in source


def test_phase_durations_are_whole_240_hz_steps():
    config = _load_yaml(CONFIG_PATH)
    control = config["control"]
    rate = control["physics_rate_hz"]
    durations = [
        value
        for key, value in control.items()
        if key.endswith("_s") and key != "physics_rate_hz"
    ]
    assert durations
    assert all(value > 0.0 for value in durations)
    assert all(value * rate == round(value * rate) for value in durations)


def test_physx_ab_uses_split_open_and_contact_gains():
    config = _load_yaml(CONFIG_PATH)
    control = config["control"]
    assert control["open_hand_stiffness"] == 25.0
    assert control["open_hand_damping"] == 2.0
    assert control["grip_hand_stiffness"] == 5.0
    assert control["grip_hand_damping"] == 1.0
    assert control["initial_open_settle_s"] == 1.0
    assert control["open_tare_s"] == 0.5
    assert control["mixed_closure_s"] == 3.5

    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert "from kcg_connector.robot_model import named_joint_target" in source
    assert "initial_positions = named_joint_target(" in source
    assert "def set_hand_gains" in source
    assert 'phase = "initial_open_settle"' in source
    assert 'phase = "prepare_mixed_carry_grip"' in source
    assert '"physical_initial_mixed_closure_included": True' in source
