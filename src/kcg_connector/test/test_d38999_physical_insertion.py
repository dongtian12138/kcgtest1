"""Pure regression tests for measured-offset D38999 insertion."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from kcg_connector.d38999_physical_insertion import (
    axial_gap_waypoints,
    compensated_tcp_transform,
    compensated_tcp_position,
    load_d38999_physical_insertion,
    measure_alignment,
    pose_transform,
    solve_fixed_q7_tcp_pose,
    verify_insertion_inputs,
)
from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE_ROOT.parents[1]
CONFIG = PACKAGE_ROOT / "config/d38999_physical_insertion_v1.yaml"


def test_contract_loads_and_hash_locks_existing_inputs():
    contract = load_d38999_physical_insertion(CONFIG)
    assert contract.enabled is True
    assert contract.motion.runtime_body_in_tcp_compensation is True
    assert set(verify_insertion_inputs(contract, REPOSITORY)) == {
        "tabletop_pick",
        "assembly_baseline",
        "ik_family",
    }
    assert contract.proxy_collision_filter.mode == "proxy_false_contacts_only"
    assert contract.proxy_collision_filter.expected_filtered_pair_count == 500


def test_nominal_targets_are_downward_and_keep_q7_fixed():
    contract = load_d38999_physical_insertion(CONFIG)
    for arm in (
        contract.motion.axis_high_arm_rad,
        contract.motion.preinsert_arm_rad,
        contract.motion.engage_arm_rad,
    ):
        transform = np.asarray(iiwa14_grasp_tcp_transform(arm))
        assert transform[:3, 2] == pytest.approx((0.0, 0.0, -1.0), abs=5e-6)
        assert arm[6] == pytest.approx(contract.motion.fixed_q7_rad)


def test_measured_body_offset_generates_expected_nominal_tcp_targets():
    local_body = (0.0, 0.0, 0.04848)
    contract = load_d38999_physical_insertion(CONFIG)
    preinsert = compensated_tcp_position(
        (0.550, 0.185, 0.2735), local_body, contract.motion.preinsert_arm_rad
    )
    engage = compensated_tcp_position(
        (0.550, 0.185, 0.2645), local_body, contract.motion.engage_arm_rad
    )
    assert preinsert == pytest.approx((0.550, 0.185, 0.32198), abs=2e-7)
    assert engage == pytest.approx((0.550, 0.185, 0.31298), abs=2e-7)


def test_fixed_q7_local_ik_corrects_millimetre_pick_offset():
    contract = load_d38999_physical_insertion(CONFIG)
    target = (0.5507, 0.1844, 0.3228)
    solved = solve_fixed_q7_tcp_pose(contract.motion.preinsert_arm_rad, target)
    transform = np.asarray(iiwa14_grasp_tcp_transform(solved))
    assert transform[:3, 3] == pytest.approx(target, abs=1e-7)
    assert transform[:3, 2] == pytest.approx((0.0, 0.0, -1.0), abs=5e-6)
    assert solved[6] == pytest.approx(contract.motion.fixed_q7_rad)


def test_full_grasp_transform_preserves_measured_translation_and_rotation():
    measured_tcp = pose_transform(
        (0.52, -0.21, 0.36), (0.999998, 0.001, -0.001, 0.0)
    )
    measured_body = pose_transform(
        (0.5201, -0.2102, 0.31152), (0.999999, 0.0, 0.001, 0.0)
    )
    desired_body = pose_transform(
        (0.55, 0.185, 0.2735), (1.0, 0.0, 0.0, 0.0)
    )
    desired_tcp = compensated_tcp_transform(
        desired_body, measured_tcp, measured_body
    )
    tcp_to_body = np.linalg.inv(measured_tcp) @ measured_body
    assert desired_tcp @ tcp_to_body == pytest.approx(desired_body, abs=1e-12)


def test_fixed_q7_ik_accepts_explicit_nearby_orientation():
    contract = load_d38999_physical_insertion(CONFIG)
    seed_transform = np.asarray(
        iiwa14_grasp_tcp_transform(contract.motion.preinsert_arm_rad)
    )
    target_rotation = seed_transform[:3, :3].copy()
    angle = 0.001
    correction = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    target_rotation = correction @ target_rotation
    solved = solve_fixed_q7_tcp_pose(
        contract.motion.preinsert_arm_rad,
        (0.5502, 0.1848, 0.3221),
        target_rotation=target_rotation,
    )
    result = np.asarray(iiwa14_grasp_tcp_transform(solved))
    assert result[:3, 3] == pytest.approx(
        (0.5502, 0.1848, 0.3221), abs=1e-7
    )
    assert result[:3, :3] == pytest.approx(target_rotation, abs=1e-7)


def test_alignment_separates_gap_lateral_axis_and_combined_error():
    result = measure_alignment(
        (0.5501, 0.1849, 0.2735),
        (math.sin(0.005), 0.0, math.cos(0.005)),
        (0.550, 0.185, 0.2615),
        (0.0, 0.0, 1.0),
        0.010,
    )
    assert result.gap_m == pytest.approx(0.012)
    assert result.lateral_error_m == pytest.approx(math.sqrt(2.0) * 1e-4)
    assert result.axis_error_rad == pytest.approx(0.005)
    assert result.combined_entry_error_m == pytest.approx(
        result.lateral_error_m + 0.010 * math.sin(0.005)
    )


def test_runtime_numpy_vectors_share_the_strict_sequence_contract():
    transform = pose_transform(
        np.asarray((0.1, 0.2, 0.3)),
        np.asarray((1.0, 0.0, 0.0, 0.0)),
    )
    assert transform[:3, 3] == pytest.approx((0.1, 0.2, 0.3))


def test_axial_servo_waypoints_are_monotonic_bounded_and_endpoint_exact():
    waypoints = axial_gap_waypoints(0.017, 0.012, 0.00025)
    assert len(waypoints) == 20
    assert waypoints[-1] == pytest.approx(0.012)
    values = (0.017,) + waypoints
    assert all(left > right for left, right in zip(values, values[1:]))
    maximum_step = max(
        abs(left - right) for left, right in zip(values, values[1:])
    )
    assert maximum_step <= 0.00025 + 1e-12


def test_axial_servo_waypoints_reject_zero_travel_and_bad_step():
    with pytest.raises(ValueError):
        axial_gap_waypoints(0.012, 0.012, 0.00025)
    with pytest.raises(ValueError):
        axial_gap_waypoints(0.012, 0.003, 0.0)


@pytest.mark.parametrize(
    "section,key,value",
    (
        ("root", "extra", 1),
        ("motion", "runtime_body_in_tcp_compensation", False),
        ("acceptance", "maximum_lateral_error_m", float("nan")),
        ("proxy_collision_filter", "expected_filtered_pair_count", 499),
        ("boundaries", "vision_included", True),
    ),
)
def test_contract_fails_closed_on_mutations(tmp_path, section, key, value):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    target = document if section == "root" else document[section]
    target[key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_d38999_physical_insertion(path)
