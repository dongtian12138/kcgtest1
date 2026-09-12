"""Measured rod closure and differential kinematics on the source robot tree."""

from dataclasses import replace
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kcg_connector.grasp.robust.finger_fourbar import FingerFourBar
from kcg_connector.grasp.robust.hand_model import HandModelError, ThreeFingerHandModel


REPOSITORY = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def hand_models():
    root = ET.Element("robot", name="source_handarm_fourbar_check")
    for name in ("iiwa14.xacro", "hand.xacro"):
        # These source files contain direct numeric URDF joints and links.
        root.extend(ET.parse(REPOSITORY / "src/iiwa_description/urdf" / name).getroot())
    historical = ThreeFingerHandModel.from_urdf(ET.tostring(root), base_link="world")
    rows = json.loads((REPOSITORY / "src/kcg_connector/config/hand_fourbar_20260912.json").read_text())["finger_joints"]
    couplings = {row["follower_joint"]: FingerFourBar.from_mapping(row) for row in rows.values()}
    return historical, historical.with_fourbar_couplings(couplings), rows


def _pose(model, fraction=0.5):
    lower, upper = model.joint_limit_vectors()
    return lower + fraction * (upper - lower)


def _point(transform, local):
    return transform[:3, :3] @ local + transform[:3, 3]


def test_opt_in_retains_independent_coordinates_and_historical_model(hand_models):
    historical, measured, rows = hand_models
    assert historical.independent_joint_names == measured.independent_joint_names
    assert len(measured.independent_joint_names) == 11
    assert not historical.fourbar_couplings
    assert len(measured.fourbar_couplings) == 3
    q = _pose(historical)
    old = historical.resolve_joint_positions(q)
    new = measured.resolve_joint_positions(q)
    for row in rows.values():
        assert old[row["follower_joint"]] == old[row["source_joint"]]
        assert new[row["follower_joint"]] != pytest.approx(old[row["follower_joint"]], abs=1e-6)
    assert old["f3j1"] == old["f1j1"]
    restored = measured.with_fourbar_couplings({})
    assert dict(restored.resolve_joint_positions(q)) == dict(old)
    for name, transform in historical.forward_kinematics(q).items():
        np.testing.assert_array_equal(restored.forward_kinematics(q)[name], transform)
    assert dict(restored.independent_joint_limits) == dict(historical.independent_joint_limits)
    # Legacy velocity callers still need no position argument.
    velocity = historical.resolve_joint_velocities(np.full(11, 0.2))
    assert velocity["f1j3"] == velocity["f1j2"] == 0.2
    assert velocity["f3j1"] == 0.2


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_fk_anchors_preserve_the_measured_68mm_rods(hand_models, fraction):
    _, model, rows = hand_models
    q = _pose(model, fraction)
    links = model.forward_kinematics(q)
    # The 3D anchors use different parent frames for side and middle fingers;
    # this independently checks the planar closure's URDF coordinate mapping.
    for row in rows.values():
        base = _point(links[row["parent_link"]], row["base_anchor_parent_local_m"])
        distal = _point(links[row["distal_link"]], row["distal_anchor_local_m"])
        assert np.linalg.norm(distal - base) == pytest.approx(row["rod_length_m"], abs=2e-12)


@pytest.mark.parametrize("fraction", [0.15, 0.55, 0.85])
@pytest.mark.parametrize("measured", [False, True])
def test_every_terminal_jacobian_column_matches_fk(hand_models, fraction, measured):
    model = hand_models[int(measured)]
    q = _pose(model, fraction)
    point = np.array([0.004, -0.013, 0.006])
    h = 1e-6
    for finger in model.fingers.values():
        actual = model.geometric_jacobian(finger.terminal_link, q, point_local_m=point)
        finite = np.empty_like(actual)
        for column in range(len(q)):
            plus, minus = q.copy(), q.copy()
            plus[column] += h
            minus[column] -= h
            high = model.forward_kinematics(plus)[finger.terminal_link]
            low = model.forward_kinematics(minus)[finger.terminal_link]
            finite[:, column] = np.r_[
                (_point(high, point) - _point(low, point)) / (2 * h),
                Rotation.from_matrix(high[:3, :3] @ low[:3, :3].T).as_rotvec() / (2 * h),
            ]
        np.testing.assert_allclose(actual, finite, atol=2e-9, rtol=2e-7)


def test_velocity_and_pad_domains_use_current_fourbar_derivatives(hand_models):
    _, model, _ = hand_models
    q = _pose(model, 0.4)
    qdot = np.linspace(-0.12, 0.18, len(q))
    with pytest.raises(HandModelError, match="positions are required"):
        model.resolve_joint_velocities(qdot)
    velocity = model.resolve_joint_velocities(qdot, positions=q)
    h = 1e-6
    high = model.resolve_joint_positions(q + h * qdot)
    low = model.resolve_joint_positions(q - h * qdot)
    for name in velocity:
        assert velocity[name] == pytest.approx((high[name] - low[name]) / (2 * h), abs=2e-9)
    domains = model.pad_kinematic_normal_domains(q, qdot)
    high_pads, low_pads = model.pad_transforms(q + h * qdot), model.pad_transforms(q - h * qdot)
    for name, domain in domains.items():
        finite = (high_pads[name][:3, 3] - low_pads[name][:3, 3]) / (2 * h)
        np.testing.assert_allclose(domain.closing_velocity_base_m_s, finite, atol=2e-9)
    incorrect = {name: value for name, value in velocity.items() if model.joints[name].movable}
    incorrect["f1j3"] += 0.01
    with pytest.raises(HandModelError, match="inconsistent"):
        model.resolve_joint_velocities(incorrect, positions=q)


def test_nonlinear_limit_intersection_and_measured_overshoot(hand_models):
    historical, model, _ = hand_models
    source, follower = "f1j2", "f1j3"
    joints = dict(historical.joints)
    joints[follower] = replace(joints[follower], limit=replace(joints[follower].limit, lower=0.2, upper=0.6))
    narrow = ThreeFingerHandModel(
        base_link=historical.base_link, joints=joints, joint_order=historical.joint_order,
        finger_joint_names={name: finger.joint_names for name, finger in historical.fingers.items()},
        pads=historical.pads, fourbar_couplings=model.fourbar_couplings,
    )
    source_limit = narrow.independent_joint_limits[source]
    column = narrow.independent_joint_names.index(source)
    q = _pose(narrow)
    for source_angle, follower_angle in ((source_limit.lower, 0.2), (source_limit.upper, 0.6)):
        q[column] = source_angle
        assert narrow.resolve_joint_positions(q)[follower] == pytest.approx(follower_angle, abs=2e-14)
    assert source_limit.lower != pytest.approx(0.2, abs=1e-6)
    assert source_limit.upper != pytest.approx(0.6, abs=1e-6)
    q[column] = source_limit.lower - 1e-5
    assert not narrow.within_joint_limits(q)
    with pytest.raises(HandModelError, match="violates"):
        narrow.geometric_jacobian("f1Link3", q)
    assert narrow.resolve_joint_positions(q, enforce_limits=False)[follower] < 0.2
    assert np.isfinite(narrow.geometric_jacobian("f1Link3", q, enforce_limits=False)).all()


def test_invalid_contract_positions_and_velocity_limits_are_rejected(hand_models):
    historical, model, _ = hand_models
    coupling = model.fourbar_couplings["f1j3"]
    for key, value, message in (
        ("unknown", coupling, "follower"),
        ("f1j3", replace(coupling, source_joint="f2j1"), "adjacent"),
        ("f1j3", replace(coupling, rod_length_m=0.8), "infeasible"),
    ):
        with pytest.raises(HandModelError, match=message):
            historical.with_fourbar_couplings({key: value})
    q = _pose(model)
    invalid = {name: value for name, value in model.resolve_joint_positions(q).items() if model.joints[name].movable}
    invalid["f1j3"] = invalid["f1j2"]
    with pytest.raises(HandModelError, match="inconsistent"):
        model.resolve_joint_positions(invalid)
    with pytest.raises(HandModelError, match="finite"):
        model.resolve_joint_velocities(np.zeros(11), positions=np.full(11, np.nan))
    # Source is below its limit but the measured derivative makes the follower
    # exceed its own velocity bound at this pose.
    qdot = np.zeros(11)
    column = model.independent_joint_names.index("f1j2")
    derivative = coupling.position_and_derivative(q[column])[1]
    assert derivative > 1.0
    qdot[column] = 4.0 / derivative + 0.01
    assert qdot[column] < 4.0
    with pytest.raises(HandModelError, match="exceeds"):
        model.resolve_joint_velocities(qdot, positions=q)
