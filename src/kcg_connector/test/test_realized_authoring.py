'''Pure tests for realized authoring helpers.'''

from __future__ import annotations

import numpy as np
import pytest

from kcg_connector.d38999_physical_insertion import solve_fixed_q7_tcp_pose
from kcg_connector.d38999_tabletop_pick import (
    iiwa14_grasp_tcp_transform,
    load_d38999_tabletop_pick_config,
)
from kcg_connector.grasp.realized_authoring import (
    RandomizationValidationConfig,
    closure_onset_plan,
    compose_loose_plug_transform,
    minimum_jerk_blend,
    synchronous_contact_stability,
    validate_offset_arm_targets,
)

PICK_CONFIG = "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"


def _pick():
    return load_d38999_tabletop_pick_config(PICK_CONFIG)


def _validation_config():
    return RandomizationValidationConfig()


def test_compose_transform_preserves_z_and_offsets_xy():
    matrix = compose_loose_plug_transform(
        (0.43, -0.08, 0.25), 0.0004, -0.0002, 0.0
    )
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0))
    assert matrix[2, 3] == pytest.approx(0.25)
    assert matrix[0, 3] == pytest.approx(0.43 + 0.0004)
    assert matrix[1, 3] == pytest.approx(-0.08 - 0.0002)
    assert np.allclose(matrix[:3, :3], np.eye(3))


def test_compose_yaw_rotates_only_about_local_z():
    matrix = compose_loose_plug_transform(
        (0.0, 0.0, 0.0), 0.0, 0.0, 90.0
    )
    # A point at +X in the root frame rotates to +Y in world; the local
    # frame stays at the origin and the Z axis is untouched.
    point = np.asarray((1.0, 0.0, 0.3, 1.0))
    rotated = matrix @ point
    assert np.allclose(rotated[:3], (0.0, 1.0, 0.3))
    assert np.allclose(matrix[:3, 2], (0.0, 0.0, 1.0))
    assert matrix[2, 3] == pytest.approx(0.0)


def test_compose_validation_rejects_bad_inputs():
    with pytest.raises(ValueError, match="nominal_origin"):
        compose_loose_plug_transform((0.0, 0.0), 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        compose_loose_plug_transform((0.0, float("nan"), 0.0), 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite and non-bool"):
        compose_loose_plug_transform((0.0, 0.0, 0.0), True, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite and non-bool"):
        compose_loose_plug_transform((0.0, 0.0, 0.0), 0.0, 0.0, float("inf"))


def _offset_realized(nominal, dx, dy):
    nominal_tcp = np.asarray(
        iiwa14_grasp_tcp_transform(tuple(float(v) for v in nominal)),
        dtype=np.float64,
    )
    requested = nominal_tcp[:3, 3].copy()
    requested[0] += dx
    requested[1] += dy
    realized = np.asarray(
        solve_fixed_q7_tcp_pose(
            tuple(float(v) for v in nominal),
            tuple(float(v) for v in requested),
            target_rotation=nominal_tcp[:3, :3],
        ),
        dtype=np.float64,
    )
    return realized, requested


@pytest.mark.parametrize(
    "dx,dy",
    (
        (0.0003, 0.0003),
        (-0.0003, 0.0003),
        (0.0003, -0.0003),
        (-0.0003, -0.0003),
    ),
)
def test_offset_arm_targets_pass_at_interval_corners(dx, dy):
    pick = _pick()
    for nominal in (
        np.asarray(pick.motion.grasp_arm_rad, dtype=np.float64),
        np.asarray(
            pick.motion.closure_clearance_arm_rad, dtype=np.float64
        ),
        np.asarray(
            pick.motion.approach_segments[-1].target_arm_rad,
            dtype=np.float64,
        ),
    ):
        realized, requested = _offset_realized(nominal, dx, dy)
        residuals = validate_offset_arm_targets(
            nominal,
            realized,
            _validation_config(),
            fk=iiwa14_grasp_tcp_transform,
            requested_position_m=requested,
        )
        assert residuals["maximum_joint_delta_rad"] <= 0.05
        assert residuals["q7_delta_rad"] <= 1.0e-9
        assert residuals["position_residual_m"] <= 1.0e-7
        assert residuals["rotation_residual_rad"] <= 1.0e-7


def test_offset_validation_rejects_q7_motion_and_joint_jumps():
    pick = _pick()
    nominal = np.asarray(pick.motion.grasp_arm_rad, dtype=np.float64)
    realized, requested = _offset_realized(nominal, 0.0003, 0.0003)
    moved_q7 = realized.copy()
    moved_q7[6] += 1.0e-3
    with pytest.raises(ValueError, match="q7"):
        validate_offset_arm_targets(
            nominal,
            moved_q7,
            _validation_config(),
            fk=iiwa14_grasp_tcp_transform,
            requested_position_m=requested,
        )
    big_jump = realized.copy()
    big_jump[1] += 0.2
    with pytest.raises(ValueError, match="joint_delta"):
        validate_offset_arm_targets(
            nominal,
            big_jump,
            _validation_config(),
            fk=iiwa14_grasp_tcp_transform,
            requested_position_m=requested,
        )


def test_offset_validation_rejects_unbounded_fk_residuals():
    pick = _pick()
    nominal = np.asarray(pick.motion.grasp_arm_rad, dtype=np.float64)
    realized, requested = _offset_realized(nominal, 0.0003, 0.0003)

    def broken_fk(arm_rad):
        return np.eye(4, dtype=np.float64) * 2.0

    with pytest.raises(ValueError, match="position residual"):
        validate_offset_arm_targets(
            nominal,
            realized,
            _validation_config(),
            fk=broken_fk,
            requested_position_m=requested,
        )


def test_offset_validation_input_validation():
    config = _validation_config()
    with pytest.raises(ValueError, match="7-vector"):
        validate_offset_arm_targets(
            (0.0,) * 6,
            (0.0,) * 7,
            config,
            fk=iiwa14_grasp_tcp_transform,
            requested_position_m=(0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="finite"):
        validate_offset_arm_targets(
            (0.0,) * 7,
            (float("nan"),) * 7,
            config,
            fk=iiwa14_grasp_tcp_transform,
            requested_position_m=(0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="requested_position"):
        validate_offset_arm_targets(
            (0.0,) * 7,
            (0.0,) * 7,
            config,
            fk=iiwa14_grasp_tcp_transform,
            requested_position_m=(0.0, float("inf"), 0.0),
        )


def test_validation_config_is_frozen_sim_tuning():
    with pytest.raises(ValueError, match="SIM_TUNING_ONLY"):
        RandomizationValidationConfig(threshold_label="HARDWARE")
    with pytest.raises(ValueError, match="maximum_arm_joint_delta_rad"):
        RandomizationValidationConfig(maximum_arm_joint_delta_rad=0.0)
    with pytest.raises(ValueError, match="maximum_arm_joint_delta_rad"):
        RandomizationValidationConfig(maximum_arm_joint_delta_rad=True)
    with pytest.raises(ValueError, match="maximum_fk_position_error_m"):
        RandomizationValidationConfig(
            maximum_fk_position_error_m=float("inf")
        )
    with pytest.raises(ValueError, match="maximum_fk_rotation_error_rad"):
        RandomizationValidationConfig(
            maximum_fk_rotation_error_rad=float("nan")
        )


def test_onset_plan_full_duration_and_total_steps():
    total, plan = closure_onset_plan(10, (0, 5, 2))
    assert total == 15
    assert len(plan) == 15
    # Channel 0 starts immediately and plays the full nominal profile.
    for step in range(10):
        assert plan[step][0] == pytest.approx(
            minimum_jerk_blend(float(step + 1) / 10.0)
        )
    # Channels with delays hold open (None) until their onset.
    for step in range(5):
        assert plan[step][1] is None
    for step in range(2):
        assert plan[step][2] is None
    # The last row has every channel at (or holding) the closed target.
    assert plan[-1] == (1.0, 1.0, 1.0)
    # Every row is a 3-tuple: the f1j1 spread joint is never in the plan.
    assert all(len(row) == 3 for row in plan)


def test_onset_plan_validates_inputs():
    with pytest.raises(ValueError, match="closure_steps"):
        closure_onset_plan(0, (0, 0, 0))
    with pytest.raises(ValueError, match="closure_steps"):
        closure_onset_plan(True, (0, 0, 0))
    with pytest.raises(ValueError, match="three values"):
        closure_onset_plan(10, (0, 0))
    with pytest.raises(ValueError, match="non-negative"):
        closure_onset_plan(10, (0, -1, 0))
    with pytest.raises(ValueError, match="non-negative"):
        closure_onset_plan(10, (0, True, 0))


def test_synchronous_contact_stability_requires_final_confirmed_states():
    assert synchronous_contact_stability(
        ("f1", "f2", "f3"),
        {
            "f1": "CONTACT_CONFIRMED",
            "f2": "CONTACT_CONFIRMED",
            "f3": "CONTACT_CONFIRMED",
        },
    )
    assert not synchronous_contact_stability(
        ("f1", "f2"),
        {"f1": "CONTACT_CONFIRMED", "f2": "CONTACT_CONFIRMED"},
    )
    # A historical contact order can never mask a terminal slip.
    assert not synchronous_contact_stability(
        ("f1", "f2", "f3"),
        {
            "f1": "CONTACT_CONFIRMED",
            "f2": "CONTACT_CONFIRMED",
            "f3": "SLIP_SUSPECTED",
        },
    )
    assert not synchronous_contact_stability(
        ("f1", "f2", "f3"),
        {
            "f1": "CONTACT_CONFIRMED",
            "f2": "CONTACT_CONFIRMED",
            "f3": "FAILED",
        },
    )


from kcg_connector.grasp.realized_authoring import (
    float32_readback_evidence,
)


def test_float32_readback_accepts_seed0_table_friction():
    intended = 0.8306600843204117
    readback = 0.8306601047515869  # the USD float32 storage readback
    evidence = float32_readback_evidence(
        intended, readback, label="table_static_friction"
    )
    assert evidence["verified"] is True
    assert evidence["storage_type"] == "float32"
    assert evidence["storage_expected"] == pytest.approx(
        float(np.float32(intended))
    )
    assert evidence["maximum_quantization_error"] == pytest.approx(
        2.04e-8, abs=1.0e-10
    )
    assert evidence["maximum_storage_error"] == 0.0


def test_float32_readback_accepts_08_and_scaled_mass():
    for intended in (0.08, 0.08 * 1.1):
        readback = float(np.float32(intended))
        evidence = float32_readback_evidence(
            intended, readback, label="mass"
        )
        assert evidence["verified"] is True
        assert evidence["storage_expected"] == pytest.approx(
            float(np.float32(intended))
        )
    # The asset-authored 0.08 reads back as its float32 storage, which is
    # NOT equal to 0.08 in float64: the helper must accept exactly this.
    asset_readback = 0.07999999821186066
    evidence = float32_readback_evidence(
        0.08, asset_readback, label="original_mass"
    )
    assert evidence["verified"] is True


def test_float32_readback_accepts_com_vector():
    intended = (0.001, -0.001, 0.0)
    readback = tuple(float(np.float32(value)) for value in intended)
    evidence = float32_readback_evidence(
        intended, readback, label="com"
    )
    assert evidence["verified"] is True
    assert evidence["shape"] == [3]
    assert evidence["storage_expected"] == [
        float(np.float32(value)) for value in intended
    ]


def test_float32_readback_fails_on_adjacent_float32():
    intended = 0.8306600843204117
    stored = np.float32(intended)
    adjacent = float(np.nextafter(stored, np.float32(np.inf)))
    evidence = float32_readback_evidence(
        intended, adjacent, label="table_static_friction"
    )
    assert evidence["verified"] is False
    assert evidence["maximum_storage_error"] > 0.0


def test_float32_readback_rejects_malformed_inputs():
    with pytest.raises(ValueError, match="shapes differ"):
        float32_readback_evidence(
            (1.0, 2.0), 1.0, label="bad_shape"
        )
    with pytest.raises(ValueError, match="finite"):
        float32_readback_evidence(
            float("nan"), 1.0, label="bad_nan"
        )
    with pytest.raises(ValueError, match="finite"):
        float32_readback_evidence(
            1.0, float("inf"), label="bad_inf"
        )
    with pytest.raises(ValueError, match="booleans"):
        float32_readback_evidence(
            True, 1.0, label="bad_bool"
        )
    with pytest.raises(ValueError, match="booleans"):
        float32_readback_evidence(
            1.0, True, label="bad_bool_readback"
        )


def test_float32_readback_rejects_numpy_bool_scalar_and_array():
    with pytest.raises(ValueError, match="booleans"):
        float32_readback_evidence(
            np.bool_(True), 1.0, label="np_bool_scalar"
        )
    with pytest.raises(ValueError, match="booleans"):
        float32_readback_evidence(
            1.0, np.bool_(False), label="np_bool_readback"
        )
    with pytest.raises(ValueError, match="booleans"):
        float32_readback_evidence(
            (1.0, np.bool_(True), 2.0),
            (1.0, 1.0, 2.0),
            label="np_bool_in_vector",
        )
    with pytest.raises(ValueError, match="booleans"):
        float32_readback_evidence(
            np.asarray([1.0, 2.0]),
            np.asarray([np.bool_(True), 2.0], dtype=object),
            label="np_bool_in_array",
        )
