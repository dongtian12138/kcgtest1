import math

import numpy as np
import pytest

from kcg_connector.postgrasp_error_injection import (
    assembly_tcp_from_grasp_tcp,
    PostGraspError,
    compose_nominal_with_error,
    injection_error,
    integrate_assembly_twist_on_grasp_tcp,
    measure_error_from_nominal,
)


@pytest.mark.parametrize(
    "translation,rotation",
    [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.001, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((-0.001, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.001, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, -0.001, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (math.radians(1.0), 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (-math.radians(1.0), 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, math.radians(1.0), 0.0)),
        ((0.0, 0.0, 0.0), (0.0, -math.radians(1.0), 0.0)),
        ((0.001, -0.002, 0.0003), (0.01, -0.02, 0.03)),
    ],
)
def test_requested_error_round_trips(translation, rotation):
    requested = PostGraspError(translation, rotation)
    pose = compose_nominal_with_error((0.0, 0.0, 0.44848), requested)
    actual = measure_error_from_nominal(
        pose[:3, 3], pose[:3, :3], (0.0, 0.0, 0.44848)
    )
    difference = injection_error(requested, actual)
    assert difference["translation_error_norm_m"] <= 1.0e-12
    assert difference["rotation_error_norm_rad"] <= 1.0e-12


def test_error_is_preserved_when_hand_pose_changes():
    requested = PostGraspError(
        (0.001, -0.002, 0.0003), (0.01, -0.02, 0.03)
    )
    hand_from_plug = compose_nominal_with_error(
        (0.0, 0.0, 0.44848), requested
    )
    world_from_hand_a = np.eye(4)
    world_from_hand_b = np.eye(4)
    world_from_hand_b[:3, :3] = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    world_from_hand_b[:3, 3] = (0.4, -0.2, 0.8)
    for world_from_hand in (world_from_hand_a, world_from_hand_b):
        world_from_plug = world_from_hand @ hand_from_plug
        recovered = np.linalg.inv(world_from_hand) @ world_from_plug
        actual = measure_error_from_nominal(
            recovered[:3, 3], recovered[:3, :3], (0.0, 0.0, 0.44848)
        )
        assert injection_error(requested, actual)[
            "translation_error_norm_m"
        ] <= 1.0e-12
        assert injection_error(requested, actual)[
            "rotation_error_norm_rad"
        ] <= 1.0e-12


def test_invalid_source_fails_closed():
    with pytest.raises(ValueError, match="only allowed at simulation reset"):
        PostGraspError(source="controller_runtime")


def test_angular_assembly_twist_keeps_mating_face_pivot_fixed():
    grasp = np.eye(4)
    offset = np.asarray((0.0, 0.0, 0.04848))
    before = assembly_tcp_from_grasp_tcp(grasp, offset)
    moved_grasp = integrate_assembly_twist_on_grasp_tcp(
        grasp,
        (0.0, 0.0, 0.0, math.radians(-0.4), 0.0, 0.0),
        np.eye(3),
        offset,
        1.0,
    )
    after = assembly_tcp_from_grasp_tcp(moved_grasp, offset)
    assert after[:3, 3] == pytest.approx(before[:3, 3], abs=1.0e-12)
    # Rotating around the hand instead would create about 0.338 mm of Y
    # motion at this offset; ensure the compensating grasp motion is present.
    assert abs(moved_grasp[1, 3]) > 0.00033


def test_assembly_twist_translation_is_observed_at_offset_tcp():
    grasp = np.eye(4)
    offset = np.asarray((0.0, 0.0, 0.04848))
    moved = integrate_assembly_twist_on_grasp_tcp(
        grasp,
        (0.001, -0.002, 0.003, 0.0, 0.0, 0.0),
        np.eye(3),
        offset,
        0.25,
    )
    delta = (
        assembly_tcp_from_grasp_tcp(moved, offset)[:3, 3]
        - assembly_tcp_from_grasp_tcp(grasp, offset)[:3, 3]
    )
    assert delta == pytest.approx((0.00025, -0.0005, 0.00075))
