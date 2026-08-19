import math

import numpy as np
import pytest

from kcg_connector.d38999_physical_insertion import solve_fixed_q7_tcp_pose
from kcg_connector.d38999_tabletop_pick import (
    iiwa14_grasp_tcp_transform,
    load_d38999_tabletop_pick_config,
)


def _rotation_error(first, second):
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.acos(float(cosine))


def test_fixed_q7_stage_targets_are_spatial_2_12_52_mm_not_time_blends():
    pick = load_d38999_tabletop_pick_config(
        "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
    )
    seed = tuple(float(value) for value in pick.motion.grasp_arm_rad)
    initial = np.asarray(iiwa14_grasp_tcp_transform(seed), dtype=np.float64)
    current = seed
    for cumulative_lift_m in (0.002, 0.012, 0.052):
        target_position = initial[:3, 3].copy()
        target_position[2] += cumulative_lift_m
        current = solve_fixed_q7_tcp_pose(
            current,
            tuple(float(value) for value in target_position),
            target_rotation=initial[:3, :3],
        )
        actual = np.asarray(
            iiwa14_grasp_tcp_transform(current), dtype=np.float64
        )
        assert actual[2, 3] - initial[2, 3] == pytest.approx(
            cumulative_lift_m, abs=1.0e-7
        )
        assert np.linalg.norm(actual[:2, 3] - initial[:2, 3]) <= 1.0e-7
        assert _rotation_error(initial[:3, :3], actual[:3, :3]) <= 1.0e-7
