from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.robust.hand_model import HandModelError


def test_measured_state_jacobian_retains_sensor_overshoot_and_planning_guard():
    repository = Path(__file__).resolve().parents[3]
    model = load_v2_inputs(
        repository,
        config_path=repository / "artifacts/kcg_connector/isaac/te_body_grasp_palm_shift_0p17mm_v1/nominal/config.yaml",
        object_id="te_deutsch_d38999_26fj35pn_step",
    ).robot_model
    q = np.zeros(len(model.independent_joint_names))
    column = model.independent_joint_names.index("f1j2")
    q[column] = -9.49888089963e-6  # actual preflight sensor value, retained
    original = q.copy()
    with pytest.raises(HandModelError, match="violates"):
        model.geometric_jacobian("f1Link3", q)
    jacobian = model.geometric_jacobian("f1Link3", q, enforce_limits=False)
    h = 1e-6
    plus, minus = q.copy(), q.copy()
    plus[column] += h; minus[column] -= h
    high = model.forward_kinematics(plus, enforce_limits=False)["f1Link3"]
    low = model.forward_kinematics(minus, enforce_limits=False)["f1Link3"]
    finite_difference = np.r_[
        (high[:3, 3]-low[:3, 3])/(2*h),
        Rotation.from_matrix(high[:3, :3] @ low[:3, :3].T).as_rotvec()/(2*h),
    ]
    np.testing.assert_allclose(jacobian[:, column], finite_difference, atol=1e-8, rtol=1e-7)
    np.testing.assert_array_equal(q, original)
    with pytest.raises(HandModelError, match="finite"):
        model.geometric_jacobian("f1Link3", np.full(len(q), np.nan), enforce_limits=False)
