import ast
from pathlib import Path

import pytest

from kcg_connector.connector_pose import (
    DEFAULT_POSE_CONTRACT_CONFIG_PATH,
    ConnectorPoseRole,
    ConnectorPoseSource,
    load_connector_pose_contract,
    pair_connector_pose_observations,
)
from kcg_connector.sim_pose_provider import (
    isaac_wxyz_to_contract_xyzw,
    make_sim_ground_truth_observation,
)


MODULE_PATH = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "sim_pose_provider.py"
)


def _contract():
    return load_connector_pose_contract(DEFAULT_POSE_CONTRACT_CONFIG_PATH)


def _observation(model_id, role, timestamp=2.0):
    return make_sim_ground_truth_observation(
        _contract(),
        model_id=model_id,
        role=role,
        timestamp_s=timestamp,
        now_s=2.0,
        frame_id="world",
        position_xyz_m=(0.52, -0.21, 0.20),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def test_provider_has_no_ros_isaac_omni_torch_or_numpy_imports():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint(
        {"rclpy", "isaacsim", "omni", "pxr", "torch", "numpy"}
    )


def test_quaternion_conversion_is_explicit_wxyz_to_xyzw():
    assert isaac_wxyz_to_contract_xyzw((0.5, 0.1, 0.2, 0.3)) == (
        0.1,
        0.2,
        0.3,
        0.5,
    )


def test_d38999_sim_truth_pair_passes_shared_contract():
    contract = _contract()
    loose = _observation(
        "d38999_26kj61sn_proxy_v1", ConnectorPoseRole.LOOSE_PLUG
    )
    fixed = _observation(
        "d38999_20kj61pn_proxy_v1",
        ConnectorPoseRole.FIXED_RECEPTACLE,
    )
    pair = pair_connector_pose_observations(
        loose, fixed, contract, now_s=2.0
    )
    assert pair.loose_plug.source is ConnectorPoseSource.SIM_GROUND_TRUTH
    assert pair.fixed_receptacle.source is (
        ConnectorPoseSource.SIM_GROUND_TRUTH
    )
    assert pair.loose_plug.quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert pair.loose_plug.covariance_6x6 == ((0.0,) * 6,) * 6
    assert contract.object_target_transforms == ()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("position_xyz_m", (0.0, 0.0), "3 finite"),
        ("quaternion_wxyz", (1.0, 0.0, 0.0), "4 finite"),
        ("translation_variance_m2", -1.0, "non-negative"),
        ("rotation_variance_rad2", float("nan"), "finite"),
        ("confidence", True, "finite"),
    ),
)
def test_provider_rejects_invalid_numerical_input(field, value, message):
    arguments = {
        "model_id": "d38999_26kj61sn_proxy_v1",
        "role": "loose_plug",
        "timestamp_s": 2.0,
        "now_s": 2.0,
        "frame_id": "world",
        "position_xyz_m": (0.52, -0.21, 0.20),
        "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        make_sim_ground_truth_observation(_contract(), **arguments)


def test_provider_rejects_model_role_mismatch():
    with pytest.raises(ValueError, match="role"):
        _observation(
            "d38999_20kj61pn_proxy_v1",
            ConnectorPoseRole.LOOSE_PLUG,
        )
