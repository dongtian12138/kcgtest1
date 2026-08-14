from copy import deepcopy
import ast
import math
from pathlib import Path

import pytest
import yaml

from kcg_connector.connector_pose import (
    DEFAULT_POSE_CONTRACT_CONFIG_PATH,
    OBJECT_TARGET_TRANSFORM_SCHEMA_VERSION,
    POSE_OBSERVATION_SCHEMA_VERSION,
    ConnectorPoseRole,
    ConnectorPoseSource,
    ObjectTargetKind,
    load_connector_pose_contract,
    pair_connector_pose_observations,
    parse_connector_pose_observation,
    parse_object_target_transform,
    resolve_object_target_pose,
)


MODULE_PATH = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "connector_pose.py"
)


def _contract():
    return load_connector_pose_contract(DEFAULT_POSE_CONTRACT_CONFIG_PATH)


def _covariance():
    return [
        [1.0e-6 if row == column else 0.0 for column in range(6)]
        for row in range(6)
    ]


def _observation(
    *,
    role="loose_plug",
    model_id="synthetic_plug_v1",
    timestamp_s=10.0,
    frame_id="world",
    source="sim_ground_truth",
):
    return {
        "schema_version": POSE_OBSERVATION_SCHEMA_VERSION,
        "model_id": model_id,
        "role": role,
        "timestamp_s": timestamp_s,
        "frame_id": frame_id,
        "position_xyz_m": [1.0, 2.0, 3.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "covariance_6x6": _covariance(),
        "confidence": 0.95,
        "symmetry_class": "keyed_order_1",
        "source": source,
    }


def _fixed_observation(**updates):
    value = _observation(
        role="fixed_receptacle",
        model_id="synthetic_receptacle_v1",
        timestamp_s=10.01,
    )
    value.update(updates)
    return value


def _transform(
    *,
    model_id="synthetic_plug_v1",
    role="loose_plug",
    target_kind="grasp",
    parent_object_frame_id="synthetic_plug_object",
):
    half_sqrt = math.sqrt(0.5)
    return {
        "schema_version": OBJECT_TARGET_TRANSFORM_SCHEMA_VERSION,
        "transform_id": "measured_plug_grasp_v1",
        "model_id": model_id,
        "role": role,
        "target_kind": target_kind,
        "parent_object_frame_id": parent_object_frame_id,
        "child_target_frame_id": "plug_grasp_target",
        "translation_xyz_m": [1.0, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, half_sqrt, half_sqrt],
    }


def _parse(value):
    return parse_connector_pose_observation(value, _contract(), now_s=10.1)


def test_module_has_no_ros_isaac_torch_or_numpy_imports():
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
        {"rclpy", "omni", "isaacsim", "torch", "numpy"}
    )


def test_shipped_contract_is_strict_and_has_no_guessed_target_transform():
    contract = _contract()
    assert contract.observation_schema_version == (
        POSE_OBSERVATION_SCHEMA_VERSION
    )
    assert contract.object_target_transforms == ()
    assert {
        (model.model_id, model.role)
        for model in contract.model_registry
    } == {
        ("synthetic_plug_v1", ConnectorPoseRole.LOOSE_PLUG),
        (
            "synthetic_receptacle_v1",
            ConnectorPoseRole.FIXED_RECEPTACLE,
        ),
        (
            "d38999_26kj61sn_proxy_v1",
            ConnectorPoseRole.LOOSE_PLUG,
        ),
        (
            "d38999_20kj61pn_proxy_v1",
            ConnectorPoseRole.FIXED_RECEPTACLE,
        ),
    }


def test_d38999_proxy_pose_pair_is_registered_but_not_calibrated():
    contract = _contract()
    loose_document = _observation(
        model_id="d38999_26kj61sn_proxy_v1",
        timestamp_s=10.0,
    )
    fixed_document = _fixed_observation(
        model_id="d38999_20kj61pn_proxy_v1",
        timestamp_s=10.01,
    )
    loose = parse_connector_pose_observation(
        loose_document, contract, now_s=10.1
    )
    fixed = parse_connector_pose_observation(
        fixed_document, contract, now_s=10.1
    )
    pair = pair_connector_pose_observations(
        loose, fixed, contract, now_s=10.1
    )
    assert pair.loose_plug.model_id == "d38999_26kj61sn_proxy_v1"
    assert pair.fixed_receptacle.model_id == (
        "d38999_20kj61pn_proxy_v1"
    )
    assert contract.object_target_transforms == ()


@pytest.mark.parametrize("source", ("sim_ground_truth", "vision"))
def test_sim_truth_and_future_vision_share_one_observation_schema(source):
    parsed = _parse(_observation(source=source))
    assert parsed.source is ConnectorPoseSource(source)
    assert parsed.position_xyz_m == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("field", ("model_id", "covariance_6x6", "source"))
def test_observation_rejects_missing_field(field):
    value = _observation()
    del value[field]
    with pytest.raises(ValueError, match="keys differ"):
        _parse(value)


def test_observation_rejects_extra_field():
    value = _observation()
    value["unversioned_guess"] = True
    with pytest.raises(ValueError, match="keys differ"):
        _parse(value)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("timestamp_s",), True),
        (("confidence",), False),
        (("position_xyz_m", 0), True),
        (("quaternion_xyzw", 3), True),
        (("covariance_6x6", 0, 0), True),
    ),
)
def test_observation_rejects_boolean_as_number(path, value):
    document = _observation()
    parent = document
    for component in path[:-1]:
        parent = parent[component]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match="boolean"):
        _parse(document)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("timestamp_s",), float("nan")),
        (("confidence",), float("inf")),
        (("position_xyz_m", 1), float("-inf")),
        (("quaternion_xyzw", 0), float("nan")),
        (("covariance_6x6", 2, 2), float("nan")),
    ),
)
def test_observation_rejects_nonfinite_values(path, value):
    document = _observation()
    parent = document
    for component in path[:-1]:
        parent = parent[component]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match="finite"):
        _parse(document)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("position_xyz_m", [0.0, 0.0]),
        ("quaternion_xyzw", [0.0, 0.0, 1.0]),
        ("covariance_6x6", [[0.0] * 6 for _ in range(5)]),
        (
            "covariance_6x6",
            [[0.0] * 6 for _ in range(5)] + [[0.0] * 5],
        ),
    ),
)
def test_observation_rejects_wrong_vector_or_covariance_shape(field, value):
    document = _observation()
    document[field] = value
    with pytest.raises(ValueError, match="contain|6x6"):
        _parse(document)


def test_quaternion_is_normalized_and_uses_one_canonical_sign():
    document = _observation()
    document["quaternion_xyzw"] = [0.0, 0.0, 0.0, -1.0]
    assert _parse(document).quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
    document["quaternion_xyzw"] = [-1.0, 0.0, 0.0, 0.0]
    assert _parse(document).quaternion_xyzw == (1.0, 0.0, 0.0, 0.0)


def test_observation_rejects_nonunit_quaternion():
    document = _observation()
    document["quaternion_xyzw"] = [0.0, 0.0, 0.0, 2.0]
    with pytest.raises(ValueError, match="normalized"):
        _parse(document)


def test_observation_rejects_asymmetric_covariance():
    document = _observation()
    document["covariance_6x6"][0][1] = 1.0e-5
    with pytest.raises(ValueError, match="symmetric"):
        _parse(document)


def test_observation_rejects_indefinite_covariance():
    document = _observation()
    document["covariance_6x6"][0][0] = 1.0
    document["covariance_6x6"][1][1] = 1.0
    document["covariance_6x6"][0][1] = 2.0
    document["covariance_6x6"][1][0] = 2.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        _parse(document)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"timestamp_s": 9.0}, "stale"),
        ({"timestamp_s": 10.2}, "future"),
        ({"confidence": 0.79}, "confidence"),
    ),
)
def test_observation_rejects_stale_future_or_low_confidence(
    updates, message
):
    document = _observation()
    document.update(updates)
    with pytest.raises(ValueError, match=message):
        _parse(document)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"model_id": "unknown"}, "unknown pose model"),
        ({"role": "fixed_receptacle"}, "role"),
        ({"symmetry_class": "continuous_axial"}, "symmetry"),
        ({"source": "camera_guess"}, "source"),
    ),
)
def test_observation_rejects_registry_or_enum_mismatch(updates, message):
    document = _observation()
    document.update(updates)
    with pytest.raises(ValueError, match=message):
        _parse(document)


def test_pair_accepts_reversed_input_and_returns_canonical_roles():
    contract = _contract()
    loose = parse_connector_pose_observation(
        _observation(), contract, now_s=10.1
    )
    fixed = parse_connector_pose_observation(
        _fixed_observation(), contract, now_s=10.1
    )
    pair = pair_connector_pose_observations(
        fixed, loose, contract, now_s=10.1
    )
    assert pair.loose_plug.model_id == "synthetic_plug_v1"
    assert pair.fixed_receptacle.model_id == "synthetic_receptacle_v1"


@pytest.mark.parametrize(
    ("fixed_updates", "message"),
    (
        ({"frame_id": "camera"}, "common frame"),
        ({"timestamp_s": 10.08}, "timestamps"),
    ),
)
def test_pair_rejects_frame_or_timestamp_mismatch(fixed_updates, message):
    contract = _contract()
    loose = parse_connector_pose_observation(
        _observation(), contract, now_s=10.1
    )
    fixed = parse_connector_pose_observation(
        _fixed_observation(**fixed_updates), contract, now_s=10.1
    )
    with pytest.raises(ValueError, match=message):
        pair_connector_pose_observations(
            loose, fixed, contract, now_s=10.1
        )


def test_pair_rejects_two_observations_with_the_same_role():
    contract = _contract()
    first = parse_connector_pose_observation(
        _observation(), contract, now_s=10.1
    )
    second_document = _observation(timestamp_s=10.01)
    second = parse_connector_pose_observation(
        second_document, contract, now_s=10.1
    )
    with pytest.raises(ValueError, match="one loose plug"):
        pair_connector_pose_observations(
            first, second, contract, now_s=10.1
        )


def test_explicit_versioned_transform_composes_parent_t_child():
    contract = _contract()
    observation_document = _observation()
    half_sqrt = math.sqrt(0.5)
    observation_document["quaternion_xyzw"] = [
        0.0,
        0.0,
        half_sqrt,
        half_sqrt,
    ]
    observation = parse_connector_pose_observation(
        observation_document, contract, now_s=10.1
    )
    transform = parse_object_target_transform(_transform(), contract)
    target = resolve_object_target_pose(
        observation, transform, contract, now_s=10.1
    )
    assert target.target_kind is ObjectTargetKind.GRASP
    assert target.target_frame_id == "plug_grasp_target"
    assert target.position_xyz_m == pytest.approx((1.0, 3.0, 3.0))
    assert target.quaternion_xyzw == pytest.approx((0.0, 0.0, 1.0, 0.0))


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"model_id": "unknown"}, "unknown model"),
        ({"role": "fixed_receptacle"}, "role"),
        ({"parent_object_frame_id": "guessed_frame"}, "parent"),
        ({"schema_version": "unversioned"}, "unsupported"),
    ),
)
def test_transform_rejects_unknown_or_mismatched_registration(
    updates, message
):
    document = _transform()
    document.update(updates)
    with pytest.raises(ValueError, match=message):
        parse_object_target_transform(document, _contract())


def test_transform_rejects_grasp_target_for_fixed_receptacle():
    document = _transform(
        model_id="synthetic_receptacle_v1",
        role="fixed_receptacle",
        target_kind="grasp",
        parent_object_frame_id="synthetic_receptacle_object",
    )
    with pytest.raises(ValueError, match="cannot register a grasp"):
        parse_object_target_transform(document, _contract())


def test_resolve_rejects_observation_transform_model_mismatch():
    contract = _contract()
    fixed = parse_connector_pose_observation(
        _fixed_observation(), contract, now_s=10.1
    )
    loose_transform = parse_object_target_transform(_transform(), contract)
    with pytest.raises(ValueError, match="model IDs differ"):
        resolve_object_target_pose(
            fixed, loose_transform, contract, now_s=10.1
        )


def test_contract_loader_rejects_extra_key_and_fabricated_bad_transform(
    tmp_path
):
    with DEFAULT_POSE_CONTRACT_CONFIG_PATH.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    extra = deepcopy(document)
    extra["implicit_defaults"] = True
    extra_path = tmp_path / "extra.yaml"
    extra_path.write_text(yaml.safe_dump(extra), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        load_connector_pose_contract(extra_path)

    bad_transform = deepcopy(document)
    bad_transform["object_target_transforms"] = [_transform()]
    bad_transform["object_target_transforms"][0][
        "parent_object_frame_id"
    ] = "unregistered_guess"
    bad_path = tmp_path / "bad_transform.yaml"
    bad_path.write_text(
        yaml.safe_dump(bad_transform), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="parent"):
        load_connector_pose_contract(bad_path)
