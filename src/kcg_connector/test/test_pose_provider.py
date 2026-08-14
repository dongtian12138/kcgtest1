"""Pure contracts for the replaceable connector pose-provider boundary."""

from copy import deepcopy
import ast
import json
import math
from pathlib import Path

import pytest

from kcg_connector.connector_pose import (
    ConnectorPoseRole,
    load_connector_pose_contract,
    pair_connector_pose_observations,
)
from kcg_connector.pose_provider import (
    POSE_PROVIDER_SAMPLE_SCHEMA_VERSION,
    PoseProvider,
    PoseProviderPurpose,
    PoseProviderSample,
    parse_pose_provider_sample,
    pose_provider_sample_to_mapping,
    validate_pose_provider_sample,
)
from kcg_connector.sim_pose_provider import (
    make_sim_ground_truth_observation,
)


MODULE_PATH = (
    Path(__file__).parents[1] / "kcg_connector" / "pose_provider.py"
)
CALIBRATION_SHA256 = "a" * 64
_DEFAULT_PAIR = object()


def _contract():
    return load_connector_pose_contract()


def _sim_pair(timestamp_s=10.0):
    contract = _contract()
    loose = make_sim_ground_truth_observation(
        contract,
        model_id="d38999_26kj61sn_proxy_v1",
        role=ConnectorPoseRole.LOOSE_PLUG,
        timestamp_s=timestamp_s,
        now_s=timestamp_s,
        frame_id="world",
        position_xyz_m=(0.52, -0.21, 0.20),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation_variance_m2=1.0e-6,
        rotation_variance_rad2=1.0e-4,
        confidence=0.95,
    )
    fixed = make_sim_ground_truth_observation(
        contract,
        model_id="d38999_20kj61pn_proxy_v1",
        role=ConnectorPoseRole.FIXED_RECEPTACLE,
        timestamp_s=timestamp_s,
        now_s=timestamp_s,
        frame_id="world",
        position_xyz_m=(0.55, 0.185, 0.2615),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation_variance_m2=1.0e-6,
        rotation_variance_rad2=1.0e-4,
        confidence=0.95,
    )
    return pair_connector_pose_observations(
        loose, fixed, contract, now_s=timestamp_s
    )


def _sample_object(pair):
    return PoseProviderSample(
        schema_version=POSE_PROVIDER_SAMPLE_SCHEMA_VERSION,
        purpose=PoseProviderPurpose.CONTROL,
        provider_id="sim_truth",
        provider_version="v1",
        capture_id="capture-0001",
        clock_domain="isaac_sim_time",
        control_frame="world",
        calibration_sha256=CALIBRATION_SHA256,
        pair=pair,
        reference_truth_pair=None,
        full_6d=True,
        keyed_orientation_observed=True,
        uses_truth_position=True,
        uses_truth_orientation=True,
        control_authorized=True,
        preflight_passed=True,
        diagnostics={
            "finite": True,
            "visible_pixels": [820, 910],
            "xy_error_m": 0.0025,
        },
    )


def _pair(source="sim_ground_truth", timestamp_s=10.0):
    document = pose_provider_sample_to_mapping(
        _sample_object(_sim_pair(timestamp_s))
    )["pair"]
    if source != "sim_ground_truth":
        document["loose_plug"]["source"] = source
        document["fixed_receptacle"]["source"] = source
    return document


def _sample(
    *,
    purpose="control",
    pair=_DEFAULT_PAIR,
    reference_truth_pair=None,
    full_6d=True,
    keyed_orientation_observed=True,
    uses_truth_position=True,
    uses_truth_orientation=True,
    control_authorized=True,
    preflight_passed=True,
):
    document = pose_provider_sample_to_mapping(_sample_object(_sim_pair()))
    document.update(
        {
            "purpose": purpose,
            "reference_truth_pair": reference_truth_pair,
            "full_6d": full_6d,
            "keyed_orientation_observed": keyed_orientation_observed,
            "uses_truth_position": uses_truth_position,
            "uses_truth_orientation": uses_truth_orientation,
            "control_authorized": control_authorized,
            "preflight_passed": preflight_passed,
        }
    )
    if pair is not _DEFAULT_PAIR:
        document["pair"] = pair
    return document


def _parse(document, purpose="control", now_s=10.1):
    return parse_pose_provider_sample(
        document,
        _contract(),
        purpose=purpose,
        now_s=now_s,
        expected_clock_domain="isaac_sim_time",
        expected_control_frame="world",
    )


def test_module_is_pure_and_protocol_names_observe_pair_contract():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {"isaacsim", "omni", "pxr", "rclpy", "torch", "numpy"}
    )
    assert "observe_pair" in PoseProvider.__dict__


def test_explicit_control_allows_complete_disclosed_sim_truth_pair():
    sample = _parse(_sample())
    assert sample.purpose is PoseProviderPurpose.CONTROL
    assert sample.control_authorized is True
    assert sample.pair.loose_plug.source.value == "sim_ground_truth"
    encoded = pose_provider_sample_to_mapping(sample)
    json.dumps(encoded, allow_nan=False)
    assert validate_pose_provider_sample(
        sample,
        _contract(),
        purpose="control",
        now_s=10.1,
        expected_clock_domain="isaac_sim_time",
        expected_control_frame="world",
    ) == sample


def test_sim_truth_control_authorization_requires_explicit_control_purpose():
    document = _sample(purpose="preflight")
    with pytest.raises(ValueError, match="control purpose"):
        _parse(document, purpose="preflight")
    with pytest.raises(ValueError, match="differs from request"):
        _parse(_sample(), purpose="preflight")


def test_masked_rgbd_with_truth_orientation_is_preflight_only():
    document = _sample(
        purpose="preflight",
        pair=None,
        reference_truth_pair=_pair(),
        full_6d=False,
        keyed_orientation_observed=False,
        uses_truth_position=False,
        uses_truth_orientation=True,
        control_authorized=False,
    )
    document["provider_id"] = "masked_rgbd_xy_truth_orientation"
    sample = _parse(document, purpose="preflight")
    assert sample.preflight_passed is True
    assert sample.pair is None
    assert sample.reference_truth_pair is not None
    assert sample.uses_truth_orientation is True
    assert sample.control_authorized is False

    dishonest = deepcopy(document)
    dishonest["purpose"] = "control"
    dishonest["control_authorized"] = True
    with pytest.raises(ValueError, match="keyed full-6D"):
        _parse(dishonest, purpose="control")


def test_full_keyed_vision_pair_without_truth_can_authorize_control():
    document = _sample(
        pair=_pair(source="vision"),
        uses_truth_position=False,
        uses_truth_orientation=False,
    )
    document["provider_id"] = "future_foundation_pose"
    sample = _parse(document)
    assert sample.pair.loose_plug.source.value == "vision"
    assert sample.control_authorized is True


def test_vision_pair_cannot_hide_truth_components():
    document = _sample(
        pair=_pair(source="vision"),
        uses_truth_position=False,
        uses_truth_orientation=True,
    )
    with pytest.raises(ValueError, match="cannot use truth"):
        _parse(document)


def test_pair_endpoints_must_use_one_source():
    mixed = _pair()
    mixed["fixed_receptacle"]["source"] = "vision"
    with pytest.raises(ValueError, match="one source"):
        _parse(_sample(pair=mixed))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("calibration_sha256", "ABC", "lowercase SHA-256"),
        ("control_authorized", 1, "boolean"),
        ("clock_domain", "isaac sim time", "stable identifier"),
    ),
)
def test_strict_sidecar_types(field, value, message):
    document = _sample()
    document[field] = value
    with pytest.raises(ValueError, match=message):
        _parse(document)


def test_exact_schema_and_json_safe_diagnostics_are_fail_closed():
    missing = _sample()
    del missing["capture_id"]
    with pytest.raises(ValueError, match="keys differ"):
        _parse(missing)

    extra = _sample()
    extra["unversioned_hint"] = True
    with pytest.raises(ValueError, match="keys differ"):
        _parse(extra)

    non_text_key = _sample()
    non_text_key[1] = "invalid"
    with pytest.raises(ValueError, match="keys must be strings"):
        _parse(non_text_key)

    for bad in (float("nan"), (1, 2), {1: "not a string key"}):
        document = _sample()
        document["diagnostics"] = {"bad": bad}
        with pytest.raises(ValueError, match="finite|JSON-safe|strings"):
            _parse(document)


def test_clock_frame_age_and_capture_skew_cross_gates():
    with pytest.raises(ValueError, match="clock domain"):
        parse_pose_provider_sample(
            _sample(),
            _contract(),
            purpose="control",
            now_s=10.1,
            expected_clock_domain="wall_time",
            expected_control_frame="world",
        )
    with pytest.raises(ValueError, match="control frame"):
        parse_pose_provider_sample(
            _sample(),
            _contract(),
            purpose="control",
            now_s=10.1,
            expected_clock_domain="isaac_sim_time",
            expected_control_frame="robot_world",
        )
    with pytest.raises(ValueError, match="stale"):
        _parse(_sample(), now_s=10.251)

    skewed = _sample(reference_truth_pair=_pair(timestamp_s=10.06))
    with pytest.raises(ValueError, match="skewed"):
        _parse(skewed)


def test_reference_pair_must_be_truth_and_match_models():
    vision_reference = _sample(
        reference_truth_pair=_pair(source="vision")
    )
    with pytest.raises(ValueError, match="must be sim_ground_truth"):
        _parse(vision_reference)

    mismatched = _sample(reference_truth_pair=_pair())
    mismatched["reference_truth_pair"]["loose_plug"]["model_id"] = (
        "synthetic_plug_v1"
    )
    mismatched["reference_truth_pair"]["loose_plug"][
        "symmetry_class"
    ] = "keyed_order_1"
    with pytest.raises(ValueError, match="compatible|model IDs differ"):
        _parse(mismatched)


def test_boolean_relationships_and_empty_passing_evidence_reject():
    keyed_partial = _sample(full_6d=False)
    with pytest.raises(ValueError, match="keyed orientation"):
        _parse(keyed_partial)

    empty = _sample(
        purpose="preflight",
        pair=None,
        reference_truth_pair=None,
        full_6d=False,
        keyed_orientation_observed=False,
        uses_truth_position=False,
        uses_truth_orientation=False,
        control_authorized=False,
    )
    with pytest.raises(ValueError, match="needs pose or truth evidence"):
        _parse(empty, purpose="preflight")


def test_diagnostics_reject_nonfinite_nested_number():
    document = _sample()
    document["diagnostics"] = {
        "nested": [{"speed": math.inf}],
    }
    with pytest.raises(ValueError, match="finite"):
        _parse(document)
