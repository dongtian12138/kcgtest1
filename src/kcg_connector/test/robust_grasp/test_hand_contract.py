from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import yaml

from kcg_connector.grasp.robust.hand_contract import (
    HandContractError,
    OBJECT_CONTACT_NORMAL_POLICY,
    PAD_SURFACE_NORMAL_POLICY,
    PAD_TRIANGLE_WINDING_SOURCE,
    load_carts_hand_contract,
)


REPOSITORY = Path(__file__).resolve().parents[4]
CONTRACT_PATH = REPOSITORY / "src/kcg_connector/config/carts_hand_contact_v1.yaml"


def _source_mapping() -> dict[str, Any]:
    value = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_contract(tmp_path: Path, value: dict[str, Any]) -> Path:
    path = tmp_path / "hand_contract.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _load(path: Path):
    return load_carts_hand_contract(path, repository_root=REPOSITORY)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_real_contract_verifies_bytes_whole_pad_meshes_and_urdf_mapping() -> None:
    contract = _load(CONTRACT_PATH)

    assert contract.schema_version == "carts_hand_contact_v1"
    assert contract.method == "CARTS-Grasp"
    assert contract.hardware_authorized is False
    assert contract.dynamic_validation_complete is False
    assert contract.dynamic_use_allowed is False
    assert contract.physical_calibration_complete is False
    assert contract.simulator_readback_complete is False
    assert contract.truth_firewall_all_false
    assert contract.force_capacity_value_source == (
        "USER_PROVIDED_HARDWARE_CAPABILITY_PENDING_CALIBRATION"
    )
    assert contract.force_capacity_role == (
        "OPTIMIZATION_UPPER_BOUND_NOT_BINARY_GRASP_THRESHOLD"
    )
    assert contract.urdf.sha256 == _sha256(contract.urdf.absolute_path)
    assert contract.source_manifest.sha256 == _sha256(
        contract.source_manifest.absolute_path
    )

    expected = {
        "finger_1_pad": ("f1", "f1Link3"),
        "finger_2_pad": ("f2", "f2Link2"),
        "finger_3_pad": ("f3", "f3Link3"),
    }
    assert set(contract.pad_by_name) == set(expected)
    for name, (finger, link) in expected.items():
        pad = contract.pad_by_name[name]
        assert (pad.finger_name, pad.link_name) == (finger, link)
        assert pad.coordinate_frame == link
        assert pad.unit == "m"
        assert pad.mesh.sha256 == _sha256(pad.mesh.absolute_path)
        assert pad.normal_force_capacity_n == 15.0
        assert pad.points_local_m.shape == (1250, 3)
        assert pad.faces.shape == (2479, 3)
        assert pad.points_local_m.dtype == np.float64
        assert pad.faces.dtype == np.int64
        assert np.all(np.isfinite(pad.points_local_m))
        assert int(np.min(pad.faces)) >= 0
        assert int(np.max(pad.faces)) < pad.vertex_count
        assert not pad.points_local_m.flags.writeable
        assert not pad.faces.flags.writeable
        with pytest.raises(ValueError):
            pad.points_local_m[0, 0] = 0.0
        with pytest.raises(ValueError):
            pad.points_local_m.flags.writeable = True
        with pytest.raises(ValueError):
            pad.faces.flags.writeable = True
        assert not hasattr(pad, "pad_contact_face_ids")
        assert not hasattr(pad, "pad_side_face_ids")

    model_input = contract.to_hand_model_pad_contract()
    encoded = json.dumps(model_input, sort_keys=True)
    assert "pad_contact_face_ids" not in encoded
    assert "pad_side_face_ids" not in encoded
    assert "0p90" not in encoded
    assert "alignment" not in encoded
    model = contract.build_hand_model()
    assert {
        name: (pad.finger_name, pad.link_name)
        for name, pad in model.pads.items()
    } == expected
    assert all(pad.contact_normal_pad is None for pad in model.pads.values())
    assert contract.closure_actuation_method == (
        "PRE_REGISTERED_SEQUENTIAL_FINGER_EXCLUSIVE_JOINT_PATHS"
    )
    assert contract.shared_independent_joint_role == (
        "PREGRASP_CONFIGURATION_ONLY_NOT_CLOSURE"
    )
    assert contract.object_contact_normal_policy == OBJECT_CONTACT_NORMAL_POLICY
    assert contract.pad_surface_normal_policy == PAD_SURFACE_NORMAL_POLICY
    assert contract.pad_triangle_winding_source == PAD_TRIANGLE_WINDING_SOURCE
    assert contract.pad_triangle_winding_consistency_required is True
    closure = contract.closing_actuation_directions_unit(model)
    assert closure.shape == (3, len(model.independent_joint_names))
    expected_closure = np.zeros_like(closure)
    for row_index, joint_name in enumerate(("f1j2", "f2j1", "f3j2")):
        expected_closure[row_index, model.independent_joint_names.index(joint_name)] = 1.0
    np.testing.assert_array_equal(closure, expected_closure)
    assert not closure.flags.writeable


@pytest.mark.parametrize("joint_name", ("f1j1", "f2j1"))
def test_shared_or_foreign_closure_joint_is_rejected(
    tmp_path: Path, joint_name: str
) -> None:
    value = _source_mapping()
    value["closure_actuation"]["rows"]["f1"]["joint_weights"] = {
        joint_name: 1.0
    }
    with pytest.raises(HandContractError, match="shared or foreign"):
        _load(_write_contract(tmp_path, value))


def test_missing_required_field_is_rejected_without_default(
    tmp_path: Path,
) -> None:
    value = _source_mapping()
    del value["pads"]["finger_1_pad"]["footprint"]["unit"]
    with pytest.raises(HandContractError, match="missing=.*unit"):
        _load(_write_contract(tmp_path, value))


@pytest.mark.parametrize(
    "path_value",
    (
        "/etc/passwd",
        "../outside.npz",
        "artifacts//non_normalized.npz",
        r"artifacts\windows_style.npz",
    ),
)
def test_non_repository_relative_or_non_normalized_pad_path_is_rejected(
    tmp_path: Path, path_value: str
) -> None:
    value = _source_mapping()
    value["pads"]["finger_1_pad"]["footprint"]["mesh_uri"] = path_value
    with pytest.raises(HandContractError, match="path"):
        _load(_write_contract(tmp_path, value))


def test_declared_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    value = _source_mapping()
    value["pads"]["finger_2_pad"]["footprint"]["mesh_sha256"] = "0" * 64
    with pytest.raises(HandContractError, match="SHA-256 mismatch"):
        _load(_write_contract(tmp_path, value))


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (
            lambda value: value["pad_semantics"].__setitem__(
                "authority", "AUTOMATIC_COMPONENT_GUESS"
            ),
            "pad_semantics.authority",
        ),
        (
            lambda value: value["force_capacity"].__setitem__(
                "value_source", "CALIBRATED_HARDWARE_TRUTH"
            ),
            "force_capacity.value_source",
        ),
        (
            lambda value: value["force_capacity"].__setitem__(
                "role", "BINARY_GRASP_PASS_GATE"
            ),
            "force_capacity.role",
        ),
        (
            lambda value: value["kinematics"].__setitem__(
                "object_contact_normal_feasibility",
                PAD_SURFACE_NORMAL_POLICY,
            ),
            "kinematics.object_contact_normal_feasibility",
        ),
        (
            lambda value: value["kinematics"].__setitem__(
                "pad_surface_normal_feasibility",
                OBJECT_CONTACT_NORMAL_POLICY,
            ),
            "kinematics.pad_surface_normal_feasibility",
        ),
    ),
)
def test_wrong_semantic_or_force_source_is_rejected(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    value = _source_mapping()
    mutator(value)
    with pytest.raises(HandContractError, match=message):
        _load(_write_contract(tmp_path, value))


@pytest.mark.parametrize("truth_field", (
    "ground_truth_object_pose_allowed",
    "collision_name_allowed",
    "physx_contact_point_allowed",
    "physx_contact_normal_allowed",
    "semantic_contact_role_allowed",
))
def test_any_online_truth_pollution_is_rejected(
    tmp_path: Path, truth_field: str
) -> None:
    value = _source_mapping()
    value["online_truth_firewall"][truth_field] = True
    with pytest.raises(HandContractError, match=truth_field):
        _load(_write_contract(tmp_path, value))


@pytest.mark.parametrize(
    ("section", "field"),
    (
        (None, "hardware_authorized"),
        ("force_capacity", "physical_tactile_or_load_cell_calibration_complete"),
        ("force_capacity", "simulator_drive_and_jacobian_readback_complete"),
        ("force_capacity", "dynamic_use_allowed"),
    ),
)
def test_static_contract_cannot_be_upgraded_to_hardware_or_dynamic_truth(
    tmp_path: Path, section: str | None, field: str
) -> None:
    value = _source_mapping()
    target = value if section is None else value[section]
    target[field] = True
    with pytest.raises(HandContractError, match=field):
        _load(_write_contract(tmp_path, value))


def test_pad_finger_link_and_coordinate_frame_are_one_to_one(
    tmp_path: Path,
) -> None:
    for field, wrong_value in (
        ("finger_name", "f1"),
        ("link_name", "f1Link3"),
    ):
        value = _source_mapping()
        value["pads"]["finger_2_pad"][field] = wrong_value
        with pytest.raises(HandContractError, match=field):
            _load(_write_contract(tmp_path, value))

    value = _source_mapping()
    value["pads"]["finger_2_pad"]["footprint"]["coordinate_frame"] = "f1Link3"
    with pytest.raises(HandContractError, match="coordinate_frame"):
        _load(_write_contract(tmp_path, value))


def test_old_face_or_alignment_rule_cannot_reenter_verified_contract(
    tmp_path: Path,
) -> None:
    value = _source_mapping()
    value["pads"]["finger_1_pad"]["footprint"]["pad_contact_face_ids"] = [1, 2]
    with pytest.raises(HandContractError, match="extra=.*pad_contact_face_ids"):
        _load(_write_contract(tmp_path, value))

    value = _source_mapping()
    value["pad_semantics"]["old_pad_contact_face_ids_used"] = True
    with pytest.raises(HandContractError, match="old_pad_contact_face_ids_used"):
        _load(_write_contract(tmp_path, value))

    value = _source_mapping()
    value["pad_semantics"]["old_closure_alignment_0p90_used"] = True
    with pytest.raises(HandContractError, match="old_closure_alignment_0p90_used"):
        _load(_write_contract(tmp_path, value))


def test_wrong_unit_and_non_null_invented_pad_normal_are_rejected(
    tmp_path: Path,
) -> None:
    value = _source_mapping()
    value["pads"]["finger_3_pad"]["footprint"]["unit"] = "mm"
    with pytest.raises(HandContractError, match="unit"):
        _load(_write_contract(tmp_path, value))

    value = _source_mapping()
    value["pads"]["finger_3_pad"]["contact_normal_pad"] = [1.0, 0.0, 0.0]
    with pytest.raises(HandContractError, match="must remain null"):
        _load(_write_contract(tmp_path, value))


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    path = tmp_path / "duplicate.yaml"
    path.write_text(text + "\nmethod: CARTS-Grasp\n", encoding="utf-8")
    with pytest.raises(HandContractError, match="duplicate YAML key"):
        _load(path)


def test_boolean_strings_do_not_satisfy_fail_closed_flags(tmp_path: Path) -> None:
    value = copy.deepcopy(_source_mapping())
    value["online_truth_firewall"]["ground_truth_object_pose_allowed"] = "false"
    with pytest.raises(HandContractError, match="must be exactly False"):
        _load(_write_contract(tmp_path, value))


def test_repository_relative_contract_is_independent_of_process_cwd(
    tmp_path: Path,
) -> None:
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        loaded = load_carts_hand_contract(
            "src/kcg_connector/config/carts_hand_contact_v1.yaml",
            repository_root=REPOSITORY,
        )
    finally:
        os.chdir(previous)
    assert loaded.urdf.sha256 == _sha256(loaded.urdf.absolute_path)
