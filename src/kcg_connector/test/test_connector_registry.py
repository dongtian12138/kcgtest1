from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from kcg_connector.connector_registry import (
    ConnectorGender,
    ConnectorRole,
    REGISTRY_SCHEMA_VERSION,
    load_connector_model_registry,
)


REGISTRY_PATH = (
    Path(__file__).parents[1]
    / "config"
    / "connector_model_registry_v1.yaml"
)
TASK_PATH = Path(__file__).parents[1] / "config" / "connector_task.yaml"
PROFILE_ID = "synthetic_thread_proxy_pair_v1"


def _document():
    with REGISTRY_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write(tmp_path, document):
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _operational_document():
    document = deepcopy(_document())
    document["registry_enabled"] = True
    profile = document["assembly_profiles"][0]
    profile["enabled"] = True
    safety = profile["safety_limits"]
    safety["maximum_insertion_axial_force_n"] = 20.0
    safety["maximum_lateral_force_n"] = 5.0
    safety["maximum_bending_moment_nm"] = 2.0
    safety["maximum_tightening_torque_nm"] = 1.0
    return document


def _assign(document, path, value):
    parent = document
    for item in path[:-1]:
        parent = parent[item]
    parent[path[-1]] = value


def test_shipped_registry_is_synthetic_only_and_disabled():
    registry = load_connector_model_registry(REGISTRY_PATH)
    assert registry.schema_version == REGISTRY_SCHEMA_VERSION
    assert registry.scope == "synthetic_curriculum_only"
    assert registry.enabled is False
    assert {model.model_id for model in registry.models} == {
        "synthetic_plug_v1",
        "synthetic_receptacle_v1",
    }
    assert all(
        model.model_id.startswith("synthetic_") for model in registry.models
    )
    assert all(
        model.provenance == "synthetic_curriculum_only"
        for model in registry.models
    )


def test_shipped_pair_is_reciprocal_male_loose_to_female_fixed():
    registry = load_connector_model_registry(REGISTRY_PATH)
    profile = registry.profile(PROFILE_ID)
    loose = registry.model(profile.loose_model_id)
    fixed = registry.model(profile.fixed_model_id)
    assert loose.deployment_role is ConnectorRole.LOOSE
    assert fixed.deployment_role is ConnectorRole.FIXED
    assert loose.gender is ConnectorGender.MALE
    assert fixed.gender is ConnectorGender.FEMALE
    assert fixed.model_id in loose.compatible_model_ids
    assert loose.model_id in fixed.compatible_model_ids


def test_default_activation_reports_unknown_force_limits_and_kill_switches():
    registry = load_connector_model_registry(REGISTRY_PATH)
    blockers = set(registry.activation_blockers(PROFILE_ID))
    assert {
        "registry_disabled",
        "profile_disabled",
        "safety_maximum_insertion_axial_force_n_missing",
        "safety_maximum_lateral_force_n_missing",
        "safety_maximum_bending_moment_nm_missing",
        "safety_maximum_tightening_torque_nm_missing",
    }.issubset(blockers)
    with pytest.raises(ValueError, match="not enabled"):
        registry.require_enabled_profile(PROFILE_ID)


def test_synthetic_pair_parameters_match_current_task_contract():
    registry = load_connector_model_registry(REGISTRY_PATH)
    profile = registry.profile(PROFILE_ID)
    with TASK_PATH.open(encoding="utf-8") as stream:
        task = yaml.safe_load(stream)
    success = task["success"]
    assert profile.insertion.minimum_engage_depth_m == success["engage_depth"]
    assert profile.insertion.maximum_lateral_error_m == (
        success["lateral_alignment_tolerance"]
    )
    assert profile.insertion.maximum_angular_error_degrees == (
        success["angular_alignment_tolerance_degrees"]
    )
    assert profile.insertion.maximum_key_error_degrees == (
        success["key_alignment_tolerance_degrees"]
    )
    assert profile.fastening.target_angle_degrees == (
        success["target_coupling_angle_degrees"]
    )
    assert profile.fastening.angle_tolerance_degrees == (
        success["coupling_angle_tolerance_degrees"]
    )
    assert profile.fastening.lead_m_per_revolution == (
        success["helical_lead_per_revolution"]
    )
    assert profile.safety_limits.maximum_finger_base_torque_nm == (
        success["maximum_absolute_finger_torque"]
    )


def test_fully_populated_synthetic_profile_can_pass_explicit_gates(tmp_path):
    registry = load_connector_model_registry(
        _write(tmp_path, _operational_document())
    )
    assert registry.require_enabled_profile(PROFILE_ID).enabled is True


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    (
        (("models", 0, "assembly_frame"), None, "assembly_frame_missing"),
        (
            ("models", 1, "rotational_symmetry"),
            None,
            "rotational_symmetry_missing",
        ),
        (("models", 0, "grasp_regions"), [], "grasp_region_missing"),
        (
            (
                "assembly_profiles",
                0,
                "insertion",
                "direction_in_fixed_assembly_frame",
            ),
            None,
            "insertion_direction",
        ),
        (
            (
                "assembly_profiles",
                0,
                "fastening",
                "tightening_sign_about_axis",
            ),
            None,
            "fastening_tightening_sign",
        ),
        (
            (
                "assembly_profiles",
                0,
                "fastening",
                "target_angle_degrees",
            ),
            None,
            "fastening_target_angle",
        ),
        (
            (
                "assembly_profiles",
                0,
                "fastening",
                "lead_m_per_revolution",
            ),
            None,
            "fastening_lead",
        ),
        (
            (
                "assembly_profiles",
                0,
                "safety_limits",
                "maximum_insertion_axial_force_n",
            ),
            None,
            "maximum_insertion_axial_force",
        ),
    ),
)
def test_enabled_profile_rejects_missing_operational_data(
    tmp_path, path, value, expected
):
    document = _operational_document()
    _assign(document, path, value)
    with pytest.raises(ValueError, match=expected):
        load_connector_model_registry(_write(tmp_path, document))


def test_loader_rejects_nonreciprocal_compatibility(tmp_path):
    document = _document()
    document["models"][1]["compatible_model_ids"] = [
        "synthetic_receptacle_v1"
    ]
    with pytest.raises(ValueError, match="does not declare"):
        load_connector_model_registry(_write(tmp_path, document))


def test_loader_rejects_wrong_mating_gender(tmp_path):
    document = _document()
    document["models"][1]["gender"] = "male"
    with pytest.raises(ValueError, match="must be female"):
        load_connector_model_registry(_write(tmp_path, document))


def test_loader_rejects_unmeasured_real_model_entry(tmp_path):
    document = _document()
    document["models"][0]["model_id"] = "real_connector_guess"
    document["models"][1]["compatible_model_ids"] = [
        "real_connector_guess"
    ]
    document["assembly_profiles"][0][
        "loose_model_id"
    ] = "real_connector_guess"
    with pytest.raises(ValueError, match="only synthetic model IDs"):
        load_connector_model_registry(_write(tmp_path, document))


def test_loader_rejects_missing_schema_field_even_while_disabled(tmp_path):
    document = _document()
    del document["assembly_profiles"][0]["fastening"][
        "lead_m_per_revolution"
    ]
    with pytest.raises(ValueError, match="keys differ"):
        load_connector_model_registry(_write(tmp_path, document))


def test_loader_rejects_non_collinear_insertion_and_fastening_axes(tmp_path):
    document = _document()
    document["assembly_profiles"][0]["fastening"][
        "axis_in_fixed_assembly_frame"
    ] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="must be collinear"):
        load_connector_model_registry(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (
            (
                "assembly_profiles",
                0,
                "safety_limits",
                "maximum_finger_base_torque_nm",
            ),
            True,
        ),
        (("models", 0, "assembly_frame", "translation_m"), [0, False, 0]),
    ),
)
def test_loader_rejects_boolean_values_in_numeric_fields(
    tmp_path, path, value
):
    document = _document()
    _assign(document, path, value)
    with pytest.raises(ValueError, match="finite"):
        load_connector_model_registry(_write(tmp_path, document))
