"""Pure-CPU gates for the keyed-v2 formal tabletop profile."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_tabletop_pick import (
    D38999_PICK_SCHEMA_VERSION_KEYED_V2,
    D38999_PICK_SCHEMA_VERSION_KEYED_V2_XYCOMP_CANDIDATE,
    load_d38999_tabletop_pick_config,
    verify_d38999_pick_dependencies,
)
from kcg_connector.d38999_tabletop_scene import (
    D38999_TABLETOP_SCHEMA_VERSION,
    D38999_TABLETOP_SCHEMA_VERSION_KEYED_V2,
    load_d38999_tabletop_scene,
    verify_d38999_tabletop_asset,
)
from kcg_connector.grasp.physical_grasp_config import (
    load_physical_grasp_experiment_config,
)


REPOSITORY = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCENE = PACKAGE_ROOT / "config/d38999_tabletop_scene_v1.yaml"
KEYED_SCENE = (
    PACKAGE_ROOT / "config/d38999_keyed_v2_tabletop_scene_v1.yaml"
)
KEYED_PICK = PACKAGE_ROOT / "config/d38999_keyed_v2_tabletop_pick_v1.yaml"
KEYED_XYCOMP_PICK = (
    PACKAGE_ROOT
    / "config/d38999_keyed_v2_tabletop_pick_xycomp_candidate_v1.yaml"
)
KEYED_PHYSICAL = (
    PACKAGE_ROOT
    / "config/d38999_keyed_v2_tabletop_physical_grasp_v1.yaml"
)
KEYED_XYCOMP_PHYSICAL = (
    PACKAGE_ROOT
    / "config/"
    "d38999_keyed_v2_tabletop_physical_grasp_xycomp_candidate_v1.yaml"
)
PAIR_MODEL_ID = "d38999_shell25j_25_61_n_keyed_physical_pair_v3"


def _document(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write(tmp_path, name, document):
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def test_legacy_scene_contract_and_serialization_stay_compatible():
    scene = load_d38999_tabletop_scene(LEGACY_SCENE)
    serialized = scene.as_dict()
    assert scene.schema_version == D38999_TABLETOP_SCHEMA_VERSION
    assert scene.asset_profile.profile_id == "d38999_shell25j_61_pair_proxy_v1"
    assert scene.asset_profile.body_mass_kg == pytest.approx(0.08)
    assert scene.asset_profile.nut_mass_kg == pytest.approx(0.04)
    assert scene.asset_profile.expected_body_collider_count == 21
    assert scene.asset_profile.expected_nut_collider_count == 24
    assert "asset_profile" not in serialized
    assert "sha256" not in serialized["asset"]
    assert scene.loose_settled_origin_m == pytest.approx(
        (0.520, -0.210, 0.200)
    )


def test_keyed_scene_freezes_face_up_orientations_mass_and_topology():
    scene = load_d38999_tabletop_scene(KEYED_SCENE)
    profile = scene.asset_profile
    assert scene.schema_version == D38999_TABLETOP_SCHEMA_VERSION_KEYED_V2
    assert profile.profile_id == PAIR_MODEL_ID
    assert profile.loose_endpoint_orientation == (
        "MATING_FACE_UP_RX_180_FOR_REAR_DOWN"
    )
    assert profile.fixed_endpoint_orientation == (
        "MATING_FACE_UP_RX_180_FOR_DOWNWARD_INSERTION"
    )
    assert profile.loose_endpoint_rotation_degrees_xyz == (180.0, 0.0, 0.0)
    assert profile.fixed_endpoint_rotation_degrees_xyz == (180.0, 0.0, 0.0)
    assert (profile.body_mass_kg, profile.nut_mass_kg) == pytest.approx(
        (0.23, 0.08)
    )
    assert profile.expected_body_collider_count == 7577
    assert profile.expected_nut_collider_count == 294
    assert "keyed_v3_physical_r11" in scene.asset.local_path
    assert scene.loose_endpoint.initial_bottom_z_m == pytest.approx(0.215)
    assert scene.loose_settled_origin_m == pytest.approx(
        (0.520, -0.210, 0.2305)
    )
    assert (
        scene.fixed_endpoint.receptacle_origin_m[2]
        - scene.fixed_endpoint.receptacle_bottom_offset_m
    ) == pytest.approx(0.240)
    assert "sha256" not in _document(KEYED_SCENE)["asset"]
    assert "sha256" not in scene.as_dict()["asset"]


def test_keyed_scene_source_identity_resolves_without_new_hash_contract():
    scene = load_d38999_tabletop_scene(KEYED_SCENE)
    asset = verify_d38999_tabletop_asset(scene, REPOSITORY)
    assert asset.name == (
        "d38999_shell25j_25_61_n_keyed_physical_v3_r11.usda"
    )
    source = asset.read_text(encoding="utf-8")
    assert f'custom string kcg:pairModelId = "{PAIR_MODEL_ID}"' in source
    assert (
        'custom string kcg:loosePlugModelId = '
        '"d38999_26kj61sn_physical_proxy_v3"'
        in source
    )
    assert (
        'custom string kcg:fixedReceptacleModelId = '
        '"d38999_20kj61pn_physical_proxy_v3"'
        in source
    )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda doc: doc["asset_profile"].update(body_mass_kg=0.08),
        lambda doc: doc["asset_profile"].update(
            expected_body_collider_count=21
        ),
        lambda doc: doc["asset_profile"].update(
            fixed_endpoint_rotation_degrees_xyz=[0.0, 0.0, 0.0]
        ),
        lambda doc: doc["asset_profile"].update(
            profile_id="d38999_shell25j_61_pair_proxy_v1"
        ),
        lambda doc: doc["asset"].update(
            local_path=(
                "artifacts/kcg_connector/isaac/"
                "d38999_shell25j_25_61_n_keyed_public_spec_v2.usda"
            )
        ),
        lambda doc: doc["loose_endpoint"].update(
            initial_origin_m=[0.520, -0.210, 0.215]
        ),
        lambda doc: doc["fixed_endpoint"].update(
            receptacle_origin_m=[0.550, 0.185, 0.240]
        ),
    ),
)
def test_keyed_scene_rejects_profile_identity_or_pose_drift(
    tmp_path, mutator
):
    document = deepcopy(_document(KEYED_SCENE))
    mutator(document)
    path = _write(tmp_path, "invalid_keyed_scene.yaml", document)
    with pytest.raises(ValueError):
        load_d38999_tabletop_scene(path)


def test_keyed_pick_is_bound_to_keyed_scene_and_geometry():
    pick = load_d38999_tabletop_pick_config(KEYED_PICK)
    geometry = pick.geometry_candidate
    assert pick.schema_version == D38999_PICK_SCHEMA_VERSION_KEYED_V2
    assert pick.scene.tabletop_config == (
        "d38999_keyed_v2_tabletop_scene_v1.yaml"
    )
    assert pick.scene.proxy_config == (
        "d38999_keyed_v2_physical_model_contract_v1.yaml"
    )
    assert geometry.loose_settled_origin_m == pytest.approx(
        (0.520, -0.210, 0.2305)
    )
    assert geometry.rear_body_radius_m == pytest.approx(0.02220)
    assert geometry.rear_body_world_z_interval_m == pytest.approx(
        (0.200, 0.21545)
    )
    assert geometry.coupling_nut_world_z_interval_m == pytest.approx(
        (0.200, 0.2325)
    )
    assert geometry.grip_local_z_interval_m == pytest.approx(
        (0.42903, 0.44448)
    )
    assert geometry.proposed_clearance_tcp_position_m == pytest.approx(
        (0.520, -0.210, 0.24448)
    )
    assert geometry.predicted_closure_sweep_table_clearance_m == pytest.approx(
        0.007807
    )
    assert geometry.dynamics_validated is False
    dependencies = verify_d38999_pick_dependencies(
        pick, KEYED_PICK, REPOSITORY
    )
    assert "source_model" in dependencies
    assert "proxy" not in dependencies
    assert dependencies["tabletop"].asset_profile.profile_id == PAIR_MODEL_ID
    assert (
        dependencies["source_model"].document["identity"]["pair_model_id"]
        == PAIR_MODEL_ID
    )


def test_xycomp_pick_is_separate_and_only_offsets_commanded_xy():
    baseline = load_d38999_tabletop_pick_config(KEYED_PICK)
    candidate = load_d38999_tabletop_pick_config(KEYED_XYCOMP_PICK)
    assert baseline.schema_version == D38999_PICK_SCHEMA_VERSION_KEYED_V2
    assert candidate.schema_version == (
        D38999_PICK_SCHEMA_VERSION_KEYED_V2_XYCOMP_CANDIDATE
    )
    assert baseline.motion.grasp_tcp_position_m == pytest.approx(
        (0.520, -0.210, 0.24448)
    )
    assert candidate.motion.grasp_tcp_position_m == pytest.approx(
        (0.520574651334, -0.210265856035, 0.24448)
    )
    assert candidate.motion.grasp_tcp_position_m[2] == pytest.approx(
        baseline.motion.grasp_tcp_position_m[2]
    )
    assert candidate.geometry_candidate.loose_settled_origin_m == pytest.approx(
        baseline.geometry_candidate.loose_settled_origin_m
    )
    dependencies = verify_d38999_pick_dependencies(
        candidate, KEYED_XYCOMP_PICK, REPOSITORY
    )
    assert dependencies["tabletop"].asset_profile.profile_id == PAIR_MODEL_ID


def test_xycomp_physical_profile_binds_only_the_xycomp_pick():
    physical = load_physical_grasp_experiment_config(KEYED_XYCOMP_PHYSICAL)
    document = _document(KEYED_XYCOMP_PHYSICAL)
    assert physical.pick_config == (
        "src/kcg_connector/config/"
        "d38999_keyed_v2_tabletop_pick_xycomp_candidate_v1.yaml"
    )
    assert document["lift"]["maximum_wrist_moment_nm"] == pytest.approx(0.30)
    assert document["lift"]["maximum_wrist_force_n"] == pytest.approx(8.0)
    assert document["randomization"] == _document(KEYED_PHYSICAL)[
        "randomization"
    ]


@pytest.mark.parametrize(
    "mutator",
    (
        lambda doc: doc["scene"].update(
            tabletop_config="d38999_tabletop_scene_v1.yaml"
        ),
        lambda doc: doc["scene"].update(
            proxy_config="d38999_shell25j_proxy_v1.yaml"
        ),
        lambda doc: doc["geometry_candidate"].update(
            loose_settled_origin_m=[0.520, -0.210, 0.200]
        ),
        lambda doc: doc["geometry_candidate"].update(
            rear_body_world_z_interval_m=[0.217, 0.231]
        ),
        lambda doc: doc.update(schema_version="unsupported_profile"),
    ),
)
def test_keyed_pick_rejects_legacy_mix_or_geometry_drift(tmp_path, mutator):
    document = deepcopy(_document(KEYED_PICK))
    mutator(document)
    path = _write(tmp_path, "invalid_keyed_pick.yaml", document)
    with pytest.raises(ValueError):
        load_d38999_tabletop_pick_config(path)


def test_keyed_physical_grasp_binds_new_chain_and_stays_fail_closed():
    physical = load_physical_grasp_experiment_config(KEYED_PHYSICAL)
    document = _document(KEYED_PHYSICAL)
    assert physical.pick_config == (
        "src/kcg_connector/config/d38999_keyed_v2_tabletop_pick_v1.yaml"
    )
    assert document["base"]["tabletop_config"] == (
        "src/kcg_connector/config/d38999_keyed_v2_tabletop_scene_v1.yaml"
    )
    assert document["lift"]["maximum_wrist_force_n"] == pytest.approx(8.0)
    assert document["lift"]["maximum_wrist_moment_nm"] == pytest.approx(0.30)
    nominal = document["randomization"]
    assert nominal["plug_x_offset_m"] == [0.0, 0.0]
    assert nominal["plug_y_offset_m"] == [0.0, 0.0]
    assert nominal["plug_yaw_deg"] == [0.0, 0.0]
    assert nominal["arm_center_error_x_m"] == [0.0, 0.0]
    assert nominal["arm_center_error_y_m"] == [0.0, 0.0]
    assert nominal["finger_start_delay_steps"] == [0]
    assert nominal["plug_mass_scale"] == [1.0, 1.0]
    assert nominal["center_of_mass_offset_m"] == [0.0, 0.0]
    assert nominal["lift_speed_scale"] == [1.0, 1.0]
    assert all(value is False for value in document["boundaries"].values())
    assert physical.post_grasp_stabilization_proxy_enabled is False
