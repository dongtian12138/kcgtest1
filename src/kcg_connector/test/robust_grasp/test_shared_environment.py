from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from kcg_connector.grasp.robust.material_boundary import (
    certify_single_embedded_material_boundary,
)
from kcg_connector.grasp.robust.shared_environment import (
    CLAIM_LIMITATIONS,
    EXPECTED_OBJECT_IDS,
    SharedEnvironmentError,
    load_shared_table_fixture_world,
)


REPOSITORY = Path(__file__).resolve().parents[4]
CONFIG = REPOSITORY / "src/kcg_connector/config/carts_shared_table_fixture_world_v1.yaml"


@pytest.fixture(scope="module")
def real_environment():
    return load_shared_table_fixture_world(CONFIG, repository_root=REPOSITORY)


def test_real_sources_bind_one_shared_world_for_both_objects(real_environment) -> None:
    result = real_environment
    assert result.registered_object_ids == EXPECTED_OBJECT_IDS
    assert result.root_frame == "world"
    assert result.robot_base_origin_m == (0.0, 0.0, 0.0)
    assert result.obstacle_count == 2
    assert tuple(row.name for row in result.obstacles) == ("table", "fixture")
    assert tuple(row.role for row in result.obstacles) == ("TABLE", "FIXTURE")
    assert result.table_fixture_world_binding_complete is True
    assert result.claim_limitations == CLAIM_LIMITATIONS
    assert result.audit["selected_values_match_both_sources"] is True


def test_table_and_fixture_have_the_frozen_world_dimensions(real_environment) -> None:
    table, fixture = real_environment.obstacles
    assert table.center_m == (0.55, 0.0, 0.16)
    assert table.size_m == (0.8, 0.9, 0.08)
    assert fixture.center_m == (0.55, 0.185, 0.22)
    assert fixture.size_m == (0.14, 0.14, 0.04)
    assert np.min(table.triangles_world_m[:, :, 2]) == pytest.approx(0.12)
    assert np.max(table.triangles_world_m[:, :, 2]) == pytest.approx(0.20)
    assert np.min(fixture.triangles_world_m[:, :, 2]) == pytest.approx(0.20)
    assert np.max(fixture.triangles_world_m[:, :, 2]) == pytest.approx(0.24)


def test_each_box_is_an_exact_embedded_material_boundary(real_environment) -> None:
    for obstacle in real_environment.obstacles:
        triangles = obstacle.triangles_world_m
        vertices, inverse = np.unique(
            triangles.reshape(-1, 3), axis=0, return_inverse=True
        )
        faces = inverse.reshape(-1, 3)
        certificate = certify_single_embedded_material_boundary(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int64),
        )
        assert certificate.source_vertex_count == 8
        assert certificate.source_face_count == 12
        assert certificate.formal_material_boundary_eligible is True


def test_non_static_inputs_remain_absent_and_certificate_is_immutable(
    real_environment,
) -> None:
    result = real_environment
    assert result.fixed_receptacle_geometry_included is False
    assert result.loose_object_initial_pose_included is False
    assert result.candidate_specific_robot_route_included is False
    assert result.isaac_dynamic_state_included is False
    assert result.hardware_state_included is False
    with pytest.raises(FrozenInstanceError):
        result.table_fixture_world_binding_complete = False  # type: ignore[misc]
    with pytest.raises(ValueError):
        replace(result, candidate_specific_robot_route_included=True)


def test_loading_is_deterministic_and_source_drift_fails_closed(
    real_environment,
    tmp_path: Path,
) -> None:
    repeated = load_shared_table_fixture_world(CONFIG, repository_root=REPOSITORY)
    assert repeated.certificate_sha256 == real_environment.certificate_sha256
    assert tuple(row.geometry_sha256 for row in repeated.obstacles) == tuple(
        row.geometry_sha256 for row in real_environment.obstacles
    )

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["obstacles"]["table"]["size_m"][0] = 0.81
    changed = tmp_path / "changed_environment.yaml"
    changed.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(SharedEnvironmentError) as error:
        load_shared_table_fixture_world(changed, repository_root=REPOSITORY)
    assert error.value.code == "SELECTED_SOURCE_MISMATCH"

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["source_bindings"]["tabletop_scene"]["sha256"] = "0" * 64
    changed_hash = tmp_path / "changed_hash_environment.yaml"
    changed_hash.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(SharedEnvironmentError) as hash_error:
        load_shared_table_fixture_world(changed_hash, repository_root=REPOSITORY)
    assert hash_error.value.code == "SOURCE_SHA256_MISMATCH"
