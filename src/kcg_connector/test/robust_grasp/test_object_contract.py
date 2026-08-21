from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import yaml

import kcg_connector.grasp.robust.object_contract as object_contract_module
from kcg_connector.grasp.robust.object_contract import (
    ObjectContractError,
    load_object_contract,
    mass_distribution_rms_radius,
)
from kcg_connector.grasp.robust.object_model import CARTS_VISUAL_SUBTREE_NPZ
from kcg_connector.grasp.robust.surface_orientation import (
    SurfaceBoundaryRole,
    audit_surface_orientation,
)


REPOSITORY = Path(__file__).resolve().parents[4]
CONTRACT = REPOSITORY / "src/kcg_connector/config/carts_grasp_objects_v1.yaml"
CURRENT_OBJECT = "current_d38999_26kj61sn_public_spec"
TRANSFER_OBJECT = "te_deutsch_d38999_26fj35pn_step"


def test_current_real_object_contract_is_hash_bound_and_not_dynamic_pass() -> None:
    loaded = load_object_contract(
        CONTRACT,
        object_id=CURRENT_OBJECT,
        repository_root=REPOSITORY,
    )
    assert loaded.model.provenance.source_format == CARTS_VISUAL_SUBTREE_NPZ
    assert loaded.model.provenance.source_sha256 == (
        "ff3dea949aa5c2f320bd4c2907d78fa86a5930cbbbdf3739f50d7f4a1848201e"
    )
    assert len(loaded.model.mesh.vertices_m) == 88078
    assert len(loaded.model.mesh.faces) == 145588
    assert loaded.model.mass_kg == pytest.approx(0.31)
    assert loaded.characteristic_radius_m > 0.0
    assert np.array_equal(loaded.task_frame_rotation_object, np.eye(3))
    assert np.array_equal(
        loaded.nominal_validation_gravity_direction_object,
        (0.0, 0.0, -1.0),
    )
    assert loaded.task_frame_source == "PUBLIC_SPEC_MODEL_CAD_DATUM"
    assert loaded.contact_material_uncertainty.friction_coefficient_interval == (
        0.45,
        0.55,
    )
    assert not loaded.contact_material_uncertainty.probability_distribution_claimed
    assert not loaded.contact_material_uncertainty.vendor_friction_claimed
    assert loaded.verified_source_sha256["contact_material_source"] == (
        "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e"
    )
    assert (
        loaded.model.provenance.source_sha256
        == loaded.verified_source_sha256["planning_geometry"]
    )
    orientation = loaded.orientation_certificate
    assert orientation.role is (
        SurfaceBoundaryRole.SOURCE_INDEXED_CLOSED_COMPONENT_SOUP
    )
    assert orientation.source_vertex_count == 88078
    assert orientation.source_face_count == 145588
    assert orientation.source_edge_count == 218382
    assert orientation.component_count == 7642
    assert orientation.source_same_direction_edge_count == 0
    assert sum(
        record.positive_volume_winding_flip_count
        for record in orientation.components
    ) == 0
    assert orientation.source_indexed_mesh_sha256 == (
        "562b6275e4b3d10d90f4991145c078d1bcc45079dcc5a857a8284d80deeccb24"
    )
    assert orientation.canonical_sha256 == (
        "8dff5ff370ab000c342719aea558724abe9e68da3ad564a2b86c487b7f1fc858"
    )
    assert not orientation.formal_outward_eligible
    with pytest.raises(FrozenInstanceError):
        orientation.formal_outward_eligible = True  # type: ignore[misc]
    assert loaded.task_frame_rotation_object.flags.writeable is False
    assert loaded.nominal_validation_gravity_direction_object.flags.writeable is False
    assert set(loaded.verified_source_sha256) == {
        "planning_geometry",
        "contact_material_source",
        "planning_geometry_manifest",
        "planning_geometry_source_stage",
        "physical_source_contract",
    }
    assert not loaded.uncertainty_calibrated
    assert not loaded.dynamic_eligible
    assert loaded.dynamic_ineligibility_reason.startswith("PENDING_")


def test_transfer_object_uses_the_same_hash_bound_material_interval_without_scores() -> None:
    loaded = load_object_contract(
        CONTRACT,
        object_id=TRANSFER_OBJECT,
        repository_root=REPOSITORY,
    )
    assert loaded.contact_material_uncertainty.friction_coefficient_interval == (
        0.45,
        0.55,
    )
    assert loaded.contact_material_uncertainty.source_class == (
        "SHARED_STUDY_SIMULATION_ASSUMPTION_FROM_DEVELOPMENT_MATERIAL_ROLE"
    )
    assert not loaded.contact_material_uncertainty.vendor_friction_claimed
    assert (
        loaded.model.provenance.source_sha256
        == loaded.verified_source_sha256["planning_geometry"]
    )
    orientation = loaded.orientation_certificate
    assert orientation.role is SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY
    assert orientation.source_vertex_count == 343520
    assert orientation.source_face_count == 687036
    assert orientation.source_edge_count == 1030554
    assert orientation.component_count == 1
    assert orientation.source_same_direction_edge_count == 0
    assert orientation.components[0].positive_volume_winding_flip_count == 0
    assert orientation.source_indexed_mesh_sha256 == (
        "36fd3849e8ff0856b0598aae672376812edf80cb7cddbba822038d80f6d3814b"
    )
    assert orientation.canonical_sha256 == (
        "f221241afd04e1ecb5c73c8bff437fdcb23856b3fd6a483a3fad1ccd714a6b63"
    )
    assert not orientation.formal_outward_eligible
    assert not loaded.dynamic_eligible


def _write_modified_contract(tmp_path: Path, mutate) -> Path:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    mutate(document)
    output = tmp_path / "objects.yaml"
    output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return output


def test_geometry_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    def mutate(document) -> None:
        document["objects"][CURRENT_OBJECT]["planning_geometry"]["sha256"] = "0" * 64

    modified = _write_modified_contract(tmp_path, mutate)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_object_contract(
            modified,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )


def test_geometry_type_role_mapping_and_bad_topology_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_roles = MappingProxyType(
        {
            "CARTS_GRASP_VISUAL_SUBTREE_NPZ_V1": (
                SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY
            ),
            "BINARY_STL_TESSELLATION_FROM_ORIGINAL_STEP": (
                SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY
            ),
        }
    )
    monkeypatch.setattr(
        object_contract_module,
        "_ORIENTATION_ROLE_BY_GEOMETRY_FORMAT",
        wrong_roles,
    )
    with pytest.raises(
        ObjectContractError,
        match="BOUNDARY_ROLE_COMPONENT_COUNT_MISMATCH",
    ):
        load_object_contract(
            CONTRACT,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )

    monkeypatch.undo()
    real_audit = object_contract_module.audit_surface_orientation
    open_vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    open_faces = np.asarray(((0, 1, 2),), dtype=np.int64)

    def audit_open_topology(
        _vertices: np.ndarray,
        _faces: np.ndarray,
        *,
        role: SurfaceBoundaryRole,
    ):
        return real_audit(open_vertices, open_faces, role=role)

    monkeypatch.setattr(
        object_contract_module,
        "audit_surface_orientation",
        audit_open_topology,
    )
    with pytest.raises(ObjectContractError, match="OPEN_SOURCE_INDEX_TOPOLOGY"):
        load_object_contract(
            CONTRACT,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )


def test_positive_volume_winding_repair_remains_evidence_not_outward_truth() -> None:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    positive_faces = np.asarray(
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        dtype=np.int64,
    )
    source_faces = np.array(positive_faces, copy=True)
    source_faces[2, 1], source_faces[2, 2] = (
        source_faces[2, 2],
        source_faces[2, 1],
    )
    certificate = audit_surface_orientation(
        vertices,
        source_faces,
        role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
    )
    repaired = np.array(source_faces, copy=True)
    flipped = np.flatnonzero(
        np.asarray(
            certificate.positive_volume_winding_sign_by_source_face,
            dtype=np.int8,
        )
        == -1
    )
    repaired[flipped, 1], repaired[flipped, 2] = (
        repaired[flipped, 2].copy(),
        repaired[flipped, 1].copy(),
    )
    assert np.array_equal(repaired, positive_faces)
    assert not certificate.formal_outward_eligible


def test_pending_calibration_cannot_be_marked_dynamic_eligible(tmp_path: Path) -> None:
    def mutate(document) -> None:
        document["objects"][CURRENT_OBJECT]["dynamic_eligibility"]["allowed"] = True
        document["objects"][CURRENT_OBJECT]["dynamic_eligibility"]["reason"] = "READY"

    modified = _write_modified_contract(tmp_path, mutate)
    with pytest.raises(ValueError, match="pending contracts"):
        load_object_contract(
            modified,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )


def test_task_frame_must_be_proper_and_share_the_assembly_axis(
    tmp_path: Path,
) -> None:
    def reflect(document) -> None:
        document["objects"][CURRENT_OBJECT]["frames"][
            "task_frame_rotation_object"
        ][0][0] = -1.0

    reflected = _write_modified_contract(tmp_path, reflect)
    with pytest.raises(ValueError, match="proper orthonormal"):
        load_object_contract(
            reflected,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )

    def rotate_third_axis(document) -> None:
        document["objects"][CURRENT_OBJECT]["frames"][
            "task_frame_rotation_object"
        ] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]

    wrong_axis = _write_modified_contract(tmp_path, rotate_third_axis)
    with pytest.raises(ValueError, match="third axis"):
        load_object_contract(
            wrong_axis,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )


def test_nonphysical_principal_inertia_is_rejected(tmp_path: Path) -> None:
    def mutate(document) -> None:
        document["objects"][CURRENT_OBJECT]["physical_properties"][
            "planning_rigid_composition"
        ]["inertia_kg_m2"] = [
            [3.0e-4, 0.0, 0.0],
            [0.0, 1.0e-4, 0.0],
            [0.0, 0.0, 1.0e-4],
        ]

    modified = _write_modified_contract(tmp_path, mutate)
    with pytest.raises(ValueError, match="triangle inequality"):
        load_object_contract(
            modified,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )


def test_material_interval_is_hash_bound_and_cannot_claim_probability_or_vendor_truth(
    tmp_path: Path,
) -> None:
    def reverse_interval(document) -> None:
        document["objects"][CURRENT_OBJECT]["contact_material_uncertainty"][
            "friction_coefficient"
        ] = [0.55, 0.45]

    reversed_contract = _write_modified_contract(tmp_path, reverse_interval)
    with pytest.raises(ObjectContractError, match="nonnegative ordered interval"):
        load_object_contract(
            reversed_contract,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )

    def claim_probability(document) -> None:
        document["objects"][CURRENT_OBJECT]["contact_material_uncertainty"][
            "probability_distribution_claimed"
        ] = True

    probability_contract = _write_modified_contract(tmp_path, claim_probability)
    with pytest.raises(ObjectContractError, match="must be exactly False"):
        load_object_contract(
            probability_contract,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )

    def claim_vendor_truth(document) -> None:
        document["objects"][CURRENT_OBJECT]["contact_material_uncertainty"][
            "vendor_friction_claimed"
        ] = True

    vendor_contract = _write_modified_contract(tmp_path, claim_vendor_truth)
    with pytest.raises(ObjectContractError, match="must be exactly False"):
        load_object_contract(
            vendor_contract,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )


def test_material_source_hash_and_yaml_duplicate_keys_fail_closed(
    tmp_path: Path,
) -> None:
    def corrupt_source_hash(document) -> None:
        document["objects"][CURRENT_OBJECT]["contact_material_uncertainty"][
            "source_sha256"
        ] = "0" * 64

    corrupted = _write_modified_contract(tmp_path, corrupt_source_hash)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_object_contract(
            corrupted,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        CONTRACT.read_text(encoding="utf-8")
        + "\nschema_version: carts_grasp_objects_v1\n",
        encoding="utf-8",
    )
    with pytest.raises(ObjectContractError, match="duplicate YAML key"):
        load_object_contract(
            duplicate,
            object_id=CURRENT_OBJECT,
            repository_root=REPOSITORY,
        )


def test_mass_distribution_rms_radius_is_exact_and_rigid_transform_invariant() -> None:
    loaded = load_object_contract(
        CONTRACT,
        object_id=CURRENT_OBJECT,
        repository_root=REPOSITORY,
    )
    angle = 0.47
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = (0.7, -0.2, 0.4)
    transformed = loaded.model.transformed(transform)
    expected = np.sqrt(
        np.trace(loaded.model.inertia_kg_m2)
        / (2.0 * loaded.model.mass_kg)
    )
    assert loaded.characteristic_radius_m == pytest.approx(expected)
    assert mass_distribution_rms_radius(transformed) == pytest.approx(
        loaded.characteristic_radius_m,
        rel=2.0e-14,
    )
