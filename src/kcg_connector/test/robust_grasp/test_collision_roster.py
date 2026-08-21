from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest
import yaml

from kcg_connector.grasp.robust.collision_roster import (
    CollisionRosterError,
    load_authoritative_collision_link_roster,
)


REPOSITORY = Path(__file__).resolve().parents[4]
CONTRACT = REPOSITORY / "src/kcg_connector/config/carts_collision_roster_v1.yaml"


def _document() -> dict:
    value = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "roster.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_real_aggregate_roster_is_exact_17_links_136_pairs_but_not_formal() -> None:
    roster = load_authoritative_collision_link_roster(
        CONTRACT, repository_root=REPOSITORY
    )
    assert len(roster.links) == 17
    assert len(roster.all_self_pairs) == 136
    assert roster.link_names[:8] == tuple(f"iiwa_link_{index}" for index in range(8))
    assert roster.link_names[8:] == (
        "handbase_link",
        "f1Link1",
        "f1Link2",
        "f1Link3",
        "f2Link1",
        "f2Link2",
        "f3Link1",
        "f3Link2",
        "f3Link3",
    )
    assert roster.excluded_noncollision_links == ("world", "iiwa_link_ee", "grasp_tcp")
    assert roster.motion_binding_complete is False
    assert roster.terminal_pad_role_binding_complete is False
    assert roster.solid_boundary_binding_complete is False
    assert roster.formal_collision_roster_eligible is False
    assert roster.audit["formal_collision_roster_eligible"] is False
    assert roster.roster_sha256 == (
        "eb240a2fb81869c6ffe1148db367e01f2b570f8d72f6292a84531a784b2bac83"
    )
    with pytest.raises(TypeError):
        replace(roster, formal_collision_roster_eligible=True)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["links"].pop(),
        lambda value: value["links"].append(
            {
                **value["links"][-1],
                "ordinal": 17,
                "link_name": "grasp_tcp",
            }
        ),
        lambda value: value["links"][0].__setitem__("unit", "mm"),
        lambda value: value["links"][0].__setitem__("ordinal", False),
        lambda value: value["links"][0]["origin_xyz_m"].__setitem__(0, True),
        lambda value: value["links"][0].__setitem__("sha256", "0" * 64),
        lambda value: value["links"].reverse(),
        lambda value: value["excluded_noncollision_links"].remove("grasp_tcp"),
        lambda value: value["motion_binding"].__setitem__("complete", True),
        lambda value: value["motion_binding"].__setitem__(
            "missing_evidence", None
        ),
        lambda value: value["motion_binding"].__setitem__(
            "fixed_arm_during_local_finger_closure_allowed_only_after_arm_pose_binding",
            False,
        ),
        lambda value: value["terminal_pad_role_binding"].__setitem__(
            "reason", "CERTIFIED"
        ),
        lambda value: value["solid_boundary_binding"].__setitem__(
            "reason", "CERTIFIED"
        ),
        lambda value: value["package_roots"].__setitem__("unused", "src"),
        lambda value: value["terminal_pad_role_binding"].__setitem__(
            "nearest_surface_or_tolerance_mapping_allowed", True
        ),
        lambda value: value.__setitem__("formal_collision_roster_eligible", True),
    ),
)
def test_declared_link_unit_order_role_or_hash_drift_fails_closed(
    tmp_path: Path, mutation
) -> None:
    value = copy.deepcopy(_document())
    mutation(value)
    with pytest.raises(CollisionRosterError):
        load_authoritative_collision_link_roster(
            _write(tmp_path, value), repository_root=REPOSITORY
        )


def _copy_required_repository(tmp_path: Path) -> tuple[Path, Path, dict]:
    copied_root = tmp_path / "repository"
    value = _document()
    required_paths = {
        value["aggregate_source"]["path"],
        *(row["path"] for row in value["include_sources"]),
        *(row["repository_path"] for row in value["links"]),
    }
    for relative in required_paths:
        source = REPOSITORY / relative
        destination = copied_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    contract_copy = copied_root / "src/kcg_connector/config/carts_collision_roster_v1.yaml"
    contract_copy.parent.mkdir(parents=True, exist_ok=True)
    contract_copy.write_text(CONTRACT.read_text(encoding="utf-8"), encoding="utf-8")
    return copied_root, contract_copy, value


@pytest.mark.parametrize(
    "mutation",
    ("second_geometry", "second_origin", "unknown_mesh_attribute", "nested_link"),
)
def test_unmodelled_collision_xml_structure_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    copied_root, contract_copy, value = _copy_required_repository(tmp_path)
    include = copied_root / value["include_sources"][0]["path"]
    tree = ET.parse(include)
    root = tree.getroot()
    collision = root.find("link/collision")
    assert collision is not None
    if mutation == "second_geometry":
        geometry = collision.find("geometry")
        assert geometry is not None
        collision.append(copy.deepcopy(geometry))
    elif mutation == "second_origin":
        origin = collision.find("origin")
        assert origin is not None
        collision.append(copy.deepcopy(origin))
    elif mutation == "unknown_mesh_attribute":
        mesh = collision.find("geometry/mesh")
        assert mesh is not None
        mesh.set("unit", "mm")
    else:
        link = root.find("link")
        assert link is not None
        wrapper = ET.SubElement(root, "wrapper")
        wrapper.append(copy.deepcopy(link))
    tree.write(include, encoding="utf-8", xml_declaration=True)
    value["include_sources"][0]["sha256"] = hashlib.sha256(
        include.read_bytes()
    ).hexdigest()
    contract_copy.write_text(
        yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(CollisionRosterError):
        load_authoritative_collision_link_roster(
            contract_copy, repository_root=copied_root
        )


def test_unmodelled_aggregate_include_attribute_fails_closed(
    tmp_path: Path,
) -> None:
    copied_root, contract_copy, value = _copy_required_repository(tmp_path)
    aggregate = copied_root / value["aggregate_source"]["path"]
    tree = ET.parse(aggregate)
    include = list(tree.getroot())[0]
    include.set("optional", "true")
    tree.write(aggregate, encoding="utf-8", xml_declaration=True)
    value["aggregate_source"]["sha256"] = hashlib.sha256(
        aggregate.read_bytes()
    ).hexdigest()
    contract_copy.write_text(
        yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(CollisionRosterError):
        load_authoritative_collision_link_roster(
            contract_copy, repository_root=copied_root
        )


def test_source_hash_and_duplicate_yaml_key_fail_closed(tmp_path: Path) -> None:
    value = _document()
    value["aggregate_source"]["sha256"] = "0" * 64
    with pytest.raises(CollisionRosterError, match="SHA-256 mismatch"):
        load_authoritative_collision_link_roster(
            _write(tmp_path, value), repository_root=REPOSITORY
        )

    duplicate = CONTRACT.read_text(encoding="utf-8") + "\nmethod_id: duplicate\n"
    path = tmp_path / "duplicate.yaml"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(CollisionRosterError, match="duplicate YAML key"):
        load_authoritative_collision_link_roster(path, repository_root=REPOSITORY)


def test_actual_mesh_byte_tamper_is_rejected(tmp_path: Path) -> None:
    copied_root, contract_copy, value = _copy_required_repository(tmp_path)

    mesh = copied_root / value["links"][0]["repository_path"]
    payload = bytearray(mesh.read_bytes())
    payload[-1] ^= 1
    mesh.write_bytes(payload)
    assert hashlib.sha256(payload).hexdigest() != value["links"][0]["sha256"]
    with pytest.raises(CollisionRosterError, match="differs from aggregate XML or mesh bytes"):
        load_authoritative_collision_link_roster(
            contract_copy, repository_root=copied_root
        )


def test_loaded_mesh_bytes_are_an_immutable_snapshot(tmp_path: Path) -> None:
    copied_root, contract_copy, value = _copy_required_repository(tmp_path)
    roster = load_authoritative_collision_link_roster(
        contract_copy, repository_root=copied_root
    )
    snapshot = roster.links[0].content_bytes
    mesh = copied_root / value["links"][0]["repository_path"]
    payload = bytearray(mesh.read_bytes())
    payload[-1] ^= 1
    mesh.write_bytes(payload)
    assert roster.links[0].content_bytes == snapshot
    assert hashlib.sha256(snapshot).hexdigest() == roster.links[0].sha256
    with pytest.raises(CollisionRosterError):
        load_authoritative_collision_link_roster(
            contract_copy, repository_root=copied_root
        )
