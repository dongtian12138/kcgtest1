#!/usr/bin/env python3
"""Bind the accepted nail-free compound convex meshes into a runtime URDF."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


_LINKS = ("f1Link3", "f2Link2", "f3Link3")
_REGIONS = ("global", "task_grip_surface", "table_facing_at_qp60")
_MANIFEST = Path(
    "artifacts/carts_v2/opposition60_isaac/"
    "research_collision_asset_residual_vertex64_20260826_run01/"
    "RESIDUAL_REPAIRED_COLLISION_ASSET_MANIFEST.json"
)
_TEMPLATE = Path(
    "artifacts/carts_v2/nailfree_height_projected/hand_model_audit/"
    "sdf_runtime_preparation/handarm_connector_no_nail_sdf_audit.urdf"
)
_OUTPUT = Path(
    "artifacts/carts_v2/opposition60_isaac/"
    "runtime_handarm_connector_no_nail_residual_vertex64_20260826_run03"
)
_RUNTIME_ROBOT_NAME = "handarm_connector_no_nail"
_LOCAL_HAND_ROBOT_NAME = "hand_connector_no_nail_local"
_LOCAL_HAND_LINKS = (
    "handbase_link",
    "f1Link1", "f1Link2", "f1Link3",
    "f2Link1", "f2Link2",
    "f3Link1", "f3Link2", "f3Link3",
    "grasp_tcp",
)
_LOCAL_HAND_REVOLUTE_JOINTS = (
    "f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3",
)
_LOCAL_HAND_MIMICS = {
    "f1j3": "f1j2", "f2j2": "f2j1", "f3j1": "f1j1", "f3j3": "f3j2",
}
_LOCAL_HAND_FIXED_JOINT = "handbase_to_grasp_tcp"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_manifest(root: Path, path: Path) -> tuple[dict, dict, dict]:
    manifest = _load_json(path)
    _require(
        manifest.get("schema_version")
        == "carts_physx_vhacd_residual_repaired_nailfree_v1",
        "collision manifest schema changed",
    )
    _require(
        manifest.get("status") == "STATIC_GEOMETRY_ASSET_CANDIDATE"
        and manifest.get("static_geometry_asset_candidate") is True
        and manifest.get("runtime_binding_accepted") is False
        and manifest.get("formal_dynamic_pass") is False
        and manifest.get("hardware_authorized") is False,
        "collision manifest is not the accepted simulation-only static candidate",
    )
    executed = manifest.get("executed_source") or {}
    executed_path = _resolve(root, executed.get("script", ""))
    _require(
        executed_path.is_file() and _sha256(executed_path) == executed.get("sha256"),
        "collision manifest executed-source binding changed",
    )
    baseline_path = _resolve(root, manifest["source_baseline_manifest"])
    _require(
        baseline_path.is_file()
        and _sha256(baseline_path) == manifest["source_baseline_manifest_sha256"],
        "collision manifest baseline binding changed",
    )
    baseline = _load_json(baseline_path)
    audit_path = _resolve(root, baseline["source_hand_audit"])
    _require(
        audit_path.is_file()
        and _sha256(audit_path) == baseline["source_hand_audit_sha256"],
        "nail-free hand audit binding changed",
    )
    audit = _load_json(audit_path)
    _require(
        audit.get("hand_variant") == "CONNECTOR_GRASP_NO_NAIL"
        and audit.get("hardware_authorized") is False,
        "nail-free hand audit identity changed",
    )
    return manifest, baseline, audit


def _verify_link(root: Path, row: dict) -> list[dict]:
    link = row.get("link")
    hulls, files = row.get("hulls", []), row.get("hull_files", [])
    _require(
        row.get("hull_count") == 64
        and len(hulls) == 64
        and len(files) == 64
        and row.get("hull_validation_pass") is True
        and row.get("static_geometry_asset_candidate") is True,
        f"{link}: compound hull acceptance changed",
    )
    removed = row.get("removed_nail_exclusive_audit") or {}
    _require(
        removed.get("pass") is True
        and removed.get("occupied_exclusive_sample_count") == 0,
        f"{link}: removed-nail zero-occupancy gate changed",
    )
    gates = row.get("registered_p95_gates") or {}
    _require(
        set(gates) == set(_REGIONS)
        and all(
            gates[name].get("pass") is True
            and gates[name].get("absolute_limit_m") == 0.002
            and gates[name].get("degradation_limit_m") == 0.00025
            for name in _REGIONS
        ),
        f"{link}: registered p95 gate changed",
    )
    verified = []
    for index, (record, binding) in enumerate(zip(hulls, files)):
        path = _resolve(root, binding["path"])
        _require(
            record.get("index") == index
            and record.get("pass") is True
            and record.get("convex") is True
            and record.get("watertight") is True
            and record.get("finite") is True
            and record.get("vertex_limit_pass") is True
            and int(record.get("vertex_count", 65)) <= 64
            and float(record.get("volume_m3", 0.0)) > 0.0,
            f"{link}: hull {index} validation changed",
        )
        _require(
            path.is_file() and _sha256(path) == binding["sha256"],
            f"{link}: hull {index} hash changed",
        )
        verified.append({"path": path, "sha256": binding["sha256"]})
    _require(len({item["path"] for item in verified}) == 64, f"{link}: duplicate hull")
    return verified


def _audit_rows(audit: dict) -> dict[str, dict]:
    return {Path(row["source_path"]).stem: row for row in audit["links"]}


def _verify_inertial(link: ET.Element, audit_row: dict) -> dict:
    inertial = link.find("inertial")
    _require(inertial is not None, f"{link.get('name')}: inertial missing")
    mass = float(inertial.find("mass").get("value"))
    origin = [float(value) for value in inertial.find("origin").get("xyz").split()]
    inertia = inertial.find("inertia").attrib
    estimate = audit_row["mass_properties"]
    expected_matrix = estimate["new_inertia_at_com_kg_m2"]
    expected = {
        "ixx": expected_matrix[0][0], "ixy": expected_matrix[0][1],
        "ixz": expected_matrix[0][2], "iyy": expected_matrix[1][1],
        "iyz": expected_matrix[1][2], "izz": expected_matrix[2][2],
    }
    _require(abs(mass - estimate["new_mass_kg"]) <= 1e-15, "mass changed")
    _require(
        all(abs(a - b) <= 1e-15 for a, b in zip(origin, estimate["new_com_m"])),
        "center of mass changed",
    )
    _require(
        all(abs(float(inertia[key]) - value) <= 1e-15 for key, value in expected.items()),
        "inertia changed",
    )
    return {"mass_kg": mass, "center_of_mass_m": origin, "inertia_kg_m2": expected}


def _replace_terminal_collisions(
    root: Path, robot: ET.Element, manifest: dict, audit: dict
) -> dict:
    links = {link.get("name"): link for link in robot.findall("link")}
    rows = {row["link"]: row for row in manifest["links"]}
    _require(set(rows) == set(_LINKS), "terminal-link set changed")
    audit_by_link = _audit_rows(audit)
    result = {}
    for name in _LINKS:
        _require(name in links and name in audit_by_link, f"{name}: link audit missing")
        link = links[name]
        visuals = link.findall("./visual/geometry/mesh")
        _require(len(visuals) == 1, f"{name}: expected one nail-free visual")
        visual_path = _resolve(root, visuals[0].get("filename"))
        _require(
            visual_path.is_file()
            and _sha256(visual_path) == rows[name]["source_mesh_sha256"]
            and _sha256(visual_path) == audit_by_link[name]["visual_output_sha256"],
            f"{name}: visual no longer matches the registered nail-free source",
        )
        inertial = _verify_inertial(link, audit_by_link[name])
        replaced = [mesh.get("filename") for mesh in link.findall("./collision/geometry/mesh")]
        for collision in link.findall("collision"):
            link.remove(collision)
        hulls = _verify_link(root, rows[name])
        for index, hull in enumerate(hulls):
            collision = ET.SubElement(
                link, "collision", {"name": f"{name}_compound_hull_{index:02d}"}
            )
            ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "mesh", {"filename": str(hull["path"])})
        result[name] = {
            "visual_mesh": str(visual_path),
            "visual_sha256": _sha256(visual_path),
            "inertial": inertial,
            "replaced_collision_meshes": replaced,
            "collision_count": len(hulls),
            "collision_hulls": [
                {"index": index, "path": str(item["path"]), "sha256": item["sha256"]}
                for index, item in enumerate(hulls)
            ],
        }
    return result


def _extract_local_hand_tree(robot: ET.Element) -> tuple[ET.ElementTree, dict]:
    local = deepcopy(robot)
    local.set("name", _LOCAL_HAND_ROBOT_NAME)
    keep_links = set(_LOCAL_HAND_LINKS)
    keep_joints = {*_LOCAL_HAND_REVOLUTE_JOINTS, _LOCAL_HAND_FIXED_JOINT}
    for link in local.findall("link"):
        if link.get("name") not in keep_links:
            local.remove(link)
    for joint in local.findall("joint"):
        if joint.get("name") not in keep_joints:
            local.remove(joint)

    links = {link.get("name"): link for link in local.findall("link")}
    joints = {joint.get("name"): joint for joint in local.findall("joint")}
    _require(set(links) == keep_links, "local-hand link inventory changed")
    _require(set(joints) == keep_joints, "local-hand joint inventory changed")
    revolute = {name for name, joint in joints.items() if joint.get("type") == "revolute"}
    _require(revolute == set(_LOCAL_HAND_REVOLUTE_JOINTS), "local-hand revolute joints changed")
    mimic = {
        name: joint.find("mimic").get("joint")
        for name, joint in joints.items()
        if joint.find("mimic") is not None
    }
    _require(mimic == _LOCAL_HAND_MIMICS, "local-hand mimic relationships changed")
    endpoints = {
        name: (joint.find("parent").get("link"), joint.find("child").get("link"))
        for name, joint in joints.items()
    }
    _require(
        all(parent in keep_links and child in keep_links for parent, child in endpoints.values()),
        "local-hand joint references a removed link",
    )
    _require(
        endpoints[_LOCAL_HAND_FIXED_JOINT] == ("handbase_link", "grasp_tcp"),
        "local-hand grasp_tcp fixed joint changed",
    )
    children = {child for _, child in endpoints.values()}
    _require(keep_links - children == {"handbase_link"}, "handbase_link is not the sole root")
    terminal_collisions = sum(len(links[name].findall("collision")) for name in _LINKS)
    total_collisions = sum(len(link.findall("collision")) for link in links.values())
    materials = [material.get("name") for material in local.findall("material")]
    _require(terminal_collisions == 192, "local-hand terminal compound collision count changed")
    _require(total_collisions == 198, "local-hand complete collision count changed")
    return ET.ElementTree(local), {
        "robot_name": _LOCAL_HAND_ROBOT_NAME,
        "link_names": list(_LOCAL_HAND_LINKS),
        "link_count": len(links),
        "joint_names": list(_LOCAL_HAND_REVOLUTE_JOINTS) + [_LOCAL_HAND_FIXED_JOINT],
        "joint_count": len(joints),
        "revolute_joint_count": len(revolute),
        "mimic_joint_count": len(mimic),
        "mimic_relationships": mimic,
        "material_names": materials,
        "material_count": len(materials),
        "terminal_compound_collision_count": terminal_collisions,
        "total_collision_count": total_collisions,
        "root_link": "handbase_link",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--collision-manifest", type=Path, default=_MANIFEST)
    parser.add_argument("--template-urdf", type=Path, default=_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT)
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve()
    source = _resolve(root, arguments.collision_manifest)
    template = _resolve(root, arguments.template_urdf)
    output = _resolve(root, arguments.output_dir)
    _require(source.is_file() and template.is_file(), "bound input is missing")
    _require(not output.exists(), f"refusing to overwrite one-shot output: {output}")
    manifest, baseline, audit = _verify_manifest(root, source)
    tree = ET.parse(template)
    robot = tree.getroot()
    _require(robot.tag == "robot", "template root is not a URDF robot")
    previous_name = robot.get("name")
    robot.set("name", _RUNTIME_ROBOT_NAME)
    link_bindings = _replace_terminal_collisions(root, robot, manifest, audit)
    robot.insert(0, ET.Comment(
        f" generated from collision manifest sha256={_sha256(source)}; runtime gates not run "
    ))
    ET.indent(tree, space="  ")
    output.mkdir(parents=True, exist_ok=False)
    urdf_path = output / "handarm_connector_no_nail.urdf"
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    local_tree, local_hand = _extract_local_hand_tree(robot)
    ET.indent(local_tree, space="  ")
    local_hand_path = output / "hand_connector_no_nail_local.urdf"
    local_tree.write(local_hand_path, encoding="utf-8", xml_declaration=True)
    local_hand.update({
        "path": str(local_hand_path),
        "sha256": _sha256(local_hand_path),
        "world_from_handbase": None,
        "world_from_handbase_required_source": "RUNTIME_CANDIDATE",
        "world_from_handbase_hardcoded": False,
    })
    generator = Path(__file__).resolve()
    binding = {
        "schema_version": "carts_opposition60_runtime_urdf_binding_v1",
        "status": "RUNTIME_URDF_GENERATED_NOT_IMPORTED",
        "evidence_level": "STATIC_URDF_BINDING_ONLY",
        "runtime_binding_accepted": False,
        "runtime_required_gates": [
            "ISAAC_IMPORT", "INITIAL_PENETRATION", "OPPOSITION60_REPLAY", "PHYSX_HEALTH"
        ],
        "formal_dynamic_pass": False,
        "hardware_authorized": False,
        "generator": str(generator.relative_to(root)),
        "generator_sha256": _sha256(generator),
        "collision_manifest": str(source),
        "collision_manifest_sha256": _sha256(source),
        "source_baseline_manifest": manifest["source_baseline_manifest"],
        "source_baseline_manifest_sha256": manifest["source_baseline_manifest_sha256"],
        "nailfree_hand_audit": baseline["source_hand_audit"],
        "nailfree_hand_audit_sha256": baseline["source_hand_audit_sha256"],
        "template_urdf": str(template),
        "template_urdf_sha256": _sha256(template),
        "template_robot_name": previous_name,
        "runtime_robot_name": _RUNTIME_ROBOT_NAME,
        "runtime_urdf": str(urdf_path),
        "runtime_urdf_sha256": _sha256(urdf_path),
        "local_hand_urdf": local_hand,
        "terminal_links": link_bindings,
    }
    binding_path = output / "RUNTIME_URDF_BINDING.json"
    binding_path.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "runtime_urdf": str(urdf_path),
        "runtime_urdf_sha256": binding["runtime_urdf_sha256"],
        "local_hand_urdf": str(local_hand_path),
        "local_hand_urdf_sha256": local_hand["sha256"],
        "local_hand_counts": {
            "links": local_hand["link_count"],
            "joints": local_hand["joint_count"],
            "revolute": local_hand["revolute_joint_count"],
            "mimic": local_hand["mimic_joint_count"],
            "terminal_compound_collisions": local_hand["terminal_compound_collision_count"],
            "all_collisions": local_hand["total_collision_count"],
        },
        "binding": str(binding_path),
        "collision_counts": {name: row["collision_count"] for name, row in link_bindings.items()},
        "runtime_binding_accepted": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
