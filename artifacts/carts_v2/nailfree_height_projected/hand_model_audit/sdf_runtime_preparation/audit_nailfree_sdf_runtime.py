#!/usr/bin/env python3
"""Prepare and fail-closed audit the fixed PhysX SDF nail-free f1 collider.

Preparation never starts Isaac. Runtime mode is an explicit f1-only asset
diagnostic and authors SDF only in memory; it never accepts a global binding.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ElementTree

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sdf_audit_geometry import (  # noqa: E402
    find_identity_collision, geometry_audit, read_binary_stl, verify_usd_mass,
)


_AUDIT_SHA256 = "619ab0a85ef41b5a0e3a79dc0bd4e7da550094c73c4d10e738a5bc5a251a98f8"
_XACRO_SHA256 = {
    "handarm.urdf.xacro": "b4ee760ec9d2306e172dc71f697a087f5788c32f89dd3c2e80a2f1e7897af399",
    "hand.xacro": "b7ddf62c62b8127816e5ba8ecefb72c257622952e3aa2211355a6a61f4d1097f",
    "iiwa14.xacro": "0bda9afbdb5204d6606ece542e79590c780b5b0d84f3478ea38a37a640d2297d",
}
_LINKS = {
    "f1Link3": (
        "7a33a6ab46729a2237dd13d99be3bcefb92bb3d4b77bbf9e69d884509cffcdb0",
        "965d327c466bec40b898fc4228f8ca240386bab3e8a79af6b48c798db1a0071a",
    ),
    "f2Link2": (
        "1758619f7ef1369fc3342c7032edee07222f9bdccc187c33830f9fa59bd508b3",
        "3d11ab9797c2ed6e4c622c3ba6c63b2c9fb8258dbf968326828046efc788e893",
    ),
    "f3Link3": (
        "93645443cff113b8c6e5a0280e3270192831d04246233cc45d9745c6e3c7d16e",
        "62d7aa934e7516f83d884adfe6f518e446d3db612d41bd81edfee96ed2f7e27b",
    ),
}
_REMOVED_RANGES = [[11836, 12911], [12912, 13551], [13552, 14191]]
_SDF = {
    "resolution": 256, "subgrid_resolution": 6,
    "bits_per_subgrid_pixel": "BitsPerPixel16",
    "narrow_band_thickness": 0.01, "margin": 0.01,
    "enable_remeshing": False, "triangle_count_reduction_factor": 1.0,
}
_BOUNDARY_M, _MAX_ERROR_M, _COVER_M, _BATCH = 1.0e-5, 1.0e-3, 2.5e-4, 65536
_TASK_MANIFEST = (
    "artifacts/carts_v2/nailfree_height_projected/task_grip_surface_audit/"
    "TASK_GRIP_SURFACE_MANIFEST.json"
)
_TASK_MANIFEST_SHA256 = "5d173d8446a4dc2f7648f8f9cd394c3061da5e83dc381f2e9ac01874dcadce17"
_TASK_F1_NPZ = (
    "artifacts/carts_v2/nailfree_height_projected/task_grip_surface_audit/"
    "f1Link3_TASK_GRIP_SURFACE_local_m.npz"
)
_TASK_F1_NPZ_SHA256 = "200f93aaa4cf4fadcad9c4838717b12d8d14237fd696d56c95c189983dd1b85a"
_MAIN_BODY_IDENTITY = {
    "migration_identity": "F1_MAIN_BODY_NORMAL_CROSSING_V2",
    "component_index": 0, "face_count": 7191, "task_face_count": 518,
    "triangle_digest": "74283647c5e38a00b4602f386d6d84afdcb42b6ccbf3173cd99c318c7172daf6",
    "other_sample_count": 24, "task_sample_count": 12,
    "auxiliary_components": {
        "152": {"face_count": 200,
            "triangle_digest": "60bd86312c210018f1a3035879e7a86fef4950f85130db68b489228d28db09a3"},
        "153": {"face_count": 888,
            "triangle_digest": "75f25117d9c30a34df385c4fb652d5bb1a4ea28698c1f538434b44cf4d1c4280"},
        "154": {"face_count": 888,
            "triangle_digest": "b9a010e0438dee99d6a4e1b59ca43ac3c52b63306f3c61386e699f714a3e364a"},
    },
}


def _root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "AGENTS.md").is_file() and (parent / ".git").exists():
            return parent
    raise RuntimeError("repository root was not found")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def _write_json(path: Path, value: dict) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_inputs(root: Path) -> dict[str, dict]:
    path = root / "artifacts/carts_v2/nailfree_height_projected/hand_model_audit/NAILFREE_HAND_MODEL_AUDIT.json"
    if _sha(path) != _AUDIT_SHA256:
        raise ValueError("upstream nail-free audit hash changed")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("collision_runtime_binding_accepted") is not False:
        raise ValueError("upstream runtime binding is no longer fail-closed")
    rows = {Path(row["source_path"]).stem: row for row in audit["links"]}
    if set(rows) != set(_LINKS):
        raise ValueError("terminal-link identity changed")
    for link, (source_sha, visual_sha) in _LINKS.items():
        row = rows[link]
        source, visual = root / row["source_path"], root / row["visual_output"]
        ranges = [item["source_face_index_range_inclusive"] for item in row["removed_components"]]
        if (
            _sha(source) != source_sha or row["source_sha256"] != source_sha
            or _sha(visual) != visual_sha or row["visual_output_sha256"] != visual_sha
            or row["retained_face_count"] != 11836 or ranges != _REMOVED_RANGES
        ):
            raise ValueError(f"{link}: mesh/removal identity changed")
        source_faces, visual_faces = read_binary_stl(source), read_binary_stl(visual)
        if len(source_faces) != 14192 or len(visual_faces) != 11836:
            raise ValueError(f"{link}: face count changed")
        if not np.array_equal(source_faces[:11836], visual_faces):
            raise ValueError(f"{link}: retained triangle stream changed")
        mass = row["mass_properties"]
        inertia = np.asarray(mass["new_inertia_at_com_kg_m2"])
        eigen = np.linalg.eigvalsh(inertia)
        if (
            mass["new_mass_kg"] <= 0 or mass.get("hardware_calibration_claimed") is not False
            or not np.allclose(inertia, inertia.T, atol=1.0e-12) or np.any(eigen <= 0)
            or eigen[0] + eigen[1] < eigen[2] - 1.0e-12
        ):
            raise ValueError(f"{link}: provisional inertial identity is invalid")
    return rows


def _load_task_surface(root: Path, row: dict) -> dict:
    manifest_path, npz_path = root / _TASK_MANIFEST, root / _TASK_F1_NPZ
    if _sha(manifest_path) != _TASK_MANIFEST_SHA256 or _sha(npz_path) != _TASK_F1_NPZ_SHA256:
        raise ValueError("f1 TASK_GRIP_SURFACE evidence hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    links = [item for item in manifest.get("links", []) if item.get("link_name") == "f1Link3"]
    if len(links) != 1:
        raise ValueError("f1 TASK_GRIP_SURFACE record is not unique")
    link = links[0]
    whitelist = link.get("terminal_semantic_whitelist", {})
    if (manifest.get("schema_version") != "carts_task_grip_surface_v2"
            or manifest.get("semantic") != "TASK_GRIP_SURFACE"
            or manifest.get("hardware_authorized") is not False
            or manifest.get("object_specific_selection_used") is not False
            or link.get("source_mesh_sha256") != row["visual_output_sha256"]
            or link.get("source_face_count") != 11836 or link.get("task_face_count") != 2200
            or link.get("surface_npz") != _TASK_F1_NPZ
            or link.get("surface_npz_sha256") != _TASK_F1_NPZ_SHA256
            or link.get("minimum_normal_motion_m_per_phase") != _BOUNDARY_M
            or whitelist.get("terminal_body_component") != 0
            or whitelist.get("legacy_pad_component") != 75
            or whitelist.get("joint_housing_components_rejected") != [152, 153, 154]):
        raise ValueError("f1 TASK_GRIP_SURFACE semantic identity changed")
    exact = read_binary_stl(root / row["visual_output"])
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    required = {"points_local_m", "faces", "source_face_indices", "face_normals_local",
                "patch_indices", "legacy_blue_pad_face_mask"}
    if set(arrays) != required:
        raise ValueError("f1 TASK_GRIP_SURFACE NPZ schema changed")
    points, faces = arrays["points_local_m"], arrays["faces"]
    source_indices, normals = arrays["source_face_indices"], arrays["face_normals_local"]
    if (points.shape != (6600, 3) or faces.shape != (2200, 3)
            or source_indices.shape != (2200,) or normals.shape != (2200, 3)
            or len(np.unique(source_indices)) != 2200
            or np.any(source_indices < 0) or np.any(source_indices >= len(exact))):
        raise ValueError("f1 TASK_GRIP_SURFACE array shape/index changed")
    triangles = points[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    computed = cross / np.linalg.norm(cross, axis=1)[:, None]
    if (not np.array_equal(triangles, exact[source_indices])
            or not np.allclose(normals, computed, rtol=0.0, atol=1.0e-12)):
        raise ValueError("f1 TASK_GRIP_SURFACE triangle/normal binding changed")
    return {"source_face_indices": source_indices, "face_normals_local": normals,
        "binding": {"manifest": _TASK_MANIFEST, "manifest_sha256": _TASK_MANIFEST_SHA256,
            "surface_npz": _TASK_F1_NPZ, "surface_npz_sha256": _TASK_F1_NPZ_SHA256,
            "task_face_count": 2200, "main_body_task_face_count": 518,
            "minimum_normal_motion_m_per_phase": _BOUNDARY_M,
            "joint_housing_components_rejected": [152, 153, 154]}}


def _set_inertial(link: ElementTree.Element, row: dict) -> None:
    inertial = link.find("inertial")
    origin, mass_node, tensor = inertial.find("origin"), inertial.find("mass"), inertial.find("inertia")
    if origin is None or mass_node is None or tensor is None:
        raise ValueError(f"{link.get('name')}: incomplete inertial")
    mass = row["mass_properties"]
    origin.set("xyz", " ".join(f"{value:.17g}" for value in mass["new_com_m"]))
    origin.set("rpy", "0 0 0")
    mass_node.set("value", f"{mass['new_mass_kg']:.17g}")
    matrix = mass["new_inertia_at_com_kg_m2"]
    values = {"ixx":matrix[0][0], "ixy":matrix[0][1], "ixz":matrix[0][2],
              "iyy":matrix[1][1], "iyz":matrix[1][2], "izz":matrix[2][2]}
    for key, value in values.items():
        tensor.set(key, f"{value:.17g}")


def _prepare(root: Path, urdf: Path, manifest: Path) -> dict:
    if urdf.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite SDF preparation evidence")
    rows, urdf_dir = _load_inputs(root), root / "src/iiwa_description/urdf"
    task_surface = _load_task_surface(root, rows["f1Link3"])
    for name, expected in _XACRO_SHA256.items():
        if _sha(urdf_dir / name) != expected:
            raise ValueError(f"authoritative Xacro changed: {name}")
    xacro = shutil.which("xacro")
    if xacro is None:
        raise RuntimeError("xacro executable is unavailable")
    expanded = subprocess.run(
        [xacro, str(urdf_dir / "handarm.urdf.xacro")], check=True,
        capture_output=True, text=True,
    ).stdout
    sys.path.insert(0, str(root / "src/kcg_connector"))
    from kcg_connector.export_isaac_urdf import sanitize_urdf
    robot = ElementTree.fromstring(sanitize_urdf(
        expanded, {"iiwa_description": root / "src/iiwa_description"}
    ))
    robot.set("name", "handarm_connector_no_nail_sdf_audit")
    for link, row in rows.items():
        nodes = robot.findall(f"./link[@name='{link}']")
        if len(nodes) != 1:
            raise ValueError(f"{link}: expanded link count is {len(nodes)}")
        meshes = nodes[0].findall("./visual/geometry/mesh") + nodes[0].findall("./collision/geometry/mesh")
        if len(meshes) != 2:
            raise ValueError(f"{link}: visual/collision mesh count changed")
        exact = str((root / row["visual_output"]).resolve())
        for mesh in meshes:
            mesh.set("filename", exact); mesh.attrib.pop("scale", None)
        _set_inertial(nodes[0], row)
    ElementTree.indent(robot, space="  ")
    _write_text(urdf, '<?xml version="1.0"?>\n' + ElementTree.tostring(robot, encoding="unicode") + "\n")
    helper = Path(__file__).with_name("_sdf_audit_geometry.py")
    document = {
        "schema_version":"carts_nailfree_sdf_runtime_preparation_v2",
        "migration_identity":"F1_MAIN_BODY_NORMAL_CROSSING_V2",
        "status":"STATIC_INPUT_BOUND_RUNTIME_NOT_RUN", "evidence_level":"STATIC_PREPARATION_ONLY",
        "created_utc":datetime.now(timezone.utc).isoformat(), "hardware_authorized":False,
        "research_dynamic_pass":False, "formal_dynamic_pass":False,
        "runtime_binding_accepted":False, "script":str(Path(__file__).resolve().relative_to(root)),
        "script_sha256":_sha(Path(__file__).resolve()),
        "private_helper":str(helper.relative_to(root)), "private_helper_sha256":_sha(helper),
        "upstream_audit_sha256":_AUDIT_SHA256,
        "temporary_urdf":str(urdf.relative_to(root)), "temporary_urdf_sha256":_sha(urdf),
        "xacro_sha256":_XACRO_SHA256, "sdf_parameters":_SDF,
        "task_grip_surface_binding":task_surface["binding"],
        "main_body_sign_gate":_MAIN_BODY_IDENTITY,
        "runtime_asset_binding":{
            "method":"REQUIRED_PRELAUNCH_SHA256_PLUS_REALIZED_GEOMETRY_AND_MASS_READBACK",
            "asset_sha256":None,
        },
        "links": {link:{"source_sha256":row["source_sha256"],
            "exact_nailfree_mesh_sha256":row["visual_output_sha256"],
            "retained_face_count":11836, "mass_properties":row["mass_properties"]}
            for link, row in rows.items()},
        "audit_thresholds":{"maximum_bidirectional_surface_error_m":_MAX_ERROR_M,
            "removed_exclusive_boundary_tolerance_m":_BOUNDARY_M,
            "retained_surface_cover_radius_m":_COVER_M},
        "limitations":["Preparation and f1 diagnostic do not persist SDF into the USD asset.",
            "Global runtime_binding_accepted remains false regardless of f1 result.",
            "Mass, COM and inertia are simulation estimates, not hardware calibration."],
    }
    _write_json(manifest, document)
    return document


def _apply_sdf(prim) -> None:
    from pxr import PhysxSchema, UsdPhysics
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    sdf.CreateSdfResolutionAttr().Set(256); sdf.CreateSdfSubgridResolutionAttr().Set(6)
    sdf.CreateSdfBitsPerSubgridPixelAttr().Set(PhysxSchema.Tokens.BitsPerPixel16)
    sdf.CreateSdfNarrowBandThicknessAttr().Set(0.01); sdf.CreateSdfMarginAttr().Set(0.01)
    sdf.CreateSdfEnableRemeshingAttr().Set(False)
    sdf.CreateSdfTriangleCountReductionFactorAttr().Set(1.0)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(PhysxSchema.Tokens.sdf)


def _cube(stage):
    from pxr import Gf, UsdGeom, UsdPhysics
    body = UsdGeom.Xform.Define(stage, "/World/SdfCalibrationBody")
    body.AddTranslateOp().Set(Gf.Vec3d(10.0, 10.0, 10.0))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr().Set(True)
    half = 0.01
    points = [(-half,-half,-half),(half,-half,-half),(half,half,-half),(-half,half,-half),
              (-half,-half,half),(half,-half,half),(half,half,half),(-half,half,half)]
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    mesh = UsdGeom.Mesh.Define(stage, "/World/SdfCalibrationBody/mesh")
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr().Set([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set([value for face in faces for value in face])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none); _apply_sdf(mesh.GetPrim())
    return mesh.GetPrim(), half


def _cooking(interface) -> dict:
    value = interface.get_cooking_statistics()
    names = ("total_scheduled_tasks","total_finished_tasks",
             "total_finished_cache_hit_tasks","total_finished_cache_miss_tasks")
    return {name:int(getattr(value, name)) for name in names}


def _wait_cooking(app, interface, before) -> dict:
    deadline = time.monotonic()+60
    while time.monotonic() < deadline:
        app.update(); after = _cooking(interface)
        if after["total_scheduled_tasks"] > before["total_scheduled_tasks"] and after["total_scheduled_tasks"] == after["total_finished_tasks"]:
            return {"before":before, "after":after, "completed":True,
                "scheduled_delta":after["total_scheduled_tasks"]-before["total_scheduled_tasks"],
                "cache_hit_delta":after["total_finished_cache_hit_tasks"]-before["total_finished_cache_hit_tasks"],
                "cache_miss_delta":after["total_finished_cache_miss_tasks"]-before["total_finished_cache_miss_tasks"]}
    raise TimeoutError("PhysX SDF cooking did not complete within 60 s")


def _query(view, points, channel, batch):
    import torch
    result = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), batch):
        row = np.asarray(points[start:start+batch], dtype=np.float32)
        padded = np.empty((batch, 3), dtype=np.float32)
        padded[:len(row)] = row; padded[len(row):] = row[-1]
        output = view.get_sdf_and_gradients(
            torch.as_tensor(padded, device="cuda:0").reshape(1,batch,3)
        ).detach().cpu().numpy().reshape(batch,4)
        result[start:start+len(row)] = output[:len(row),channel]
    return result


def _calibrate(simulation, path, half):
    import torch
    points = np.asarray([[0,0,0],[.015,0,0],[0,-.006,0],[.014,.013,0],
                         [-.012,.004,.016],[.003,-.014,-.005]], dtype=np.float32)
    view = simulation.create_sdf_shape_view(path, len(points))
    if view.count != 1 or not view.check(): raise RuntimeError("calibration cube SDF view is invalid")
    output = view.get_sdf_and_gradients(
        torch.as_tensor(points,device="cuda:0").reshape(1,len(points),3)
    ).detach().cpu().numpy().reshape(len(points),4)
    delta = np.abs(points)-half
    expected = np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(np.max(delta,axis=1),0)
    errors = [float(np.max(np.abs(output[:,index]-expected))) for index in range(4)]
    matches = [index for index,error in enumerate(errors) if error <= 5e-4]
    if len(matches) != 1: raise RuntimeError(f"SDF channel/sign calibration is not unique: {errors}")
    return matches[0], {"distance_channel":matches[0],"channel_max_abs_errors":errors,"pass":True}


def _load_preparation(root: Path, path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    helper = Path(__file__).with_name("_sdf_audit_geometry.py")
    if (value.get("schema_version") != "carts_nailfree_sdf_runtime_preparation_v2"
        or value.get("migration_identity") != "F1_MAIN_BODY_NORMAL_CROSSING_V2"
        or value.get("runtime_binding_accepted") is not False or value.get("hardware_authorized") is not False
        or value.get("script_sha256") != _sha(Path(__file__).resolve())
        or value.get("private_helper_sha256") != _sha(helper)
        or value.get("upstream_audit_sha256") != _AUDIT_SHA256
        or value.get("task_grip_surface_binding", {}).get("manifest_sha256") != _TASK_MANIFEST_SHA256
        or value.get("task_grip_surface_binding", {}).get("surface_npz_sha256") != _TASK_F1_NPZ_SHA256
        or value.get("main_body_sign_gate") != _MAIN_BODY_IDENTITY):
        raise ValueError("preparation manifest identity/boundary failed")
    urdf = root/value["temporary_urdf"]
    if _sha(urdf) != value["temporary_urdf_sha256"]: raise ValueError("temporary URDF SHA changed")
    value["manifest_path"] = str(path)
    return value


def _run_f1(root, asset, asset_sha, preparation, task_surface, app):
    if _sha(asset) != asset_sha: raise ValueError("runtime asset differs from prelaunch SHA")
    world = None
    try:
        import omni.kit.commands
        import omni.physics.tensors as tensors
        import omni.physxcommands
        from omni.physx.scripts.ifaces import get_physx_cooking_private_interface
        from pxr import PhysxSchema, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        sys.path.insert(0, str(root/"src/kcg_connector/isaac"))
        from carts_v2.engine_health import (audit_physx_log,current_engine_log_path,
            gpu_backend_record,gpu_world_parameters,load_runtime_resources,synchronize_engine_log)
        row = _load_inputs(root)["f1Link3"]
        exact, source = read_binary_stl(root/row["visual_output"]), read_binary_stl(root/row["source_path"])
        resources = load_runtime_resources(root/"src/kcg_connector/config/carts_v2_isaac_runtime.json")
        World.clear_instance(); SimulationManager.set_physics_sim_device("cuda:0")
        world = World(stage_units_in_meters=1.0,physics_dt=1/120,rendering_dt=1/60,
                      **gpu_world_parameters(resources))
        stage = get_current_stage(); add_reference_to_stage(str(asset),"/World/HandArm")
        owner, collider, frame = find_identity_collision(stage,"f1Link3",exact,11836)
        mass = verify_usd_mass(owner,row); cooking = get_physx_cooking_private_interface()
        before = _cooking(cooking); _apply_sdf(collider); cube, half = _cube(stage)
        omni.kit.commands.execute("CookPhysxColliders")
        cooking_record = _wait_cooking(app,cooking,before)
        world.reset(); world.play(); world.step(render=False)
        backend = gpu_backend_record(world,world.get_physics_context())
        if not backend["pass"]: raise RuntimeError(f"GPU backend failed: {backend}")
        simulation = tensors.create_simulation_view("torch"); simulation.set_subspace_roots("/")
        channel, calibration = _calibrate(simulation,str(cube.GetPath()),half)
        view = simulation.create_sdf_shape_view(str(collider.GetPath()),_BATCH)
        if view.count != 1 or not view.check(): raise RuntimeError("f1 SDF view is invalid")
        geometry = geometry_audit(lambda p:_query(view,p,channel,_BATCH), exact, source,
            _REMOVED_RANGES,256,_COVER_M,_BOUNDARY_M,_MAX_ERROR_M,
            task_surface,_MAIN_BODY_IDENTITY)
        sdf = PhysxSchema.PhysxSDFMeshCollisionAPI(collider)
        schema = {"approximation":str(UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()),
            "has_sdf_api":bool(sdf),"resolution":int(sdf.GetSdfResolutionAttr().Get()),
            "subgrid_resolution":int(sdf.GetSdfSubgridResolutionAttr().Get()),
            "bits_per_subgrid_pixel":str(sdf.GetSdfBitsPerSubgridPixelAttr().Get()),
            "narrow_band_thickness":float(sdf.GetSdfNarrowBandThicknessAttr().Get()),
            "margin":float(sdf.GetSdfMarginAttr().Get()),
            "enable_remeshing":bool(sdf.GetSdfEnableRemeshingAttr().Get()),
            "triangle_count_reduction_factor":float(sdf.GetSdfTriangleCountReductionFactorAttr().Get())}
        schema_pass = bool(schema["approximation"] == "sdf" and schema["has_sdf_api"]
            and schema["resolution"] == 256 and schema["subgrid_resolution"] == 6
            and schema["bits_per_subgrid_pixel"] == "BitsPerPixel16"
            and math.isclose(schema["narrow_band_thickness"],.01,rel_tol=0,abs_tol=1e-8)
            and math.isclose(schema["margin"],.01,rel_tol=0,abs_tol=1e-8)
            and schema["enable_remeshing"] is False
            and math.isclose(schema["triangle_count_reduction_factor"],1.0,rel_tol=0,abs_tol=1e-8))
        log_path = current_engine_log_path(); sync = synchronize_engine_log(log_path)
        log = audit_physx_log(log_path,cutoff_bytes=sync["audit_byte_count"],required_marker=sync["marker"])
        engine_clean = bool(log.get("scan_complete") is True and log.get("capacity_warning_count") == 0
                            and not log.get("all_physx_warning_lines") and not log.get("physx_error_lines"))
        passed = bool(geometry["accepted"] and calibration["pass"] and cooking_record["completed"]
                      and backend["pass"] and schema_pass and engine_clean)
        return {"f1_sdf_diagnostic_pass":passed,"runtime_binding_accepted":False,
            "sdf_persisted_to_asset":False,"stage_mutation_scope":"IN_MEMORY_DIAGNOSTIC_ONLY",
            "asset":str(asset),"asset_sha256":asset_sha,
            "temporary_urdf_sha256":preparation["temporary_urdf_sha256"],
            "preparation_manifest_sha256":_sha(Path(preparation["manifest_path"])),
            "collision_frame":frame,"usd_geometry_exact_match":True,"usd_mass_properties":mass,
            "sdf_schema_readback":schema,"sdf_schema_pass":schema_pass,"cooking":cooking_record,
            "cube_calibration":calibration,"geometry":geometry,"gpu_backend":backend,
            "engine_log":log,"engine_clean":engine_clean}
    finally:
        if world is not None:
            try: world.stop()
            except Exception: pass


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only",action="store_true")
    action.add_argument("--run-f1-sdf-audit",action="store_true")
    parser.add_argument("--asset",type=Path); parser.add_argument("--asset-sha256")
    parser.add_argument("--result",type=Path)
    args, root, directory = parser.parse_args(), _root(), Path(__file__).resolve().parent
    urdf = directory/"handarm_connector_no_nail_sdf_audit.urdf"
    manifest = directory/"SDF_RUNTIME_PREPARATION_MANIFEST.json"
    if args.prepare_only:
        value = _prepare(root,urdf,manifest)
        print(json.dumps({"status":value["status"],"isaac_started":False},indent=2)); return 0
    if args.asset is None or not args.asset_sha256 or len(args.asset_sha256) != 64:
        raise ValueError("runtime requires --asset and 64-character --asset-sha256")
    asset = args.asset.expanduser().resolve()
    if not asset.is_file(): raise FileNotFoundError(asset)
    result_path = args.result.expanduser().resolve() if args.result else directory/"SDF_RUNTIME_F1_RESULT.json"
    if result_path.exists(): raise FileExistsError(f"refusing to overwrite {result_path}")
    result = {"schema_version":"carts_nailfree_f1_sdf_diagnostic_v2","status":"FAILED_CLOSED",
        "evidence_level":"ISAAC_ASSET_DIAGNOSTIC_ONLY","hardware_authorized":False,
        "research_dynamic_pass":False,"formal_dynamic_pass":False,
        "runtime_binding_accepted":False,"f1_sdf_diagnostic_pass":False,
        "sdf_persisted_to_asset":False,"created_utc":datetime.now(timezone.utc).isoformat()}
    app = None
    try:
        preparation = _load_preparation(root,manifest)
        rows = _load_inputs(root)
        task_surface = _load_task_surface(root, rows["f1Link3"])
        from isaacsim import SimulationApp
        app = SimulationApp({"headless":True,"multi_gpu":False,"active_gpu":0,
                             "physics_gpu":0,"fast_shutdown":True})
        result.update(_run_f1(root,asset,args.asset_sha256.lower(),preparation,task_surface,app))
        result["status"] = "F1_SDF_DIAGNOSTIC_PASS" if result["f1_sdf_diagnostic_pass"] else "FAILED_CLOSED"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"; result["traceback"] = traceback.format_exc()
    result["runtime_binding_accepted"] = False
    _write_json(result_path,result)
    print(json.dumps({"status":result["status"],"result":str(result_path)},indent=2),flush=True)
    exit_code = 0 if result["f1_sdf_diagnostic_pass"] else 2
    if app is not None: app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
