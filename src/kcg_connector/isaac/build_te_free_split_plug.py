#!/usr/bin/env python3

"""Build the free TE J35 plug as a body plus a coaxially rotating coupling nut."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import coacd
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation
import trimesh


REPOSITORY = Path(__file__).resolve().parents[3]
BODY_VISUAL = REPOSITORY / (
    "artifacts/kcg_connector/isaac/te_j35_engineering_v1/visual/"
    "D38999_26FJ35PN_BODY_VISUAL.usdc"
)
NUT_VISUAL = REPOSITORY / (
    "artifacts/kcg_connector/isaac/te_j35_engineering_v1/visual/"
    "D38999_26FJ35PN_COUPLING_NUT_VISUAL.usdc"
)
DEFAULT_OUTPUT = REPOSITORY / (
    "artifacts/kcg_connector/isaac/te_j35_free_split_tabletop_v1"
)
VELOCITY_BRAKE_DAMPING_NM_S_PER_DEG = 1.0e10
REFERENCE_TOTAL_MASS_KG = 0.062459669349
MASS_REFERENCE = REPOSITORY / (
    "src/kcg_connector/config/te_j35_mass_reference_v1.yaml"
)
PARTS = (
    ("Body", BODY_VISUAL, 0.027548618179969777, 64),
    ("CouplingNut", NUT_VISUAL, 0.03491105116903022, 64),
)
COACD_PARAMETERS = {
    "threshold": 0.05,
    "preprocess_mode": "auto",
    "preprocess_resolution": 50,
    "resolution": 2000,
    "mcts_nodes": 20,
    "mcts_iterations": 150,
    "mcts_max_depth": 3,
    "pca": False,
    "merge": True,
    "decimate": False,
    "max_ch_vertex": 256,
    "extrude": False,
    "seed": 20260902,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_single_usd_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stage = Usd.Stage.Open(str(path))
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError(f"expected one mesh in {path}, found {len(meshes)}")
    mesh = UsdGeom.Mesh(meshes[0])
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    if counts.ndim != 1 or not np.all(counts == 3):
        raise ValueError(f"source visual is not a triangle mesh: {path}")
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    faces = np.asarray(
        mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32
    ).reshape(-1, 3)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"source visual has invalid vertices: {path}")
    return vertices, faces


def _close_hidden_planar_boundaries(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Cap only the two supplier-split interfaces hidden inside the assembly."""

    directed = np.vstack(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)])
    )
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    boundary = directed[counts[inverse] == 1]
    next_vertex = {int(start): int(end) for start, end in boundary}
    if len(next_vertex) != len(boundary):
        raise ValueError("split interface boundary is not a set of simple loops")

    closed_vertices = list(vertices)
    closed_faces = list(faces)
    records: list[dict[str, object]] = []
    while next_vertex:
        start = next(iter(next_vertex))
        current = start
        loop: list[tuple[int, int]] = []
        while True:
            if current not in next_vertex:
                raise ValueError("split interface contains an open boundary chain")
            following = next_vertex.pop(current)
            loop.append((current, following))
            current = following
            if current == start:
                break
        indices = np.asarray([edge[0] for edge in loop], dtype=np.int64)
        loop_vertices = vertices[indices]
        z_span = float(np.ptp(loop_vertices[:, 2]))
        if z_span > 1.0e-9:
            raise ValueError("supplier split interface is not planar")
        center = np.mean(loop_vertices, axis=0)
        center_index = len(closed_vertices)
        closed_vertices.append(center)
        # Reverse every existing boundary edge so the new cap has consistent
        # winding. The cap lies entirely at the hidden body/nut interface.
        closed_faces.extend(
            (end, begin, center_index) for begin, end in loop
        )
        records.append(
            {
                "edge_count": len(loop),
                "plane_z_m": float(center[2]),
                "radial_extent_m": float(
                    np.max(np.linalg.norm(loop_vertices[:, :2], axis=1))
                ),
            }
        )

    closed_vertices_array = np.asarray(closed_vertices, dtype=np.float64)
    closed_faces_array = np.asarray(closed_faces, dtype=np.int32)
    mesh = trimesh.Trimesh(
        vertices=closed_vertices_array,
        faces=closed_faces_array,
        process=False,
    )
    if len(records) != 2 or not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("hidden-interface closure did not create a closed part")
    return closed_vertices_array, closed_faces_array, records


def _mass_properties(
    vertices: np.ndarray, faces: np.ndarray, mass_kg: float
) -> dict[str, object]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    properties = mesh.mass_properties
    if properties.mass <= 0.0 or properties.volume <= 0.0:
        raise ValueError("closed part has invalid signed volume")
    inertia = np.asarray(properties.inertia, dtype=np.float64) * (
        float(mass_kg) / float(properties.mass)
    )
    diagonal, axes = np.linalg.eigh(0.5 * (inertia + inertia.T))
    if np.linalg.det(axes) < 0.0:
        axes[:, 0] *= -1.0
    quaternion_xyzw = Rotation.from_matrix(axes).as_quat()
    if np.any(diagonal <= 0.0):
        raise ValueError("part principal inertia is not positive")
    return {
        "mass_kg": float(mass_kg),
        "volume_m3": float(properties.volume),
        "center_of_mass_m": np.asarray(
            properties.center_mass, dtype=np.float64
        ).tolist(),
        "inertia_tensor_kg_m2": inertia.tolist(),
        "principal_inertia_kg_m2": diagonal.tolist(),
        "principal_axes_quaternion_wxyz": [
            float(quaternion_xyzw[3]),
            float(quaternion_xyzw[0]),
            float(quaternion_xyzw[1]),
            float(quaternion_xyzw[2]),
        ],
        "method": (
            "UNIFORM_EFFECTIVE_DENSITY_OVER_SUPPLIER_CAD_PARTITION_"
            "SCALED_TO_OFFICIAL_REFERENCE_TOTAL"
        ),
    }


def _decompose(
    vertices: np.ndarray, faces: np.ndarray, maximum_hulls: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    hulls = coacd.run_coacd(
        coacd.Mesh(vertices, faces),
        max_convex_hull=maximum_hulls,
        **COACD_PARAMETERS,
    )
    if not hulls:
        raise RuntimeError("CoACD returned no collision hulls")
    return [
        (np.asarray(points, dtype=np.float64), np.asarray(triangles, dtype=np.int32))
        for points, triangles in hulls
    ]


def _author_rigid_part(
    stage: Usd.Stage,
    *,
    root_path: str,
    visual_path: Path,
    output_asset: Path,
    mass: dict[str, object],
    hulls: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    part = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    rigid = UsdPhysics.RigidBodyAPI.Apply(part)
    rigid.CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(part)
    mass_api.CreateMassAttr(float(mass["mass_kg"]))
    mass_api.CreateCenterOfMassAttr(
        Gf.Vec3f(*map(float, mass["center_of_mass_m"]))
    )
    mass_api.CreateDiagonalInertiaAttr(
        Gf.Vec3f(*map(float, mass["principal_inertia_kg_m2"]))
    )
    quaternion = mass["principal_axes_quaternion_wxyz"]
    mass_api.CreatePrincipalAxesAttr(
        Gf.Quatf(
            float(quaternion[0]),
            Gf.Vec3f(*map(float, quaternion[1:])),
        )
    )

    visual = UsdGeom.Xform.Define(stage, root_path + "/Visual").GetPrim()
    visual.GetReferences().AddReference(
        os.path.relpath(visual_path, output_asset.parent)
    )
    collision_root = UsdGeom.Scope.Define(stage, root_path + "/Collision")
    for index, (points, triangles) in enumerate(hulls):
        mesh = UsdGeom.Mesh.Define(
            stage, f"{collision_root.GetPath()}/Hull_{index:03d}"
        )
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, row)) for row in points])
        mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
        mesh.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
            UsdPhysics.Tokens.convexHull
        )


def _rotational_resistance_record(rotational_resistance_nm: float) -> dict[str, object]:
    return {
        "model": "SYMMETRIC_CAPPED_ZERO_VELOCITY_BRAKE",
        "assumed_resisting_torque_nm": rotational_resistance_nm,
        "drive_type": "force",
        "target_velocity_deg_s": 0.0,
        "stiffness_nm_per_deg": 0.0,
        "damping_nm_s_per_deg": (
            VELOCITY_BRAKE_DAMPING_NM_S_PER_DEG
            if rotational_resistance_nm > 0.0
            else 0.0
        ),
        "maximum_drive_torque_nm": rotational_resistance_nm,
        "angle_preference_or_limit_authored": False,
        "physical_status": (
            "ENGINEERING_ASSUMPTION_PUBLIC_NUMERIC_VALUE_UNAVAILABLE"
            if rotational_resistance_nm > 0.0
            else "UNMODELED"
        ),
    }


def _author_rotational_resistance(
    stage: Usd.Stage, rotational_resistance_nm: float
) -> None:
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError("split plug asset has no default prim")
    root.CreateAttribute(
        "kcg:rotationalResistanceModel", Sdf.ValueTypeNames.String
    ).Set(
        "ASSUMED_SYMMETRIC_CAPPED_ZERO_VELOCITY_BRAKE"
        if rotational_resistance_nm > 0.0
        else "UNSPECIFIED_NO_AUTHORED_DRIVE_OR_JOINT_FRICTION"
    )
    joint_prim = stage.GetPrimAtPath(
        str(root.GetPath()) + "/Joints/CouplingNutRevolute"
    )
    if not joint_prim.IsA(UsdPhysics.RevoluteJoint):
        raise ValueError("split plug asset has no coupling-nut revolute joint")
    if joint_prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
        raise ValueError("source split plug already has an angular drive")
    if rotational_resistance_nm <= 0.0:
        return
    # A force-limited zero-velocity drive behaves as a symmetric brake:
    # it can hold the joint while the external torque is below the cap,
    # and it saturates at the requested resisting torque once motion starts.
    # Stiffness remains zero, so no angle is preferred or restored.
    brake = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    brake.CreateTypeAttr("force")
    brake.CreateTargetVelocityAttr(0.0)
    brake.CreateDampingAttr(VELOCITY_BRAKE_DAMPING_NM_S_PER_DEG)
    brake.CreateStiffnessAttr(0.0)
    brake.CreateMaxForceAttr(rotational_resistance_nm)


def _replace_mass_properties_with_reference(
    stage: Usd.Stage, manifest: dict[str, object]
) -> None:
    """Replace mass/inertia only; leave visual and collision geometry untouched."""

    target_masses = {name: float(mass_kg) for name, _, mass_kg, _ in PARTS}
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError("split plug asset has no default prim")
    parts = manifest.get("parts")
    if not isinstance(parts, dict):
        raise ValueError("base manifest has no part records")

    for name, target_mass_kg in target_masses.items():
        part_record = parts.get(name)
        if not isinstance(part_record, dict):
            raise ValueError(f"base manifest has no {name} record")
        mass_record = part_record.get("mass_properties")
        if not isinstance(mass_record, dict):
            raise ValueError(f"base manifest has no {name} mass record")
        old_mass_kg = float(mass_record["mass_kg"])
        if old_mass_kg <= 0.0:
            raise ValueError(f"base manifest has invalid {name} mass")
        scale = target_mass_kg / old_mass_kg

        inertia = np.asarray(
            mass_record["inertia_tensor_kg_m2"], dtype=np.float64
        ) * scale
        principal_inertia = np.asarray(
            mass_record["principal_inertia_kg_m2"], dtype=np.float64
        ) * scale
        mass_record["mass_kg"] = target_mass_kg
        mass_record["inertia_tensor_kg_m2"] = inertia.tolist()
        mass_record["principal_inertia_kg_m2"] = principal_inertia.tolist()
        mass_record["method"] = (
            "UNIFORM_EFFECTIVE_DENSITY_OVER_SUPPLIER_CAD_PARTITION_"
            "SCALED_TO_OFFICIAL_REFERENCE_TOTAL"
        )

        prim = stage.GetPrimAtPath(str(root.GetPath()) + f"/{name}")
        if not prim or not prim.HasAPI(UsdPhysics.MassAPI):
            raise ValueError(f"split plug asset has no MassAPI on {name}")
        mass_api = UsdPhysics.MassAPI(prim)
        mass_api.GetMassAttr().Set(target_mass_kg)
        mass_api.GetDiagonalInertiaAttr().Set(
            Gf.Vec3f(*map(float, principal_inertia))
        )

    manifest["mass_model"] = {
        "total_mass_kg": REFERENCE_TOTAL_MASS_KG,
        "source": str(MASS_REFERENCE.relative_to(REPOSITORY)),
        "source_sha256": _sha256(MASS_REFERENCE),
        "reference_scope": (
            "AMPHENOL_OFFICIAL_SAME_MIL_SLASH_CONFIGURATION_"
            "WEIGHT_INCLUDING_CONTACTS"
        ),
        "component_split_method": (
            "UNIFORM_EFFECTIVE_DENSITY_OVER_SUPPLIER_CAD_PARTITION"
        ),
        "component_split_is_manufacturer_measurement": False,
    }


def _build_from_existing_asset(
    output_directory: Path,
    rotational_resistance_nm: float,
    base_manifest_path: Path,
    replace_with_reference_mass: bool,
) -> tuple[Path, Path]:
    base_manifest_path = base_manifest_path.resolve()
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if (
        base_manifest.get("schema_version")
        != "kcg_te_j35_free_split_tabletop_asset_v1"
        or base_manifest.get("product_id") != "D38999/26FJ35PN"
        or base_manifest.get("hardware_authorized") is not False
        or base_manifest.get("joint", {}).get("drive_authored") is not False
        or base_manifest.get("joint", {}).get("joint_friction_authored") is not False
    ):
        raise ValueError("base manifest is not the zero-resistance split plug")
    base_asset_sha256 = str(base_manifest["asset_sha256"])
    base_asset = (REPOSITORY / str(base_manifest["asset"])).resolve()
    if not base_asset.is_file() or _sha256(base_asset) != base_asset_sha256:
        raise ValueError("base split-plug asset differs from its manifest")

    output_directory.mkdir(parents=True, exist_ok=True)
    output_asset = output_directory / "TE_J35_FREE_SPLIT_PLUG_V1.usdc"
    manifest_path = output_directory / "MANIFEST.json"
    if output_asset.resolve() == base_asset:
        raise ValueError("resistance asset must not overwrite the zero-resistance asset")
    shutil.copyfile(base_asset, output_asset)
    stage = Usd.Stage.Open(str(output_asset))
    if stage is None:
        raise ValueError("copied split-plug asset cannot be opened")
    if replace_with_reference_mass:
        _replace_mass_properties_with_reference(stage, base_manifest)
    _author_rotational_resistance(stage, rotational_resistance_nm)
    stage.GetRootLayer().Save()

    base_manifest["asset"] = str(output_asset.relative_to(REPOSITORY))
    base_manifest["asset_sha256"] = _sha256(output_asset)
    base_manifest["joint"]["drive_authored"] = rotational_resistance_nm > 0.0
    base_manifest["joint"]["joint_friction_authored"] = False
    base_manifest["joint"]["rotational_resistance"] = (
        _rotational_resistance_record(rotational_resistance_nm)
    )
    base_manifest["joint"]["rotational_resistance_evidence"] = (
        "ENGINEERING_ASSUMPTION_PUBLIC_NUMERIC_VALUE_UNAVAILABLE"
        if rotational_resistance_nm > 0.0
        else "UNKNOWN_NOT_PUBLISHED_OR_MEASURED"
    )
    base_manifest["derived_from_zero_resistance_asset"] = {
        "manifest": str(base_manifest_path.relative_to(REPOSITORY)),
        "asset_sha256": base_asset_sha256,
        "geometry_and_collision_records_unchanged": True,
        "mass_records_unchanged": not replace_with_reference_mass,
    }
    manifest_path.write_text(
        json.dumps(base_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_asset, manifest_path


def build(
    output_directory: Path,
    rotational_resistance_nm: float = 0.0,
    base_manifest_path: Path | None = None,
    replace_with_reference_mass: bool = False,
) -> tuple[Path, Path]:
    rotational_resistance_nm = float(rotational_resistance_nm)
    if (
        not np.isfinite(rotational_resistance_nm)
        or rotational_resistance_nm < 0.0
    ):
        raise ValueError("rotational resistance must be one nonnegative torque")
    if base_manifest_path is not None:
        return _build_from_existing_asset(
            output_directory,
            rotational_resistance_nm,
            base_manifest_path,
            replace_with_reference_mass,
        )
    if replace_with_reference_mass:
        raise ValueError("mass replacement requires --base-manifest")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_asset = output_directory / "TE_J35_FREE_SPLIT_PLUG_V1.usdc"
    manifest_path = output_directory / "MANIFEST.json"

    built_parts: dict[str, dict[str, object]] = {}
    for name, visual_path, mass_kg, maximum_hulls in PARTS:
        vertices, faces = _load_single_usd_mesh(visual_path)
        closed_vertices, closed_faces, caps = _close_hidden_planar_boundaries(
            vertices, faces
        )
        hulls = _decompose(closed_vertices, closed_faces, maximum_hulls)
        built_parts[name] = {
            "visual_path": visual_path,
            "source_vertices": vertices,
            "source_faces": faces,
            "closed_vertices": closed_vertices,
            "closed_faces": closed_faces,
            "caps": caps,
            "mass": _mass_properties(closed_vertices, closed_faces, mass_kg),
            "hulls": hulls,
            "maximum_hulls": maximum_hulls,
        }

    stage = Usd.Stage.CreateNew(str(output_asset))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root_path = "/TE_J35FreeSplitPlug"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)
    root.CreateAttribute("kcg:productId", Sdf.ValueTypeNames.String).Set(
        "D38999/26FJ35PN"
    )
    root.CreateAttribute("kcg:hardwareAuthorized", Sdf.ValueTypeNames.Bool).Set(False)
    root.CreateAttribute("kcg:relativeMotion", Sdf.ValueTypeNames.String).Set(
        "COAXIAL_ROTATION_ONLY"
    )
    root.CreateAttribute(
        "kcg:rotationalResistanceModel", Sdf.ValueTypeNames.String
    ).Set("UNSPECIFIED_NO_AUTHORED_DRIVE_OR_JOINT_FRICTION")

    body_path = root_path + "/Body"
    nut_path = root_path + "/CouplingNut"
    for name, path in (("Body", body_path), ("CouplingNut", nut_path)):
        record = built_parts[name]
        _author_rigid_part(
            stage,
            root_path=path,
            visual_path=record["visual_path"],
            output_asset=output_asset,
            mass=record["mass"],
            hulls=record["hulls"],
        )

    joint_path = root_path + "/Joints/CouplingNutRevolute"
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateAxisAttr(UsdPhysics.Tokens.z)
    joint.CreateBody0Rel().SetTargets((Sdf.Path(body_path),))
    joint.CreateBody1Rel().SetTargets((Sdf.Path(nut_path),))
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    joint.CreateCollisionEnabledAttr(False)
    _author_rotational_resistance(stage, rotational_resistance_nm)
    stage.GetRootLayer().Save()

    manifest = {
        "schema_version": "kcg_te_j35_free_split_tabletop_asset_v1",
        "product_id": "D38999/26FJ35PN",
        "hardware_authorized": False,
        "asset": str(output_asset.relative_to(REPOSITORY)),
        "asset_sha256": _sha256(output_asset),
        "reference_prim_path": "/World/TE_J35FreeSplitPlug",
        "body_relative_prim_path": "Body",
        "coupling_nut_relative_prim_path": "CouplingNut",
        "joint_relative_prim_path": "Joints/CouplingNutRevolute",
        "component_bottom_offsets_m": [0.0, 0.0015493999235332012],
        "legal_grasp_contact_part": "CouplingNut",
        "joint": {
            "type": "REVOLUTE",
            "axis_in_supplier_object_frame": [0.0, 0.0, 1.0],
            "allowed_relative_motion": "ROTATION_ABOUT_COMMON_Z_AXIS_ONLY",
            "locked_relative_motion": [
                "TRANSLATION_X",
                "TRANSLATION_Y",
                "TRANSLATION_Z",
                "ROTATION_X",
                "ROTATION_Y",
            ],
            "lower_limit_authored": False,
            "upper_limit_authored": False,
            "drive_authored": rotational_resistance_nm > 0.0,
            "joint_friction_authored": False,
            "rotational_resistance": _rotational_resistance_record(
                rotational_resistance_nm
            ),
            "rotational_resistance_evidence": (
                "ENGINEERING_ASSUMPTION_PUBLIC_NUMERIC_VALUE_UNAVAILABLE"
                if rotational_resistance_nm > 0.0
                else "UNKNOWN_NOT_PUBLISHED_OR_MEASURED"
            ),
        },
        "mass_model": {
            "total_mass_kg": REFERENCE_TOTAL_MASS_KG,
            "source": str(MASS_REFERENCE.relative_to(REPOSITORY)),
            "source_sha256": _sha256(MASS_REFERENCE),
            "reference_scope": (
                "AMPHENOL_OFFICIAL_SAME_MIL_SLASH_CONFIGURATION_"
                "WEIGHT_INCLUDING_CONTACTS"
            ),
            "component_split_method": (
                "UNIFORM_EFFECTIVE_DENSITY_OVER_SUPPLIER_CAD_PARTITION"
            ),
            "component_split_is_manufacturer_measurement": False,
        },
        "parts": {},
        "official_structure_sources": [
            {
                "claim": "series_iii_uses_triple_start_coupling_and_anti_decoupling_ratcheting",
                "url": "https://www.te.com/en/products/connectors/circular-connectors/intersection/mil-dtl-38999-series-iii-connectors.html?tab=pgp-story",
            },
            {
                "claim": "plug_coupling_ring_rotates_approximately_360_degrees_and_has_anti_decoupling_device",
                "url": "https://landandmaritimeapps.dla.mil/Downloads/MilSpec/Docs/MIL-DTL-38999/dtl38999.pdf",
            },
            {
                "claim": "slash_26_is_a_straight_threaded_plug",
                "url": "https://landandmaritimeapps.dla.mil/Downloads/MilSpec/Docs/MIL-DTL-38999/dtl38999ss26.pdf",
            },
        ],
        "evidence_limit": (
            "PUBLIC_DOCUMENTS_CONFIRM_ARCHITECTURE_BUT_NOT_THIS_SPECIMENS_"
            "ROTATIONAL_RESISTANCE_OR_BACKLASH"
        ),
    }
    for name, record in built_parts.items():
        hulls = record["hulls"]
        manifest["parts"][name] = {
            "visual_usd": str(record["visual_path"].relative_to(REPOSITORY)),
            "visual_usd_sha256": _sha256(record["visual_path"]),
            "source_mesh": {
                "vertex_count": int(len(record["source_vertices"])),
                "triangle_count": int(len(record["source_faces"])),
                "supplier_split_open_boundary": True,
            },
            "hidden_interface_caps": record["caps"],
            "closed_collision_source": {
                "vertex_count": int(len(record["closed_vertices"])),
                "triangle_count": int(len(record["closed_faces"])),
                "watertight": True,
                "winding_consistent": True,
            },
            "collision": {
                "method": "COACD_FROM_EXACT_SPLIT_SUPPLIER_SURFACE",
                "hull_count": len(hulls),
                "maximum_hulls": int(record["maximum_hulls"]),
                "output_vertex_count": int(sum(len(hull[0]) for hull in hulls)),
                "output_triangle_count": int(sum(len(hull[1]) for hull in hulls)),
                **COACD_PARAMETERS,
            },
            "mass_properties": record["mass"],
        }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_asset, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rotational-resistance-nm", type=float, default=0.0)
    parser.add_argument("--base-manifest")
    parser.add_argument("--replace-with-reference-mass", action="store_true")
    arguments = parser.parse_args()
    asset, manifest = build(
        Path(arguments.output_directory).resolve(),
        rotational_resistance_nm=arguments.rotational_resistance_nm,
        base_manifest_path=(
            None
            if arguments.base_manifest is None
            else Path(arguments.base_manifest).resolve()
        ),
        replace_with_reference_mass=arguments.replace_with_reference_mass,
    )
    print(
        json.dumps(
            {
                "asset": str(asset),
                "manifest": str(manifest),
                "rotational_resistance_nm": arguments.rotational_resistance_nm,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
