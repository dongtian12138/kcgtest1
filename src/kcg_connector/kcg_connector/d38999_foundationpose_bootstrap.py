"""Fail-closed FoundationPose asset bootstrap for the D38999 proxy.

The current end-to-end controller does not import this module.  It performs
three intentionally limited jobs without Isaac Sim, ROS, TensorRT, or GPU
imports:

* verify the content-addressed NVIDIA ONNX files kept under ``artifacts/``;
* generate deterministic OBJ meshes from the existing proxy dimensions; and
* report whether an isolated FoundationPose runtime is actually available.

Even a successful artifact check is not a 6D-pose or control claim.  The proxy
has no unique polarization key and is at least 180-degree yaw symmetric.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import yaml

from kcg_connector.d38999_proxy import (
    D38999Shell25JProxy,
    load_d38999_shell25j_proxy,
)


SCHEMA_VERSION = "kcg_d38999_foundationpose_bootstrap_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "d38999_foundationpose_bootstrap_v1.yaml"
)
DEFAULT_ARTIFACT_ROOT = Path(
    "artifacts/kcg_connector/foundationpose_1.0.1_onnx_local_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_MODEL_VERSION = "1.0.1_onnx"
_EXPECTED_OBJECT_IDS = {
    "loose_body",
    "coupling_nut",
    "fixed_receptacle",
}


@dataclass(frozen=True)
class ContentAddressedFile:
    """One repository-relative file bound to exact content."""

    path: Path
    sha256: str
    size_bytes: int | None = None
    url: str | None = None


@dataclass(frozen=True)
class MeshMapping:
    """Mapping from one OBJ coordinate frame to source and runtime prims."""

    mesh_id: str
    file: ContentAddressedFile
    model_id: str
    asset_prim_path: str
    runtime_prim_path: str
    frame_origin: str
    source_components: tuple[str, ...]
    rotational_symmetry_order: int
    unique_polarization_key_present: bool
    control_orientation_qualified: bool


@dataclass(frozen=True)
class FoundationPoseBootstrapContract:
    """Validated, disabled preparation contract."""

    schema_version: str
    enabled: bool
    status: str
    inputs: dict[str, ContentAddressedFile]
    official_release: dict[str, Any]
    official_sources: dict[str, str]
    model_version: str
    model_license: str
    model_storage_scope: str
    models: dict[str, ContentAddressedFile]
    meshes: dict[str, MeshMapping]
    radial_sections: int
    engine_outputs: dict[str, Path]
    runtime: dict[str, Any]
    observability: dict[str, Any]
    boundaries: dict[str, bool]


@dataclass(frozen=True)
class MeshStats:
    """Small parser-independent evidence about one generated OBJ."""

    vertex_count: int
    triangle_count: int
    bounds_min_xyz_m: tuple[float, float, float]
    bounds_max_xyz_m: tuple[float, float, float]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{label} keys differ; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty unpadded text")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _positive_real(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _sha(value: Any, label: str) -> str:
    result = _text(value, label)
    if not _SHA256.fullmatch(result) or result == "0" * 64:
        raise ValueError(f"{label} must be a non-zero lowercase SHA-256")
    return result


def _relative_path(value: Any, label: str) -> Path:
    result = Path(_text(value, label))
    if result.is_absolute() or ".." in result.parts:
        raise ValueError(f"{label} must be repository-relative")
    return result


def _https(value: Any, label: str) -> str:
    result = _text(value, label)
    if not result.startswith("https://"):
        raise ValueError(f"{label} must use HTTPS")
    return result


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{label} must be non-empty and unique")
    return result


def sha256_file(path: Path) -> str:
    """Return SHA-256 without loading large models into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_content_file(
    value: Any,
    label: str,
    *,
    require_size: bool,
    require_url: bool,
) -> ContentAddressedFile:
    document = _mapping(value, label)
    keys = {"path", "sha256"}
    if require_size:
        keys.add("size_bytes")
    if require_url:
        keys.add("url")
    _exact(document, keys, label)
    return ContentAddressedFile(
        path=_relative_path(document["path"], f"{label}.path"),
        sha256=_sha(document["sha256"], f"{label}.sha256"),
        size_bytes=(
            _positive_int(document["size_bytes"], f"{label}.size_bytes")
            if require_size
            else None
        ),
        url=(
            _https(document["url"], f"{label}.url")
            if require_url
            else None
        ),
    )


def load_foundationpose_bootstrap_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> FoundationPoseBootstrapContract:
    """Load the strict contract without probing the host or artifacts."""

    config_path = Path(path).expanduser().resolve()
    document = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "document",
    )
    _exact(
        document,
        {
            "schema_version",
            "enabled",
            "status",
            "inputs",
            "official_release",
            "model_bundle",
            "mesh_bundle",
            "runtime",
            "observability",
            "boundaries",
        },
        "document",
    )
    schema = _text(document["schema_version"], "schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    enabled = _boolean(document["enabled"], "enabled")
    if enabled:
        raise ValueError("FoundationPose bootstrap must remain disabled")

    inputs_document = _mapping(document["inputs"], "inputs")
    if set(inputs_document) != {
        "proxy_config",
        "proxy_asset",
        "tabletop_scene",
    }:
        raise ValueError("inputs must bind proxy config, asset, and scene")
    inputs = {
        name: _load_content_file(
            value,
            f"inputs.{name}",
            require_size=False,
            require_url=False,
        )
        for name, value in inputs_document.items()
    }

    release = dict(_mapping(document["official_release"], "official_release"))
    _exact(
        release,
        {
            "isaac_ros_release",
            "ros_distribution",
            "supported_x86_host_os",
            "minimum_gpu_memory_gb",
            "minimum_compute_capability",
            "minimum_driver_major",
            "tensorrt_engine_precision",
            "sources",
        },
        "official_release",
    )
    if (
        _text(release["isaac_ros_release"], "Isaac ROS release")
        != "release-4.5"
    ):
        raise ValueError("Isaac ROS release must remain pinned to release-4.5")
    if _text(release["ros_distribution"], "ROS distribution") != "jazzy":
        raise ValueError("release-4.5 contract requires ROS Jazzy")
    if _text(release["supported_x86_host_os"], "host OS") != "Ubuntu 24.04":
        raise ValueError("release-4.5 x86 contract requires Ubuntu 24.04")
    _positive_real(release["minimum_gpu_memory_gb"], "GPU memory")
    _positive_real(
        release["minimum_compute_capability"], "compute capability"
    )
    _positive_int(release["minimum_driver_major"], "driver major")
    if (
        _text(release["tensorrt_engine_precision"], "engine precision")
        != "FP32"
    ):
        raise ValueError(
            "release-4.5 FoundationPose contract uses FP32 engines"
        )
    sources_document = _mapping(release.pop("sources"), "official sources")
    if set(sources_document) != {
        "model_card",
        "model_eula",
        "quickstart",
        "system_requirements",
        "ros_source",
    }:
        raise ValueError("official source set is incomplete")
    official_sources = {
        name: _https(value, f"official source {name}")
        for name, value in sources_document.items()
    }

    model = _mapping(document["model_bundle"], "model_bundle")
    _exact(
        model,
        {
            "version",
            "license",
            "download_is_acceptance",
            "storage_scope",
            "standalone_redistribution_allowed",
            "commit_or_package_allowed",
            "files",
        },
        "model_bundle",
    )
    model_version = _text(model["version"], "model version")
    if model_version != _EXPECTED_MODEL_VERSION:
        raise ValueError("FoundationPose model version is not pinned")
    if not _boolean(model["download_is_acceptance"], "download acceptance"):
        raise ValueError("model download acceptance boundary was weakened")
    if _boolean(
        model["standalone_redistribution_allowed"],
        "standalone redistribution",
    ):
        raise ValueError("stand-alone model redistribution must be forbidden")
    if _boolean(model["commit_or_package_allowed"], "commit/package"):
        raise ValueError("model files must never be committed or packaged")
    storage_scope = _text(model["storage_scope"], "model storage scope")
    if storage_scope != "local_gitignored_artifact_only":
        raise ValueError("model storage must remain local and gitignored")
    model_files = _mapping(model["files"], "model files")
    if set(model_files) != {"refine_model", "score_model"}:
        raise ValueError("both official ONNX files are required")
    models = {
        name: _load_content_file(
            value,
            f"model_bundle.files.{name}",
            require_size=True,
            require_url=True,
        )
        for name, value in model_files.items()
    }
    for item in models.values():
        if DEFAULT_ARTIFACT_ROOT not in item.path.parents:
            raise ValueError(
                "ONNX files must stay under the local artifact root"
            )

    mesh_bundle = _mapping(document["mesh_bundle"], "mesh_bundle")
    _exact(
        mesh_bundle,
        {
            "generator_revision",
            "units",
            "meters_per_unit",
            "up_axis",
            "mating_axis",
            "radial_sections",
            "objects",
        },
        "mesh_bundle",
    )
    if (
        _text(mesh_bundle["generator_revision"], "mesh generator")
        != "d38999_proxy_obj_v1"
    ):
        raise ValueError("mesh generator revision changed")
    if _text(mesh_bundle["units"], "mesh units") != "metre":
        raise ValueError("FoundationPose meshes must use metres")
    if float(mesh_bundle["meters_per_unit"]) != 1.0:
        raise ValueError("meters_per_unit must be exactly 1")
    if _text(mesh_bundle["up_axis"], "up axis") != "z":
        raise ValueError("mesh up axis must be z")
    if _text(mesh_bundle["mating_axis"], "mating axis") != "positive_z":
        raise ValueError("mesh mating axis must be positive z")
    radial_sections = _positive_int(
        mesh_bundle["radial_sections"], "radial_sections"
    )
    if radial_sections < 24 or radial_sections % 24:
        raise ValueError("radial_sections must be a multiple of 24")

    mesh_documents = _mapping(mesh_bundle["objects"], "mesh objects")
    if set(mesh_documents) != _EXPECTED_OBJECT_IDS:
        raise ValueError("mesh object mapping must cover three rigid objects")
    meshes: dict[str, MeshMapping] = {}
    for mesh_id, raw in mesh_documents.items():
        label = f"mesh_bundle.objects.{mesh_id}"
        item = _mapping(raw, label)
        _exact(
            item,
            {
                "path",
                "sha256",
                "model_id",
                "asset_prim_path",
                "runtime_prim_path",
                "frame_origin",
                "source_components",
                "rotational_symmetry_order",
                "unique_polarization_key_present",
                "control_orientation_qualified",
            },
            label,
        )
        file = ContentAddressedFile(
            path=_relative_path(item["path"], f"{label}.path"),
            sha256=_sha(item["sha256"], f"{label}.sha256"),
        )
        if DEFAULT_ARTIFACT_ROOT not in file.path.parents:
            raise ValueError(
                "OBJ files must stay under the local artifact root"
            )
        asset_prim = _text(item["asset_prim_path"], f"{label}.asset prim")
        runtime_prim = _text(
            item["runtime_prim_path"], f"{label}.runtime prim"
        )
        if not asset_prim.startswith("/World/D38999Shell25JProxy/"):
            raise ValueError("asset prim mapping is outside the source proxy")
        if not runtime_prim.startswith(
            "/World/D38999TabletopV1/D38999Pair/"
        ):
            raise ValueError(
                "runtime prim mapping is outside the tabletop pair"
            )
        unique_key = _boolean(
            item["unique_polarization_key_present"],
            f"{label}.unique key",
        )
        qualified = _boolean(
            item["control_orientation_qualified"],
            f"{label}.orientation qualification",
        )
        if unique_key or qualified:
            raise ValueError(
                "proxy meshes cannot claim unique keyed orientation"
            )
        meshes[mesh_id] = MeshMapping(
            mesh_id=mesh_id,
            file=file,
            model_id=_text(item["model_id"], f"{label}.model_id"),
            asset_prim_path=asset_prim,
            runtime_prim_path=runtime_prim,
            frame_origin=_text(item["frame_origin"], f"{label}.origin"),
            source_components=_string_list(
                item["source_components"], f"{label}.components"
            ),
            rotational_symmetry_order=_positive_int(
                item["rotational_symmetry_order"], f"{label}.symmetry"
            ),
            unique_polarization_key_present=unique_key,
            control_orientation_qualified=qualified,
        )

    runtime = dict(_mapping(document["runtime"], "runtime"))
    _exact(
        runtime,
        {
            "preferred_isolation",
            "container_runtime_candidates",
            "required_executables_inside_environment",
            "required_ros_packages",
            "engine_outputs",
            "expected_engine_build_minutes",
            "expected_isolated_environment_setup_hours",
            "expected_first_sim_inference_hours_after_environment",
        },
        "runtime",
    )
    if _text(runtime["preferred_isolation"], "runtime isolation") != (
        "dedicated_ubuntu_24_04_isaac_ros_environment"
    ):
        raise ValueError("FoundationPose must stay in a dedicated environment")
    container_candidates = _string_list(
        runtime["container_runtime_candidates"], "container candidates"
    )
    if not {"docker", "podman"}.issubset(container_candidates):
        raise ValueError("container runtime candidates are incomplete")
    _string_list(
        runtime["required_executables_inside_environment"],
        "required executables",
    )
    required_packages = _string_list(
        runtime["required_ros_packages"], "required ROS packages"
    )
    if "isaac_ros_foundationpose" not in required_packages:
        raise ValueError("Isaac ROS FoundationPose package is required")
    engine_document = _mapping(runtime["engine_outputs"], "engine outputs")
    if set(engine_document) != {"refine", "score"}:
        raise ValueError("both TensorRT engine paths are required")
    engine_outputs = {
        name: _relative_path(path, f"engine_outputs.{name}")
        for name, path in engine_document.items()
    }
    runtime["container_runtime_candidates"] = container_candidates
    runtime["required_ros_packages"] = required_packages
    runtime["required_executables_inside_environment"] = _string_list(
        runtime["required_executables_inside_environment"],
        "required executables",
    )
    runtime.pop("engine_outputs")

    observability = dict(
        _mapping(document["observability"], "observability")
    )
    _exact(
        observability,
        {
            "required_inputs",
            "current_proxy_minimum_yaw_symmetry_order",
            "equivalent_yaw_period_rad",
            "unique_key_geometry_present",
            "unique_key_yaw_observable",
            "foundationpose_can_invent_missing_key_geometry",
        },
        "observability",
    )
    required_inputs = _string_list(
        observability["required_inputs"], "required pose inputs"
    )
    if set(required_inputs) != {
        "registered_rgb",
        "registered_depth",
        "camera_info",
        "instance_mask",
        "object_mesh",
    }:
        raise ValueError("FoundationPose input set is incomplete")
    if _positive_int(
        observability["current_proxy_minimum_yaw_symmetry_order"],
        "proxy yaw symmetry",
    ) < 2:
        raise ValueError("proxy yaw symmetry must remain explicit")
    if not math.isclose(
        float(observability["equivalent_yaw_period_rad"]),
        math.pi,
        abs_tol=1.0e-15,
    ):
        raise ValueError("proxy equivalent yaw period must remain pi")
    for key in (
        "unique_key_geometry_present",
        "unique_key_yaw_observable",
        "foundationpose_can_invent_missing_key_geometry",
    ):
        if _boolean(observability[key], f"observability.{key}"):
            raise ValueError(f"{key} must remain false")
    observability["required_inputs"] = required_inputs

    boundaries_document = _mapping(document["boundaries"], "boundaries")
    expected_boundaries = {
        "host_package_install_allowed",
        "gpu_execution_performed",
        "tensorrt_engine_build_performed",
        "foundationpose_inference_performed",
        "e2e_runner_integration_allowed",
        "active_usd_modification_allowed",
        "full_6d_pose_claimed",
        "unique_key_yaw_claimed",
        "vision_control_authorized",
        "real_assembly_success_claimed",
    }
    if set(boundaries_document) != expected_boundaries:
        raise ValueError("boundary set differs")
    boundaries = {
        name: _boolean(value, f"boundaries.{name}")
        for name, value in boundaries_document.items()
    }
    if any(boundaries.values()):
        raise ValueError("all bootstrap claim/action boundaries must be false")

    return FoundationPoseBootstrapContract(
        schema_version=schema,
        enabled=enabled,
        status=_text(document["status"], "status"),
        inputs=inputs,
        official_release=release,
        official_sources=official_sources,
        model_version=model_version,
        model_license=_text(model["license"], "model license"),
        model_storage_scope=storage_scope,
        models=models,
        meshes=meshes,
        radial_sections=radial_sections,
        engine_outputs=engine_outputs,
        runtime=runtime,
        observability=observability,
        boundaries=boundaries,
    )


class _ObjBuilder:
    """Minimal deterministic triangle-only OBJ builder."""

    def __init__(self, object_name: str):
        self.object_name = object_name
        self.vertices: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]] = []

    def _vertices(
        self, points: Sequence[tuple[float, float, float]]
    ) -> tuple[int, ...]:
        first = len(self.vertices) + 1
        self.vertices.extend(points)
        return tuple(range(first, first + len(points)))

    def _quad(self, a: int, b: int, c: int, d: int) -> None:
        self.faces.append((a, b, c))
        self.faces.append((a, c, d))

    def add_box(
        self,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        rotation_z_rad: float = 0.0,
    ) -> None:
        cx, cy, cz = center
        sx, sy, sz = (0.5 * value for value in size)
        cosine = math.cos(rotation_z_rad)
        sine = math.sin(rotation_z_rad)
        points = []
        for z in (-sz, sz):
            for y in (-sy, sy):
                for x in (-sx, sx):
                    points.append(
                        (
                            cx + cosine * x - sine * y,
                            cy + sine * x + cosine * y,
                            cz + z,
                        )
                    )
        v = self._vertices(points)
        # Indices per z layer: (-y,-x), (-y,+x), (+y,-x), (+y,+x).
        self._quad(v[0], v[1], v[3], v[2])
        self._quad(v[4], v[6], v[7], v[5])
        self._quad(v[0], v[4], v[5], v[1])
        self._quad(v[2], v[3], v[7], v[6])
        self._quad(v[0], v[2], v[6], v[4])
        self._quad(v[1], v[5], v[7], v[3])

    def add_cylinder(
        self,
        radius: float,
        z_min: float,
        z_max: float,
        sections: int,
        *,
        center_xy: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        cx, cy = center_xy
        rings = []
        for z in (z_min, z_max):
            rings.append(
                self._vertices(
                    [
                        (
                            cx
                            + radius
                            * math.cos(2.0 * math.pi * i / sections),
                            cy
                            + radius
                            * math.sin(2.0 * math.pi * i / sections),
                            z,
                        )
                        for i in range(sections)
                    ]
                )
            )
        bottom, top = rings
        bottom_center, top_center = self._vertices(
            [(cx, cy, z_min), (cx, cy, z_max)]
        )
        for i in range(sections):
            j = (i + 1) % sections
            self._quad(bottom[i], bottom[j], top[j], top[i])
            self.faces.append((bottom_center, bottom[j], bottom[i]))
            self.faces.append((top_center, top[i], top[j]))

    def document(self) -> bytes:
        lines = [
            "# kcg deterministic D38999 FoundationPose proxy OBJ",
            "# units: metre; up axis: +Z; mating axis: +Z",
            f"o {self.object_name}",
        ]
        lines.extend(
            f"v {x:.9f} {y:.9f} {z:.9f}"
            for x, y, z in self.vertices
        )
        lines.extend(f"f {a} {b} {c}" for a, b, c in self.faces)
        return ("\n".join(lines) + "\n").encode("ascii")


def _contact_positions(count: int) -> tuple[tuple[float, float], ...]:
    """Match the USD generator's visual-only 1+6+12+18+24 layout."""

    if count != 61:
        raise ValueError("the simplified OBJ supports the v1 61-contact proxy")
    positions = [(0.0, 0.0)]
    for ring_index, ring_count in enumerate((6, 12, 18, 24), start=1):
        radius_fraction = 0.20 * ring_index
        offset = 0.5 * math.pi / ring_count
        for index in range(ring_count):
            angle = 2.0 * math.pi * index / ring_count + offset
            positions.append(
                (
                    radius_fraction * math.cos(angle),
                    radius_fraction * math.sin(angle),
                )
            )
    return tuple(positions)


def _add_ring_segments(
    builder: _ObjBuilder,
    *,
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    count: int,
) -> None:
    radius = 0.5 * (inner_radius + outer_radius)
    radial_size = outer_radius - inner_radius
    tangential_size = 0.91 * 2.0 * math.pi * radius / count
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        builder.add_box(
            (
                radius * math.cos(angle),
                radius * math.sin(angle),
                0.5 * (z_min + z_max),
            ),
            (radial_size, tangential_size, z_max - z_min),
            angle,
        )


def build_proxy_mesh_documents(
    proxy: D38999Shell25JProxy,
    *,
    radial_sections: int = 48,
) -> dict[str, bytes]:
    """Generate three rigid-object meshes without opening or editing USD."""

    if radial_sections < 24 or radial_sections % 24:
        raise ValueError("radial_sections must be a multiple of 24")
    plug = proxy.plug_geometry_m
    receptacle = proxy.receptacle_geometry_m
    contacts = _contact_positions(plug.contact_count)

    loose = _ObjBuilder("d38999_26kj61sn_loose_body_proxy_v1")
    loose.add_cylinder(
        plug.rear_body_radius,
        plug.overall_length - plug.rear_body_length,
        plug.overall_length,
        radial_sections,
    )
    _add_ring_segments(
        loose,
        inner_radius=plug.mating_shell_inner_radius,
        outer_radius=plug.mating_shell_outer_radius,
        z_min=0.0,
        z_max=plug.mating_shell_length,
        count=20,
    )
    loose.add_cylinder(
        plug.contact_face_radius, 0.0, 0.0010, radial_sections
    )
    contact_scale = 0.90 * plug.contact_face_radius
    for x_fraction, y_fraction in contacts:
        loose.add_cylinder(
            plug.contact_visual_radius,
            -0.00035,
            0.00115,
            12,
            center_xy=(
                contact_scale * x_fraction,
                contact_scale * y_fraction,
            ),
        )

    nut = _ObjBuilder("d38999_26kj61sn_coupling_nut_proxy_v1")
    nut_center_z = 0.5 * plug.overall_length
    _add_ring_segments(
        nut,
        inner_radius=plug.coupling_nut_inner_radius,
        outer_radius=plug.coupling_nut_outer_radius,
        z_min=nut_center_z - 0.5 * plug.coupling_nut_length,
        z_max=nut_center_z + 0.5 * plug.coupling_nut_length,
        count=plug.grip_segment_count,
    )

    fixed = _ObjBuilder("d38999_20kj61pn_fixed_receptacle_proxy_v1")
    fixed.add_box(
        (0.0, 0.0, -0.5 * receptacle.flange_thickness),
        (
            receptacle.flange_side,
            receptacle.flange_side,
            receptacle.flange_thickness,
        ),
    )
    _add_ring_segments(
        fixed,
        inner_radius=receptacle.entry_radius,
        outer_radius=receptacle.shell_outer_radius,
        z_min=0.0,
        z_max=receptacle.front_shell_length,
        count=20,
    )
    fixed.add_cylinder(
        receptacle.rear_body_radius,
        -receptacle.rear_body_length,
        0.0,
        radial_sections,
    )
    fixed.add_cylinder(
        receptacle.contact_face_radius, 0.0, 0.0010, radial_sections
    )
    pin_scale = 0.90 * receptacle.contact_face_radius
    for x_fraction, y_fraction in contacts:
        fixed.add_cylinder(
            receptacle.pin_visual_radius,
            0.0010,
            0.0010 + receptacle.pin_visual_length,
            12,
            center_xy=(
                pin_scale * x_fraction,
                pin_scale * y_fraction,
            ),
        )

    return {
        "loose_body": loose.document(),
        "coupling_nut": nut.document(),
        "fixed_receptacle": fixed.document(),
    }


def validate_obj_document(document: bytes) -> MeshStats:
    """Validate the deterministic triangle subset accepted by this contract."""

    try:
        text = document.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("OBJ must be ASCII") from exc
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    object_count = 0
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "o" and len(fields) == 2:
            object_count += 1
        elif fields[0] == "v" and len(fields) == 4:
            point = tuple(float(value) for value in fields[1:])
            if not all(math.isfinite(value) for value in point):
                raise ValueError(
                    f"non-finite OBJ vertex at line {line_number}"
                )
            vertices.append(point)  # type: ignore[arg-type]
        elif fields[0] == "f" and len(fields) == 4:
            try:
                face = tuple(int(value) for value in fields[1:])
            except ValueError as exc:
                raise ValueError(
                    f"invalid OBJ face at line {line_number}"
                ) from exc
            faces.append(face)  # type: ignore[arg-type]
        else:
            raise ValueError(f"unsupported OBJ record at line {line_number}")
    if object_count != 1 or not vertices or not faces:
        raise ValueError("OBJ must contain one non-empty triangle object")
    for face in faces:
        if len(set(face)) != 3 or min(face) < 1 or max(face) > len(vertices):
            raise ValueError("OBJ face index is invalid")
    mins = tuple(min(point[i] for point in vertices) for i in range(3))
    maxes = tuple(max(point[i] for point in vertices) for i in range(3))
    if any(high <= low for low, high in zip(mins, maxes)):
        raise ValueError("OBJ bounds must have positive extent")
    return MeshStats(
        vertex_count=len(vertices),
        triangle_count=len(faces),
        bounds_min_xyz_m=mins,
        bounds_max_xyz_m=maxes,
    )


def generate_and_verify_meshes(
    contract: FoundationPoseBootstrapContract,
    repository: str | Path,
) -> dict[str, MeshStats]:
    """Write only missing, exactly pinned OBJ artifacts.

    Existing mismatched files are never overwritten.  The active USD is read
    only indirectly through its hash contract; geometry comes from the proxy
    YAML so this operation cannot mutate the simulation asset.
    """

    root = Path(repository).expanduser().resolve()
    proxy = load_d38999_shell25j_proxy(
        root / contract.inputs["proxy_config"].path
    )
    documents = build_proxy_mesh_documents(
        proxy, radial_sections=contract.radial_sections
    )
    stats: dict[str, MeshStats] = {}
    for mesh_id, mapping in contract.meshes.items():
        document = documents[mesh_id]
        actual = hashlib.sha256(document).hexdigest()
        if actual != mapping.file.sha256:
            raise ValueError(
                f"generated {mesh_id} hash {actual} differs from contract "
                f"{mapping.file.sha256}"
            )
        stats[mesh_id] = validate_obj_document(document)
        output = root / mapping.file.path
        if output.exists():
            if not output.is_file() or sha256_file(output) != actual:
                raise FileExistsError(
                    f"refusing to overwrite mismatched mesh: {output}"
                )
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(document)
    return stats


def _verify_file(
    repository: Path, item: ContentAddressedFile
) -> dict[str, Any]:
    path = repository / item.path
    evidence: dict[str, Any] = {
        "path": item.path.as_posix(),
        "exists": path.is_file(),
        "expected_sha256": item.sha256,
        "actual_sha256": None,
        "expected_size_bytes": item.size_bytes,
        "actual_size_bytes": None,
        "verified": False,
    }
    if not path.is_file():
        return evidence
    evidence["actual_size_bytes"] = path.stat().st_size
    evidence["actual_sha256"] = sha256_file(path)
    size_ok = item.size_bytes is None or path.stat().st_size == item.size_bytes
    evidence["verified"] = size_ok and evidence["actual_sha256"] == item.sha256
    return evidence


def _os_release() -> dict[str, str]:
    result: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"')
    return result


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    result: dict[str, Any] = {
        "available": executable is not None,
        "gpus": [],
        "error": None,
    }
    if executable is None:
        return result
    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        for line in completed.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) != 4:
                raise ValueError("unexpected nvidia-smi field count")
            result["gpus"].append(
                {
                    "name": fields[0],
                    "memory_mib": int(fields[1]),
                    "driver_version": fields[2],
                    "compute_capability": float(fields[3]),
                }
            )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def probe_host_runtime(
    contract: FoundationPoseBootstrapContract,
) -> dict[str, Any]:
    """Collect host evidence without starting a container or GPU job."""

    os_info = _os_release()
    executables = {
        name: shutil.which(name)
        for name in (
            "docker",
            "podman",
            "trtexec",
            "isaac-ros",
            "ros2",
        )
    }
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("tensorrt", "onnx", "onnxruntime")
    }
    ros_package_available = False
    if executables["ros2"]:
        try:
            subprocess.run(
                [
                    executables["ros2"],
                    "pkg",
                    "prefix",
                    "isaac_ros_foundationpose",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            ros_package_available = True
        except (OSError, subprocess.SubprocessError):
            pass
    gpu = _nvidia_smi()
    release = contract.official_release
    hardware_pass = any(
        item["memory_mib"] >= 1024.0 * float(release["minimum_gpu_memory_gb"])
        and item["compute_capability"]
        >= float(release["minimum_compute_capability"])
        and int(item["driver_version"].split(".", 1)[0])
        >= int(release["minimum_driver_major"])
        for item in gpu["gpus"]
    )
    host_os_pass = (
        os_info.get("ID") == "ubuntu"
        and os_info.get("VERSION_ID") == "24.04"
        and platform.machine() == "x86_64"
    )
    return {
        "probe_is_read_only": True,
        "os": {
            "id": os_info.get("ID"),
            "version_id": os_info.get("VERSION_ID"),
            "pretty_name": os_info.get("PRETTY_NAME"),
            "architecture": platform.machine(),
            "official_release_4_5_x86_host_supported": host_os_pass,
        },
        "executables": executables,
        "python_modules": modules,
        "isaac_ros_foundationpose_package_available": ros_package_available,
        "gpu": gpu,
        "official_gpu_driver_memory_gate_passed": hardware_pass,
        "container_runtime_available": any(
            executables[name] is not None for name in ("docker", "podman")
        ),
        "host_tensorrt_available": (
            executables["trtexec"] is not None or modules["tensorrt"]
        ),
    }


def evaluate_foundationpose_readiness(
    contract: FoundationPoseBootstrapContract,
    repository: str | Path,
) -> dict[str, Any]:
    """Return evidence and blockers without claiming that inference ran."""

    root = Path(repository).expanduser().resolve()
    input_evidence = {
        name: _verify_file(root, item)
        for name, item in contract.inputs.items()
    }
    model_evidence = {
        name: _verify_file(root, item)
        for name, item in contract.models.items()
    }
    mesh_evidence: dict[str, dict[str, Any]] = {}
    for name, mapping in contract.meshes.items():
        evidence = _verify_file(root, mapping.file)
        evidence.update(
            {
                "model_id": mapping.model_id,
                "asset_prim_path": mapping.asset_prim_path,
                "runtime_prim_path": mapping.runtime_prim_path,
                "frame_origin": mapping.frame_origin,
                "rotational_symmetry_order": (
                    mapping.rotational_symmetry_order
                ),
                "unique_polarization_key_present": False,
                "control_orientation_qualified": False,
                "mesh_stats": None,
            }
        )
        path = root / mapping.file.path
        if evidence["verified"]:
            stats = validate_obj_document(path.read_bytes())
            evidence["mesh_stats"] = {
                "vertex_count": stats.vertex_count,
                "triangle_count": stats.triangle_count,
                "bounds_min_xyz_m": list(stats.bounds_min_xyz_m),
                "bounds_max_xyz_m": list(stats.bounds_max_xyz_m),
            }
        mesh_evidence[name] = evidence

    engines = {
        name: {
            "path": path.as_posix(),
            "exists": (root / path).is_file(),
        }
        for name, path in contract.engine_outputs.items()
    }
    host = probe_host_runtime(contract)
    inputs_verified = all(item["verified"] for item in input_evidence.values())
    models_verified = all(item["verified"] for item in model_evidence.values())
    meshes_verified = all(item["verified"] for item in mesh_evidence.values())
    artifact_bundle_verified = (
        inputs_verified and models_verified and meshes_verified
    )
    runtime_environment_ready = all(
        (
            host["os"]["official_release_4_5_x86_host_supported"],
            host["official_gpu_driver_memory_gate_passed"],
            host["container_runtime_available"],
            host["isaac_ros_foundationpose_package_available"],
            all(item["exists"] for item in engines.values()),
        )
    )

    blockers: list[str] = []
    if not inputs_verified:
        blockers.append("source_proxy_or_scene_hash_mismatch")
    if not models_verified:
        blockers.append("official_onnx_bundle_missing_or_hash_mismatch")
    if not meshes_verified:
        blockers.append("proxy_obj_bundle_missing_or_hash_mismatch")
    if not host["os"]["official_release_4_5_x86_host_supported"]:
        blockers.append("host_ubuntu_22_04_outside_release_4_5_support_matrix")
    if not host["official_gpu_driver_memory_gate_passed"]:
        blockers.append("gpu_driver_memory_or_compute_gate_failed")
    if not host["container_runtime_available"]:
        blockers.append("isolated_container_runtime_unavailable")
    if not host["isaac_ros_foundationpose_package_available"]:
        blockers.append("isaac_ros_foundationpose_runtime_unavailable")
    if not all(item["exists"] for item in engines.values()):
        blockers.append("tensorrt_engine_plans_not_built")
    blockers.extend(
        (
            "registered_rgb_depth_camera_info_mask_bridge_not_qualified",
            "proxy_unique_polarization_key_geometry_absent",
            "yaw_has_at_least_180_degree_equivalent_hypothesis",
            "bootstrap_contract_disabled_and_not_e2e_integrated",
        )
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "ARTIFACTS_VERIFIED_RUNTIME_BLOCKED"
            if artifact_bundle_verified
            else "ARTIFACT_BUNDLE_INVALID"
        ),
        "config_enabled": False,
        "model_version": contract.model_version,
        "model_license": contract.model_license,
        "model_storage_scope": contract.model_storage_scope,
        "official_sources": contract.official_sources,
        "inputs": input_evidence,
        "models": model_evidence,
        "meshes": mesh_evidence,
        "engines": engines,
        "host_runtime": host,
        "gates": {
            "artifact_bundle_verified": artifact_bundle_verified,
            "runtime_environment_ready": runtime_environment_ready,
            "foundationpose_inference_ready": False,
            "full_6d_keyed_pose_observable": False,
            "vision_control_authorized": False,
        },
        "blockers": blockers,
        "claims": {
            "gpu_execution_performed": False,
            "tensorrt_engine_build_performed": False,
            "foundationpose_inference_performed": False,
            "full_6d_pose_claimed": False,
            "unique_key_yaw_claimed": False,
            "vision_control_authorized": False,
            "real_assembly_success_claimed": False,
        },
        "next_isolated_step": {
            "requires_user_authority_for_host_or_separate_supported_machine": (
                True
            ),
            "environment": "Ubuntu 24.04 x86_64 + Isaac ROS release-4.5/Jazzy",
            "sequence": [
                (
                    "provide a supported isolated Isaac ROS environment and "
                    "container runtime"
                ),
                (
                    "mount this gitignored artifact bundle read-only into "
                    "ISAAC_ROS_WS"
                ),
                (
                    "build both FP32 TensorRT plans with the official "
                    "trtexec shapes"
                ),
                (
                    "run the official rosbag quickstart before connecting "
                    "simulation RGB-D"
                ),
                "add registered RGB/depth/camera_info/instance-mask bridge",
                (
                    "evaluate pose modulo declared symmetry against withheld "
                    "sim truth"
                ),
                (
                    "add measured unique key geometry before any keyed-yaw "
                    "control claim"
                ),
            ],
            "official_refine_trtexec": (
                "/usr/src/tensorrt/bin/trtexec --onnx=refine_model.onnx "
                "--saveEngine=refine_trt_engine.plan "
                "--minShapes=input1:1x160x160x6,input2:1x160x160x6 "
                "--optShapes=input1:1x160x160x6,input2:1x160x160x6 "
                "--maxShapes=input1:42x160x160x6,input2:42x160x160x6"
            ),
            "official_score_trtexec": (
                "/usr/src/tensorrt/bin/trtexec --onnx=score_model.onnx "
                "--saveEngine=score_trt_engine.plan "
                "--minShapes=input1:1x160x160x6,input2:1x160x160x6 "
                "--optShapes=input1:1x160x160x6,input2:1x160x160x6 "
                "--maxShapes=input1:252x160x160x6,input2:252x160x160x6"
            ),
            "estimated_engine_build_minutes": contract.runtime[
                "expected_engine_build_minutes"
            ],
            "estimated_environment_setup_hours": contract.runtime[
                "expected_isolated_environment_setup_hours"
            ],
            "estimated_first_sim_inference_hours_after_environment": (
                contract.runtime[
                    "expected_first_sim_inference_hours_after_environment"
                ]
            ),
        },
    }
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the disabled D38999 FoundationPose bootstrap"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--repository", default=str(Path(__file__).resolve().parents[3])
    )
    parser.add_argument(
        "--generate-meshes",
        action="store_true",
        help="write only missing meshes after their generated hashes match",
    )
    parser.add_argument(
        "--report", help="optional new or existing JSON report"
    )
    parser.add_argument(
        "--require",
        choices=("artifacts", "runtime", "vision_control"),
        default="runtime",
        help="gate that determines the process exit code",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    repository = Path(arguments.repository).expanduser().resolve()
    contract = load_foundationpose_bootstrap_contract(arguments.config)
    if arguments.generate_meshes:
        generate_and_verify_meshes(contract, repository)
    report = evaluate_foundationpose_readiness(contract, repository)
    if arguments.report:
        output = Path(arguments.report).expanduser().resolve()
        if output.exists() and not output.is_file():
            raise FileExistsError(f"report path is not a file: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    gate = {
        "artifacts": report["gates"]["artifact_bundle_verified"],
        "runtime": report["gates"]["runtime_environment_ready"],
        "vision_control": report["gates"]["vision_control_authorized"],
    }[arguments.require]
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
