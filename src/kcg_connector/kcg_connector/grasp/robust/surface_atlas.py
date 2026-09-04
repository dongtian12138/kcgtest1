"""Frozen STEP contact charts for the continuous robust-grasp study.

The atlas is intentionally separate from the historical STL object model.  It
separates proven searchable, proven functionally forbidden, and unresolved
STEP faces.  Unresolved faces remain in the outer search domain; absence from
the proven-searchable list is never treated as a physical prohibition.  The
atlas does not certify collision clearance, dynamic contact, exact B-rep
distance, or grasp success.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from kcg_connector.grasp.robust.object_model import meters_per_unit


SCHEMA_VERSION = "kcg_step_contact_atlas_v2"
CLAIM_SCOPE = (
    "SIMULATION_ONLY_RESEARCH_CONTACT_CONTRACT_NOT_VENDOR_OR_HARDWARE_FACT"
)
ATLAS_HASH_DOMAIN = b"KCG_STEP_CONTACT_ATLAS_V2\0"
EXPECTED_DEVELOPMENT_OBJECT = "te_deutsch_d38999_26fj35pn_step"
EXPECTED_SEARCHABLE_LABEL = "PROVEN_SEARCHABLE_EXTERNAL_STRUCTURAL_SURFACE"
EXPECTED_FORBIDDEN_LABEL = "PROVEN_FUNCTIONAL_HARD_FORBIDDEN_SURFACE"
EXPECTED_UNRESOLVED_LABEL = "UNRESOLVED_INCLUDED_IN_OUTER_SEARCH_DOMAIN"


class SurfaceAtlasError(ValueError):
    """Raised when the frozen contact-chart evidence cannot be reconstructed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise SurfaceAtlasError("YAML keys must be scalar and hashable") from error
        if duplicate:
            raise SurfaceAtlasError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise SurfaceAtlasError(f"{label} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    missing = sorted(set(expected).difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise SurfaceAtlasError(
            f"{label} schema mismatch; missing={missing}, extra={extra}"
        )


def _exact_bool(value: Any, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise SurfaceAtlasError(f"{label} must be exactly {expected}")
    return value


def _positive_finite(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise SurfaceAtlasError(f"{label} must be finite and positive")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_file(root: Path, relative: Any, label: str) -> Path:
    raw = Path(str(relative))
    if raw.is_absolute():
        raise SurfaceAtlasError(f"{label} must be repository-relative")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SurfaceAtlasError(f"{label} escapes the repository") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {path}")
    return path


def _hex_digest(value: Any, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise SurfaceAtlasError(f"{label} must be one SHA-256 digest")
    return digest


def _one_based_face_indices(
    value: Any, label: str, *, allow_empty: bool = False
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SurfaceAtlasError(f"{label} must be a sequence")
    result = tuple(int(item) for item in value)
    if (not result and not allow_empty) or any(item < 1 for item in result):
        raise SurfaceAtlasError(f"{label} must contain positive one-based indices")
    if len(set(result)) != len(result) or tuple(sorted(result)) != result:
        raise SurfaceAtlasError(f"{label} must be unique and sorted")
    return result


def _readonly(value: Any, dtype: str, shape_tail: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim < len(shape_tail) or (
        shape_tail and tuple(array.shape[-len(shape_tail) :]) != shape_tail
    ):
        raise SurfaceAtlasError(f"{label} must end in shape {shape_tail}")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise SurfaceAtlasError(f"{label} contains a non-finite value")
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def _atlas_sha256(
    source_step_sha256: str,
    triangles_m: np.ndarray,
    triangle_uv: np.ndarray,
    parent_face_index: np.ndarray,
    parent_triangle_index: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(ATLAS_HASH_DOMAIN)
    digest.update(source_step_sha256.encode("ascii"))
    arrays = (
        ("triangles_m", np.asarray(triangles_m, dtype="<f8")),
        ("triangle_uv", np.asarray(triangle_uv, dtype="<f8")),
        ("parent_face_index", np.asarray(parent_face_index, dtype="<i8")),
        ("parent_triangle_index", np.asarray(parent_triangle_index, dtype="<i8")),
    )
    for name, array in arrays:
        canonical = np.ascontiguousarray(array)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(struct.pack("<I", canonical.ndim))
        digest.update(struct.pack("<" + "Q" * canonical.ndim, *canonical.shape))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class StepContactAtlasContract:
    contract_path: Path
    repository_root: Path
    development_object: str
    source_step_path: Path
    source_step_sha256: str
    source_unit: str
    expected_solid_count: int
    expected_face_count: int
    ocp_package: str
    ocp_package_version: str
    linear_deflection_mm: float
    angular_deflection_rad: float
    relative: bool
    parallel: bool
    proven_searchable_parent_faces: tuple[int, ...]
    hard_forbidden_parent_faces: tuple[int, ...]
    searchable_label: str
    forbidden_label: str
    unresolved_label: str
    expected_allowed_brep_area_mm2: float
    expected_allowed_triangle_count: int
    expected_allowed_atlas_sha256: str
    mesh_clearance_mm: float
    numerical_margin_mm: float
    total_clearance_mm: float

    @property
    def allowed_parent_faces(self) -> tuple[int, ...]:
        """Outer-domain faces: proven searchable plus every unresolved face."""

        hard = frozenset(self.hard_forbidden_parent_faces)
        return tuple(
            index
            for index in range(1, self.expected_face_count + 1)
            if index not in hard
        )

    @property
    def unresolved_parent_faces(self) -> tuple[int, ...]:
        classified = frozenset(
            self.proven_searchable_parent_faces + self.hard_forbidden_parent_faces
        )
        return tuple(
            index
            for index in range(1, self.expected_face_count + 1)
            if index not in classified
        )

    @property
    def forbidden_parent_faces(self) -> tuple[int, ...]:
        return self.hard_forbidden_parent_faces


@dataclass(frozen=True)
class StepTriangleRolePartition:
    """Frozen STEP tessellation separated by functional role.

    This is an explicit collision-query view, not an exact B-rep distance
    certificate.  In particular, ``unresolved`` is not promoted to searchable
    and a collision on a shared tessellation boundary still requires an exact
    local review.
    """

    contract: StepContactAtlasContract
    proven_searchable_triangles_m: np.ndarray
    proven_searchable_parent_face_index: np.ndarray
    unresolved_triangles_m: np.ndarray
    unresolved_parent_face_index: np.ndarray
    hard_forbidden_triangles_m: np.ndarray
    hard_forbidden_parent_face_index: np.ndarray

    def __post_init__(self) -> None:
        roles = (
            (
                "proven_searchable",
                self.proven_searchable_triangles_m,
                self.proven_searchable_parent_face_index,
                frozenset(self.contract.proven_searchable_parent_faces),
            ),
            (
                "unresolved",
                self.unresolved_triangles_m,
                self.unresolved_parent_face_index,
                frozenset(self.contract.unresolved_parent_faces),
            ),
            (
                "hard_forbidden",
                self.hard_forbidden_triangles_m,
                self.hard_forbidden_parent_face_index,
                frozenset(self.contract.hard_forbidden_parent_faces),
            ),
        )
        for label, triangle_value, parent_value, expected_parents in roles:
            triangles = np.ascontiguousarray(triangle_value, dtype=np.float64)
            parents = np.ascontiguousarray(parent_value, dtype=np.int64)
            if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
                raise SurfaceAtlasError(f"{label} STEP triangles must have shape (N,3,3)")
            if parents.shape != (len(triangles),):
                raise SurfaceAtlasError(f"{label} STEP parent coverage changed")
            if not len(triangles) or not np.all(np.isfinite(triangles)):
                raise SurfaceAtlasError(f"{label} STEP triangle set is empty or non-finite")
            if not set(map(int, np.unique(parents))).issubset(expected_parents):
                raise SurfaceAtlasError(f"{label} STEP triangles contain a wrong-role parent")
            triangles.setflags(write=False)
            parents.setflags(write=False)
            object.__setattr__(self, f"{label}_triangles_m", triangles)
            object.__setattr__(self, f"{label}_parent_face_index", parents)
        nonhard_count = len(self.proven_searchable_triangles_m) + len(
            self.unresolved_triangles_m
        )
        if nonhard_count != self.contract.expected_allowed_triangle_count:
            raise SurfaceAtlasError(
                "non-hard STEP triangle count differs from the frozen atlas contract"
            )


@dataclass(frozen=True)
class ParentSurfaceFrame:
    """Analytic carrier and outward-normal convention for one allowed STEP face."""

    face_index: int
    kind: str
    origin_m: tuple[float, float, float]
    x_direction: tuple[float, float, float]
    y_direction: tuple[float, float, float]
    axis_direction: tuple[float, float, float]
    radius_m: float | None
    outward_sign: float
    uv_length_scale_m: float

    def __post_init__(self) -> None:
        if self.face_index < 1 or self.kind not in ("plane", "cylinder"):
            raise SurfaceAtlasError("parent surface identity is invalid")
        origin = np.asarray(self.origin_m, dtype=np.float64)
        x_direction = np.asarray(self.x_direction, dtype=np.float64)
        y_direction = np.asarray(self.y_direction, dtype=np.float64)
        axis_direction = np.asarray(self.axis_direction, dtype=np.float64)
        if any(value.shape != (3,) for value in (origin, x_direction, y_direction, axis_direction)):
            raise SurfaceAtlasError("parent surface frame vectors must have shape (3,)")
        if not all(
            np.all(np.isfinite(value))
            for value in (origin, x_direction, y_direction, axis_direction)
        ):
            raise SurfaceAtlasError("parent surface frame contains a non-finite value")
        frame = np.column_stack((x_direction, y_direction, axis_direction))
        if not np.allclose(frame.T @ frame, np.eye(3), rtol=0.0, atol=1.0e-12):
            raise SurfaceAtlasError("parent surface frame is not orthonormal")
        if float(np.linalg.det(frame)) < 1.0 - 1.0e-12:
            raise SurfaceAtlasError("parent surface frame is not right handed")
        radius = self.radius_m
        if self.kind == "plane" and radius is not None:
            raise SurfaceAtlasError("a planar parent surface cannot have a radius")
        if self.kind == "cylinder" and (
            radius is None or not math.isfinite(radius) or radius <= 0.0
        ):
            raise SurfaceAtlasError("a cylindrical parent surface needs a positive radius")
        if self.outward_sign not in (-1.0, 1.0):
            raise SurfaceAtlasError("parent outward sign must be exactly -1 or +1")
        if not math.isfinite(self.uv_length_scale_m) or self.uv_length_scale_m <= 0.0:
            raise SurfaceAtlasError("parent UV length scale must be finite and positive")
        object.__setattr__(self, "origin_m", tuple(float(value) for value in origin))
        object.__setattr__(
            self, "x_direction", tuple(float(value) for value in x_direction)
        )
        object.__setattr__(
            self, "y_direction", tuple(float(value) for value in y_direction)
        )
        object.__setattr__(
            self, "axis_direction", tuple(float(value) for value in axis_direction)
        )

    def point_from_uv(self, u: float, v: float) -> np.ndarray:
        """Evaluate the exact plane/cylinder carrier in the frozen object frame."""

        u_value, v_value = float(u), float(v)
        if not math.isfinite(u_value) or not math.isfinite(v_value):
            raise SurfaceAtlasError("parent surface UV must be finite")
        origin = np.asarray(self.origin_m)
        x_direction = np.asarray(self.x_direction)
        y_direction = np.asarray(self.y_direction)
        axis_direction = np.asarray(self.axis_direction)
        if self.kind == "plane":
            point = origin + self.uv_length_scale_m * (
                u_value * x_direction + v_value * y_direction
            )
        else:
            assert self.radius_m is not None
            point = (
                origin
                + self.radius_m
                * (math.cos(u_value) * x_direction + math.sin(u_value) * y_direction)
                + self.uv_length_scale_m * v_value * axis_direction
            )
        point.setflags(write=False)
        return point

    def normal_from_uv(self, u: float, v: float) -> np.ndarray:
        """Return the exact outward unit normal; ``v`` is accepted for one API."""

        del v
        if self.kind == "plane":
            normal = self.outward_sign * np.asarray(self.axis_direction)
        else:
            u_value = float(u)
            if not math.isfinite(u_value):
                raise SurfaceAtlasError("parent cylinder U must be finite")
            normal = self.outward_sign * (
                math.cos(u_value) * np.asarray(self.x_direction)
                + math.sin(u_value) * np.asarray(self.y_direction)
            )
        normal = np.asarray(normal, dtype=np.float64)
        normal.setflags(write=False)
        return normal


@dataclass(frozen=True)
class StepContactAtlas:
    """Continuous barycentric charts for every frozen allowed STEP triangle."""

    contract: StepContactAtlasContract
    triangles_m: np.ndarray
    triangle_uv: np.ndarray
    parent_face_index: np.ndarray
    parent_triangle_index: np.ndarray
    parent_surfaces: tuple[ParentSurfaceFrame, ...]
    allowed_brep_area_m2: float
    atlas_sha256: str

    def __post_init__(self) -> None:
        triangles = _readonly(self.triangles_m, "<f8", (3, 3), "triangles_m")
        uv = _readonly(self.triangle_uv, "<f8", (3, 2), "triangle_uv")
        parents = _readonly(self.parent_face_index, "<i8", (), "parent_face_index")
        local = _readonly(
            self.parent_triangle_index, "<i8", (), "parent_triangle_index"
        )
        count = int(triangles.shape[0])
        if uv.shape[0] != count or parents.shape != (count,) or local.shape != (count,):
            raise SurfaceAtlasError("atlas arrays do not have a shared triangle count")
        if count != self.contract.expected_allowed_triangle_count:
            raise SurfaceAtlasError(
                "allowed triangle count differs from the contract: "
                f"{count} != {self.contract.expected_allowed_triangle_count}"
            )
        if not set(int(v) for v in parents).issubset(
            self.contract.allowed_parent_faces
        ):
            raise SurfaceAtlasError("atlas contains a forbidden parent face")
        frames = tuple(self.parent_surfaces)
        frame_by_face = {frame.face_index: frame for frame in frames}
        if len(frame_by_face) != len(frames) or not set(frame_by_face).issubset(
            self.contract.allowed_parent_faces
        ):
            raise SurfaceAtlasError(
                "analytic parent surfaces are duplicated or outside the outer domain"
            )
        areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        if np.any(areas <= 0.0):
            raise SurfaceAtlasError("atlas contains a degenerate triangle")
        maximum_carrier_residual_m = 0.0
        minimum_normal_alignment = 1.0
        triangle_normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        triangle_normals /= np.linalg.norm(triangle_normals, axis=1)[:, None]
        for index, face_index in enumerate(parents):
            frame = frame_by_face.get(int(face_index))
            if frame is None:
                continue
            exact_vertices = np.asarray(
                [frame.point_from_uv(*coordinates) for coordinates in uv[index]]
            )
            maximum_carrier_residual_m = max(
                maximum_carrier_residual_m,
                float(np.max(np.linalg.norm(exact_vertices - triangles[index], axis=1))),
            )
            uv_center = np.mean(uv[index], axis=0)
            exact_normal = frame.normal_from_uv(*uv_center)
            minimum_normal_alignment = min(
                minimum_normal_alignment,
                float(triangle_normals[index] @ exact_normal),
            )
        if maximum_carrier_residual_m > 1.0e-8:
            raise SurfaceAtlasError(
                "atlas UV does not reproduce its analytic parent carrier within 10 nm"
            )
        if minimum_normal_alignment < math.cos(self.contract.angular_deflection_rad + 1.0e-9):
            raise SurfaceAtlasError("atlas winding disagrees with analytic parent normals")
        digest = _atlas_sha256(
            self.contract.source_step_sha256, triangles, uv, parents, local
        )
        if digest != self.contract.expected_allowed_atlas_sha256:
            raise SurfaceAtlasError(
                "allowed atlas SHA-256 differs from the frozen contract: "
                f"{digest} != {self.contract.expected_allowed_atlas_sha256}"
            )
        object.__setattr__(self, "triangles_m", triangles)
        object.__setattr__(self, "triangle_uv", uv)
        object.__setattr__(self, "parent_face_index", parents)
        object.__setattr__(self, "parent_triangle_index", local)
        object.__setattr__(self, "parent_surfaces", frames)
        object.__setattr__(self, "atlas_sha256", digest)

    @property
    def triangle_count(self) -> int:
        return int(self.triangles_m.shape[0])

    @property
    def triangle_areas_m2(self) -> np.ndarray:
        areas = 0.5 * np.linalg.norm(
            np.cross(
                self.triangles_m[:, 1] - self.triangles_m[:, 0],
                self.triangles_m[:, 2] - self.triangles_m[:, 0],
            ),
            axis=1,
        )
        areas.setflags(write=False)
        return areas

    def parent_surface(self, face_index: int) -> ParentSurfaceFrame:
        for frame in self.parent_surfaces:
            if frame.face_index == int(face_index):
                return frame
        raise KeyError(
            f"STEP face {face_index} has no implemented plane/cylinder analytic carrier"
        )

    def point_from_barycentric(
        self, triangle_index: int, barycentric: Sequence[float]
    ) -> np.ndarray:
        """Map one continuous simplex coordinate to the object frame in metres."""

        index = int(triangle_index)
        if index < 0 or index >= self.triangle_count:
            raise IndexError("triangle_index is outside the contact atlas")
        weights = np.asarray(barycentric, dtype=np.float64)
        tolerance = 128.0 * np.finfo(np.float64).eps
        if (
            weights.shape != (3,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < -tolerance)
            or not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=tolerance)
        ):
            raise SurfaceAtlasError("barycentric coordinates must belong to Delta^2")
        point = weights @ self.triangles_m[index]
        point.setflags(write=False)
        return point


def load_step_contact_atlas_contract(
    contract_path: str | Path, *, repository_root: str | Path
) -> StepContactAtlasContract:
    root = Path(repository_root).resolve(strict=True)
    supplied = Path(contract_path)
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"surface atlas contract is unavailable: {path}")
    try:
        document = _mapping(
            yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader),
            "surface atlas contract",
        )
    except yaml.YAMLError as error:
        raise SurfaceAtlasError("surface atlas contract is not valid YAML") from error
    _exact_keys(
        document,
        (
            "schema_version",
            "claim_scope",
            "hardware_authorized",
            "development_object",
            "source_step",
            "tessellation",
            "contact_semantics",
            "research_clearance",
        ),
        "surface atlas contract",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise SurfaceAtlasError("surface atlas schema_version changed")
    if document["claim_scope"] != CLAIM_SCOPE:
        raise SurfaceAtlasError("surface atlas claim_scope changed")
    _exact_bool(document["hardware_authorized"], False, "hardware_authorized")
    if document["development_object"] != EXPECTED_DEVELOPMENT_OBJECT:
        raise SurfaceAtlasError("TE must remain the development object")

    source = _mapping(document["source_step"], "source_step")
    _exact_keys(
        source,
        (
            "path",
            "sha256",
            "length_unit",
            "expected_solid_count",
            "expected_face_count",
        ),
        "source_step",
    )
    source_path = _repository_file(root, source["path"], "source_step.path")
    source_hash = _hex_digest(source["sha256"], "source_step.sha256")
    actual_hash = _sha256_file(source_path)
    if actual_hash != source_hash:
        raise SurfaceAtlasError(
            f"source STEP SHA-256 mismatch: {actual_hash} != {source_hash}"
        )
    source_unit = str(source["length_unit"])
    meters_per_unit(source_unit)
    solid_count = int(source["expected_solid_count"])
    face_count = int(source["expected_face_count"])
    if solid_count < 1 or face_count < 1:
        raise SurfaceAtlasError("expected STEP topology counts must be positive")

    tessellation = _mapping(document["tessellation"], "tessellation")
    _exact_keys(
        tessellation,
        (
            "package",
            "package_version",
            "linear_deflection_mm",
            "angular_deflection_rad",
            "relative",
            "parallel",
        ),
        "tessellation",
    )
    relative = _exact_bool(tessellation["relative"], False, "tessellation.relative")
    parallel = _exact_bool(tessellation["parallel"], False, "tessellation.parallel")

    semantics = _mapping(document["contact_semantics"], "contact_semantics")
    _exact_keys(
        semantics,
        (
            "policy",
            "searchable_label",
            "forbidden_label",
            "unresolved_label",
            "proven_searchable_parent_faces",
            "proven_hard_forbidden_parent_faces",
            "unlisted_parent_face_policy",
            "outer_domain_policy",
            "parent_face_ids_are_locators_only",
            "shared_boundary_policy",
            "expected_allowed_brep_area_mm2",
            "expected_allowed_triangle_count",
            "expected_allowed_atlas_sha256",
        ),
        "contact_semantics",
    )
    if semantics["policy"] != "FUNCTIONAL_ROLE_PARTITION_V1":
        raise SurfaceAtlasError("functional contact policy changed")
    if semantics["searchable_label"] != EXPECTED_SEARCHABLE_LABEL:
        raise SurfaceAtlasError("searchable contact semantic changed")
    if semantics["forbidden_label"] != EXPECTED_FORBIDDEN_LABEL:
        raise SurfaceAtlasError("forbidden contact semantic changed")
    if semantics["unresolved_label"] != EXPECTED_UNRESOLVED_LABEL:
        raise SurfaceAtlasError("unresolved contact semantic changed")
    if semantics["unlisted_parent_face_policy"] != "UNRESOLVED":
        raise SurfaceAtlasError("unlisted STEP faces must remain unresolved")
    if semantics["outer_domain_policy"] != "SEARCHABLE_PLUS_UNRESOLVED":
        raise SurfaceAtlasError("outer domain must retain unresolved STEP faces")
    _exact_bool(
        semantics["parent_face_ids_are_locators_only"],
        True,
        "contact_semantics.parent_face_ids_are_locators_only",
    )
    if semantics["shared_boundary_policy"] != "NO_INTRINSIC_BAN":
        raise SurfaceAtlasError("shared-boundary policy changed")
    searchable_faces = _one_based_face_indices(
        semantics["proven_searchable_parent_faces"],
        "contact_semantics.proven_searchable_parent_faces",
    )
    forbidden_faces = _one_based_face_indices(
        semantics["proven_hard_forbidden_parent_faces"],
        "contact_semantics.proven_hard_forbidden_parent_faces",
        allow_empty=True,
    )
    if (
        set(searchable_faces) & set(forbidden_faces)
        or max(searchable_faces + forbidden_faces, default=0) > face_count
    ):
        raise SurfaceAtlasError(
            "functional STEP face sets overlap or exceed face_count"
        )

    clearance = _mapping(document["research_clearance"], "research_clearance")
    _exact_keys(
        clearance,
        (
            "mesh_clearance_mm",
            "numerical_margin_mm",
            "total_clearance_mm",
            "exact_step_hausdorff_enclosure_claimed",
        ),
        "research_clearance",
    )
    _exact_bool(
        clearance["exact_step_hausdorff_enclosure_claimed"],
        False,
        "research_clearance.exact_step_hausdorff_enclosure_claimed",
    )
    mesh_clearance = _positive_finite(
        clearance["mesh_clearance_mm"], "research_clearance.mesh_clearance_mm"
    )
    numerical_margin = _positive_finite(
        clearance["numerical_margin_mm"], "research_clearance.numerical_margin_mm"
    )
    total_clearance = _positive_finite(
        clearance["total_clearance_mm"], "research_clearance.total_clearance_mm"
    )
    if not np.isclose(
        mesh_clearance + numerical_margin,
        total_clearance,
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps,
    ):
        raise SurfaceAtlasError("research clearance components do not sum exactly")

    expected_triangles = int(semantics["expected_allowed_triangle_count"])
    if expected_triangles < 1:
        raise SurfaceAtlasError("expected allowed triangle count must be positive")
    return StepContactAtlasContract(
        contract_path=path,
        repository_root=root,
        development_object=str(document["development_object"]),
        source_step_path=source_path,
        source_step_sha256=source_hash,
        source_unit=source_unit,
        expected_solid_count=solid_count,
        expected_face_count=face_count,
        ocp_package=str(tessellation["package"]),
        ocp_package_version=str(tessellation["package_version"]),
        linear_deflection_mm=_positive_finite(
            tessellation["linear_deflection_mm"], "tessellation.linear_deflection_mm"
        ),
        angular_deflection_rad=_positive_finite(
            tessellation["angular_deflection_rad"], "tessellation.angular_deflection_rad"
        ),
        relative=relative,
        parallel=parallel,
        proven_searchable_parent_faces=searchable_faces,
        hard_forbidden_parent_faces=forbidden_faces,
        searchable_label=str(semantics["searchable_label"]),
        forbidden_label=str(semantics["forbidden_label"]),
        unresolved_label=str(semantics["unresolved_label"]),
        expected_allowed_brep_area_mm2=_positive_finite(
            semantics["expected_allowed_brep_area_mm2"],
            "contact_semantics.expected_allowed_brep_area_mm2",
        ),
        expected_allowed_triangle_count=expected_triangles,
        expected_allowed_atlas_sha256=_hex_digest(
            semantics["expected_allowed_atlas_sha256"],
            "contact_semantics.expected_allowed_atlas_sha256",
        ),
        mesh_clearance_mm=mesh_clearance,
        numerical_margin_mm=numerical_margin,
        total_clearance_mm=total_clearance,
    )


def build_step_contact_atlas(contract: StepContactAtlasContract) -> StepContactAtlas:
    """Reconstruct the allowed charts from the exact STEP and frozen OCP build."""

    try:
        installed_version = version(contract.ocp_package)
    except PackageNotFoundError as error:
        raise SurfaceAtlasError(
            f"required ordinary dependency is missing: {contract.ocp_package}"
        ) from error
    if installed_version != contract.ocp_package_version:
        raise SurfaceAtlasError(
            "OCP package version differs from the frozen contract: "
            f"{installed_version} != {contract.ocp_package_version}"
        )
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
    except ImportError as error:
        raise SurfaceAtlasError("OCP STEP/tessellation API is unavailable") from error

    reader = STEPControl_Reader()
    if reader.ReadFile(str(contract.source_step_path)) != IFSelect_RetDone:
        raise SurfaceAtlasError("OCP could not read the frozen STEP")
    if int(reader.TransferRoots()) < 1:
        raise SurfaceAtlasError("OCP could not transfer a STEP root")
    shape = reader.OneShape()

    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_count = 0
    while solids.More():
        solid_count += 1
        solids.Next()
    if solid_count != contract.expected_solid_count:
        raise SurfaceAtlasError(
            f"STEP solid count changed: {solid_count} != {contract.expected_solid_count}"
        )

    mesher = BRepMesh_IncrementalMesh(
        shape,
        contract.linear_deflection_mm,
        contract.relative,
        contract.angular_deflection_rad,
        contract.parallel,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise SurfaceAtlasError("OCP did not complete the frozen tessellation")

    allowed = frozenset(contract.allowed_parent_faces)
    analytic_faces = frozenset(contract.proven_searchable_parent_faces)
    triangles: list[list[list[float]]] = []
    triangle_uv: list[list[list[float]]] = []
    parents: list[int] = []
    local_indices: list[int] = []
    parent_surfaces: list[ParentSurfaceFrame] = []
    brep_area_mm2 = 0.0
    face_count = 0
    source_scale_m = meters_per_unit(contract.source_unit)

    def xyz(value: Any) -> tuple[float, float, float]:
        return (float(value.X()), float(value.Y()), float(value.Z()))

    faces = TopExp_Explorer(shape, TopAbs_FACE)
    while faces.More():
        face_count += 1
        face = TopoDS.Face_s(faces.Current())
        if face_count in allowed:
            surface = BRepAdaptor_Surface(face)
            surface_type = surface.GetType()
            if face_count not in analytic_faces:
                position = None
                radius_m = None
                kind = None
            elif surface_type == GeomAbs_Plane:
                position = surface.Plane().Position()
                radius_m = None
                kind = "plane"
            elif surface_type == GeomAbs_Cylinder:
                cylinder = surface.Cylinder()
                position = cylinder.Position()
                radius_m = float(cylinder.Radius()) * source_scale_m
                kind = "cylinder"
            else:
                position = None
                radius_m = None
                kind = None
            if position is not None and kind is not None:
                parent_surfaces.append(
                    ParentSurfaceFrame(
                        face_index=face_count,
                        kind=kind,
                        origin_m=tuple(
                            component * source_scale_m
                            for component in xyz(position.Location())
                        ),
                        x_direction=xyz(position.XDirection()),
                        y_direction=xyz(position.YDirection()),
                        axis_direction=xyz(position.Direction()),
                        radius_m=radius_m,
                        outward_sign=(
                            -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
                        ),
                        uv_length_scale_m=source_scale_m,
                    )
                )
            properties = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, properties)
            brep_area_mm2 += float(properties.Mass())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)
            if triangulation is None or not triangulation.HasUVNodes():
                raise SurfaceAtlasError(
                    f"allowed STEP face {face_count} lacks triangulation or UV nodes"
                )
            transform = location.Transformation()
            reversed_face = face.Orientation() == TopAbs_REVERSED
            for local_index in range(1, triangulation.NbTriangles() + 1):
                node_indices = list(triangulation.Triangle(local_index).Get())
                if reversed_face:
                    node_indices[1], node_indices[2] = (
                        node_indices[2],
                        node_indices[1],
                    )
                triangle: list[list[float]] = []
                uv_row: list[list[float]] = []
                for node_index in node_indices:
                    point = triangulation.Node(node_index).Transformed(transform)
                    triangle.append(
                        [
                            float(point.X()) * meters_per_unit(contract.source_unit),
                            float(point.Y()) * meters_per_unit(contract.source_unit),
                            float(point.Z()) * meters_per_unit(contract.source_unit),
                        ]
                    )
                    uv = triangulation.UVNode(node_index)
                    uv_row.append([float(uv.X()), float(uv.Y())])
                triangles.append(triangle)
                triangle_uv.append(uv_row)
                parents.append(face_count)
                local_indices.append(local_index)
        faces.Next()
    if face_count != contract.expected_face_count:
        raise SurfaceAtlasError(
            f"STEP face count changed: {face_count} != {contract.expected_face_count}"
        )
    area_tolerance_mm2 = max(
        1.0e-9, 512.0 * np.finfo(np.float64).eps * brep_area_mm2
    )
    if not np.isclose(
        brep_area_mm2,
        contract.expected_allowed_brep_area_mm2,
        rtol=0.0,
        atol=area_tolerance_mm2,
    ):
        raise SurfaceAtlasError(
            "allowed B-rep area differs from the frozen contract: "
            f"{brep_area_mm2} != {contract.expected_allowed_brep_area_mm2} mm^2"
        )
    triangle_array = np.asarray(triangles, dtype="<f8")
    uv_array = np.asarray(triangle_uv, dtype="<f8")
    parent_array = np.asarray(parents, dtype="<i8")
    local_array = np.asarray(local_indices, dtype="<i8")
    digest = _atlas_sha256(
        contract.source_step_sha256,
        triangle_array,
        uv_array,
        parent_array,
        local_array,
    )
    return StepContactAtlas(
        contract=contract,
        triangles_m=triangle_array,
        triangle_uv=uv_array,
        parent_face_index=parent_array,
        parent_triangle_index=local_array,
        parent_surfaces=tuple(parent_surfaces),
        allowed_brep_area_m2=brep_area_mm2 * 1.0e-6,
        atlas_sha256=digest,
    )


def load_step_contact_atlas(
    contract_path: str | Path, *, repository_root: str | Path
) -> StepContactAtlas:
    return build_step_contact_atlas(
        load_step_contact_atlas_contract(
            contract_path, repository_root=repository_root
        )
    )


def build_step_triangle_role_partition(
    contract: StepContactAtlasContract,
) -> StepTriangleRolePartition:
    """Tessellate every frozen STEP face and retain its functional role.

    The contact atlas intentionally omits hard-forbidden faces to keep the
    global outer model compact.  Collision screening needs the complementary
    geometry as well, so this explicit loader is kept separate and invoked
    only by checks that need the full partition.
    """

    try:
        installed_version = version(contract.ocp_package)
    except PackageNotFoundError as error:
        raise SurfaceAtlasError(
            f"required ordinary dependency is missing: {contract.ocp_package}"
        ) from error
    if installed_version != contract.ocp_package_version:
        raise SurfaceAtlasError(
            "OCP package version differs from the frozen contract: "
            f"{installed_version} != {contract.ocp_package_version}"
        )
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
    except ImportError as error:
        raise SurfaceAtlasError("OCP STEP/tessellation API is unavailable") from error

    reader = STEPControl_Reader()
    if reader.ReadFile(str(contract.source_step_path)) != IFSelect_RetDone:
        raise SurfaceAtlasError("OCP could not read the frozen STEP")
    if int(reader.TransferRoots()) < 1:
        raise SurfaceAtlasError("OCP could not transfer a STEP root")
    shape = reader.OneShape()
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_count = 0
    while solids.More():
        solid_count += 1
        solids.Next()
    if solid_count != contract.expected_solid_count:
        raise SurfaceAtlasError(
            f"STEP solid count changed: {solid_count} != {contract.expected_solid_count}"
        )

    mesher = BRepMesh_IncrementalMesh(
        shape,
        contract.linear_deflection_mm,
        contract.relative,
        contract.angular_deflection_rad,
        contract.parallel,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise SurfaceAtlasError("OCP did not complete STEP tessellation")

    searchable_faces = frozenset(contract.proven_searchable_parent_faces)
    hard_faces = frozenset(contract.hard_forbidden_parent_faces)
    rows: dict[str, list[list[list[float]]]] = {
        "proven_searchable": [],
        "unresolved": [],
        "hard_forbidden": [],
    }
    parents: dict[str, list[int]] = {name: [] for name in rows}
    scale_m = meters_per_unit(contract.source_unit)
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while explorer.More():
        face_index += 1
        face = TopoDS.Face_s(explorer.Current())
        if face_index in searchable_faces:
            role = "proven_searchable"
        elif face_index in hard_faces:
            role = "hard_forbidden"
        else:
            role = "unresolved"
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            raise SurfaceAtlasError(f"STEP face {face_index} has no mesh")
        transform = location.Transformation()
        reversed_face = face.Orientation() == TopAbs_REVERSED
        for local_index in range(1, triangulation.NbTriangles() + 1):
            node_indices = list(triangulation.Triangle(local_index).Get())
            if reversed_face:
                node_indices[1], node_indices[2] = node_indices[2], node_indices[1]
            triangle: list[list[float]] = []
            for node_index in node_indices:
                point = triangulation.Node(node_index).Transformed(transform)
                triangle.append(
                    [
                        float(point.X()) * scale_m,
                        float(point.Y()) * scale_m,
                        float(point.Z()) * scale_m,
                    ]
                )
            rows[role].append(triangle)
            parents[role].append(face_index)
        explorer.Next()
    if face_index != contract.expected_face_count:
        raise SurfaceAtlasError(
            f"STEP face count changed: {face_index} != {contract.expected_face_count}"
        )

    def triangles(role: str) -> np.ndarray:
        return np.ascontiguousarray(rows[role], dtype=np.float64).reshape(-1, 3, 3)

    def parent_indices(role: str) -> np.ndarray:
        return np.ascontiguousarray(parents[role], dtype=np.int64)

    return StepTriangleRolePartition(
        contract=contract,
        proven_searchable_triangles_m=triangles("proven_searchable"),
        proven_searchable_parent_face_index=parent_indices("proven_searchable"),
        unresolved_triangles_m=triangles("unresolved"),
        unresolved_parent_face_index=parent_indices("unresolved"),
        hard_forbidden_triangles_m=triangles("hard_forbidden"),
        hard_forbidden_parent_face_index=parent_indices("hard_forbidden"),
    )


def load_step_triangle_role_partition(
    contract_path: str | Path, *, repository_root: str | Path
) -> StepTriangleRolePartition:
    return build_step_triangle_role_partition(
        load_step_contact_atlas_contract(
            contract_path, repository_root=repository_root
        )
    )


__all__ = [
    "CLAIM_SCOPE",
    "ParentSurfaceFrame",
    "SCHEMA_VERSION",
    "StepContactAtlas",
    "StepContactAtlasContract",
    "StepTriangleRolePartition",
    "SurfaceAtlasError",
    "build_step_contact_atlas",
    "build_step_triangle_role_partition",
    "load_step_contact_atlas",
    "load_step_contact_atlas_contract",
    "load_step_triangle_role_partition",
]
