"""Numerical one-coordinate contact search for a frozen TE high-opening seed.

This module answers one deliberately narrow Stage-2 question: with the palm
shape, axial placement, pregrasp, and closure ranges read from the historical
high-opening TE seed, is there a connector-axis yaw and one stopping position
per finger at which all three complete blue pads intersect an allowed STEP
surface while remaining more than 61 micrometres from every forbidden STEP
surface?

The first coordinate can be full connector-axis yaw or the shared outer-finger
palm joint while the other is fixed.  The search therefore branches over a
continuous ``common x q1 x q2 x q3`` box.  Its box pruning uses a conservative
rigid-motion bound, but the centre distances come from floating-point FCL and
do not have a proved outward-rounding error bound.
Consequently an exhausted search is *not* a mathematical infeasibility proof.
An incumbent is only a triangulated, numerical mode witness for subsequent
fixed-mode refinement; it is not a force, PD, path, Isaac, lift, or robustness
result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from kcg_connector.grasp.robust.analytic_outer_master import (
    _affine_source,
    _ancestor_joint_names,
    load_analytic_envelope_contract,
)
from kcg_connector.grasp.robust.hand_contract import (
    CARTSHandContract,
    VerifiedPad,
    load_carts_hand_contract,
)
from kcg_connector.grasp.robust.hand_model import ThreeFingerHandModel
from kcg_connector.grasp.robust.surface_atlas import (
    StepContactAtlasContract,
    load_step_contact_atlas_contract,
)


CLAIM_SCOPE = "SIMULATION_ONLY_FLOATING_POINT_TRIANGULATED_INNER_MODE_SEARCH"
REQUIRED_FORBIDDEN_CLEARANCE_M = 61.0e-6
DEFAULT_ANALYTIC_CONTRACT = (
    "src/kcg_connector/config/te_continuous_grasp_analytic_envelope_v1.yaml"
)
DEFAULT_SEED_RESULT = (
    "artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828/"
    "finite_gdelta_candidate_replay/te_p15_y20_z25_robust_run01/"
    "search_result.json"
)
EXPECTED_JOINTS = ("f1j1", "f1j2", "f2j1", "f3j2")
EXPECTED_CLOSING_JOINTS = ("f1j2", "f2j1", "f3j2")


class FixedSeedYawSearchError(ValueError):
    """Raised when the frozen study inputs no longer match this experiment."""


@dataclass(frozen=True)
class FrozenSeed:
    source_path: Path
    object_from_hand: np.ndarray
    seed_yaw_rad: float
    pregrasp: np.ndarray
    endpoint: np.ndarray


@dataclass(frozen=True)
class SearchBox:
    lower: tuple[float, float, float, float]
    upper: tuple[float, float, float, float]
    depth: int

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (np.asarray(self.lower) + np.asarray(self.upper))

    @property
    def half_width(self) -> np.ndarray:
        return 0.5 * (np.asarray(self.upper) - np.asarray(self.lower))


@dataclass(frozen=True)
class BoxEvaluation:
    allowed_distance_m: tuple[float, float, float]
    forbidden_distance_m: tuple[float, float, float]
    allowed_collision: tuple[bool, bool, bool]
    motion_radius_m: tuple[float, float, float]
    common_motion_radius_m: tuple[float, float, float]
    radial_radius_m: tuple[float, float, float]
    impossible_reason: str | None
    centre_rank: int
    centre_score_m: float


def _repository_path(root: Path, value: str | Path, label: str) -> Path:
    supplied = Path(value)
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise FixedSeedYawSearchError(f"{label} escapes the repository") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {path}")
    return path


def _load_seed(root: Path, value: str | Path) -> FrozenSeed:
    path = _repository_path(root, value, "seed result")
    document = json.loads(path.read_text(encoding="utf-8"))
    selected = document.get("selected_candidate")
    if not isinstance(selected, dict):
        raise FixedSeedYawSearchError("seed result lacks selected_candidate")
    control = selected.get("control_plan")
    if not isinstance(control, dict):
        raise FixedSeedYawSearchError("seed result lacks control_plan")
    transform = np.asarray(control.get("object_from_hand_row_major"), dtype=np.float64)
    pregrasp = np.asarray(control.get("pregrasp_joint_positions_rad"), dtype=np.float64)
    endpoint = np.asarray(control.get("final_joint_positions_rad"), dtype=np.float64)
    if transform.shape != (16,) or pregrasp.shape != (4,) or endpoint.shape != (4,):
        raise FixedSeedYawSearchError("seed transform or joint vectors changed shape")
    transform = transform.reshape(4, 4)
    if (
        not np.all(np.isfinite(transform))
        or not np.all(np.isfinite(pregrasp))
        or not np.all(np.isfinite(endpoint))
        or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-14)
        or not np.allclose(transform[:2, 2], 0.0, rtol=0.0, atol=1.0e-14)
        or not np.allclose(transform[2, :2], 0.0, rtol=0.0, atol=1.0e-14)
        or not np.isclose(transform[2, 2], 1.0, rtol=0.0, atol=1.0e-14)
        or not np.allclose(transform[:2, 3], 0.0, rtol=0.0, atol=1.0e-14)
    ):
        raise FixedSeedYawSearchError(
            "the fixed-yaw study requires the original axis-aligned seed"
        )
    if not np.isclose(pregrasp[0], endpoint[0], rtol=0.0, atol=1.0e-14):
        raise FixedSeedYawSearchError("palm joint must remain frozen during closure")
    if np.any(endpoint[1:] <= pregrasp[1:]):
        raise FixedSeedYawSearchError("all three frozen closing intervals must be positive")
    yaw = math.atan2(float(transform[1, 0]), float(transform[0, 0])) % (2.0 * math.pi)
    transform.setflags(write=False)
    pregrasp.setflags(write=False)
    endpoint.setflags(write=False)
    return FrozenSeed(
        source_path=path,
        object_from_hand=transform,
        seed_yaw_rad=yaw,
        pregrasp=pregrasp,
        endpoint=endpoint,
    )


def _joint_motion_bound(
    hand: ThreeFingerHandModel,
    pad: VerifiedPad,
    independent_joint: str,
    *,
    require_positive: bool,
) -> float:
    """Return a configuration-independent pad point-speed bound in m/rad."""

    path = _ancestor_joint_names(hand, pad.link_name)
    point_radius = float(np.max(np.linalg.norm(pad.points_local_m, axis=1)))
    affine_cache: dict[str, tuple[str, float]] = {}
    result = 0.0
    for index, joint_name in enumerate(path):
        joint = hand.joints[joint_name]
        if not joint.movable:
            continue
        source, multiplier = _affine_source(hand, joint_name, affine_cache)
        if source != independent_joint:
            continue
        if joint.joint_type in ("revolute", "continuous"):
            downstream_radius = point_radius + sum(
                float(np.linalg.norm(hand.joints[name].origin_xyz_m))
                for name in path[index + 1 :]
            )
            result += abs(multiplier) * downstream_radius
        elif joint.joint_type == "prismatic":
            result += abs(multiplier)
        else:
            raise FixedSeedYawSearchError(
                f"unsupported movable joint type {joint.joint_type!r}"
            )
    if not math.isfinite(result) or result < 0.0:
        raise FixedSeedYawSearchError(f"invalid joint-motion bound for {pad.name}")
    if require_positive and result <= 0.0:
        raise FixedSeedYawSearchError(f"no positive closing bound for {pad.name}")
    return result


def _closing_motion_bound(
    hand: ThreeFingerHandModel,
    pad: VerifiedPad,
    closing_joint: str,
) -> float:
    return _joint_motion_bound(
        hand, pad, closing_joint, require_positive=True
    )


def _load_complete_step_triangles(
    contract: StepContactAtlasContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tessellate every STEP face and retain parent identities."""

    try:
        installed_version = version(contract.ocp_package)
    except PackageNotFoundError as error:
        raise FixedSeedYawSearchError(
            f"required ordinary dependency is missing: {contract.ocp_package}"
        ) from error
    if installed_version != contract.ocp_package_version:
        raise FixedSeedYawSearchError(
            "OCP package version differs from the frozen contract: "
            f"{installed_version} != {contract.ocp_package_version}"
        )
    # The hand contract is deliberately loaded before this function imports
    # OCP/fcl.  That preserves the project's known XML-loader compatibility.
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    reader = STEPControl_Reader()
    if reader.ReadFile(str(contract.source_step_path)) != IFSelect_RetDone:
        raise FixedSeedYawSearchError("OCP could not read the frozen STEP")
    if int(reader.TransferRoots()) < 1:
        raise FixedSeedYawSearchError("OCP could not transfer a STEP root")
    shape = reader.OneShape()
    mesher = BRepMesh_IncrementalMesh(
        shape,
        contract.linear_deflection_mm,
        contract.relative,
        contract.angular_deflection_rad,
        contract.parallel,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise FixedSeedYawSearchError("OCP did not complete STEP tessellation")

    allowed_faces = frozenset(contract.allowed_parent_faces)
    rows: dict[str, list[list[list[float]]]] = {"allowed": [], "forbidden": []}
    parents: dict[str, list[int]] = {"allowed": [], "forbidden": []}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while explorer.More():
        face_index += 1
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            raise FixedSeedYawSearchError(f"STEP face {face_index} has no mesh")
        transform = location.Transformation()
        reversed_face = face.Orientation() == TopAbs_REVERSED
        kind = "allowed" if face_index in allowed_faces else "forbidden"
        for local_index in range(1, triangulation.NbTriangles() + 1):
            node_indices = list(triangulation.Triangle(local_index).Get())
            if reversed_face:
                node_indices[1], node_indices[2] = node_indices[2], node_indices[1]
            triangle: list[list[float]] = []
            for node_index in node_indices:
                point = triangulation.Node(node_index).Transformed(transform)
                triangle.append(
                    [
                        float(point.X()) * 1.0e-3,
                        float(point.Y()) * 1.0e-3,
                        float(point.Z()) * 1.0e-3,
                    ]
                )
            rows[kind].append(triangle)
            parents[kind].append(face_index)
        explorer.Next()
    if face_index != contract.expected_face_count:
        raise FixedSeedYawSearchError(
            f"STEP face count changed: {face_index} != {contract.expected_face_count}"
        )
    allowed = np.ascontiguousarray(rows["allowed"], dtype=np.float64).reshape(-1, 3, 3)
    forbidden = np.ascontiguousarray(rows["forbidden"], dtype=np.float64).reshape(-1, 3, 3)
    allowed_parent = np.ascontiguousarray(parents["allowed"], dtype=np.int64)
    forbidden_parent = np.ascontiguousarray(parents["forbidden"], dtype=np.int64)
    if len(allowed) != contract.expected_allowed_triangle_count:
        raise FixedSeedYawSearchError("allowed STEP triangle count changed")
    return allowed, allowed_parent, forbidden, forbidden_parent


class _FCLScene:
    def __init__(
        self,
        hand: ThreeFingerHandModel,
        hand_contract: CARTSHandContract,
        seed: FrozenSeed,
        allowed_triangles: np.ndarray,
        allowed_parent: np.ndarray,
        forbidden_triangles: np.ndarray,
    ) -> None:
        import fcl

        self.fcl = fcl
        self.hand = hand
        self.hand_contract = hand_contract
        self.seed = seed
        self.allowed_parent = allowed_parent
        self.allowed = fcl.CollisionObject(self._triangle_soup(allowed_triangles))
        self.forbidden = (
            None
            if not len(forbidden_triangles)
            else fcl.CollisionObject(self._triangle_soup(forbidden_triangles))
        )
        self.pads = tuple(
            fcl.CollisionObject(self._indexed_mesh(pad.points_local_m, pad.faces))
            for pad in hand_contract.pads
        )
        self.distance_request = fcl.DistanceRequest(enable_nearest_points=False)
        self.collision_request = fcl.CollisionRequest(
            num_max_contacts=1, enable_contact=False
        )
        self.transforms: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _indexed_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Any:
        model = self.fcl.BVHModel()
        vertex_values = np.ascontiguousarray(vertices, dtype=np.float64)
        face_values = np.ascontiguousarray(faces, dtype=np.int32)
        model.beginModel(len(face_values), len(vertex_values))
        model.addSubModel(vertex_values, face_values)
        model.endModel()
        return model

    def _triangle_soup(self, triangles: np.ndarray) -> Any:
        vertices = np.ascontiguousarray(triangles.reshape(-1, 3), dtype=np.float64)
        faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
        return self._indexed_mesh(vertices, faces)

    def object_from_hand(self, yaw_rad: float) -> np.ndarray:
        delta = float(yaw_rad) - self.seed.seed_yaw_rad
        cosine, sine = math.cos(delta), math.sin(delta)
        rotation = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation @ self.seed.object_from_hand[:3, :3]
        result[:3, 3] = rotation @ self.seed.object_from_hand[:3, 3]
        return result

    def set_state(self, yaw_rad: float, joints: Sequence[float]) -> None:
        transforms = self.hand.forward_kinematics(
            joints, base_transform=self.object_from_hand(yaw_rad)
        )
        selected: list[np.ndarray] = []
        for collision_object, pad in zip(self.pads, self.hand_contract.pads):
            transform = transforms[pad.link_name]
            collision_object.setTransform(
                self.fcl.Transform(transform[:3, :3], transform[:3, 3])
            )
            selected.append(transform)
        self.transforms = (selected[0], selected[1], selected[2])

    def distance(self, finger_index: int, kind: str) -> float:
        target = self.allowed if kind == "allowed" else self.forbidden
        if target is None:
            return math.inf
        result = self.fcl.DistanceResult()
        value = float(
            self.fcl.distance(
                self.pads[finger_index], target, self.distance_request, result
            )
        )
        return max(0.0, value)

    def collides_allowed(self, finger_index: int) -> bool:
        return bool(
            self.fcl.collide(
                self.pads[finger_index],
                self.allowed,
                self.collision_request,
                self.fcl.CollisionResult(),
            )
        )

    def radial_radius(self, finger_index: int) -> float:
        if self.transforms is None:
            raise RuntimeError("scene state was not set")
        pad = self.hand_contract.pads[finger_index]
        transform = self.transforms[finger_index]
        points = pad.points_local_m @ transform[:3, :3].T + transform[:3, 3]
        return float(np.max(np.linalg.norm(points[:, :2], axis=1)))

    def allowed_contact_modes(self, finger_index: int) -> list[dict[str, int]]:
        request = self.fcl.CollisionRequest(num_max_contacts=256, enable_contact=True)
        result = self.fcl.CollisionResult()
        self.fcl.collide(self.pads[finger_index], self.allowed, request, result)
        modes = {
            (int(contact.b1), int(contact.b2))
            for contact in result.contacts
        }
        return [
            {
                "pad_triangle_zero_based": pad_triangle,
                "object_triangle_zero_based": object_triangle,
                "object_parent_face_one_based": int(
                    self.allowed_parent[object_triangle]
                ),
            }
            for pad_triangle, object_triangle in sorted(modes)
        ]


def _evaluate_box(
    scene: _FCLScene,
    box: SearchBox,
    seed: FrozenSeed,
    closing_bounds: Sequence[float],
    common_joint_bounds: Sequence[float],
    *,
    search_variable: str,
    fixed_yaw_rad: float,
    clearance_m: float,
    assumed_distance_error_m: float,
) -> BoxEvaluation:
    center = box.center
    half = box.half_width
    if search_variable == "yaw":
        yaw_rad = float(center[0])
        palm_rad = float(seed.pregrasp[0])
    elif search_variable == "palm":
        yaw_rad = fixed_yaw_rad
        palm_rad = float(center[0])
    else:
        raise FixedSeedYawSearchError(f"unsupported search variable {search_variable!r}")
    joints = np.asarray((palm_rad, center[1], center[2], center[3]))
    scene.set_state(yaw_rad, joints)
    allowed_distance: list[float] = []
    forbidden_distance: list[float] = []
    allowed_collision: list[bool] = []
    motion_radius: list[float] = []
    common_motion_radius: list[float] = []
    radial_radius: list[float] = []
    impossible = None
    for index in range(3):
        allowed = scene.distance(index, "allowed")
        forbidden = scene.distance(index, "forbidden")
        collision = scene.collides_allowed(index)
        radial = scene.radial_radius(index)
        if search_variable == "yaw":
            common_motion = 2.0 * radial * math.sin(
                min(float(half[0]), math.pi) / 2.0
            )
        else:
            common_motion = float(common_joint_bounds[index]) * float(half[0])
        joint_motion = float(closing_bounds[index]) * float(half[index + 1])
        motion = common_motion + joint_motion
        allowed_distance.append(allowed)
        forbidden_distance.append(forbidden)
        allowed_collision.append(collision)
        motion_radius.append(motion)
        common_motion_radius.append(common_motion)
        radial_radius.append(radial)
        if allowed - assumed_distance_error_m - motion > 0.0:
            impossible = f"FINGER_{index + 1}_NO_ALLOWED_INTERSECTION_IN_BOX"
            break
        if forbidden + assumed_distance_error_m + motion < clearance_m:
            impossible = f"FINGER_{index + 1}_ENTIRE_BOX_IN_FORBIDDEN_BUFFER"
            break
    while len(allowed_distance) < 3:
        allowed_distance.append(math.nan)
        forbidden_distance.append(math.nan)
        allowed_collision.append(False)
        motion_radius.append(math.nan)
        common_motion_radius.append(math.nan)
        radial_radius.append(math.nan)
    rank = sum(allowed_collision) + sum(
        value > clearance_m for value in forbidden_distance if math.isfinite(value)
    )
    scores = [
        forbidden_distance[index] - clearance_m - allowed_distance[index]
        for index in range(3)
        if math.isfinite(allowed_distance[index])
    ]
    return BoxEvaluation(
        allowed_distance_m=tuple(allowed_distance),  # type: ignore[arg-type]
        forbidden_distance_m=tuple(forbidden_distance),  # type: ignore[arg-type]
        allowed_collision=tuple(allowed_collision),  # type: ignore[arg-type]
        motion_radius_m=tuple(motion_radius),  # type: ignore[arg-type]
        common_motion_radius_m=tuple(common_motion_radius),  # type: ignore[arg-type]
        radial_radius_m=tuple(radial_radius),  # type: ignore[arg-type]
        impossible_reason=impossible,
        centre_rank=rank,
        centre_score_m=min(scores) if scores else -math.inf,
    )


def _prefix_forbidden_clearance_audit(
    scene: _FCLScene,
    seed: FrozenSeed,
    yaw_rad: float,
    joints_at_stop: np.ndarray,
    finger_index: int,
    closing_bound: float,
    *,
    clearance_m: float,
    assumed_distance_error_m: float,
    minimum_half_width_rad: float,
    maximum_boxes: int,
) -> dict[str, Any]:
    lower = float(seed.pregrasp[finger_index + 1])
    upper = float(joints_at_stop[finger_index + 1])
    stack = [(lower, upper)]
    visited = 0
    minimum_sampled = math.inf
    while stack:
        start, stop = stack.pop()
        visited += 1
        if visited > maximum_boxes:
            return {
                "status": "UNRESOLVED_BOX_LIMIT",
                "visited_boxes": visited,
                "minimum_sampled_clearance_m": minimum_sampled,
            }
        center = 0.5 * (start + stop)
        half = 0.5 * (stop - start)
        joints = np.array(joints_at_stop, copy=True)
        joints[finger_index + 1] = center
        scene.set_state(yaw_rad, joints)
        distance = scene.distance(finger_index, "forbidden")
        minimum_sampled = min(minimum_sampled, distance)
        lower_bound = distance - assumed_distance_error_m - closing_bound * half
        if lower_bound > clearance_m:
            continue
        if distance - assumed_distance_error_m <= clearance_m:
            return {
                "status": "SAMPLED_CLEARANCE_NOT_CERTIFIED",
                "visited_boxes": visited,
                "joint_position_rad": center,
                "sampled_clearance_m": distance,
                "minimum_sampled_clearance_m": minimum_sampled,
            }
        if half <= minimum_half_width_rad:
            return {
                "status": "UNRESOLVED_MINIMUM_WIDTH",
                "visited_boxes": visited,
                "joint_interval_rad": [start, stop],
                "sampled_clearance_m": distance,
                "motion_bound_m": closing_bound * half,
                "minimum_sampled_clearance_m": minimum_sampled,
            }
        midpoint = center
        stack.append((midpoint, stop))
        stack.append((start, midpoint))
    return {
        "status": "FLOATING_POINT_LIPSCHITZ_PREFIX_CLEARANCE_AUDIT",
        "visited_boxes": visited,
        "minimum_sampled_clearance_m": minimum_sampled,
    }


def _split_box(
    box: SearchBox,
    evaluation: BoxEvaluation,
    closing_bounds: Sequence[float],
    minimum_widths: Sequence[float],
) -> tuple[SearchBox, SearchBox] | None:
    lower = np.asarray(box.lower)
    upper = np.asarray(box.upper)
    half = box.half_width
    contributions = np.asarray(
        (
            max(
                value
                for value in evaluation.common_motion_radius_m
                if math.isfinite(value)
            ),
            closing_bounds[0] * half[1],
            closing_bounds[1] * half[2],
            closing_bounds[2] * half[3],
        )
    )
    eligible = (upper - lower) > np.asarray(minimum_widths)
    contributions[~eligible] = -math.inf
    if not np.any(np.isfinite(contributions)):
        return None
    dimension = int(np.argmax(contributions))
    midpoint = 0.5 * (lower[dimension] + upper[dimension])
    first_upper = np.array(upper, copy=True)
    first_upper[dimension] = midpoint
    second_lower = np.array(lower, copy=True)
    second_lower[dimension] = midpoint
    return (
        SearchBox(tuple(lower), tuple(first_upper), box.depth + 1),
        SearchBox(tuple(second_lower), tuple(upper), box.depth + 1),
    )


def search_fixed_seed_full_yaw(
    repository_root: str | Path,
    *,
    analytic_contract_path: str | Path = DEFAULT_ANALYTIC_CONTRACT,
    seed_result_path: str | Path = DEFAULT_SEED_RESULT,
    maximum_nodes: int = 100_000,
    time_limit_s: float = 180.0,
    minimum_yaw_width_rad: float = 2.5e-5,
    minimum_joint_width_rad: float = 2.5e-6,
    assumed_fcl_distance_error_m: float = 1.0e-9,
    progress_interval: int = 2_000,
    initial_lower: Sequence[float] | None = None,
    initial_upper: Sequence[float] | None = None,
    search_variable: str = "yaw",
    fixed_yaw_rad: float | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    contract = load_analytic_envelope_contract(
        analytic_contract_path, repository_root=root
    )
    hand_contract = load_carts_hand_contract(
        contract.hand_contact_contract_path, repository_root=root
    )
    hand = hand_contract.build_hand_model()
    if tuple(hand.independent_joint_names) != EXPECTED_JOINTS:
        raise FixedSeedYawSearchError("independent hand-joint order changed")
    if tuple(contract.closing_joint_names) != EXPECTED_CLOSING_JOINTS:
        raise FixedSeedYawSearchError("closing-joint contract changed")
    seed = _load_seed(root, seed_result_path)
    atlas_contract = load_step_contact_atlas_contract(
        contract.step_contact_atlas_path, repository_root=root
    )
    allowed, allowed_parent, forbidden, _forbidden_parent = (
        _load_complete_step_triangles(atlas_contract)
    )
    scene = _FCLScene(
        hand,
        hand_contract,
        seed,
        allowed,
        allowed_parent,
        forbidden,
    )
    closing_bounds = tuple(
        _closing_motion_bound(hand, pad, joint)
        for pad, joint in zip(hand_contract.pads, EXPECTED_CLOSING_JOINTS)
    )
    palm_bounds = tuple(
        _joint_motion_bound(
            hand, pad, EXPECTED_JOINTS[0], require_positive=False
        )
        for pad in hand_contract.pads
    )
    if search_variable == "yaw":
        resolved_fixed_yaw = seed.seed_yaw_rad
        common_lower, common_upper = 0.0, 2.0 * math.pi
        common_name = "yaw"
        common_motion_rule = (
            "2*centre_pad_axis_radius*sin(common_half_width/2)"
        )
    elif search_variable == "palm":
        if fixed_yaw_rad is None or not math.isfinite(fixed_yaw_rad):
            raise FixedSeedYawSearchError(
                "palm search requires one finite fixed_yaw_rad"
            )
        resolved_fixed_yaw = float(fixed_yaw_rad) % (2.0 * math.pi)
        joint_lower, joint_upper = hand.joint_limit_vectors()
        common_lower = float(joint_lower[0])
        common_upper = float(joint_upper[0])
        common_name = EXPECTED_JOINTS[0]
        common_motion_rule = "common_joint_motion_bound*common_half_width"
    else:
        raise FixedSeedYawSearchError(
            "search_variable must be exactly 'yaw' or 'palm'"
        )
    full_lower = np.asarray((common_lower, *map(float, seed.pregrasp[1:])))
    full_upper = np.asarray((common_upper, *map(float, seed.endpoint[1:])))
    selected_lower = (
        full_lower if initial_lower is None else np.asarray(initial_lower, dtype=np.float64)
    )
    selected_upper = (
        full_upper if initial_upper is None else np.asarray(initial_upper, dtype=np.float64)
    )
    if (initial_lower is None) != (initial_upper is None):
        raise FixedSeedYawSearchError(
            "initial_lower and initial_upper must be supplied together"
        )
    if (
        selected_lower.shape != (4,)
        or selected_upper.shape != (4,)
        or not np.all(np.isfinite(selected_lower))
        or not np.all(np.isfinite(selected_upper))
        or np.any(selected_lower < full_lower)
        or np.any(selected_upper > full_upper)
        or np.any(selected_upper <= selected_lower)
    ):
        raise FixedSeedYawSearchError("selected initial box is outside the full domain")
    initial = SearchBox(
        lower=tuple(map(float, selected_lower)),
        upper=tuple(map(float, selected_upper)),
        depth=0,
    )
    minimum_widths = (
        minimum_yaw_width_rad,
        minimum_joint_width_rad,
        minimum_joint_width_rad,
        minimum_joint_width_rad,
    )
    counters = {
        "evaluated_nodes": 0,
        "pruned_no_allowed": 0,
        "pruned_forbidden_buffer": 0,
        "unresolved_minimum_width": 0,
        "centre_contact_triplets": 0,
        "prefix_audit_rejections": 0,
    }
    started = time.monotonic()
    serial = 0
    queue: list[tuple[int, float, float, int, SearchBox, BoxEvaluation]] = []

    def evaluate_and_push(box: SearchBox) -> None:
        nonlocal serial
        evaluation = _evaluate_box(
            scene,
            box,
            seed,
            closing_bounds,
            palm_bounds,
            search_variable=search_variable,
            fixed_yaw_rad=resolved_fixed_yaw,
            clearance_m=REQUIRED_FORBIDDEN_CLEARANCE_M,
            assumed_distance_error_m=assumed_fcl_distance_error_m,
        )
        counters["evaluated_nodes"] += 1
        if evaluation.impossible_reason is not None:
            if "NO_ALLOWED" in evaluation.impossible_reason:
                counters["pruned_no_allowed"] += 1
            else:
                counters["pruned_forbidden_buffer"] += 1
            return
        serial += 1
        finite_motion = [
            value for value in evaluation.motion_radius_m if math.isfinite(value)
        ]
        heapq.heappush(
            queue,
            (
                -evaluation.centre_rank,
                -evaluation.centre_score_m,
                max(finite_motion) if finite_motion else math.inf,
                serial,
                box,
                evaluation,
            ),
        )

    evaluate_and_push(initial)
    incumbent: dict[str, Any] | None = None
    stop_reason = "QUEUE_EXHAUSTED"
    while queue:
        elapsed = time.monotonic() - started
        if counters["evaluated_nodes"] >= maximum_nodes:
            stop_reason = "NODE_LIMIT"
            break
        if elapsed >= time_limit_s:
            stop_reason = "TIME_LIMIT"
            break
        _, _, _, _, box, evaluation = heapq.heappop(queue)
        center = box.center
        if (
            all(evaluation.allowed_collision)
            and all(
                value - assumed_fcl_distance_error_m
                > REQUIRED_FORBIDDEN_CLEARANCE_M
                for value in evaluation.forbidden_distance_m
            )
        ):
            counters["centre_contact_triplets"] += 1
            yaw_at_center = (
                float(center[0])
                if search_variable == "yaw"
                else resolved_fixed_yaw
            )
            palm_at_center = (
                float(seed.pregrasp[0])
                if search_variable == "yaw"
                else float(center[0])
            )
            joints = np.asarray(
                (palm_at_center, center[1], center[2], center[3])
            )
            prefix = [
                _prefix_forbidden_clearance_audit(
                    scene,
                    seed,
                    yaw_at_center,
                    joints,
                    index,
                    closing_bounds[index],
                    clearance_m=REQUIRED_FORBIDDEN_CLEARANCE_M,
                    assumed_distance_error_m=assumed_fcl_distance_error_m,
                    minimum_half_width_rad=minimum_joint_width_rad / 4.0,
                    maximum_boxes=20_000,
                )
                for index in range(3)
            ]
            if all(
                row["status"]
                == "FLOATING_POINT_LIPSCHITZ_PREFIX_CLEARANCE_AUDIT"
                for row in prefix
            ):
                scene.set_state(yaw_at_center, joints)
                incumbent = {
                    "search_variable": search_variable,
                    "common_coordinate_rad": float(center[0]),
                    "yaw_rad": yaw_at_center,
                    "yaw_deg": math.degrees(yaw_at_center),
                    "palm_rad": palm_at_center,
                    "joint_positions_rad": joints.tolist(),
                    "forbidden_clearance_m": list(
                        evaluation.forbidden_distance_m
                    ),
                    "allowed_contact_modes": [
                        scene.allowed_contact_modes(index) for index in range(3)
                    ],
                    "prefix_clearance_audit": prefix,
                    "source_box": {
                        "lower": list(box.lower),
                        "upper": list(box.upper),
                        "depth": box.depth,
                    },
                }
                stop_reason = "NUMERICAL_INNER_MODE_WITNESS_FOUND"
                break
            counters["prefix_audit_rejections"] += 1
        children = _split_box(
            box, evaluation, closing_bounds, minimum_widths
        )
        if children is None:
            counters["unresolved_minimum_width"] += 1
            continue
        for child in children:
            evaluate_and_push(child)
        if progress_interval > 0 and counters["evaluated_nodes"] % progress_interval < 2:
            print(
                json.dumps(
                    {
                        "progress": CLAIM_SCOPE,
                        "evaluated_nodes": counters["evaluated_nodes"],
                        "queued_boxes": len(queue),
                        "elapsed_s": time.monotonic() - started,
                        "best_center_rank": -queue[0][0] if queue else None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    elapsed = time.monotonic() - started
    if incumbent is not None:
        status = "NUMERICAL_INNER_MODE_WITNESS"
    elif stop_reason == "QUEUE_EXHAUSTED" and counters["unresolved_minimum_width"] == 0:
        status = "NUMERICAL_BOXES_EXHAUSTED_NOT_FORMAL_INFEASIBILITY"
    else:
        status = "UNRESOLVED"
    remaining_widths = np.asarray(
        [
            np.asarray(entry[4].upper) - np.asarray(entry[4].lower)
            for entry in queue
        ],
        dtype=np.float64,
    )
    rank_histogram: dict[str, int] = {}
    for entry in queue:
        rank = str(entry[5].centre_rank)
        rank_histogram[rank] = rank_histogram.get(rank, 0) + 1
    top_remaining: list[dict[str, Any]] = []
    for entry in sorted(queue)[:12]:
        box = entry[4]
        evaluation = entry[5]
        top_remaining.append(
            {
                "center": box.center.tolist(),
                "width": (
                    np.asarray(box.upper) - np.asarray(box.lower)
                ).tolist(),
                "depth": box.depth,
                "centre_rank_of_six": evaluation.centre_rank,
                "centre_score_m": evaluation.centre_score_m,
                "allowed_distance_m": list(evaluation.allowed_distance_m),
                "forbidden_distance_m": list(evaluation.forbidden_distance_m),
                "allowed_collision": list(evaluation.allowed_collision),
                "motion_radius_m": list(evaluation.motion_radius_m),
                "common_motion_radius_m": list(
                    evaluation.common_motion_radius_m
                ),
            }
        )
    if len(remaining_widths):
        width_summary: dict[str, Any] = {
            "minimum": np.min(remaining_widths, axis=0).tolist(),
            "median": np.median(remaining_widths, axis=0).tolist(),
            "maximum": np.max(remaining_widths, axis=0).tolist(),
        }
    else:
        width_summary = {"minimum": None, "median": None, "maximum": None}
    return {
        "schema_version": "kcg_te_fixed_seed_one_coordinate_contact_search_v2",
        "claim_scope": CLAIM_SCOPE,
        "hardware_authorized": False,
        "isaac_started": False,
        "status": status,
        "stop_reason": stop_reason,
        "incumbent": incumbent,
        "search_domain": {
            "search_variable": search_variable,
            "common_coordinate": common_name,
            "full_common_interval_rad": [common_lower, common_upper],
            "initial_box_lower": list(initial.lower),
            "initial_box_upper": list(initial.upper),
            "fixed_yaw_rad": resolved_fixed_yaw,
            "fixed_palm_joint_rad": (
                float(seed.pregrasp[0]) if search_variable == "yaw" else None
            ),
            "closing_joint_intervals_rad": [
                [float(seed.pregrasp[index]), float(seed.endpoint[index])]
                for index in range(1, 4)
            ],
            "fixed_seed_yaw_rad": seed.seed_yaw_rad,
            "fixed_object_from_hand_translation_m": seed.object_from_hand[:3, 3].tolist(),
        },
        "geometry": {
            "allowed_triangle_count": int(len(allowed)),
            "forbidden_triangle_count": int(len(forbidden)),
            "complete_pad_triangle_count": [
                int(pad.triangle_count) for pad in hand_contract.pads
            ],
            "required_forbidden_clearance_m": REQUIRED_FORBIDDEN_CLEARANCE_M,
        },
        "bounds": {
            "closing_point_motion_m_per_rad": list(closing_bounds),
            "palm_point_motion_m_per_rad": list(palm_bounds),
            "box_motion_rule": (
                common_motion_rule
                + "+closing_motion_bound*joint_half_width"
            ),
            "set_distance_lipschitz_constant": 1.0,
            "assumed_fcl_absolute_distance_error_m": assumed_fcl_distance_error_m,
            "fcl_error_bound_formally_proved": False,
            "exhaustion_is_global_proof": False,
        },
        "limits": {
            "maximum_nodes": maximum_nodes,
            "time_limit_s": time_limit_s,
            "minimum_yaw_width_rad": minimum_yaw_width_rad,
            "minimum_joint_width_rad": minimum_joint_width_rad,
        },
        "counters": counters,
        "remaining_queue_boxes": len(queue),
        "remaining_box_diagnostics": {
            "coordinate_order": [common_name, *EXPECTED_CLOSING_JOINTS],
            "width_summary": width_summary,
            "centre_rank_histogram": rank_histogram,
            "top_by_search_priority": top_remaining,
        },
        "elapsed_s": elapsed,
        "seed_result": str(seed.source_path.relative_to(root)),
        "evidence_boundary": (
            "A witness is a floating-point triangulated contact-mode seed only. "
            "No exact B-rep, force, PD, path, Isaac, lift, hold, or global "
            "robustness conclusion follows. Search exhaustion is not a proof "
            "because FCL has no established outward-rounding error enclosure."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--analytic-contract", default=DEFAULT_ANALYTIC_CONTRACT)
    parser.add_argument("--seed-result", default=DEFAULT_SEED_RESULT)
    parser.add_argument("--maximum-nodes", type=int, default=100_000)
    parser.add_argument("--time-limit-s", type=float, default=180.0)
    parser.add_argument("--minimum-yaw-width-rad", type=float, default=2.5e-5)
    parser.add_argument("--minimum-joint-width-rad", type=float, default=2.5e-6)
    parser.add_argument("--assumed-fcl-distance-error-m", type=float, default=1.0e-9)
    parser.add_argument("--progress-interval", type=int, default=2_000)
    parser.add_argument("--box-lower", type=float, nargs=4)
    parser.add_argument("--box-upper", type=float, nargs=4)
    parser.add_argument(
        "--search-variable", choices=("yaw", "palm"), default="yaw"
    )
    parser.add_argument("--fixed-yaw-rad", type=float)
    arguments = parser.parse_args(argv)
    if arguments.maximum_nodes < 1 or arguments.time_limit_s <= 0.0:
        parser.error("search limits must be positive")
    if (
        arguments.minimum_yaw_width_rad <= 0.0
        or arguments.minimum_joint_width_rad <= 0.0
        or arguments.assumed_fcl_distance_error_m < 0.0
    ):
        parser.error("widths must be positive and assumed error nonnegative")
    result = search_fixed_seed_full_yaw(
        arguments.repository_root,
        analytic_contract_path=arguments.analytic_contract,
        seed_result_path=arguments.seed_result,
        maximum_nodes=arguments.maximum_nodes,
        time_limit_s=arguments.time_limit_s,
        minimum_yaw_width_rad=arguments.minimum_yaw_width_rad,
        minimum_joint_width_rad=arguments.minimum_joint_width_rad,
        assumed_fcl_distance_error_m=arguments.assumed_fcl_distance_error_m,
        progress_interval=arguments.progress_interval,
        initial_lower=arguments.box_lower,
        initial_upper=arguments.box_upper,
        search_variable=arguments.search_variable,
        fixed_yaw_rad=arguments.fixed_yaw_rad,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["incumbent"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_SCOPE",
    "FixedSeedYawSearchError",
    "search_fixed_seed_full_yaw",
]
