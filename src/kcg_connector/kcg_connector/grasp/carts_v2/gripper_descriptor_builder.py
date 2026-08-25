"""Build object-independent KCG hand conditioning; AABBs are not collision proofs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.robust.hand_contract import CARTSHandContract
from kcg_connector.grasp.robust.hand_model import ThreeFingerHandModel


_SHARED_JOINT = "f1j1"
_REFERENCE_PAD = "finger_2_pad"
_FULL_PALM_SAMPLE_COUNT = 91
_LEGACY_PRESHAPE_SAMPLE_COUNT = 9


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite with shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DescriptorFrame:
    handbase_from_graspgenx: np.ndarray
    graspgenx_from_handbase: np.ndarray

    def __post_init__(self) -> None:
        forward = _finite_array(self.handbase_from_graspgenx, (4, 4), "frame")
        inverse = _finite_array(self.graspgenx_from_handbase, (4, 4), "inverse")
        if not np.allclose(forward @ inverse, np.eye(4), atol=1e-12):
            raise ValueError("descriptor frame transforms are not inverse")
        if not np.allclose(forward[:3, :3].T @ forward[:3, :3], np.eye(3), atol=1e-12):
            raise ValueError("descriptor frame is not orthonormal")
        if not np.isclose(np.linalg.det(forward[:3, :3]), 1.0, atol=1e-12):
            raise ValueError("descriptor frame must be right handed")
        object.__setattr__(self, "handbase_from_graspgenx", forward)
        object.__setattr__(self, "graspgenx_from_handbase", inverse)


@dataclass(frozen=True)
class KCGGraspGenXDescriptor:
    descriptor_id: str
    palm_configuration_rad: float
    conditioning_close_phase: float
    open_joint_positions_rad: Mapping[str, float]
    half_joint_positions_rad: Mapping[str, float]
    conditioning_close_joint_positions_rad: Mapping[str, float]
    frame: DescriptorFrame
    fingertip_graspgenx_m: tuple[float, float, float]
    open_aabb_extents_m: tuple[float, float, float]
    open_aabb_offset_m: tuple[float, float, float]
    half_aabb_extents_m: tuple[float, float, float]
    half_aabb_offset_m: tuple[float, float, float]

    def to_official_config(self) -> dict[str, object]:
        """Return official field names without importing GraspGenX."""

        return {
            "open": dict(self.open_joint_positions_rad),
            "close": dict(self.conditioning_close_joint_positions_rad),
            "fingertip": list(self.fingertip_graspgenx_m),
            "sweep_volume": {
                "extents": list(self.open_aabb_extents_m),
                "offset": list(self.open_aabb_offset_m),
                "extents2": list(self.half_aabb_extents_m),
                "offset2": list(self.half_aabb_offset_m),
            },
            "type": "revolute_3f",
            "base_rotation": self.frame.graspgenx_from_handbase.tolist(),
            "symmetric": False,
        }


def shared_preshape_grid(
    hand: ThreeFingerHandModel, *, sample_count: int = _FULL_PALM_SAMPLE_COUNT
) -> tuple[float, ...]:
    """Return the production 91-point closed-interval palm grid."""

    if sample_count != _FULL_PALM_SAMPLE_COUNT:
        raise ValueError("the full-palm descriptor design uses exactly 91 samples")
    limit = hand.independent_joint_limits[_SHARED_JOINT]
    values = np.linspace(limit.lower, limit.upper, _FULL_PALM_SAMPLE_COUNT)
    return tuple(float(value) for value in values)


def legacy_shared_preshape_grid(
    hand: ThreeFingerHandModel,
) -> tuple[float, ...]:
    """Return the frozen 10--90% nine-point baseline grid."""

    limit = hand.independent_joint_limits[_SHARED_JOINT]
    span = limit.upper - limit.lower
    values = np.linspace(
        limit.lower + 0.1 * span,
        limit.lower + 0.9 * span,
        _LEGACY_PRESHAPE_SAMPLE_COUNT,
    )
    return tuple(float(value) for value in values)


def select_preshape_values(
    hand: ThreeFingerHandModel,
    *,
    self_collision_free: Callable[[float], bool] | None = None,
    legal_samples_rad: Sequence[float] | None = None,
    maximum_count: int = 5,
    nominal_rad: float = 0.70,
) -> tuple[float, ...]:
    """Select at most five palm angles for the legacy baseline only."""

    if (self_collision_free is None) == (legal_samples_rad is None):
        raise ValueError("provide exactly one collision callback or legal sample list")
    grid = np.asarray(legacy_shared_preshape_grid(hand), dtype=np.float64)
    if self_collision_free is not None:
        legal = grid[[bool(self_collision_free(float(value))) for value in grid]]
    else:
        supplied = np.sort(np.asarray(legal_samples_rad, dtype=np.float64))
        if supplied.ndim != 1 or not np.all(np.isfinite(supplied)):
            raise ValueError("legal preshape samples must be one finite vector")
        mask = np.asarray(
            [np.any(np.isclose(value, supplied, atol=1e-12)) for value in grid]
        )
        if len(supplied) != int(np.sum(mask)):
            raise ValueError(
                "legal preshape samples must be a unique subset of the fixed grid"
            )
        legal = grid[mask]
    if len(legal) == 0 or not 1 <= maximum_count <= 5:
        raise ValueError(
            "at least one legal sample and maximum_count in [1, 5] are required"
        )
    selected = [int(np.argmin(np.abs(legal - float(nominal_rad))))]
    while len(selected) < min(maximum_count, len(legal)):
        distance = np.min(np.abs(legal[:, None] - legal[selected][None, :]), axis=1)
        distance[selected] = -1.0
        selected.append(int(np.argmax(distance)))
    return tuple(float(legal[index]) for index in selected)


def descriptor_joint_states(
    contract: CARTSHandContract,
    hand: ThreeFingerHandModel,
    palm_configuration_rad: float,
    conditioning_close_phase: float,
) -> tuple[Mapping[str, float], Mapping[str, float], Mapping[str, float]]:
    """Build GraspGenX open/half/close conditioning, not a physical gate."""

    if not 0.0 <= conditioning_close_phase <= 1.0:
        raise ValueError("conditioning close phase must lie in [0, 1]")
    names = tuple(hand.independent_joint_names)
    lower, upper = hand.joint_limit_vectors()
    open_values = lower.copy()
    open_values[names.index(_SHARED_JOINT)] = float(palm_configuration_rad)
    close_values = open_values.copy()
    directions = contract.closing_actuation_directions_unit(hand)
    for row in directions:
        for index in np.flatnonzero(row):
            close_values[index] = lower[index] + conditioning_close_phase * (
                upper[index] - lower[index]
            )
    half_values = 0.5 * (open_values + close_values)
    movable = {name for name in hand.joint_order if hand.joints[name].movable}
    maps = []
    for values in (open_values, half_values, close_values):
        resolved = hand.resolve_joint_positions(values, enforce_limits=True)
        maps.append(
            MappingProxyType(
                {name: float(resolved[name]) for name in resolved if name in movable}
            )
        )
    return maps[0], maps[1], maps[2]


def pad_points_in_graspgenx(
    contract: CARTSHandContract,
    hand: ThreeFingerHandModel,
    joint_positions_rad: Mapping[str, float],
    frame: DescriptorFrame,
) -> np.ndarray:
    """Transform all registered whole-PAD vertices into the descriptor frame."""

    return np.vstack(
        _pad_point_sets_in_graspgenx(contract, hand, joint_positions_rad, frame)
    )


def _pad_point_sets_in_graspgenx(
    contract: CARTSHandContract,
    hand: ThreeFingerHandModel,
    joint_positions_rad: Mapping[str, float],
    frame: DescriptorFrame,
) -> tuple[np.ndarray, ...]:
    transforms = hand.pad_transforms(joint_positions_rad)
    rows = []
    for pad in contract.pads:
        handbase_points = (
            pad.points_local_m @ transforms[pad.name][:3, :3].T
            + transforms[pad.name][:3, 3]
        )
        rows.append(
            handbase_points @ frame.graspgenx_from_handbase[:3, :3].T
            + frame.graspgenx_from_handbase[:3, 3]
        )
    if not rows or any(not np.all(np.isfinite(row)) for row in rows):
        raise ValueError("transformed PAD points are not finite")
    return tuple(rows)


def build_descriptor_frame(
    contract: CARTSHandContract,
    hand: ThreeFingerHandModel,
    open_joint_positions_rad: Mapping[str, float],
) -> DescriptorFrame:
    transforms = hand.pad_transforms(open_joint_positions_rad)
    centers = {}
    for pad in contract.pads:
        points = (
            pad.points_local_m @ transforms[pad.name][:3, :3].T
            + transforms[pad.name][:3, 3]
        )
        centers[pad.name] = np.mean(points, axis=0)
    workspace = np.mean(tuple(centers.values()), axis=0)
    workspace_norm = np.linalg.norm(workspace)
    if workspace_norm <= 1e-12:
        raise ValueError("registered PAD geometry cannot define the approach axis")
    z_axis = workspace / workspace_norm
    x_seed = workspace - centers[_REFERENCE_PAD]
    x_axis = x_seed - z_axis * float(x_seed @ z_axis)
    if np.linalg.norm(x_axis) <= 1e-12:
        raise ValueError("registered PAD geometry cannot define the GraspGenX frame")
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    link_transforms = hand.forward_kinematics(open_joint_positions_rad)
    proximal_origins = []
    for finger in hand.fingers.values():
        joint = hand.joints[finger.joint_names[0]]
        joint_frame = link_transforms[joint.parent_link] @ joint.origin_transform()
        proximal_origins.append(joint_frame[:3, 3])
    proximal_plane_depth = float(np.mean(np.asarray(proximal_origins) @ z_axis))
    forward = np.eye(4)
    forward[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    forward[:3, 3] = proximal_plane_depth * z_axis
    return DescriptorFrame(forward, np.linalg.inv(forward))


def inner_work_aabb(
    contract: CARTSHandContract,
    hand: ThreeFingerHandModel,
    joint_positions_rad: Mapping[str, float],
    frame: DescriptorFrame,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Match the official wizard's box around the space between fingertips."""

    groups = _pad_point_sets_in_graspgenx(
        contract, hand, joint_positions_rad, frame
    )
    lower_by_pad = np.asarray([np.min(points, axis=0) for points in groups])
    upper_by_pad = np.asarray([np.max(points, axis=0) for points in groups])
    centers = 0.5 * (lower_by_pad + upper_by_pad)
    # The frame was defined from the opposed finger toward the workspace
    # center, so the official canonical closing dimension is fixed at +X.
    # Re-selecting the largest spread at half-open can incorrectly switch to
    # the separation between the other two fingers of a radial three-finger hand.
    closing_axis = 0
    order = np.argsort(centers[:, closing_axis])
    inner_min = float(upper_by_pad[order[0], closing_axis])
    inner_max = float(lower_by_pad[order[-1], closing_axis])
    if inner_max <= inner_min:
        inner_min = float(centers[order[0], closing_axis])
        inner_max = float(centers[order[-1], closing_axis])
    lower, upper = np.min(lower_by_pad, axis=0), np.max(upper_by_pad, axis=0)
    lower[closing_axis], upper[closing_axis] = inner_min, inner_max
    extents = upper - lower
    if np.any(extents <= 0.0) or not np.all(np.isfinite(extents)):
        raise ValueError("inner work AABB must have positive finite extents")
    return (
        tuple(float(x) for x in extents),
        tuple(float(x) for x in 0.5 * (lower + upper)),
    )


def _build_descriptors(
    contract: CARTSHandContract,
    hand: ThreeFingerHandModel,
    *,
    palm_configurations_rad: Sequence[float],
    conditioning_close_phase_by_palm: Mapping[float, float],
    descriptor_prefix: str,
    descriptor_index_width: int,
) -> tuple[KCGGraspGenXDescriptor, ...]:
    descriptors = []
    for index, palm_configuration in enumerate(palm_configurations_rad):
        matches = [
            float(phase)
            for value, phase in conditioning_close_phase_by_palm.items()
            if np.isclose(float(value), palm_configuration, atol=1.0e-12)
        ]
        if len(matches) != 1 or not 0.0 < matches[0] <= 1.0:
            raise ValueError(
                "each palm configuration needs one conditioning close phase"
            )
        conditioning_close_phase = matches[0]
        open_map, half_map, close_map = descriptor_joint_states(
            contract, hand, palm_configuration, conditioning_close_phase
        )
        frame = build_descriptor_frame(contract, hand, open_map)
        open_extents, open_offset = inner_work_aabb(
            contract, hand, open_map, frame
        )
        half_extents, half_offset = inner_work_aabb(
            contract, hand, half_map, frame
        )
        pad_points = _pad_point_sets_in_graspgenx(
            contract, hand, open_map, frame
        )
        fingertip = (
            0.0,
            0.0,
            float(np.mean([np.max(points[:, 2]) for points in pad_points])),
        )
        descriptors.append(KCGGraspGenXDescriptor(
            descriptor_id=(
                f"{descriptor_prefix}_{index:0{descriptor_index_width}d}"
            ),
            palm_configuration_rad=float(palm_configuration),
            conditioning_close_phase=conditioning_close_phase,
            open_joint_positions_rad=open_map, half_joint_positions_rad=half_map,
            conditioning_close_joint_positions_rad=close_map, frame=frame,
            fingertip_graspgenx_m=fingertip, open_aabb_extents_m=open_extents,
            open_aabb_offset_m=open_offset, half_aabb_extents_m=half_extents,
            half_aabb_offset_m=half_offset,
        ))
    return tuple(descriptors)


def build_kcg_graspgenx_descriptors(
    contract: CARTSHandContract,
    hand: ThreeFingerHandModel,
    *,
    conditioning_close_phase_by_palm: Mapping[float, float],
) -> tuple[KCGGraspGenXDescriptor, ...]:
    """Build all 91 full-palm descriptors or fail closed."""

    palm_grid = shared_preshape_grid(hand)
    supplied = np.sort(
        np.asarray(tuple(conditioning_close_phase_by_palm), dtype=np.float64)
    )
    if (
        supplied.shape != (_FULL_PALM_SAMPLE_COUNT,)
        or not np.all(np.isfinite(supplied))
        or not np.allclose(supplied, palm_grid, atol=1.0e-12, rtol=0.0)
    ):
        raise ValueError("full-palm production requires all 91 registered angles")
    return _build_descriptors(
        contract,
        hand,
        palm_configurations_rad=palm_grid,
        conditioning_close_phase_by_palm=conditioning_close_phase_by_palm,
        descriptor_prefix="kcg_3f_palm",
        descriptor_index_width=3,
    )


def build_legacy_kcg_graspgenx_descriptors(
    contract: CARTSHandContract,
    hand: ThreeFingerHandModel,
    *,
    closure_phase_by_preshape: Mapping[float, float],
    self_collision_free: Callable[[float], bool] | None = None,
    legal_samples_rad: Sequence[float] | None = None,
) -> tuple[KCGGraspGenXDescriptor, ...]:
    """Retain the historical five-angle descriptor baseline explicitly."""

    preshapes = select_preshape_values(
        hand,
        self_collision_free=self_collision_free,
        legal_samples_rad=legal_samples_rad,
    )
    return _build_descriptors(
        contract,
        hand,
        palm_configurations_rad=preshapes,
        conditioning_close_phase_by_palm=closure_phase_by_preshape,
        descriptor_prefix="kcg_3f_preshape",
        descriptor_index_width=2,
    )
