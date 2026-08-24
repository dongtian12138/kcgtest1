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
    preshape_f1j1_rad: float
    open_joint_positions_rad: Mapping[str, float]
    half_joint_positions_rad: Mapping[str, float]
    close_joint_positions_rad: Mapping[str, float]
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
            "close": dict(self.close_joint_positions_rad),
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
    hand: ThreeFingerHandModel, *, sample_count: int = 9
) -> tuple[float, ...]:
    if sample_count != 9:
        raise ValueError("the registered descriptor design uses exactly 9 samples")
    limit = hand.independent_joint_limits[_SHARED_JOINT]
    span = limit.upper - limit.lower
    values = np.linspace(limit.lower + 0.1 * span, limit.lower + 0.9 * span, 9)
    return tuple(float(value) for value in values)


def select_preshape_values(
    hand: ThreeFingerHandModel,
    *,
    self_collision_free: Callable[[float], bool] | None = None,
    legal_samples_rad: Sequence[float] | None = None,
    maximum_count: int = 5,
    nominal_rad: float = 0.70,
) -> tuple[float, ...]:
    if (self_collision_free is None) == (legal_samples_rad is None):
        raise ValueError("provide exactly one collision callback or legal sample list")
    grid = np.asarray(shared_preshape_grid(hand), dtype=np.float64)
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
    preshape_f1j1_rad: float,
    maximum_closure_phase: float,
) -> tuple[Mapping[str, float], Mapping[str, float], Mapping[str, float]]:
    if not 0.0 <= maximum_closure_phase <= 1.0:
        raise ValueError("maximum closure phase must lie in [0, 1]")
    names = tuple(hand.independent_joint_names)
    lower, upper = hand.joint_limit_vectors()
    open_values = lower.copy()
    open_values[names.index(_SHARED_JOINT)] = float(preshape_f1j1_rad)
    close_values = open_values.copy()
    directions = contract.closing_actuation_directions_unit(hand)
    for row in directions:
        for index in np.flatnonzero(row):
            close_values[index] = lower[index] + maximum_closure_phase * (
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
    forward = np.eye(4)
    forward[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
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


def build_kcg_graspgenx_descriptors(
    contract: CARTSHandContract,
    hand: ThreeFingerHandModel,
    *,
    maximum_closure_phase: float,
    self_collision_free: Callable[[float], bool] | None = None,
    legal_samples_rad: Sequence[float] | None = None,
) -> tuple[KCGGraspGenXDescriptor, ...]:
    """Build at most five object-independent, physically registered descriptors."""

    preshapes = select_preshape_values(
        hand,
        self_collision_free=self_collision_free,
        legal_samples_rad=legal_samples_rad,
    )
    descriptors = []
    for index, preshape in enumerate(preshapes):
        open_map, half_map, close_map = descriptor_joint_states(
            contract, hand, preshape, maximum_closure_phase
        )
        frame = build_descriptor_frame(contract, hand, open_map)
        open_extents, open_offset = inner_work_aabb(
            contract, hand, open_map, frame
        )
        half_extents, half_offset = inner_work_aabb(
            contract, hand, half_map, frame
        )
        pad_centers = []
        pad_transforms = hand.pad_transforms(open_map)
        for pad in contract.pads:
            center = np.mean(pad.points_local_m, axis=0)
            base_center = (
                pad_transforms[pad.name][:3, :3] @ center
                + pad_transforms[pad.name][:3, 3]
            )
            pad_centers.append(
                frame.graspgenx_from_handbase[:3, :3] @ base_center
                + frame.graspgenx_from_handbase[:3, 3]
            )
        fingertip = tuple(float(x) for x in np.mean(pad_centers, axis=0))
        descriptors.append(KCGGraspGenXDescriptor(
            descriptor_id=f"kcg_3f_preshape_{index:02d}", preshape_f1j1_rad=preshape,
            open_joint_positions_rad=open_map, half_joint_positions_rad=half_map,
            close_joint_positions_rad=close_map, frame=frame,
            fingertip_graspgenx_m=fingertip, open_aabb_extents_m=open_extents,
            open_aabb_offset_m=open_offset, half_aabb_extents_m=half_extents,
            half_aabb_offset_m=half_offset,
        ))
    return tuple(descriptors)
