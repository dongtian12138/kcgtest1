"""Reset-only post-grasp error transforms for the D38999 proxy.

The functions in this module are simulator independent.  They define the
fixed ``GRASP_LATCH_PROXY`` transform; they are not a controller and expose no
object-truth value to a motion policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PostGraspError:
    """Requested Plug pose error in the nominal assembly/TCP frame."""

    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_xyz_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source: str = "reset_only_simulation"

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation_m, dtype=np.float64)
        rotation = np.asarray(self.rotation_xyz_rad, dtype=np.float64)
        if translation.shape != (3,) or rotation.shape != (3,):
            raise ValueError("post-grasp translation and rotation must have length three")
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
            raise ValueError("post-grasp error must be finite")
        if self.source != "reset_only_simulation":
            raise ValueError("post-grasp error is only allowed at simulation reset")


def rotation_xyz(rotation_xyz_rad) -> np.ndarray:
    """Return ``Rz @ Ry @ Rx`` for intrinsic XYZ angles."""

    rx, ry, rz = np.asarray(rotation_xyz_rad, dtype=np.float64)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation_x = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    rotation_y = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rotation_z = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return rotation_z @ rotation_y @ rotation_x


def rotation_vector_matrix(rotation_vector_rad) -> np.ndarray:
    """Rodrigues exponential for a world-frame rotation vector."""

    vector = np.asarray(rotation_vector_rad, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation vector must be finite and length three")
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-15:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / angle
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def xyz_from_rotation(rotation) -> np.ndarray:
    """Invert :func:`rotation_xyz` away from its Euler singularity."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    ry = math.asin(max(-1.0, min(1.0, -float(matrix[2, 0]))))
    if abs(math.cos(ry)) > 1.0e-9:
        rx = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        rz = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        rx = math.atan2(-float(matrix[1, 2]), float(matrix[1, 1]))
        rz = 0.0
    return np.asarray((rx, ry, rz), dtype=np.float64)


def transform(translation_m, rotation_xyz_rad) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_xyz(rotation_xyz_rad)
    result[:3, 3] = np.asarray(translation_m, dtype=np.float64)
    return result


def assembly_tcp_from_grasp_tcp(
    grasp_tcp,
    grasp_to_assembly_translation_m,
) -> np.ndarray:
    """Return FK of an offset assembly TCP from the commanded grasp TCP."""

    grasp = np.asarray(grasp_tcp, dtype=np.float64)
    offset = np.asarray(grasp_to_assembly_translation_m, dtype=np.float64)
    if grasp.shape != (4, 4) or offset.shape != (3,):
        raise ValueError("grasp pose must be 4x4 and assembly offset length three")
    result = grasp.copy()
    result[:3, 3] = grasp[:3, 3] + grasp[:3, :3] @ offset
    return result


def integrate_assembly_twist_on_grasp_tcp(
    grasp_tcp,
    twist_assembly,
    assembly_axes_world,
    grasp_to_assembly_translation_m,
    dt_s,
) -> np.ndarray:
    """Integrate a task-frame twist while rotating about the assembly TCP.

    The robot IK target is the grasp TCP, but the commanded twist belongs at
    the mating-face assembly TCP.  The exact offset update below includes the
    translational ``omega cross r`` term instead of rotating about the hand.
    """

    grasp = np.asarray(grasp_tcp, dtype=np.float64)
    twist = np.asarray(twist_assembly, dtype=np.float64)
    axes = np.asarray(assembly_axes_world, dtype=np.float64)
    offset = np.asarray(grasp_to_assembly_translation_m, dtype=np.float64)
    dt = float(dt_s)
    if (
        grasp.shape != (4, 4)
        or twist.shape != (6,)
        or axes.shape != (3, 3)
        or offset.shape != (3,)
        or not math.isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError("invalid assembly-twist integration input")
    old_assembly = assembly_tcp_from_grasp_tcp(grasp, offset)
    rotation_delta = rotation_vector_matrix(axes @ (twist[3:] * dt))
    result = grasp.copy()
    result[:3, :3] = rotation_delta @ grasp[:3, :3]
    desired_assembly_position = (
        old_assembly[:3, 3] + axes @ (twist[:3] * dt)
    )
    result[:3, 3] = desired_assembly_position - result[:3, :3] @ offset
    return result


def compose_nominal_with_error(
    nominal_translation_m,
    requested: PostGraspError,
) -> np.ndarray:
    """Return ``T_hand_plug_nominal @ T_delta_inhand``."""

    nominal = np.eye(4, dtype=np.float64)
    nominal[:3, 3] = np.asarray(nominal_translation_m, dtype=np.float64)
    return nominal @ transform(
        requested.translation_m, requested.rotation_xyz_rad
    )


def measure_error_from_nominal(
    actual_translation_m,
    actual_rotation,
    nominal_translation_m,
) -> PostGraspError:
    """Recover the post-grasp delta from a measured hand-to-Plug pose."""

    nominal = np.eye(4, dtype=np.float64)
    nominal[:3, 3] = np.asarray(nominal_translation_m, dtype=np.float64)
    actual = np.eye(4, dtype=np.float64)
    actual[:3, :3] = np.asarray(actual_rotation, dtype=np.float64)
    actual[:3, 3] = np.asarray(actual_translation_m, dtype=np.float64)
    delta = np.linalg.inv(nominal) @ actual
    return PostGraspError(
        translation_m=tuple(float(value) for value in delta[:3, 3]),
        rotation_xyz_rad=tuple(
            float(value) for value in xyz_from_rotation(delta[:3, :3])
        ),
    )


def injection_error(requested: PostGraspError, actual: PostGraspError) -> dict:
    translation_difference = np.asarray(actual.translation_m) - np.asarray(
        requested.translation_m
    )
    rotation_difference = np.asarray(actual.rotation_xyz_rad) - np.asarray(
        requested.rotation_xyz_rad
    )
    return {
        "translation_difference_m": translation_difference.tolist(),
        "rotation_difference_rad": rotation_difference.tolist(),
        "translation_error_norm_m": float(np.linalg.norm(translation_difference)),
        "rotation_error_norm_rad": float(np.linalg.norm(rotation_difference)),
    }
