"""Simulator-independent geometry helpers for connector assembly."""

import math

import numpy as np


_EPSILON = 1.0e-12


def _vector(values, size, label):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain finite values")
    return result


def normalize_quaternion(quaternion):
    """Return a normalized ROS-order quaternion ``[x, y, z, w]``."""
    result = _vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(result))
    if norm <= _EPSILON:
        raise ValueError("quaternion norm must be non-zero")
    return result / norm


def quaternion_conjugate(quaternion):
    """Return the conjugate of a normalized ROS-order quaternion."""
    x, y, z, w = normalize_quaternion(quaternion)
    return np.array([-x, -y, -z, w], dtype=np.float64)


def quaternion_multiply(first, second):
    """Multiply two ROS-order quaternions."""
    ax, ay, az, aw = normalize_quaternion(first)
    bx, by, bz, bw = normalize_quaternion(second)
    result = np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )
    return normalize_quaternion(result)


def rotate_vector(quaternion, vector):
    """Rotate a three-vector by a ROS-order quaternion."""
    x, y, z, w = normalize_quaternion(quaternion)
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation @ _vector(vector, 3, "vector")


def relative_pose(source_position, source_orientation, target_position, target_orientation):
    """Express ``target`` pose in the ``source`` frame."""
    source_position = _vector(source_position, 3, "source_position")
    target_position = _vector(target_position, 3, "target_position")
    source_orientation = normalize_quaternion(source_orientation)
    target_orientation = normalize_quaternion(target_orientation)
    inverse = quaternion_conjugate(source_orientation)
    position = rotate_vector(inverse, target_position - source_position)
    orientation = quaternion_multiply(inverse, target_orientation)
    if orientation[3] < 0.0:
        orientation = -orientation
    return position, orientation


def axis_angle_error(first_orientation, second_orientation, local_axis=(0.0, 0.0, 1.0)):
    """Return the unsigned angle in radians between two directed axes."""
    axis = _vector(local_axis, 3, "local_axis")
    norm = float(np.linalg.norm(axis))
    if norm <= _EPSILON:
        raise ValueError("local_axis norm must be non-zero")
    axis /= norm
    first_axis = rotate_vector(first_orientation, axis)
    second_axis = rotate_vector(second_orientation, axis)
    cosine = float(np.clip(np.dot(first_axis, second_axis), -1.0, 1.0))
    return math.acos(cosine)


def split_axial_error(offset, axis=(0.0, 0.0, 1.0)):
    """Split an offset into signed axial and non-negative lateral errors."""
    offset = _vector(offset, 3, "offset")
    axis = _vector(axis, 3, "axis")
    norm = float(np.linalg.norm(axis))
    if norm <= _EPSILON:
        raise ValueError("axis norm must be non-zero")
    axis /= norm
    axial = float(np.dot(offset, axis))
    lateral = float(np.linalg.norm(offset - axial * axis))
    return axial, lateral


def helical_travel(angle_radians, lead_per_revolution):
    """Return ideal screw travel for an unwrapped coupling-nut angle."""
    if not math.isfinite(angle_radians) or not math.isfinite(lead_per_revolution):
        raise ValueError("helical inputs must be finite")
    if lead_per_revolution <= 0.0:
        raise ValueError("lead_per_revolution must be positive")
    return lead_per_revolution * angle_radians / (2.0 * math.pi)


def unwrap_angle(previous_unwrapped, current_wrapped):
    """Extend a wrapped angle continuously from the previous value."""
    if not math.isfinite(previous_unwrapped) or not math.isfinite(current_wrapped):
        raise ValueError("angles must be finite")
    previous_wrapped = math.atan2(
        math.sin(previous_unwrapped), math.cos(previous_unwrapped)
    )
    delta = math.atan2(
        math.sin(current_wrapped - previous_wrapped),
        math.cos(current_wrapped - previous_wrapped),
    )
    return previous_unwrapped + delta
