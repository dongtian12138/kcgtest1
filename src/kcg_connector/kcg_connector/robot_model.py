"""Authoritative joint-name mapping shared by Isaac task scripts."""

import math

import numpy as np


ARM_JOINT_NAMES = tuple(f"iiwa_joint_{index}" for index in range(1, 8))
ACTIVE_HAND_JOINT_NAMES = ("f1j1", "f1j2", "f2j1", "f3j2")
MIMIC_HAND_JOINTS = {
    "f3j1": "f1j1",
    "f1j3": "f1j2",
    "f2j2": "f2j1",
    "f3j3": "f3j2",
}
ALL_HAND_JOINT_NAMES = (
    "f1j1",
    "f1j2",
    "f1j3",
    "f2j1",
    "f2j2",
    "f3j1",
    "f3j2",
    "f3j3",
)


def _finite_positions(values, expected_size, label):
    positions = np.asarray(values, dtype=np.float64)
    if positions.shape != (expected_size,):
        raise ValueError(f"{label} must contain {expected_size} positions")
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{label} must contain finite positions")
    return positions


def expand_active_hand_positions(active_positions):
    """Expand four physical commands to all eight modeled hand joints."""
    active = _finite_positions(
        active_positions, len(ACTIVE_HAND_JOINT_NAMES), "active hand target"
    )
    targets = dict(zip(ACTIVE_HAND_JOINT_NAMES, active))
    targets.update(
        {
            mimic_name: targets[source_name]
            for mimic_name, source_name in MIMIC_HAND_JOINTS.items()
        }
    )
    return targets


def named_joint_target(dof_names, arm_positions, active_hand_positions):
    """Build an articulation-order target without assuming importer ordering."""
    names = tuple(str(name) for name in dof_names)
    if len(names) != len(set(names)):
        raise ValueError("articulation DOF names must be unique")
    arm = _finite_positions(
        arm_positions, len(ARM_JOINT_NAMES), "arm target"
    )
    targets = dict(zip(ARM_JOINT_NAMES, arm))
    targets.update(expand_active_hand_positions(active_hand_positions))
    missing = sorted(set(targets) - set(names))
    if missing:
        raise ValueError(f"articulation is missing joints: {', '.join(missing)}")
    result = np.zeros(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        if name in targets:
            result[index] = targets[name]
    if not all(math.isfinite(float(value)) for value in result):
        raise ValueError("expanded articulation target is not finite")
    return result
