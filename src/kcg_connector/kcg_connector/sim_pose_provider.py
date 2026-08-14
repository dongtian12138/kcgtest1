"""Pure adapter from simulator world poses to the shared pose contract.

Isaac Sim reports quaternions as ``[w, x, y, z]`` while the connector pose
contract deliberately uses ``[x, y, z, w]``.  Keeping that conversion in one
strict adapter prevents a future vision provider from inheriting simulator
conventions or silently swapping quaternion components.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Sequence

from kcg_connector.connector_pose import (
    POSE_OBSERVATION_SCHEMA_VERSION,
    ConnectorPoseContract,
    ConnectorPoseObservation,
    ConnectorPoseRole,
    parse_connector_pose_observation,
)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain {size} finite numbers")
    try:
        items = tuple(value)
    except TypeError as error:
        raise ValueError(
            f"{label} must contain {size} finite numbers"
        ) from error
    if len(items) != size:
        raise ValueError(f"{label} must contain {size} finite numbers")
    return tuple(
        _finite(item, f"{label}[{index}]")
        for index, item in enumerate(items)
    )


def _variance(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _role(value: ConnectorPoseRole | str) -> ConnectorPoseRole:
    try:
        return ConnectorPoseRole(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"unsupported connector pose role: {value!r}"
        ) from error


def isaac_wxyz_to_contract_xyzw(
    quaternion_wxyz: Sequence[Real],
) -> tuple[float, float, float, float]:
    """Convert one finite Isaac quaternion without guessing its convention."""
    w_value, x_value, y_value, z_value = _vector(
        quaternion_wxyz, 4, "quaternion_wxyz"
    )
    return (x_value, y_value, z_value, w_value)


def make_sim_ground_truth_observation(
    contract: ConnectorPoseContract,
    *,
    model_id: str,
    role: ConnectorPoseRole | str,
    timestamp_s: Real,
    now_s: Real,
    frame_id: str,
    position_xyz_m: Sequence[Real],
    quaternion_wxyz: Sequence[Real],
    translation_variance_m2: Real = 0.0,
    rotation_variance_rad2: Real = 0.0,
    confidence: Real = 1.0,
) -> ConnectorPoseObservation:
    """Build and validate one ``sim_ground_truth`` pose observation.

    The adapter reads identity and symmetry from the registered model.  It
    cannot create an object-to-grasp or object-to-assembly calibration.
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be non-empty text")
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError("frame_id must be non-empty text")
    parsed_role = _role(role)
    model = contract.model(model_id.strip())
    if model.role is not parsed_role:
        raise ValueError("requested role does not match pose model registry")
    translation_variance = _variance(
        translation_variance_m2, "translation_variance_m2"
    )
    rotation_variance = _variance(
        rotation_variance_rad2, "rotation_variance_rad2"
    )
    diagonal = (translation_variance,) * 3 + (rotation_variance,) * 3
    covariance = [
        [diagonal[row] if row == column else 0.0 for column in range(6)]
        for row in range(6)
    ]
    document = {
        "schema_version": POSE_OBSERVATION_SCHEMA_VERSION,
        "model_id": model.model_id,
        "role": parsed_role.value,
        "timestamp_s": _finite(timestamp_s, "timestamp_s"),
        "frame_id": frame_id.strip(),
        "position_xyz_m": _vector(
            position_xyz_m, 3, "position_xyz_m"
        ),
        "quaternion_xyzw": isaac_wxyz_to_contract_xyzw(
            quaternion_wxyz
        ),
        "covariance_6x6": covariance,
        "confidence": _finite(confidence, "confidence"),
        "symmetry_class": model.symmetry_class,
        "source": "sim_ground_truth",
    }
    return parse_connector_pose_observation(
        document,
        contract,
        now_s=_finite(now_s, "now_s"),
    )
