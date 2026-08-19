"""Palm RGB-D keyed-v2 detection and C2 shadow-selection pipeline.

This module only composes two fail-closed CPU stages.  It never reads object
truth, semantic labels, contacts, or collider identity, and it has no control
promotion path.  The two branch directions must be projected by the existing
5-DOF/C2 estimator from its own image/FK hypotheses.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from kcg_connector.d38999_key_branch_selector import (
    select_key_branch_from_rgbd,
)
from kcg_connector.d38999_key_region_detector import (
    detect_key_region_from_palm_rgbd,
)


SCHEMA_VERSION = "kcg_d38999_key_shadow_pipeline_v1"


def run_palm_key_shadow_pipeline(
    connector_face_mask: Any,
    depth_m: Any,
    face_center_uv: Sequence[float],
    branch_directions_uv: Sequence[Sequence[float]],
    keyed_model_id: str | None,
    *,
    occlusion_mask: Any | None,
) -> dict[str, Any]:
    """Detect the unique master key and select one C2 branch for shadow."""

    detector = detect_key_region_from_palm_rgbd(
        connector_face_mask,
        depth_m,
        face_center_uv,
        keyed_model_id,
        occlusion_mask=occlusion_mask,
    )
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "PALM_RGBD_KEYED_V2_C2_SHADOW_SELECTION_ONLY",
        "status": "REJECTED",
        "reason": None,
        "rejection_code": None,
        "passed": False,
        "shadow_only": True,
        "control_authorized": False,
        "selected_for_shadow": None,
        "shadow_selected_hypothesis_id": None,
        "selected_for_control_allowed": False,
        "key_region_detection": detector,
        "key_branch_selection": None,
    }
    if detector.get("passed") is not True:
        base.update(
            status="REJECTED_KEY_REGION_DETECTION",
            reason=detector.get("reason"),
            rejection_code=detector.get("rejection_code"),
        )
        return base

    face = np.asarray(connector_face_mask)
    selector = select_key_branch_from_rgbd(
        detector["key_probability"],
        face,
        depth_m,
        face_center_uv,
        branch_directions_uv,
        keyed_model_id,
        occlusion_mask=occlusion_mask,
    )
    base["key_branch_selection"] = selector
    if selector.get("passed") is not True:
        base.update(
            status="REJECTED_KEY_BRANCH_SELECTION",
            reason=selector.get("reason"),
            rejection_code=selector.get("rejection_code"),
        )
        return base

    base.update(
        status="SHADOW_C2_BRANCH_SELECTED",
        passed=True,
        selected_for_shadow=selector["selected_for_shadow"],
        shadow_selected_hypothesis_id=selector[
            "shadow_selected_hypothesis_id"
        ],
    )
    return base


__all__ = ["SCHEMA_VERSION", "run_palm_key_shadow_pipeline"]

