"""Wrist-wrench/proprioception-only D38999 contact classification."""

from __future__ import annotations

from enum import Enum
import math
from typing import Mapping

import numpy as np


class ContactClass(str, Enum):
    NO_CONTACT = "NO_CONTACT"
    AXIAL_CONTACT = "AXIAL_CONTACT"
    SINGLE_EDGE_CONTACT = "SINGLE_EDGE_CONTACT"
    DOUBLE_EDGE_OR_JAM = "DOUBLE_EDGE_OR_JAM"
    GUIDED_ENTRY = "GUIDED_ENTRY"
    KEY_MISMATCH = "KEY_MISMATCH"
    INSERTING = "INSERTING"
    SEATED = "SEATED"
    UNKNOWN = "UNKNOWN"


def classify_contact(
    wrench_assembly: tuple[float, ...] | list[float] | np.ndarray,
    *,
    axial_progress_m: float,
    progress_in_window_m: float,
    rz_search_active: bool,
    thresholds: Mapping[str, float],
) -> ContactClass:
    wrench = np.asarray(wrench_assembly, dtype=np.float64)
    if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
        return ContactClass.UNKNOWN
    if not all(math.isfinite(float(value)) for value in (axial_progress_m, progress_in_window_m)):
        return ContactClass.UNKNOWN
    axial = abs(float(wrench[2]))
    lateral = float(np.linalg.norm(wrench[:2]))
    bending = float(np.linalg.norm(wrench[3:5]))
    torsion = abs(float(wrench[5]))
    if (
        axial_progress_m >= float(thresholds["seated_progress_m"])
        and axial >= float(thresholds["seated_axial_n"])
        and progress_in_window_m <= float(thresholds["stalled_progress_m"])
    ):
        return ContactClass.SEATED
    if rz_search_active and torsion >= float(thresholds["key_mismatch_mz_nm"]):
        return ContactClass.KEY_MISMATCH
    if (
        lateral >= float(thresholds["jam_lateral_n"])
        and bending >= float(thresholds["jam_bending_nm"])
        and progress_in_window_m <= float(thresholds["stalled_progress_m"])
    ):
        return ContactClass.DOUBLE_EDGE_OR_JAM
    if progress_in_window_m >= float(thresholds["guided_progress_m"]):
        if lateral < float(thresholds["lateral_contact_n"]) and bending < float(thresholds["bending_contact_nm"]):
            return ContactClass.INSERTING
        return ContactClass.GUIDED_ENTRY
    if lateral >= float(thresholds["lateral_contact_n"]) or bending >= float(thresholds["bending_contact_nm"]):
        return ContactClass.SINGLE_EDGE_CONTACT
    if axial >= float(thresholds["axial_contact_n"]):
        return ContactClass.AXIAL_CONTACT
    return ContactClass.NO_CONTACT


__all__ = ["ContactClass", "classify_contact"]

