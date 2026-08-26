"""Bind the B0 external load-bearing object surface without changing old evidence."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from kcg_connector.grasp.carts_v2.models import (
    FaceRoleMap, HARD_FORBIDDEN, PRIMARY_GRIP, SECONDARY_GRIP, V2Inputs,
)


B0_SURFACE_METHOD = "EXTERNAL_LOAD_BEARING_SURFACE_B0"


def b0_nominal_pickup_task_pass(quality, expected_operation_cap_n: float) -> bool:
    """Accept nominal gravity/lift balance without promoting disturbance evidence."""

    observed = quality.nominal_operation_force_cap_n
    return bool(quality.nominal_gravity_lift_balance_pass
                and observed is not None and math.isfinite(observed)
                and math.isclose(float(observed), float(expected_operation_cap_n),
                                 rel_tol=0.0, abs_tol=1.0e-12))


def b0_sampled_table_clearance_pass(
        pregrasp_clearance_m: float | None,
        closure_clearance_m: float | None,
        required_clearance_m: float,
        numerical_tolerance_m: float) -> bool:
    """Fail closed unless both sampled paths retain the registered table margin."""

    values = (pregrasp_clearance_m, closure_clearance_m,
              required_clearance_m, numerical_tolerance_m)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return False
    return bool(float(required_clearance_m) > 0.0
                and float(numerical_tolerance_m) >= 0.0
                and min(float(pregrasp_clearance_m), float(closure_clearance_m))
                >= float(required_clearance_m) - float(numerical_tolerance_m))


def _b0_role_arrays(face_semantics, allowed_labels, protected_labels, old_roles):
    semantics = np.asarray(tuple(str(value) for value in face_semantics), dtype=object)
    allowed_set = frozenset(str(value) for value in allowed_labels)
    protected_set = frozenset(str(value) for value in protected_labels)
    if allowed_set & protected_set:
        raise ValueError("B0_ALLOWED_AND_FUNCTIONAL_PROTECTED_SEMANTICS_OVERLAP")
    known = np.isin(semantics, tuple(allowed_set | protected_set))
    if not bool(np.all(known)):
        unknown = sorted(set(semantics[~known].tolist()))
        raise ValueError(f"B0_OBJECT_SURFACE_SEMANTICS_UNRESOLVED:{unknown}")
    protected = np.isin(semantics, tuple(protected_set))
    legacy_primary = np.asarray(old_roles) == PRIMARY_GRIP
    roles = np.full(len(semantics), SECONDARY_GRIP, dtype=np.uint8)
    roles[legacy_primary & ~protected] = PRIMARY_GRIP
    roles[protected] = HARD_FORBIDDEN
    reason = np.zeros(len(semantics), dtype=np.uint8)
    reason[protected] = 1
    return roles, ~protected, reason


def bind_b0_external_load_bearing_surfaces(inputs: V2Inputs) -> V2Inputs:
    """Make old geometric preferences soft; retain only explicit semantic protection."""

    model, old = inputs.object_contract.model, inputs.face_roles
    roles, allowed, reasons = _b0_role_arrays(
        model.mesh.face_semantics,
        model.allowed_contact_semantics,
        model.forbidden_contact_semantics,
        old.face_role,
    )
    area = np.asarray(model.mesh.face_areas_m2, dtype=np.float64)
    mapping = FaceRoleMap(
        object_id=inputs.object_contract.object_id,
        face_is_allowed=allowed,
        face_role=roles,
        legacy_radial_only_face_is_allowed=roles == PRIMARY_GRIP,
        reason_code=reasons,
        method=B0_SURFACE_METHOD,
        allowed_area_m2=float(np.sum(area[allowed])),
        total_area_m2=float(np.sum(area)),
    )
    return replace(inputs, face_roles=mapping)


def b0_surface_audit(inputs: V2Inputs) -> dict[str, object]:
    roles = np.asarray(inputs.face_roles.face_role)
    return {
        "method": inputs.face_roles.method,
        "external_load_bearing_face_count": int(np.count_nonzero(
            roles != HARD_FORBIDDEN)),
        "functional_protected_face_count": int(np.count_nonzero(
            roles == HARD_FORBIDDEN)),
        "legacy_primary_diagnostic_face_count": int(np.count_nonzero(
            roles == PRIMARY_GRIP)),
        "legacy_primary_secondary_are_hard_gates": False,
        "normal_alignment_is_object_semantic_hard_gate": False,
    }


__all__ = ["B0_SURFACE_METHOD", "b0_nominal_pickup_task_pass",
           "b0_sampled_table_clearance_pass", "b0_surface_audit",
           "bind_b0_external_load_bearing_surfaces"]
