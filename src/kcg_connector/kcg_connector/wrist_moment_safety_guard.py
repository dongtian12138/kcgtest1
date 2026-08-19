"""Fail-closed wrist-moment guard for the frozen D38999 grasp boundary.

The guard consumes only a canonical wrist moment, its frame id, and a
monotonic sample timestamp.  It deliberately has no object pose, contact
identity, contact normal, or assembly-event input.  The 0.30 N*m limit and
the frozen three-component decomposition are verified against repository
contracts before a guard can be constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from kcg_connector.grasp.grasp_stability_monitor import (
    WRIST_MOMENT_SEMANTICS,
    evaluate_wrist_moment_safety,
)


AUTHORIZED_WRIST_MOMENT_LIMIT_NM = 0.30
EXPECTED_SEMANTICS = "three_component_decomposition_v1"
EXPECTED_FRAME_ID = "connector_task_frame"

FROZEN_SOURCE_SHA256 = {
    "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml": (
        "68d18977d920cec3681e99ee8beabf934a19d7c69f6ab50f2aa5db5f6e1504dd"
    ),
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "src/kcg_connector/config/d38999_keyed_v3_physical_acceptance_r12_v1.yaml": (
        "dd37d07d5bd87a97c3dbefe225c0eafd409fc783a6010eec8db9da397ce2cf76"
    ),
    "src/kcg_connector/kcg_connector/grasp/grasp_stability_monitor.py": (
        "0f19f957a65f57dd173b42d96721e6d20ac07d01046a04f6bf5e52a1e63bb55f"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{source} must be positive and finite")
    return result


def _yaml(path: Path) -> Mapping[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    return document


def _require_authorized_limit(value: Any, source: str) -> float:
    limit = _number(value, source)
    if limit != AUTHORIZED_WRIST_MOMENT_LIMIT_NM:
        raise ValueError(
            f"{source} changed the authorized wrist-moment limit: {limit}"
        )
    return limit


def load_frozen_wrist_moment_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load and cross-check every frozen source for the 0.30 N*m gate."""

    root = Path(repository_root).resolve()
    resolved: dict[str, Path] = {}
    source_rows: list[dict[str, str]] = []
    for relative, expected_sha256 in FROZEN_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen wrist-moment source missing: {relative}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"frozen wrist-moment source hash mismatch: {relative}"
            )
        resolved[relative] = path
        source_rows.append(
            {
                "path": relative,
                "sha256": actual_sha256,
            }
        )

    grasp = _yaml(
        resolved[
            "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
        ]
    )
    master = _yaml(
        resolved[
            "src/kcg_connector/config/d38999_master_model_contract_v1.yaml"
        ]
    )
    acceptance = _yaml(
        resolved[
            "src/kcg_connector/config/"
            "d38999_keyed_v3_physical_acceptance_r12_v1.yaml"
        ]
    )

    values = {
        "grasp_lift_maximum_wrist_moment_nm": _require_authorized_limit(
            grasp["lift"]["maximum_wrist_moment_nm"],
            "grasp.lift.maximum_wrist_moment_nm",
        ),
        "master_torque_component_limit_nm": _require_authorized_limit(
            master["acceptance_limits"]["torque_component_limit_nm"],
            "master.acceptance_limits.torque_component_limit_nm",
        ),
        "formal_perpendicular_moment_max_nm": _require_authorized_limit(
            acceptance["shared_numeric_profile"]
            ["robot_formal_perpendicular_moment_max_nm"],
            "acceptance.shared_numeric_profile."
            "robot_formal_perpendicular_moment_max_nm",
        ),
        "p1_driver_torque_component_limit_nm": _require_authorized_limit(
            acceptance["benches"]["P1"]["inputs"]
            ["component_driver_profile"]["torque_component_limit_nm"],
            "acceptance.benches.P1.inputs.component_driver_profile."
            "torque_component_limit_nm",
        ),
    }
    if WRIST_MOMENT_SEMANTICS != EXPECTED_SEMANTICS:
        raise ValueError("frozen wrist-moment semantics changed")

    return {
        "schema_version": 1,
        "limit_nm": AUTHORIZED_WRIST_MOMENT_LIMIT_NM,
        "semantics": EXPECTED_SEMANTICS,
        "comparison": "strictly_greater_triggers_exact_equality_passes",
        "expected_frame_id": EXPECTED_FRAME_ID,
        "cross_checked_values": values,
        "sources": source_rows,
    }


def _moment(value: Any) -> tuple[float, float, float]:
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        raise ValueError("moment must be a finite 3-vector")
    ndim = getattr(value, "ndim", None)
    if ndim is not None and int(ndim) != 1:
        raise ValueError("moment must be a finite 3-vector")
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        raise ValueError("moment must be a finite 3-vector") from None
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError("moment must be a finite 3-vector")
    return result  # type: ignore[return-value]


@dataclass
class WristMomentSafetyGuard:
    """Timestamped, latched adapter around the frozen moment evaluator."""

    reference_moment_nm: Sequence[float]
    contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.reference_moment_nm = _moment(self.reference_moment_nm)
        limit = _require_authorized_limit(
            self.contract.get("limit_nm"), "guard.contract.limit_nm"
        )
        if self.contract.get("semantics") != EXPECTED_SEMANTICS:
            raise ValueError("guard contract semantics mismatch")
        if self.contract.get("expected_frame_id") != EXPECTED_FRAME_ID:
            raise ValueError("guard contract frame mismatch")
        self.limit_nm = limit
        self.latched = False
        self.failure_reason: str | None = None
        self.trigger_component: str | None = None
        self.last_timestamp_s: float | None = None
        self.last_evaluation: dict[str, Any] | None = None
        self.observation_count = 0
        self.reset_count = 0

    @classmethod
    def from_frozen_contracts(
        cls,
        reference_moment_nm: Sequence[float],
        repository_root: str | Path,
    ) -> "WristMomentSafetyGuard":
        return cls(
            reference_moment_nm=reference_moment_nm,
            contract=load_frozen_wrist_moment_contract(repository_root),
        )

    def _decision(self, sample_consumed: bool) -> dict[str, Any]:
        return {
            "safe_to_continue": not self.latched,
            "fault_latched": self.latched,
            "failure_reason": self.failure_reason,
            "trigger_component": self.trigger_component,
            "sample_consumed": sample_consumed,
            "observation_count": self.observation_count,
            "last_timestamp_s": self.last_timestamp_s,
            "last_evaluation": (
                dict(self.last_evaluation)
                if self.last_evaluation is not None
                else None
            ),
        }

    def _latch(self, reason: str, *, sample_consumed: bool) -> dict[str, Any]:
        self.latched = True
        if self.failure_reason is None:
            self.failure_reason = reason
        return self._decision(sample_consumed)

    def observe(
        self,
        current_moment_nm: Sequence[float],
        *,
        timestamp_s: float,
        frame_id: str,
    ) -> dict[str, Any]:
        """Consume one sensor moment and return a fail-closed decision."""

        if self.latched:
            return self._decision(sample_consumed=False)
        if frame_id != EXPECTED_FRAME_ID:
            return self._latch("unexpected_wrench_frame", sample_consumed=False)
        if (
            isinstance(timestamp_s, bool)
            or not isinstance(timestamp_s, (int, float))
            or not math.isfinite(float(timestamp_s))
        ):
            return self._latch("invalid_timestamp", sample_consumed=False)
        timestamp = float(timestamp_s)
        if (
            self.last_timestamp_s is not None
            and timestamp <= self.last_timestamp_s
        ):
            return self._latch(
                "nonmonotonic_wrench_timestamp", sample_consumed=False
            )
        try:
            current = _moment(current_moment_nm)
        except ValueError:
            return self._latch(
                "nonfinite_or_invalid_wrist_moment", sample_consumed=False
            )

        self.last_timestamp_s = timestamp
        self.observation_count += 1
        self.last_evaluation = evaluate_wrist_moment_safety(
            current,
            self.reference_moment_nm,
            self.limit_nm,
        )
        if self.last_evaluation["triggered"]:
            self.trigger_component = self.last_evaluation["trigger_component"]
            return self._latch("wrist_moment_limit", sample_consumed=True)
        return self._decision(sample_consumed=True)

    def reset_latch(
        self,
        *,
        explicit_authorization: bool,
        reason: str,
    ) -> None:
        """Clear a latched fault only through an explicit caller action."""

        if explicit_authorization is not True:
            raise ValueError("explicit reset authorization is required")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reset reason is required")
        self.latched = False
        self.failure_reason = None
        self.trigger_component = None
        self.last_timestamp_s = None
        self.last_evaluation = None
        self.reset_count += 1

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "OFFLINE_MONITOR_STATE",
            "limit_nm": self.limit_nm,
            "semantics": EXPECTED_SEMANTICS,
            "comparison": "strictly_greater_triggers_exact_equality_passes",
            "expected_frame_id": EXPECTED_FRAME_ID,
            "fault_latched": self.latched,
            "failure_reason": self.failure_reason,
            "trigger_component": self.trigger_component,
            "observation_count": self.observation_count,
            "reset_count": self.reset_count,
            "last_timestamp_s": self.last_timestamp_s,
            "last_evaluation": (
                dict(self.last_evaluation)
                if self.last_evaluation is not None
                else None
            ),
            "contract": dict(self.contract),
            "truth_firewall": {
                "object_pose_input": False,
                "contact_identity_input": False,
                "contact_normal_input": False,
                "assembly_event_truth_input": False
            },
            "dynamic_grasp_pass_claimed": False,
            "formal_physics_pass_claimed": False
        }
