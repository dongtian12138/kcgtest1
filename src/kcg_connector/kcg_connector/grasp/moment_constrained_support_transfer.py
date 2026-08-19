"""Moment-constrained table-support transfer for DYN-B-V3.

The online controller consumes only synchronized robot state, three finger-root
loads and the wrist sensor chain.  Object pose, contact identity/normal and
event truth are deliberately absent from the API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_d38999_moment_constrained_support_transfer_v1"
TASK_ID = "DYN-B-V3-MOMENT-CONSTRAINED-SUPPORT-TRANSFER"
THRESHOLD_LABEL = "SIM_TUNING_ONLY_DYN_B_V3"
PHASE_INTERNAL_FORCE_CENTERING = "INTERNAL_FORCE_CENTERING"
PHASE_BUMPLESS_CONTROL_TRANSFER = "BUMPLESS_CONTROL_TRANSFER"
PHASE_QUASISTATIC_UNWEIGHT = "QUASISTATIC_UNWEIGHT"
PHASE_BREAKAWAY_COMMIT = "BREAKAWAY_COMMIT"
PHASE_LIFT_READY = "LIFT_READY"


def _exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown or missing:
        raise ValueError(
            f"{label} has unknown keys {unknown} and/or missing keys {missing}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _three(values: Sequence[float], label: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    result = tuple(_finite(value, label) for value in values)
    return result  # type: ignore[return-value]


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be one SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be one SHA-256 hex digest") from exc
    return value.lower()


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must remain repository-relative")
    return value


@dataclass(frozen=True)
class EvidenceBinding:
    path: str
    sha256: str


@dataclass(frozen=True)
class MomentConstrainedSupportTransferConfig:
    enabled: bool
    task_id: str
    threshold_label: str
    evidence: tuple[EvidenceBinding, ...]
    force_scale_n: float
    moment_scale_nm: float
    jacobian_per_rad: tuple[tuple[float, ...], ...]
    damping_ratio: float
    finger_target_step_bound_rad: float
    finger_cumulative_bound_rad: float
    support_force_target_n: float
    support_profile_samples: int
    stable_confirm_steps: int
    support_advance_moment_score_limit_nm: float
    maximum_observed_gate_jump_nm: float
    minimum_normalized_root_load: float
    maximum_normalized_load_imbalance: float
    root_load_scale_nm: tuple[float, float, float]
    force_component_gate_n: float
    moment_component_gate_nm: float
    raw_hard_gate_unchanged: bool
    hard_gate_detection_delay_steps: int
    zero_sum_internal_force: bool
    object_pose_write_forbidden: bool
    object_contact_event_truth_forbidden: bool

    def __post_init__(self) -> None:
        if self.enabled is not True:
            raise ValueError("B-V3 support-transfer config must be enabled")
        if self.task_id != TASK_ID:
            raise ValueError(f"B-V3 task_id must remain {TASK_ID}")
        if self.threshold_label != THRESHOLD_LABEL:
            raise ValueError(
                f"B-V3 threshold_label must remain {THRESHOLD_LABEL}"
            )
        jacobian = np.asarray(self.jacobian_per_rad, dtype=np.float64)
        if jacobian.shape != (4, 3) or not np.all(np.isfinite(jacobian)):
            raise ValueError("B-V3 Jacobian must be one finite 4x3 matrix")
        sqrt2 = math.sqrt(2.0)
        sqrt6 = math.sqrt(6.0)
        basis = np.asarray(
            (
                (1.0 / sqrt2, 1.0 / sqrt6),
                (-1.0 / sqrt2, 1.0 / sqrt6),
                (0.0, -2.0 / sqrt6),
            ),
            dtype=np.float64,
        )
        if np.linalg.matrix_rank(jacobian @ basis) != 2:
            raise ValueError("B-V3 differential Jacobian must retain rank two")
        if abs(self.force_scale_n - 8.0) > 1.0e-12:
            raise ValueError("B-V3 force objective scale must remain 8 N")
        if abs(self.moment_scale_nm - 0.30) > 1.0e-12:
            raise ValueError("B-V3 moment objective scale must remain 0.30 N m")
        if abs(self.force_component_gate_n - 8.0) > 1.0e-12:
            raise ValueError("B-V3 force hard gate must remain 8 N")
        if abs(self.moment_component_gate_nm - 0.30) > 1.0e-12:
            raise ValueError("B-V3 moment hard gate must remain 0.30 N m")
        if abs(self.support_force_target_n - 3.0411) > 1.0e-12:
            raise ValueError("B-V3 support force must remain frozen payload weight")
        if self.support_profile_samples != 240:
            raise ValueError("B-V3 must reuse the 240-sample H18 profile")
        if self.stable_confirm_steps != 48:
            raise ValueError("B-V3 must reuse the 48-step stable-contact window")
        if abs(self.damping_ratio - 0.10) > 1.0e-12:
            raise ValueError("B-V3 must reuse the H23 damping ratio")
        if abs(self.finger_target_step_bound_rad - 0.004 / 48.0) > 1.0e-15:
            raise ValueError("B-V3 finger step must reuse the H23 safe probe rate")
        if abs(self.finger_cumulative_bound_rad - 0.030) > 1.0e-12:
            raise ValueError("B-V3 finger cumulative bound must remain 0.030 rad")
        derived_margin = self.moment_component_gate_nm - self.maximum_observed_gate_jump_nm
        if (
            derived_margin <= 0.0
            or abs(derived_margin - self.support_advance_moment_score_limit_nm)
            > 1.0e-12
        ):
            raise ValueError("B-V3 support-advance margin must be evidence-derived")
        if abs(self.minimum_normalized_root_load - 0.40) > 1.0e-12:
            raise ValueError("B-V3 minimum root load must remain 0.40")
        if abs(self.maximum_normalized_load_imbalance - 0.18) > 1.0e-12:
            raise ValueError("B-V3 root-load imbalance must remain 0.18")
        if any(abs(value - 0.30) > 1.0e-12 for value in self.root_load_scale_nm):
            raise ValueError("B-V3 root-load scales must remain 0.30 N m")
        if not self.evidence:
            raise ValueError("B-V3 requires immutable evidence bindings")
        required_true = (
            "raw_hard_gate_unchanged",
            "zero_sum_internal_force",
            "object_pose_write_forbidden",
            "object_contact_event_truth_forbidden",
        )
        for name in required_true:
            if getattr(self, name) is not True:
                raise ValueError(f"B-V3 {name} must remain true")
        if self.hard_gate_detection_delay_steps != 0:
            raise ValueError("B-V3 hard-gate delay must remain zero")


def load_moment_constrained_support_transfer_config(
    path: str | Path,
) -> MomentConstrainedSupportTransferConfig:
    """Load the one strict B-V3 support-transfer contract."""

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _mapping(document, "B-V3 document")
    _exact_keys(
        root,
        (
            "schema_version",
            "task_id",
            "enabled",
            "threshold_label",
            "evidence",
            "objective",
            "internal_force",
            "support_transfer",
            "safety",
            "truth_firewall",
        ),
        "B-V3 document",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"B-V3 schema_version must remain {SCHEMA_VERSION}")
    evidence_value = root["evidence"]
    if not isinstance(evidence_value, list) or not evidence_value:
        raise ValueError("B-V3 evidence must be a non-empty list")
    evidence = []
    for index, raw in enumerate(evidence_value):
        item = _mapping(raw, f"B-V3 evidence[{index}]")
        _exact_keys(item, ("path", "sha256"), f"B-V3 evidence[{index}]")
        evidence.append(
            EvidenceBinding(
                path=_relative_path(item["path"], "B-V3 evidence path"),
                sha256=_sha256(item["sha256"], "B-V3 evidence sha256"),
            )
        )
    objective = _mapping(root["objective"], "B-V3 objective")
    _exact_keys(objective, ("force_scale_n", "moment_scale_nm"), "B-V3 objective")
    internal = _mapping(root["internal_force"], "B-V3 internal_force")
    _exact_keys(
        internal,
        (
            "jacobian_per_rad",
            "damping_ratio",
            "finger_target_step_bound_rad",
            "finger_cumulative_bound_rad",
            "zero_sum_required",
        ),
        "B-V3 internal_force",
    )
    jacobian_raw = internal["jacobian_per_rad"]
    if not isinstance(jacobian_raw, list) or len(jacobian_raw) != 4:
        raise ValueError("B-V3 jacobian_per_rad must have four rows")
    jacobian = tuple(
        tuple(_three(row, "B-V3 Jacobian row")) for row in jacobian_raw
    )
    support = _mapping(root["support_transfer"], "B-V3 support_transfer")
    _exact_keys(
        support,
        (
            "support_force_target_n",
            "support_profile_samples",
            "stable_confirm_steps",
            "support_advance_moment_score_limit_nm",
            "maximum_observed_gate_jump_nm",
            "minimum_normalized_root_load",
            "maximum_normalized_load_imbalance",
            "root_load_scale_nm",
        ),
        "B-V3 support_transfer",
    )
    safety = _mapping(root["safety"], "B-V3 safety")
    _exact_keys(
        safety,
        (
            "force_component_gate_n",
            "moment_component_gate_nm",
            "raw_hard_gate_unchanged",
            "hard_gate_detection_delay_steps",
        ),
        "B-V3 safety",
    )
    truth = _mapping(root["truth_firewall"], "B-V3 truth_firewall")
    _exact_keys(
        truth,
        ("object_pose_write_forbidden", "object_contact_event_truth_forbidden"),
        "B-V3 truth_firewall",
    )
    return MomentConstrainedSupportTransferConfig(
        enabled=root["enabled"],
        task_id=str(root["task_id"]),
        threshold_label=str(root["threshold_label"]),
        evidence=tuple(evidence),
        force_scale_n=_positive(objective["force_scale_n"], "force_scale_n"),
        moment_scale_nm=_positive(objective["moment_scale_nm"], "moment_scale_nm"),
        jacobian_per_rad=jacobian,
        damping_ratio=_positive(internal["damping_ratio"], "damping_ratio"),
        finger_target_step_bound_rad=_positive(
            internal["finger_target_step_bound_rad"], "finger_target_step_bound_rad"
        ),
        finger_cumulative_bound_rad=_positive(
            internal["finger_cumulative_bound_rad"], "finger_cumulative_bound_rad"
        ),
        support_force_target_n=_positive(
            support["support_force_target_n"], "support_force_target_n"
        ),
        support_profile_samples=_positive_integer(
            support["support_profile_samples"], "support_profile_samples"
        ),
        stable_confirm_steps=_positive_integer(
            support["stable_confirm_steps"], "stable_confirm_steps"
        ),
        support_advance_moment_score_limit_nm=_positive(
            support["support_advance_moment_score_limit_nm"],
            "support_advance_moment_score_limit_nm",
        ),
        maximum_observed_gate_jump_nm=_positive(
            support["maximum_observed_gate_jump_nm"],
            "maximum_observed_gate_jump_nm",
        ),
        minimum_normalized_root_load=_positive(
            support["minimum_normalized_root_load"],
            "minimum_normalized_root_load",
        ),
        maximum_normalized_load_imbalance=_positive(
            support["maximum_normalized_load_imbalance"],
            "maximum_normalized_load_imbalance",
        ),
        root_load_scale_nm=_three(
            support["root_load_scale_nm"], "root_load_scale_nm"
        ),
        force_component_gate_n=_positive(
            safety["force_component_gate_n"], "force_component_gate_n"
        ),
        moment_component_gate_nm=_positive(
            safety["moment_component_gate_nm"], "moment_component_gate_nm"
        ),
        raw_hard_gate_unchanged=safety["raw_hard_gate_unchanged"],
        hard_gate_detection_delay_steps=safety["hard_gate_detection_delay_steps"],
        zero_sum_internal_force=internal["zero_sum_required"],
        object_pose_write_forbidden=truth["object_pose_write_forbidden"],
        object_contact_event_truth_forbidden=truth[
            "object_contact_event_truth_forbidden"
        ],
    )


def verify_evidence_bindings(
    config: MomentConstrainedSupportTransferConfig,
    repository: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Verify every immutable evidence input before a dynamic process."""

    root = Path(repository).resolve()
    results = []
    for binding in config.evidence:
        path = (root / binding.path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("B-V3 evidence escaped the repository") from exc
        if not path.is_file():
            raise ValueError(f"B-V3 evidence is missing: {binding.path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != binding.sha256:
            raise ValueError(
                f"B-V3 evidence hash mismatch for {binding.path}: "
                f"expected={binding.sha256}, actual={actual}"
            )
        results.append(
            {"path": binding.path, "expected_sha256": binding.sha256, "actual_sha256": actual, "verified": True}
        )
    return tuple(results)


def _project_zero_sum_box(
    requested: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if requested.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("B-V3 zero-sum projection requires three-vectors")
    if not np.all(np.isfinite(requested)) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("B-V3 zero-sum projection values must be finite")
    if np.any(lower > upper) or float(np.sum(lower)) > 1.0e-12 or float(np.sum(upper)) < -1.0e-12:
        raise ValueError("B-V3 zero-sum projection is infeasible")
    low = float(np.min(requested - upper))
    high = float(np.max(requested - lower))
    projected = np.zeros(3, dtype=np.float64)
    for _ in range(80):
        level = 0.5 * (low + high)
        projected = np.clip(requested - level, lower, upper)
        if float(np.sum(projected)) > 0.0:
            low = level
        else:
            high = level
    residual = float(np.sum(projected))
    if abs(residual) > 1.0e-14:
        for index in range(3):
            candidate = projected[index] - residual
            if lower[index] <= candidate <= upper[index]:
                projected[index] = candidate
                break
    return projected


class MomentConstrainedSupportTransfer:
    """Stateful B-V3 controller with no simulator-truth input surface."""

    def __init__(
        self,
        config: MomentConstrainedSupportTransferConfig,
        *,
        base_targets_rad: Sequence[float],
        open_targets_rad: Sequence[float],
        closed_targets_rad: Sequence[float],
    ) -> None:
        self.config = config
        self.base_targets = np.asarray(_three(base_targets_rad, "base targets"))
        self.open_targets = np.asarray(_three(open_targets_rad, "open targets"))
        self.closed_targets = np.asarray(_three(closed_targets_rad, "closed targets"))
        self.direction = np.where(self.closed_targets > self.open_targets, 1.0, -1.0)
        if np.any(self.closed_targets == self.open_targets):
            raise ValueError("B-V3 fingers require nonzero closure intervals")
        low_joint = np.minimum(self.open_targets, self.closed_targets)
        high_joint = np.maximum(self.open_targets, self.closed_targets)
        if np.any(self.base_targets < low_joint) or np.any(self.base_targets > high_joint):
            raise ValueError("B-V3 base target lies outside a finger interval")
        self.targets = self.base_targets.copy()
        self.cumulative_closure = np.zeros(3, dtype=np.float64)
        self.phase = PHASE_INTERNAL_FORCE_CENTERING
        self.stable_count = 0
        self.support_profile_index = 0
        self.record_count = 0
        self.phase_counts = {
            PHASE_INTERNAL_FORCE_CENTERING: 0,
            PHASE_BUMPLESS_CONTROL_TRANSFER: 0,
            PHASE_QUASISTATIC_UNWEIGHT: 0,
            PHASE_BREAKAWAY_COMMIT: 0,
            PHASE_LIFT_READY: 0,
        }
        self.maximum_gate_score_nm = 0.0
        self.maximum_abs_target_step_rad = 0.0
        self.maximum_abs_cumulative_rad = 0.0
        self.maximum_zero_sum_residual_rad = 0.0
        self.support_advance_count = 0
        self._jacobian = np.asarray(config.jacobian_per_rad, dtype=np.float64)
        sqrt2 = math.sqrt(2.0)
        sqrt6 = math.sqrt(6.0)
        self._basis = np.asarray(
            (
                (1.0 / sqrt2, 1.0 / sqrt6),
                (-1.0 / sqrt2, 1.0 / sqrt6),
                (0.0, -2.0 / sqrt6),
            ),
            dtype=np.float64,
        )
        differential = self._jacobian @ self._basis
        self._left, self._singular, self._right_t = np.linalg.svd(
            differential, full_matrices=False
        )

    def _objective(
        self,
        task_force_xy_n: Sequence[float],
        sensor_origin_moment_increment_xyz_nm: Sequence[float],
    ) -> np.ndarray:
        if not isinstance(task_force_xy_n, (list, tuple)) or len(task_force_xy_n) != 2:
            raise ValueError("B-V3 task force input must contain Fx and Fy")
        force = np.asarray(
            [_finite(value, "B-V3 task force") for value in task_force_xy_n],
            dtype=np.float64,
        )
        moment = np.asarray(
            _three(sensor_origin_moment_increment_xyz_nm, "B-V3 sensor moment"),
            dtype=np.float64,
        )
        return np.asarray(
            (
                force[0] / self.config.force_scale_n,
                force[1] / self.config.force_scale_n,
                moment[0] / self.config.moment_scale_nm,
                moment[1] / self.config.moment_scale_nm,
            ),
            dtype=np.float64,
        )

    def _requested_closure_delta(self, objective: np.ndarray) -> np.ndarray:
        damping = self.config.damping_ratio * float(self._singular[0])
        gains = self._singular / (
            self._singular * self._singular + damping * damping
        )
        coordinates = -(
            self._right_t.T @ (gains * (self._left.T @ objective))
        )
        requested = self._basis @ coordinates
        maximum = float(np.max(np.abs(requested)))
        if maximum > self.config.finger_target_step_bound_rad:
            requested *= self.config.finger_target_step_bound_rad / maximum
        return requested

    def _apply_internal_force_step(self, objective: np.ndarray) -> dict[str, Any]:
        requested = self._requested_closure_delta(objective)
        step = self.config.finger_target_step_bound_rad
        total = self.config.finger_cumulative_bound_rad
        physical_lower = self.direction * (self.open_targets - self.targets)
        physical_upper = self.direction * (self.closed_targets - self.targets)
        lower = np.maximum.reduce(
            (
                np.full(3, -step),
                -total - self.cumulative_closure,
                physical_lower,
            )
        )
        upper = np.minimum.reduce(
            (
                np.full(3, step),
                total - self.cumulative_closure,
                physical_upper,
            )
        )
        applied = _project_zero_sum_box(requested, lower, upper)
        self.cumulative_closure += applied
        self.targets += self.direction * applied
        low_joint = np.minimum(self.open_targets, self.closed_targets)
        high_joint = np.maximum(self.open_targets, self.closed_targets)
        if np.any(self.targets < low_joint - 1.0e-12) or np.any(self.targets > high_joint + 1.0e-12):
            raise RuntimeError("B-V3 target escaped the physical joint interval")
        predicted = objective + self._jacobian @ applied
        zero_sum = float(np.sum(applied))
        self.maximum_abs_target_step_rad = max(
            self.maximum_abs_target_step_rad, float(np.max(np.abs(applied)))
        )
        self.maximum_abs_cumulative_rad = max(
            self.maximum_abs_cumulative_rad,
            float(np.max(np.abs(self.cumulative_closure))),
        )
        self.maximum_zero_sum_residual_rad = max(
            self.maximum_zero_sum_residual_rad, abs(zero_sum)
        )
        return {
            "requested_delta_closure_rad": requested.tolist(),
            "applied_delta_closure_rad": applied.tolist(),
            "cumulative_delta_closure_rad": self.cumulative_closure.tolist(),
            "output_targets_rad": self.targets.tolist(),
            "applied_delta_sum_rad": zero_sum,
            "predicted_objective_after_step": predicted.tolist(),
            "physical_or_cumulative_bound_active": bool(
                np.any(np.isclose(applied, lower, atol=1.0e-15))
                or np.any(np.isclose(applied, upper, atol=1.0e-15))
            ),
        }

    def update(
        self,
        *,
        task_force_xy_n: Sequence[float],
        sensor_origin_moment_increment_xyz_nm: Sequence[float],
        finger_root_torque_nm: Sequence[float],
        raw_moment_gate_score_nm: float,
        raw_hard_gate_triggered: bool,
        input_global_step: int,
    ) -> dict[str, Any]:
        """Advance one command from the immediately preceding sensor sample."""

        if isinstance(input_global_step, bool) or not isinstance(input_global_step, int) or input_global_step < 0:
            raise ValueError("B-V3 input_global_step must be non-negative")
        gate_score = _finite(raw_moment_gate_score_nm, "raw_moment_gate_score_nm")
        if gate_score < 0.0:
            raise ValueError("B-V3 raw moment gate score must be non-negative")
        if type(raw_hard_gate_triggered) is not bool:
            raise ValueError("B-V3 raw_hard_gate_triggered must be boolean")
        if raw_hard_gate_triggered or gate_score > self.config.moment_component_gate_nm:
            raise RuntimeError("B-V3 raw sensor-origin hard gate triggered")
        roots = np.abs(np.asarray(_three(finger_root_torque_nm, "finger roots")))
        normalized = roots / np.asarray(self.config.root_load_scale_nm)
        minimum_load = float(np.min(normalized))
        imbalance = float(np.max(normalized) - np.min(normalized))
        objective = self._objective(
            task_force_xy_n, sensor_origin_moment_increment_xyz_nm
        )
        internal = self._apply_internal_force_step(objective)
        advance_safe = bool(
            gate_score <= self.config.support_advance_moment_score_limit_nm
            and minimum_load >= self.config.minimum_normalized_root_load
            and imbalance <= self.config.maximum_normalized_load_imbalance
        )
        self.stable_count = self.stable_count + 1 if advance_safe else 0
        phase_before = self.phase
        support_advanced = False
        commit_lift = False
        if self.phase == PHASE_INTERNAL_FORCE_CENTERING:
            if self.stable_count >= self.config.stable_confirm_steps:
                self.phase = PHASE_BUMPLESS_CONTROL_TRANSFER
                self.stable_count = 0
        elif self.phase == PHASE_BUMPLESS_CONTROL_TRANSFER:
            if advance_safe:
                self.phase = PHASE_QUASISTATIC_UNWEIGHT
                self.stable_count = 0
        elif self.phase == PHASE_QUASISTATIC_UNWEIGHT:
            if self.stable_count >= self.config.stable_confirm_steps:
                if self.support_profile_index < self.config.support_profile_samples:
                    self.support_profile_index += 1
                    self.support_advance_count += 1
                    support_advanced = True
                    self.stable_count = 0
                else:
                    self.phase = PHASE_BREAKAWAY_COMMIT
                    self.stable_count = 0
        elif self.phase == PHASE_BREAKAWAY_COMMIT:
            if self.stable_count >= self.config.stable_confirm_steps:
                self.phase = PHASE_LIFT_READY
                self.stable_count = 0
                commit_lift = True
        elif self.phase == PHASE_LIFT_READY:
            commit_lift = True
        else:
            raise RuntimeError(f"unknown B-V3 phase {self.phase}")
        self.record_count += 1
        self.phase_counts[phase_before] += 1
        self.maximum_gate_score_nm = max(self.maximum_gate_score_nm, gate_score)
        return {
            "input_global_step": input_global_step,
            "output_global_step": input_global_step + 1,
            "phase_before": phase_before,
            "phase_after": self.phase,
            "support_profile_index": self.support_profile_index,
            "support_profile_fraction": (
                float(self.support_profile_index)
                / float(self.config.support_profile_samples)
            ),
            "support_advanced": support_advanced,
            "support_force_target_n": self.config.support_force_target_n,
            "commit_lift": commit_lift,
            "advance_safe": advance_safe,
            "stable_count": self.stable_count,
            "raw_moment_gate_score_nm": gate_score,
            "support_advance_moment_score_limit_nm": self.config.support_advance_moment_score_limit_nm,
            "normalized_root_load": normalized.tolist(),
            "minimum_normalized_root_load": minimum_load,
            "normalized_root_load_imbalance": imbalance,
            **internal,
            "raw_sensor_hard_gate_unchanged": True,
            "hard_gate_detection_delay_steps": 0,
            "object_truth_used": False,
            "contact_truth_used": False,
            "contact_normal_used": False,
            "event_truth_used": False,
            "object_pose_written": False,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "record_count": self.record_count,
            "phase_counts": dict(self.phase_counts),
            "support_profile_index": self.support_profile_index,
            "support_advance_count": self.support_advance_count,
            "maximum_gate_score_nm": self.maximum_gate_score_nm,
            "maximum_abs_target_step_rad": self.maximum_abs_target_step_rad,
            "maximum_abs_cumulative_rad": self.maximum_abs_cumulative_rad,
            "maximum_zero_sum_residual_rad": self.maximum_zero_sum_residual_rad,
            "final_targets_rad": self.targets.tolist(),
            "final_cumulative_delta_closure_rad": self.cumulative_closure.tolist(),
            "raw_sensor_hard_gate_unchanged": True,
            "hard_gate_detection_delay_steps": 0,
            "object_truth_used": False,
            "contact_truth_used": False,
            "contact_normal_used": False,
            "event_truth_used": False,
            "object_pose_written": False,
        }


__all__ = [
    "MomentConstrainedSupportTransfer",
    "MomentConstrainedSupportTransferConfig",
    "PHASE_BREAKAWAY_COMMIT",
    "PHASE_BUMPLESS_CONTROL_TRANSFER",
    "PHASE_INTERNAL_FORCE_CENTERING",
    "PHASE_LIFT_READY",
    "PHASE_QUASISTATIC_UNWEIGHT",
    "load_moment_constrained_support_transfer_config",
    "verify_evidence_bindings",
]
