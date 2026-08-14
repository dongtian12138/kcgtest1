"""Pure contract and oracle proof for residual-v0 learnability challenge v1.

This module is intentionally disconnected from the Isaac backend and formal
training runner.  Loading it cannot enable the challenge or mutate runtime
state.  It only validates the versioned YAML and proves that every allowed
control-path perturbation has a bounded 4-D compensating action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


CHALLENGE_SCHEMA_VERSION = (
    "kcg_connector_residual_learning_challenge_v1"
)
RESIDUAL_INTERFACE_VERSION = "kcg_connector_twist_residual_v0"
RESIDUAL_ACTION_SIZE = 4
RESIDUAL_OBSERVATION_SIZE = 24
OPERATIONAL_STAGE_REFERENCE = "stage20"
CLAMP_JOINT_NAMES = ("f1j2", "f2j1", "f3j2")
BASE_CLAMP_NOMINAL_POSITIONS_RAD = (0.750, 0.500, 0.750)
CLAMP_RESIDUAL_ACTION_LIMITS_RAD = (0.020, 0.020, 0.020)
DEFAULT_CHALLENGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/connector_residual_learning_challenge_v1.yaml"
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: tuple[str, ...], name: str
) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise ValueError(f"{name} keys are invalid: {'; '.join(details)}")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite_tuple(
    value: Any, name: str, size: int
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must contain {size} finite numbers")
    result = tuple(
        _finite_float(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _string_tuple(
    value: Any, name: str, size: int | None = None
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(value)
    if size is not None and len(result) != size:
        raise ValueError(f"{name} must contain exactly {size} strings")
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return result


def _require_close(
    actual: float, expected: float, name: str, tolerance: float = 1.0e-12
) -> None:
    if not math.isclose(
        actual, expected, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ValueError(f"{name} must be exactly {expected}")


@dataclass(frozen=True)
class SeedRange:
    """One inclusive, non-overlapping experiment seed interval."""

    start: int
    end_inclusive: int

    @property
    def count(self) -> int:
        return self.end_inclusive - self.start + 1

    def contains(self, seed: int) -> bool:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            return False
        return self.start <= int(seed) <= self.end_inclusive


@dataclass(frozen=True)
class LearningChallengeTask:
    target_angle_degrees: float
    success_hold_duration_s: float
    hold_entry_detection_policy_steps: int
    policy_rate_hz: float
    maximum_episode_steps: int
    nominal_q7_speed_degrees_per_second: float
    q7_speed_residual_degrees_per_second: float
    maximum_q7_speed_degrees_per_second: float


@dataclass(frozen=True)
class ControlPathRandomization:
    q7_velocity_scale_lower: float
    q7_velocity_scale_upper: float
    clamp_joint_names: tuple[str, ...]
    clamp_offset_lower_rad: tuple[float, ...]
    clamp_offset_upper_rad: tuple[float, ...]
    clamp_observation_reference: str
    clamp_base_nominal_positions_rad: tuple[float, ...]
    clamp_residual_action_limits_rad: tuple[float, ...]


@dataclass(frozen=True)
class DeadlineRule:
    penalty: float
    termination_reason: str
    counts_as_safety_failure: bool


@dataclass(frozen=True)
class FixedPhysics:
    mass_randomized: bool
    friction_randomized: bool
    helical_lead_randomized: bool


@dataclass(frozen=True)
class LearningChallengeAcceptance:
    zero_policy_success_rate_minimum: float
    zero_policy_success_rate_maximum: float
    oracle_minimum_success_rate: float
    maximum_zero_policy_raw_safety_failures: int
    maximum_oracle_raw_safety_failures: int
    zero_policy_allowed_failure_reasons: tuple[str, ...]
    require_all_paired_randomization_matches: bool


@dataclass(frozen=True)
class ConnectorResidualLearningChallenge:
    """Validated, disabled, offline-only challenge definition."""

    schema_version: str
    enabled: bool
    interface_version: str
    action_size: int
    observation_size: int
    operational_stage_reference: str
    task: LearningChallengeTask
    control_path: ControlPathRandomization
    deadline: DeadlineRule
    fixed_physics: FixedPhysics
    tuning_seeds: SeedRange
    validation_seeds: SeedRange
    final_paired_seeds: SeedRange
    acceptance: LearningChallengeAcceptance

    def as_dict(self) -> dict[str, Any]:
        """Return finite JSON data for provenance without enabling runtime."""

        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


@dataclass(frozen=True)
class OracleCompensation:
    """One exact action that cancels an allowed control-path sample."""

    normalized_action: tuple[float, ...]
    q7_velocity_scale: float
    q7_commanded_speed_degrees_per_second: float
    q7_effective_speed_degrees_per_second: float
    clamp_offsets_rad: tuple[float, ...]
    compensated_clamp_positions_rad: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


@dataclass(frozen=True)
class LearnabilityProof:
    """Closed-form bounds showing the challenge is safe and reachable."""

    minimum_effective_q7_speed_degrees_per_second: float
    maximum_effective_q7_speed_degrees_per_second: float
    maximum_allowed_q7_speed_degrees_per_second: float
    effective_speed_bound_proved: bool
    q7_oracle_action_minimum: float
    q7_oracle_action_maximum: float
    q7_oracle_action_bound_proved: bool
    clamp_oracle_maximum_absolute_action: float
    clamp_oracle_action_bound_proved: bool
    oracle_motion_policy_steps: int
    hold_entry_detection_policy_steps: int
    success_hold_policy_steps: int
    oracle_required_policy_steps: int
    maximum_episode_steps: int
    oracle_budget_margin_policy_steps: int
    oracle_budget_proved: bool
    slowest_zero_policy_required_steps: int
    slowest_zero_policy_hits_deadline: bool
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


@dataclass(frozen=True)
class ChallengeAcceptanceResult:
    """Executable result for the frozen empirical acceptance gates."""

    zero_policy_success_rate_in_range: bool
    oracle_success_rate_passed: bool
    zero_policy_raw_safety_passed: bool
    oracle_raw_safety_passed: bool
    zero_policy_failure_reasons_passed: bool
    paired_randomization_matches_passed: bool
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


def _seed_range(
    document: Mapping[str, Any], name: str, expected: tuple[int, int]
) -> SeedRange:
    raw = _mapping(document[name], f"seed_ranges.{name}")
    _exact_keys(
        raw,
        ("start", "end_inclusive"),
        f"seed_ranges.{name}",
    )
    seed_range = SeedRange(
        start=_integer(raw["start"], f"seed_ranges.{name}.start"),
        end_inclusive=_integer(
            raw["end_inclusive"],
            f"seed_ranges.{name}.end_inclusive",
        ),
    )
    if (seed_range.start, seed_range.end_inclusive) != expected:
        raise ValueError(
            f"seed_ranges.{name} must be exactly {expected[0]}..{expected[1]}"
        )
    return seed_range


def load_residual_learning_challenge(
    config_path: str | Path = DEFAULT_CHALLENGE_PATH,
) -> ConnectorResidualLearningChallenge:
    """Load the exact disabled v1 schema and reject any contract drift."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = _mapping(yaml.safe_load(stream), "challenge")
    _exact_keys(
        document,
        (
            "schema_version",
            "enabled",
            "contract",
            "task",
            "control_path_randomization",
            "deadline",
            "fixed_physics",
            "seed_ranges",
            "acceptance",
        ),
        "challenge",
    )
    if document["schema_version"] != CHALLENGE_SCHEMA_VERSION:
        raise ValueError("unsupported learning challenge schema")
    enabled = _boolean(document["enabled"], "enabled")
    if enabled:
        raise ValueError(
            "learning challenge v1 must remain disabled until runtime review"
        )

    contract = _mapping(document["contract"], "contract")
    _exact_keys(
        contract,
        (
            "interface_version",
            "action_size",
            "observation_size",
            "operational_stage_reference",
        ),
        "contract",
    )
    interface_version = _string(
        contract["interface_version"], "contract.interface_version"
    )
    action_size = _integer(contract["action_size"], "contract.action_size")
    observation_size = _integer(
        contract["observation_size"], "contract.observation_size"
    )
    operational_stage_reference = _string(
        contract["operational_stage_reference"],
        "contract.operational_stage_reference",
    )
    if interface_version != RESIDUAL_INTERFACE_VERSION:
        raise ValueError("challenge must preserve residual interface v0")
    if action_size != RESIDUAL_ACTION_SIZE:
        raise ValueError("challenge action size must remain 4")
    if observation_size != RESIDUAL_OBSERVATION_SIZE:
        raise ValueError("challenge observation size must remain 24")
    if operational_stage_reference != OPERATIONAL_STAGE_REFERENCE:
        raise ValueError("challenge must reference operational stage20")

    task_raw = _mapping(document["task"], "task")
    _exact_keys(
        task_raw,
        (
            "target_angle_degrees",
            "success_hold_duration_s",
            "hold_entry_detection_policy_steps",
            "policy_rate_hz",
            "maximum_episode_steps",
            "nominal_q7_speed_degrees_per_second",
            "q7_speed_residual_degrees_per_second",
            "maximum_q7_speed_degrees_per_second",
        ),
        "task",
    )
    task = LearningChallengeTask(
        target_angle_degrees=_finite_float(
            task_raw["target_angle_degrees"],
            "task.target_angle_degrees",
        ),
        success_hold_duration_s=_finite_float(
            task_raw["success_hold_duration_s"],
            "task.success_hold_duration_s",
        ),
        hold_entry_detection_policy_steps=_integer(
            task_raw["hold_entry_detection_policy_steps"],
            "task.hold_entry_detection_policy_steps",
        ),
        policy_rate_hz=_finite_float(
            task_raw["policy_rate_hz"], "task.policy_rate_hz"
        ),
        maximum_episode_steps=_integer(
            task_raw["maximum_episode_steps"],
            "task.maximum_episode_steps",
        ),
        nominal_q7_speed_degrees_per_second=_finite_float(
            task_raw["nominal_q7_speed_degrees_per_second"],
            "task.nominal_q7_speed_degrees_per_second",
        ),
        q7_speed_residual_degrees_per_second=_finite_float(
            task_raw["q7_speed_residual_degrees_per_second"],
            "task.q7_speed_residual_degrees_per_second",
        ),
        maximum_q7_speed_degrees_per_second=_finite_float(
            task_raw["maximum_q7_speed_degrees_per_second"],
            "task.maximum_q7_speed_degrees_per_second",
        ),
    )
    for actual, expected, name in (
        (task.target_angle_degrees, 20.0, "target angle"),
        (task.success_hold_duration_s, 0.5, "success hold"),
        (task.policy_rate_hz, 10.0, "policy rate"),
        (
            task.nominal_q7_speed_degrees_per_second,
            10.0,
            "nominal q7 speed",
        ),
        (
            task.q7_speed_residual_degrees_per_second,
            2.0,
            "q7 speed residual",
        ),
        (
            task.maximum_q7_speed_degrees_per_second,
            20.0,
            "maximum q7 speed",
        ),
    ):
        _require_close(actual, expected, f"task.{name}")
    if task.maximum_episode_steps != 28:
        raise ValueError("challenge horizon must be exactly 28 policy steps")
    if task.hold_entry_detection_policy_steps != 1:
        raise ValueError(
            "challenge must include exactly one HOLD-entry detection step"
        )

    randomization = _mapping(
        document["control_path_randomization"],
        "control_path_randomization",
    )
    _exact_keys(
        randomization,
        (
            "q7_velocity_scale",
            "clamp_position_offset",
            "clamp_observation_reference",
        ),
        "control_path_randomization",
    )
    q7_scale = _mapping(
        randomization["q7_velocity_scale"],
        "control_path_randomization.q7_velocity_scale",
    )
    _exact_keys(
        q7_scale,
        ("lower", "upper"),
        "control_path_randomization.q7_velocity_scale",
    )
    scale_lower = _finite_float(
        q7_scale["lower"],
        "control_path_randomization.q7_velocity_scale.lower",
    )
    scale_upper = _finite_float(
        q7_scale["upper"],
        "control_path_randomization.q7_velocity_scale.upper",
    )
    _require_close(scale_lower, 0.85, "q7 velocity scale lower")
    _require_close(scale_upper, 1.15, "q7 velocity scale upper")

    clamp = _mapping(
        randomization["clamp_position_offset"],
        "control_path_randomization.clamp_position_offset",
    )
    _exact_keys(
        clamp,
        ("joint_names", "lower_rad", "upper_rad"),
        "control_path_randomization.clamp_position_offset",
    )
    joint_names = _string_tuple(
        clamp["joint_names"],
        "control_path_randomization.clamp_position_offset.joint_names",
        3,
    )
    if joint_names != CLAMP_JOINT_NAMES:
        raise ValueError("clamp offsets must target f1j2, f2j1, f3j2")
    offset_lower = _finite_tuple(
        clamp["lower_rad"],
        "control_path_randomization.clamp_position_offset.lower_rad",
        3,
    )
    offset_upper = _finite_tuple(
        clamp["upper_rad"],
        "control_path_randomization.clamp_position_offset.upper_rad",
        3,
    )
    if offset_lower != (-0.015,) * 3 or offset_upper != (0.015,) * 3:
        raise ValueError("clamp offsets must be exactly +/-0.015 rad")

    reference = _mapping(
        randomization["clamp_observation_reference"],
        "control_path_randomization.clamp_observation_reference",
    )
    _exact_keys(
        reference,
        ("kind", "positions_rad", "residual_action_limits_rad"),
        "control_path_randomization.clamp_observation_reference",
    )
    reference_kind = _string(
        reference["kind"],
        "control_path_randomization.clamp_observation_reference.kind",
    )
    reference_positions = _finite_tuple(
        reference["positions_rad"],
        "control_path_randomization.clamp_observation_reference.positions_rad",
        3,
    )
    action_limits = _finite_tuple(
        reference["residual_action_limits_rad"],
        "control_path_randomization.clamp_observation_reference."
        "residual_action_limits_rad",
        3,
    )
    if reference_kind != "base_nominal":
        raise ValueError("clamp observation reference must be base_nominal")
    if reference_positions != BASE_CLAMP_NOMINAL_POSITIONS_RAD:
        raise ValueError("clamp observation reference changed")
    if action_limits != CLAMP_RESIDUAL_ACTION_LIMITS_RAD:
        raise ValueError("clamp residual action limits changed")
    control_path = ControlPathRandomization(
        q7_velocity_scale_lower=scale_lower,
        q7_velocity_scale_upper=scale_upper,
        clamp_joint_names=joint_names,
        clamp_offset_lower_rad=offset_lower,
        clamp_offset_upper_rad=offset_upper,
        clamp_observation_reference=reference_kind,
        clamp_base_nominal_positions_rad=reference_positions,
        clamp_residual_action_limits_rad=action_limits,
    )

    deadline_raw = _mapping(document["deadline"], "deadline")
    _exact_keys(
        deadline_raw,
        ("penalty", "termination_reason", "counts_as_safety_failure"),
        "deadline",
    )
    deadline = DeadlineRule(
        penalty=_finite_float(deadline_raw["penalty"], "deadline.penalty"),
        termination_reason=_string(
            deadline_raw["termination_reason"],
            "deadline.termination_reason",
        ),
        counts_as_safety_failure=_boolean(
            deadline_raw["counts_as_safety_failure"],
            "deadline.counts_as_safety_failure",
        ),
    )
    _require_close(deadline.penalty, -10.0, "deadline.penalty")
    if deadline.termination_reason != "time_limit":
        raise ValueError("deadline reason must be time_limit")
    if deadline.counts_as_safety_failure:
        raise ValueError("deadline must not count as a safety failure")

    fixed_raw = _mapping(document["fixed_physics"], "fixed_physics")
    _exact_keys(
        fixed_raw,
        (
            "mass_randomized",
            "friction_randomized",
            "helical_lead_randomized",
        ),
        "fixed_physics",
    )
    fixed_physics = FixedPhysics(
        mass_randomized=_boolean(
            fixed_raw["mass_randomized"],
            "fixed_physics.mass_randomized",
        ),
        friction_randomized=_boolean(
            fixed_raw["friction_randomized"],
            "fixed_physics.friction_randomized",
        ),
        helical_lead_randomized=_boolean(
            fixed_raw["helical_lead_randomized"],
            "fixed_physics.helical_lead_randomized",
        ),
    )
    if any(asdict(fixed_physics).values()):
        raise ValueError("mass, friction and helical lead must remain fixed")

    seed_raw = _mapping(document["seed_ranges"], "seed_ranges")
    _exact_keys(
        seed_raw,
        ("tuning", "validation", "final_paired"),
        "seed_ranges",
    )
    tuning_seeds = _seed_range(seed_raw, "tuning", (20000, 20063))
    validation_seeds = _seed_range(
        seed_raw, "validation", (30000, 30127)
    )
    final_paired_seeds = _seed_range(
        seed_raw, "final_paired", (10000, 10099)
    )
    seed_sets = (
        set(range(tuning_seeds.start, tuning_seeds.end_inclusive + 1)),
        set(
            range(
                validation_seeds.start,
                validation_seeds.end_inclusive + 1,
            )
        ),
        set(
            range(
                final_paired_seeds.start,
                final_paired_seeds.end_inclusive + 1,
            )
        ),
    )
    if any(
        seed_sets[first] & seed_sets[second]
        for first, second in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("challenge seed ranges must be disjoint")

    acceptance_raw = _mapping(document["acceptance"], "acceptance")
    _exact_keys(
        acceptance_raw,
        (
            "zero_policy_success_rate",
            "oracle_minimum_success_rate",
            "maximum_zero_policy_raw_safety_failures",
            "maximum_oracle_raw_safety_failures",
            "zero_policy_allowed_failure_reasons",
            "require_all_paired_randomization_matches",
        ),
        "acceptance",
    )
    zero_rate = _mapping(
        acceptance_raw["zero_policy_success_rate"],
        "acceptance.zero_policy_success_rate",
    )
    _exact_keys(
        zero_rate,
        ("minimum", "maximum"),
        "acceptance.zero_policy_success_rate",
    )
    zero_minimum = _finite_float(
        zero_rate["minimum"],
        "acceptance.zero_policy_success_rate.minimum",
    )
    zero_maximum = _finite_float(
        zero_rate["maximum"],
        "acceptance.zero_policy_success_rate.maximum",
    )
    oracle_minimum = _finite_float(
        acceptance_raw["oracle_minimum_success_rate"],
        "acceptance.oracle_minimum_success_rate",
    )
    zero_safety = _integer(
        acceptance_raw["maximum_zero_policy_raw_safety_failures"],
        "acceptance.maximum_zero_policy_raw_safety_failures",
    )
    oracle_safety = _integer(
        acceptance_raw["maximum_oracle_raw_safety_failures"],
        "acceptance.maximum_oracle_raw_safety_failures",
    )
    failure_reasons = _string_tuple(
        acceptance_raw["zero_policy_allowed_failure_reasons"],
        "acceptance.zero_policy_allowed_failure_reasons",
    )
    require_matches = _boolean(
        acceptance_raw["require_all_paired_randomization_matches"],
        "acceptance.require_all_paired_randomization_matches",
    )
    _require_close(zero_minimum, 0.65, "zero-policy minimum success rate")
    _require_close(zero_maximum, 0.85, "zero-policy maximum success rate")
    _require_close(oracle_minimum, 0.98, "oracle minimum success rate")
    if zero_safety != 0 or oracle_safety != 0:
        raise ValueError("raw safety failure limits must both be zero")
    if failure_reasons != ("time_limit",):
        raise ValueError("zero-policy failures may only be time_limit")
    if not require_matches:
        raise ValueError("all paired randomization samples must match")
    acceptance = LearningChallengeAcceptance(
        zero_policy_success_rate_minimum=zero_minimum,
        zero_policy_success_rate_maximum=zero_maximum,
        oracle_minimum_success_rate=oracle_minimum,
        maximum_zero_policy_raw_safety_failures=zero_safety,
        maximum_oracle_raw_safety_failures=oracle_safety,
        zero_policy_allowed_failure_reasons=failure_reasons,
        require_all_paired_randomization_matches=require_matches,
    )

    challenge = ConnectorResidualLearningChallenge(
        schema_version=CHALLENGE_SCHEMA_VERSION,
        enabled=enabled,
        interface_version=interface_version,
        action_size=action_size,
        observation_size=observation_size,
        operational_stage_reference=operational_stage_reference,
        task=task,
        control_path=control_path,
        deadline=deadline,
        fixed_physics=fixed_physics,
        tuning_seeds=tuning_seeds,
        validation_seeds=validation_seeds,
        final_paired_seeds=final_paired_seeds,
        acceptance=acceptance,
    )
    if not prove_learnability(challenge).passed:
        raise ValueError("learning challenge oracle proof failed")
    challenge.as_dict()
    return challenge


def oracle_compensation(
    challenge: ConnectorResidualLearningChallenge,
    q7_velocity_scale: float,
    clamp_offsets_rad: Sequence[float],
) -> OracleCompensation:
    """Cancel one allowed scale/offset sample with a bounded 4-D action.

    The q7 term solves ``scale * (nominal + action * residual) = nominal``.
    Each clamp term solves ``base + offset + action * limit = base``.
    """

    if not isinstance(challenge, ConnectorResidualLearningChallenge):
        raise TypeError("challenge must be ConnectorResidualLearningChallenge")
    scale = _finite_float(q7_velocity_scale, "q7_velocity_scale")
    control = challenge.control_path
    if not (
        control.q7_velocity_scale_lower
        <= scale
        <= control.q7_velocity_scale_upper
    ):
        raise ValueError("q7 velocity scale is outside the challenge domain")
    offsets = _finite_tuple(clamp_offsets_rad, "clamp_offsets_rad", 3)
    for index, offset in enumerate(offsets):
        if not (
            control.clamp_offset_lower_rad[index]
            <= offset
            <= control.clamp_offset_upper_rad[index]
        ):
            raise ValueError(
                f"clamp offset {index} is outside the challenge domain"
            )

    task = challenge.task
    q7_action = (
        task.nominal_q7_speed_degrees_per_second / scale
        - task.nominal_q7_speed_degrees_per_second
    ) / task.q7_speed_residual_degrees_per_second
    clamp_actions = tuple(
        -offset / limit
        for offset, limit in zip(
            offsets, control.clamp_residual_action_limits_rad
        )
    )
    normalized_action = (q7_action,) + clamp_actions
    if any(abs(value) > 1.0 + 1.0e-12 for value in normalized_action):
        raise ValueError(
            "oracle compensation escaped normalized action bounds"
        )
    commanded_speed = (
        task.nominal_q7_speed_degrees_per_second
        + q7_action * task.q7_speed_residual_degrees_per_second
    )
    effective_speed = scale * commanded_speed
    if not (
        0.0
        < effective_speed
        <= task.maximum_q7_speed_degrees_per_second
    ):
        raise ValueError("oracle compensation violates q7 speed bounds")
    compensated_positions = tuple(
        base + offset + action * limit
        for base, offset, action, limit in zip(
            control.clamp_base_nominal_positions_rad,
            offsets,
            clamp_actions,
            control.clamp_residual_action_limits_rad,
        )
    )
    result = OracleCompensation(
        normalized_action=tuple(float(value) for value in normalized_action),
        q7_velocity_scale=scale,
        q7_commanded_speed_degrees_per_second=commanded_speed,
        q7_effective_speed_degrees_per_second=effective_speed,
        clamp_offsets_rad=offsets,
        compensated_clamp_positions_rad=compensated_positions,
    )
    result.as_dict()
    return result


def prove_learnability(
    challenge: ConnectorResidualLearningChallenge,
) -> LearnabilityProof:
    """Prove speed, normalized-action and 28-step oracle bounds.

    For all policy q7 actions in [-1, 1], the effective speed lies in
    ``[0.85 * 8, 1.15 * 12] = [6.8, 13.8] deg/s``, inside ``(0, 20]``.
    The oracle restores 10 deg/s; 20 motion steps, one HOLD-entry detection
    step and 5 stable-hold steps require 26 of 28 steps.  At the slowest
    zero-action scale, 24+1+5=30 steps, so the non-safety ``time_limit`` can
    distinguish compensation from zero action.
    """

    if not isinstance(challenge, ConnectorResidualLearningChallenge):
        raise TypeError("challenge must be ConnectorResidualLearningChallenge")
    task = challenge.task
    control = challenge.control_path
    minimum_effective_speed = (
        control.q7_velocity_scale_lower
        * (
            task.nominal_q7_speed_degrees_per_second
            - task.q7_speed_residual_degrees_per_second
        )
    )
    maximum_effective_speed = (
        control.q7_velocity_scale_upper
        * (
            task.nominal_q7_speed_degrees_per_second
            + task.q7_speed_residual_degrees_per_second
        )
    )
    effective_speed_bound = bool(
        minimum_effective_speed > 0.0
        and maximum_effective_speed
        <= task.maximum_q7_speed_degrees_per_second
    )
    zero_offsets = (0.0, 0.0, 0.0)
    low_scale_oracle = oracle_compensation(
        challenge, control.q7_velocity_scale_lower, zero_offsets
    )
    high_scale_oracle = oracle_compensation(
        challenge, control.q7_velocity_scale_upper, zero_offsets
    )
    q7_action_minimum = min(
        low_scale_oracle.normalized_action[0],
        high_scale_oracle.normalized_action[0],
    )
    q7_action_maximum = max(
        low_scale_oracle.normalized_action[0],
        high_scale_oracle.normalized_action[0],
    )
    q7_action_bound = bool(
        q7_action_minimum >= -1.0 - 1.0e-12
        and q7_action_maximum <= 1.0 + 1.0e-12
    )
    clamp_maximum_action = max(
        max(abs(lower), abs(upper)) / limit
        for lower, upper, limit in zip(
            control.clamp_offset_lower_rad,
            control.clamp_offset_upper_rad,
            control.clamp_residual_action_limits_rad,
        )
    )
    clamp_action_bound = clamp_maximum_action <= 1.0 + 1.0e-12
    oracle_motion_steps = math.ceil(
        task.target_angle_degrees
        / task.nominal_q7_speed_degrees_per_second
        * task.policy_rate_hz
    )
    hold_steps = math.ceil(
        task.success_hold_duration_s * task.policy_rate_hz
    )
    oracle_required_steps = (
        oracle_motion_steps
        + task.hold_entry_detection_policy_steps
        + hold_steps
    )
    budget_margin = task.maximum_episode_steps - oracle_required_steps
    oracle_budget = budget_margin >= 0
    slowest_zero_motion_steps = math.ceil(
        task.target_angle_degrees
        / (
            task.nominal_q7_speed_degrees_per_second
            * control.q7_velocity_scale_lower
        )
        * task.policy_rate_hz
    )
    slowest_zero_required_steps = (
        slowest_zero_motion_steps
        + task.hold_entry_detection_policy_steps
        + hold_steps
    )
    slowest_zero_hits_deadline = bool(
        slowest_zero_required_steps > task.maximum_episode_steps
    )
    passed = bool(
        effective_speed_bound
        and q7_action_bound
        and clamp_action_bound
        and oracle_budget
        and slowest_zero_hits_deadline
    )
    proof = LearnabilityProof(
        minimum_effective_q7_speed_degrees_per_second=(
            minimum_effective_speed
        ),
        maximum_effective_q7_speed_degrees_per_second=(
            maximum_effective_speed
        ),
        maximum_allowed_q7_speed_degrees_per_second=(
            task.maximum_q7_speed_degrees_per_second
        ),
        effective_speed_bound_proved=effective_speed_bound,
        q7_oracle_action_minimum=q7_action_minimum,
        q7_oracle_action_maximum=q7_action_maximum,
        q7_oracle_action_bound_proved=q7_action_bound,
        clamp_oracle_maximum_absolute_action=clamp_maximum_action,
        clamp_oracle_action_bound_proved=clamp_action_bound,
        oracle_motion_policy_steps=oracle_motion_steps,
        hold_entry_detection_policy_steps=(
            task.hold_entry_detection_policy_steps
        ),
        success_hold_policy_steps=hold_steps,
        oracle_required_policy_steps=oracle_required_steps,
        maximum_episode_steps=task.maximum_episode_steps,
        oracle_budget_margin_policy_steps=budget_margin,
        oracle_budget_proved=oracle_budget,
        slowest_zero_policy_required_steps=slowest_zero_required_steps,
        slowest_zero_policy_hits_deadline=slowest_zero_hits_deadline,
        passed=passed,
    )
    proof.as_dict()
    return proof


def evaluate_challenge_acceptance(
    challenge: ConnectorResidualLearningChallenge,
    *,
    zero_policy_success_rate: float,
    oracle_success_rate: float,
    zero_policy_raw_safety_failures: int,
    oracle_raw_safety_failures: int,
    zero_policy_failure_reasons: Sequence[str],
    paired_randomization_matches: Sequence[bool],
) -> ChallengeAcceptanceResult:
    """Apply every frozen empirical gate without simulator dependencies."""

    if not isinstance(challenge, ConnectorResidualLearningChallenge):
        raise TypeError("challenge must be ConnectorResidualLearningChallenge")
    zero_rate = _finite_float(
        zero_policy_success_rate, "zero_policy_success_rate"
    )
    oracle_rate = _finite_float(oracle_success_rate, "oracle_success_rate")
    if not 0.0 <= zero_rate <= 1.0 or not 0.0 <= oracle_rate <= 1.0:
        raise ValueError("success rates must be in [0, 1]")
    zero_safety = _integer(
        zero_policy_raw_safety_failures,
        "zero_policy_raw_safety_failures",
    )
    oracle_safety = _integer(
        oracle_raw_safety_failures,
        "oracle_raw_safety_failures",
    )
    if zero_safety < 0 or oracle_safety < 0:
        raise ValueError("raw safety failure counts must be nonnegative")
    failure_reasons = _string_tuple(
        zero_policy_failure_reasons,
        "zero_policy_failure_reasons",
    )
    if (
        isinstance(paired_randomization_matches, (str, bytes))
        or not isinstance(paired_randomization_matches, Sequence)
    ):
        raise ValueError(
            "paired_randomization_matches must be a boolean sequence"
        )
    matches = tuple(paired_randomization_matches)
    if len(matches) != challenge.final_paired_seeds.count:
        raise ValueError(
            "paired_randomization_matches must cover all final paired seeds"
        )
    if any(not isinstance(value, bool) for value in matches):
        raise ValueError(
            "paired_randomization_matches must contain only booleans"
        )

    gates = challenge.acceptance
    zero_rate_passed = bool(
        gates.zero_policy_success_rate_minimum
        <= zero_rate
        <= gates.zero_policy_success_rate_maximum
    )
    oracle_rate_passed = bool(
        oracle_rate >= gates.oracle_minimum_success_rate
    )
    zero_safety_passed = bool(
        zero_safety <= gates.maximum_zero_policy_raw_safety_failures
    )
    oracle_safety_passed = bool(
        oracle_safety <= gates.maximum_oracle_raw_safety_failures
    )
    allowed_reasons = set(gates.zero_policy_allowed_failure_reasons)
    failure_reasons_passed = all(
        reason in allowed_reasons for reason in failure_reasons
    )
    matches_passed = bool(
        not gates.require_all_paired_randomization_matches
        or all(matches)
    )
    passed = bool(
        zero_rate_passed
        and oracle_rate_passed
        and zero_safety_passed
        and oracle_safety_passed
        and failure_reasons_passed
        and matches_passed
    )
    result = ChallengeAcceptanceResult(
        zero_policy_success_rate_in_range=zero_rate_passed,
        oracle_success_rate_passed=oracle_rate_passed,
        zero_policy_raw_safety_passed=zero_safety_passed,
        oracle_raw_safety_passed=oracle_safety_passed,
        zero_policy_failure_reasons_passed=failure_reasons_passed,
        paired_randomization_matches_passed=matches_passed,
        passed=passed,
    )
    result.as_dict()
    return result


__all__ = [
    "BASE_CLAMP_NOMINAL_POSITIONS_RAD",
    "CHALLENGE_SCHEMA_VERSION",
    "CLAMP_JOINT_NAMES",
    "CLAMP_RESIDUAL_ACTION_LIMITS_RAD",
    "DEFAULT_CHALLENGE_PATH",
    "OPERATIONAL_STAGE_REFERENCE",
    "RESIDUAL_ACTION_SIZE",
    "RESIDUAL_INTERFACE_VERSION",
    "RESIDUAL_OBSERVATION_SIZE",
    "ConnectorResidualLearningChallenge",
    "ChallengeAcceptanceResult",
    "ControlPathRandomization",
    "DeadlineRule",
    "FixedPhysics",
    "LearnabilityProof",
    "LearningChallengeAcceptance",
    "LearningChallengeTask",
    "OracleCompensation",
    "SeedRange",
    "evaluate_challenge_acceptance",
    "load_residual_learning_challenge",
    "oracle_compensation",
    "prove_learnability",
]
