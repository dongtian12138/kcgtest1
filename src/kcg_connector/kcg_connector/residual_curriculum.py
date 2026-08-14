"""Pure, versioned 20/60/120 degree residual curriculum resolution.

The module only derives immutable task configuration.  It does not import
Isaac Sim, mutate a backend, change the residual-v0 action/observation shape,
or implement regrasp.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import yaml

from .residual_rl import ConnectorResidualConfig


CURRICULUM_SCHEMA_VERSION = "kcg_connector_residual_curriculum_v1"
RESIDUAL_INTERFACE_VERSION = "kcg_connector_twist_residual_v0"
DEFAULT_STAGE_NAME = "stage20"
CURRICULUM_STAGE_NAMES = ("stage20", "stage60", "stage120")
DEFAULT_Q7_COMMAND_RESERVE_RAD = math.radians(10.0)
MAXIMUM_SINGLE_STROKE_RAD = math.radians(120.0)
DEFAULT_CURRICULUM_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/connector_residual_curriculum_v1.yaml"
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_keys(
    mapping: Mapping[str, Any], expected: tuple[str, ...], name: str
) -> None:
    actual = set(mapping)
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


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True)
class ResidualCurriculumStage:
    """One immutable single-stroke stage, in SI units."""

    name: str
    target_angle_rad: float
    success_hold_duration_s: float
    maximum_episode_steps: int
    minimum_axial_progress_fraction: float
    minimum_required_episode_steps: int
    predecessor_name: str | None


@dataclass(frozen=True)
class ConnectorResidualCurriculum:
    """Validated curriculum contract shared by all three stages."""

    schema_version: str
    interface_version: str
    default_stage_name: str
    policy_rate_hz: float
    tightening_direction: int
    minimum_tightening_speed_rad_s: float
    maximum_single_stroke_rad: float
    minimum_episode_margin_policy_steps: int
    success_angle_tolerance_rad: float
    maximum_helical_error_m: float
    q7_command_reserve_rad: float
    stages: tuple[ResidualCurriculumStage, ...]

    def stage(self, name: str | None = None) -> ResidualCurriculumStage:
        selected = self.default_stage_name if name is None else name
        if not isinstance(selected, str) or not selected:
            raise ValueError("curriculum stage name must be a nonempty string")
        for stage in self.stages:
            if stage.name == selected:
                return stage
        raise ValueError(f"unknown residual curriculum stage: {selected!r}")


@dataclass(frozen=True)
class ResolvedResidualStage:
    """A stage applied to residual v0 plus its fail-closed q7 proof."""

    curriculum_schema_version: str
    stage: ResidualCurriculumStage
    residual_config: ConnectorResidualConfig
    initial_q7_rad: float
    planned_final_q7_rad: float
    q7_command_reserve_rad: float
    lower_headroom_rad: float
    upper_headroom_rad: float

    def as_dict(self) -> dict[str, Any]:
        """Return a finite JSON-safe provenance document."""

        document = {
            "curriculum_schema_version": self.curriculum_schema_version,
            "interface_version": self.residual_config.interface_version,
            "stage_name": self.stage.name,
            "predecessor_stage_name": self.stage.predecessor_name,
            "target_segment_degrees": math.degrees(
                self.stage.target_angle_rad
            ),
            "target_angle_rad": self.stage.target_angle_rad,
            "success_hold_duration_s": (
                self.stage.success_hold_duration_s
            ),
            "maximum_episode_steps": self.stage.maximum_episode_steps,
            "minimum_required_episode_steps": (
                self.stage.minimum_required_episode_steps
            ),
            "success_angle_tolerance_degrees": math.degrees(
                self.residual_config.success_angle_tolerance_rad
            ),
            "maximum_helical_error_m": (
                self.residual_config.helical_error_tolerance_m
            ),
            "minimum_axial_progress_fraction": (
                self.stage.minimum_axial_progress_fraction
            ),
            "initial_q7_rad": self.initial_q7_rad,
            "planned_final_q7_rad": self.planned_final_q7_rad,
            "q7_safe_lower_rad": self.residual_config.q7_safe_lower_rad,
            "q7_safe_upper_rad": self.residual_config.q7_safe_upper_rad,
            "q7_command_reserve_degrees": math.degrees(
                self.q7_command_reserve_rad
            ),
            "q7_command_reserve_rad": self.q7_command_reserve_rad,
            "q7_lower_headroom_rad": self.lower_headroom_rad,
            "q7_upper_headroom_rad": self.upper_headroom_rad,
            "resolved_residual_config": asdict(self.residual_config),
        }
        # Round-trip through strict JSON to recursively turn tuples into lists
        # and to reject accidental NaN/Infinity values.
        return json.loads(
            json.dumps(document, allow_nan=False, sort_keys=True)
        )


def _required_episode_steps(
    target_angle_rad: float,
    hold_duration_s: float,
    policy_rate_hz: float,
    minimum_speed_rad_s: float,
    margin_steps: int,
) -> int:
    motion_steps = math.ceil(
        target_angle_rad / minimum_speed_rad_s * policy_rate_hz
    )
    hold_steps = math.ceil(hold_duration_s * policy_rate_hz)
    return motion_steps + hold_steps + margin_steps


def load_connector_residual_curriculum(
    config_path: str | Path = DEFAULT_CURRICULUM_PATH,
) -> ConnectorResidualCurriculum:
    """Load the exact v1 schema and reject unsafe stage schedules."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = _mapping(yaml.safe_load(stream), "curriculum")
    _exact_keys(
        document,
        ("schema_version", "contract", "acceptance", "stages"),
        "curriculum",
    )
    if document["schema_version"] != CURRICULUM_SCHEMA_VERSION:
        raise ValueError(
            "unsupported residual curriculum schema: "
            f"{document['schema_version']!r}"
        )

    contract = _mapping(document["contract"], "contract")
    _exact_keys(
        contract,
        (
            "interface_version",
            "default_stage",
            "policy_rate_hz",
            "tightening_direction",
            "minimum_tightening_speed_degrees_per_second",
            "maximum_single_stroke_degrees",
            "minimum_episode_margin_policy_steps",
        ),
        "contract",
    )
    interface_version = contract["interface_version"]
    if interface_version != RESIDUAL_INTERFACE_VERSION:
        raise ValueError("curriculum must preserve the residual-v0 interface")
    default_stage = contract["default_stage"]
    if default_stage != DEFAULT_STAGE_NAME:
        raise ValueError(
            "residual curriculum v1 default stage must be stage20"
        )
    policy_rate_hz = _finite_float(
        contract["policy_rate_hz"], "contract.policy_rate_hz"
    )
    if not math.isclose(policy_rate_hz, 10.0, abs_tol=1.0e-12):
        raise ValueError("residual curriculum v1 policy rate must be 10 Hz")
    tightening_direction = contract["tightening_direction"]
    if tightening_direction != -1 or isinstance(tightening_direction, bool):
        raise ValueError("residual curriculum tightening direction must be -1")
    minimum_speed_rad_s = math.radians(
        _finite_float(
            contract[
                "minimum_tightening_speed_degrees_per_second"
            ],
            "contract.minimum_tightening_speed_degrees_per_second",
        )
    )
    maximum_single_stroke_rad = math.radians(
        _finite_float(
            contract["maximum_single_stroke_degrees"],
            "contract.maximum_single_stroke_degrees",
        )
    )
    margin_steps = _positive_integer(
        contract["minimum_episode_margin_policy_steps"],
        "contract.minimum_episode_margin_policy_steps",
    )
    if minimum_speed_rad_s <= 0.0:
        raise ValueError("minimum tightening speed must be positive")
    if not 0.0 < maximum_single_stroke_rad <= MAXIMUM_SINGLE_STROKE_RAD:
        raise ValueError("maximum single stroke must be in (0, 120] degrees")

    acceptance = _mapping(document["acceptance"], "acceptance")
    _exact_keys(
        acceptance,
        (
            "success_angle_tolerance_degrees",
            "maximum_helical_error_m",
            "q7_command_reserve_degrees",
        ),
        "acceptance",
    )
    success_angle_tolerance_rad = math.radians(
        _finite_float(
            acceptance["success_angle_tolerance_degrees"],
            "acceptance.success_angle_tolerance_degrees",
        )
    )
    maximum_helical_error_m = _finite_float(
        acceptance["maximum_helical_error_m"],
        "acceptance.maximum_helical_error_m",
    )
    q7_command_reserve_rad = math.radians(
        _finite_float(
            acceptance["q7_command_reserve_degrees"],
            "acceptance.q7_command_reserve_degrees",
        )
    )
    if success_angle_tolerance_rad <= 0.0:
        raise ValueError("success angle tolerance must be positive")
    if maximum_helical_error_m <= 0.0:
        raise ValueError("maximum helical error must be positive")
    if q7_command_reserve_rad < DEFAULT_Q7_COMMAND_RESERVE_RAD:
        raise ValueError("q7 command reserve must be at least 10 degrees")

    raw_stages = _mapping(document["stages"], "stages")
    if tuple(raw_stages) != CURRICULUM_STAGE_NAMES:
        raise ValueError(
            "curriculum stages must be ordered stage20, stage60, stage120"
        )
    stages = []
    previous_target = 0.0
    previous_hold = 0.0
    previous_maximum_steps = 0
    previous_fraction = 0.0
    for index, name in enumerate(CURRICULUM_STAGE_NAMES):
        raw = _mapping(raw_stages[name], f"stages.{name}")
        _exact_keys(
            raw,
            (
                "target_segment_degrees",
                "success_hold_duration_s",
                "maximum_episode_steps",
                "minimum_axial_progress_fraction",
            ),
            f"stages.{name}",
        )
        target_angle_rad = math.radians(
            _finite_float(
                raw["target_segment_degrees"],
                f"stages.{name}.target_segment_degrees",
            )
        )
        hold_duration_s = _finite_float(
            raw["success_hold_duration_s"],
            f"stages.{name}.success_hold_duration_s",
        )
        maximum_episode_steps = _positive_integer(
            raw["maximum_episode_steps"],
            f"stages.{name}.maximum_episode_steps",
        )
        axial_fraction = _finite_float(
            raw["minimum_axial_progress_fraction"],
            f"stages.{name}.minimum_axial_progress_fraction",
        )
        if not 0.0 < target_angle_rad <= maximum_single_stroke_rad:
            raise ValueError(
                f"{name} target must be in the single-stroke range"
            )
        if hold_duration_s <= 0.0:
            raise ValueError(f"{name} hold duration must be positive")
        if not 0.0 < axial_fraction <= 1.0:
            raise ValueError(
                f"{name} minimum axial progress fraction must be in (0, 1]"
            )
        if target_angle_rad <= success_angle_tolerance_rad:
            raise ValueError(f"{name} target must exceed angle tolerance")
        if target_angle_rad <= previous_target:
            raise ValueError("curriculum target angles must increase strictly")
        if hold_duration_s <= previous_hold:
            raise ValueError(
                "curriculum hold durations must increase strictly"
            )
        if maximum_episode_steps <= previous_maximum_steps:
            raise ValueError(
                "curriculum maximum episode steps must increase strictly"
            )
        if axial_fraction < previous_fraction:
            raise ValueError(
                "curriculum axial progress fractions must not decrease"
            )
        required_steps = _required_episode_steps(
            target_angle_rad,
            hold_duration_s,
            policy_rate_hz,
            minimum_speed_rad_s,
            margin_steps,
        )
        if maximum_episode_steps < required_steps:
            raise ValueError(
                f"{name} maximum episode steps lack the required 10 Hz "
                f"margin: need {required_steps}"
            )
        stages.append(
            ResidualCurriculumStage(
                name=name,
                target_angle_rad=target_angle_rad,
                success_hold_duration_s=hold_duration_s,
                maximum_episode_steps=maximum_episode_steps,
                minimum_axial_progress_fraction=axial_fraction,
                minimum_required_episode_steps=required_steps,
                predecessor_name=(
                    None if index == 0 else CURRICULUM_STAGE_NAMES[index - 1]
                ),
            )
        )
        previous_target = target_angle_rad
        previous_hold = hold_duration_s
        previous_maximum_steps = maximum_episode_steps
        previous_fraction = axial_fraction

    return ConnectorResidualCurriculum(
        schema_version=CURRICULUM_SCHEMA_VERSION,
        interface_version=interface_version,
        default_stage_name=default_stage,
        policy_rate_hz=policy_rate_hz,
        tightening_direction=tightening_direction,
        minimum_tightening_speed_rad_s=minimum_speed_rad_s,
        maximum_single_stroke_rad=maximum_single_stroke_rad,
        minimum_episode_margin_policy_steps=margin_steps,
        success_angle_tolerance_rad=success_angle_tolerance_rad,
        maximum_helical_error_m=maximum_helical_error_m,
        q7_command_reserve_rad=q7_command_reserve_rad,
        stages=tuple(stages),
    )


def resolve_stage(
    base_residual_config: ConnectorResidualConfig,
    stage_name: str | None,
    initial_q7_rad: float,
    reserve: float = DEFAULT_Q7_COMMAND_RESERVE_RAD,
    *,
    curriculum: ConnectorResidualCurriculum | None = None,
) -> ResolvedResidualStage:
    """Apply one stage and prove its q7 command endpoint has safe reserve.

    ``reserve`` is in radians.  Passing ``None`` as ``stage_name`` selects the
    versioned default, ``stage20``.
    """

    if not isinstance(base_residual_config, ConnectorResidualConfig):
        raise TypeError(
            "base_residual_config must be ConnectorResidualConfig"
        )
    selected_curriculum = (
        load_connector_residual_curriculum()
        if curriculum is None
        else curriculum
    )
    if not isinstance(selected_curriculum, ConnectorResidualCurriculum):
        raise TypeError("curriculum must be ConnectorResidualCurriculum")
    stage = selected_curriculum.stage(stage_name)
    initial_q7 = _finite_float(initial_q7_rad, "initial_q7_rad")
    command_reserve = _finite_float(reserve, "reserve")
    minimum_reserve = max(
        DEFAULT_Q7_COMMAND_RESERVE_RAD,
        selected_curriculum.q7_command_reserve_rad,
    )
    if command_reserve + 1.0e-12 < minimum_reserve:
        raise ValueError("q7 command reserve must be at least 10 degrees")
    if (
        base_residual_config.interface_version
        != selected_curriculum.interface_version
    ):
        raise ValueError("base residual interface differs from curriculum")
    if (
        base_residual_config.tightening_direction
        != selected_curriculum.tightening_direction
    ):
        raise ValueError("base residual tightening direction is reversed")
    if not math.isclose(
        base_residual_config.policy_rate_hz,
        selected_curriculum.policy_rate_hz,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("base residual policy rate differs from curriculum")
    minimum_speed = (
        base_residual_config.nominal_q7_speed_rad_s
        - base_residual_config.q7_speed_residual_rad_s
    )
    if not math.isclose(
        minimum_speed,
        selected_curriculum.minimum_tightening_speed_rad_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "base residual minimum tightening speed differs from curriculum"
        )

    staged_config = replace(
        base_residual_config,
        target_angle_rad=stage.target_angle_rad,
        success_hold_duration_s=stage.success_hold_duration_s,
        helical_error_tolerance_m=(
            selected_curriculum.maximum_helical_error_m
        ),
        minimum_axial_progress_fraction=(
            stage.minimum_axial_progress_fraction
        ),
        success_angle_tolerance_rad=(
            selected_curriculum.success_angle_tolerance_rad
        ),
    )
    required_steps = _required_episode_steps(
        staged_config.target_angle_rad,
        staged_config.success_hold_duration_s,
        staged_config.policy_rate_hz,
        minimum_speed,
        selected_curriculum.minimum_episode_margin_policy_steps,
    )
    if required_steps != stage.minimum_required_episode_steps:
        raise ValueError("resolved stage episode-step proof changed")
    if stage.maximum_episode_steps < required_steps:
        raise ValueError("resolved stage has insufficient episode steps")

    safe_lower = _finite_float(
        staged_config.q7_safe_lower_rad, "q7_safe_lower_rad"
    )
    safe_upper = _finite_float(
        staged_config.q7_safe_upper_rad, "q7_safe_upper_rad"
    )
    if safe_lower >= safe_upper:
        raise ValueError("base residual q7 safe window is invalid")
    reserved_lower = safe_lower + command_reserve
    reserved_upper = safe_upper - command_reserve
    if reserved_lower > reserved_upper:
        raise ValueError("q7 command reserve leaves no safe window")
    if not reserved_lower <= initial_q7 <= reserved_upper:
        raise ValueError("initial q7 lacks the required command reserve")
    planned_final_q7 = (
        initial_q7
        + staged_config.tightening_direction * staged_config.target_angle_rad
    )
    if not reserved_lower <= planned_final_q7 <= reserved_upper:
        raise ValueError(
            "planned q7 endpoint lacks the required command reserve"
        )
    lower_headroom = planned_final_q7 - safe_lower
    upper_headroom = safe_upper - planned_final_q7
    if min(lower_headroom, upper_headroom) + 1.0e-12 < command_reserve:
        raise ValueError("planned q7 endpoint reserve proof failed")

    return ResolvedResidualStage(
        curriculum_schema_version=selected_curriculum.schema_version,
        stage=stage,
        residual_config=staged_config,
        initial_q7_rad=initial_q7,
        planned_final_q7_rad=planned_final_q7,
        q7_command_reserve_rad=command_reserve,
        lower_headroom_rad=lower_headroom,
        upper_headroom_rad=upper_headroom,
    )


def resolved_stage_document(
    resolved: ResolvedResidualStage,
) -> dict[str, Any]:
    """Return the canonical JSON-safe representation of a resolution."""

    if not isinstance(resolved, ResolvedResidualStage):
        raise TypeError("resolved must be ResolvedResidualStage")
    return resolved.as_dict()


__all__ = [
    "CURRICULUM_SCHEMA_VERSION",
    "CURRICULUM_STAGE_NAMES",
    "DEFAULT_CURRICULUM_PATH",
    "DEFAULT_Q7_COMMAND_RESERVE_RAD",
    "DEFAULT_STAGE_NAME",
    "ConnectorResidualCurriculum",
    "ResidualCurriculumStage",
    "ResolvedResidualStage",
    "load_connector_residual_curriculum",
    "resolve_stage",
    "resolved_stage_document",
]
