"""Pure, fail-closed contract for the D38999 assembly baseline.

This module intentionally imports neither Isaac Sim nor ROS.  It defines the
datum and axial/thread arithmetic used by a future prepared-grip experiment;
it does not execute motion or claim physical connector fidelity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


D38999_ASSEMBLY_BASELINE_SCHEMA_VERSION = (
    "kcg_d38999_assembly_baseline_v1"
)
DEFAULT_D38999_ASSEMBLY_BASELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_assembly_baseline_v1.yaml"
)
EXPECTED_FIXED_DATUM_PRIM_PATH = (
    "/World/D38999TabletopV1/D38999Pair/D38999Shell25JProxy/"
    "FixedReceptacle"
)
EXPECTED_PLUG_DATUM_PRIM_PATH = (
    "/World/D38999TabletopV1/D38999Pair/D38999Shell25JProxy/"
    "LoosePlug/BodyAssembly"
)
EXPECTED_FIXED_DATUM_WORLD_M = (0.550, 0.185, 0.2615)
EXPECTED_INSERTION_AXIS_WORLD = (0.0, 0.0, 1.0)
EXPECTED_TORQUE_JOINT_NAMES = ("f1j2", "f2j1", "f3j2")
_ABS_TOLERANCE = 1.0e-12


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise ValueError(
            f"{label} keys are invalid; "
            f"missing={sorted(wanted - actual)}, "
            f"unexpected={sorted(actual - wanted)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three numbers")
    result = tuple(
        _finite(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    return result  # type: ignore[return-value]


def _names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a sequence of names")
    result = tuple(
        _text(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{label} names must be unique")
    return result


def _close(actual: float, expected: float) -> bool:
    return math.isclose(
        actual, expected, rel_tol=0.0, abs_tol=_ABS_TOLERANCE
    )


def _require_close(actual: float, expected: float, label: str) -> None:
    if not _close(actual, expected):
        raise ValueError(f"{label} must be exactly {expected!r}")


def _require_vector(
    actual: Sequence[float], expected: Sequence[float], label: str
) -> None:
    if len(actual) != len(expected) or not all(
        _close(left, right) for left, right in zip(actual, expected)
    ):
        raise ValueError(f"{label} does not match the v1 datum contract")


def signed_axial_gap_m(
    plug_position_world_m: Sequence[Real],
    fixed_position_world_m: Sequence[Real],
    fixed_axis_world: Sequence[Real],
) -> float:
    """Return ``g = (P - F) dot Fz`` after validating a unit Fz axis."""
    plug = _vector3(plug_position_world_m, "plug_position_world_m")
    fixed = _vector3(fixed_position_world_m, "fixed_position_world_m")
    axis = _vector3(fixed_axis_world, "fixed_axis_world")
    norm = math.sqrt(sum(component * component for component in axis))
    if not _close(norm, 1.0):
        raise ValueError("fixed_axis_world must be a unit vector")
    return sum(
        (plug_value - fixed_value) * axis_value
        for plug_value, fixed_value, axis_value in zip(plug, fixed, axis)
    )


def thread_proxy_travel_m(
    rotation_rad: Real, lead_m_per_revolution: Real
) -> float:
    """Return signed proxy travel for a signed rotation in radians."""
    rotation = _finite(rotation_rad, "rotation_rad")
    lead = _positive(lead_m_per_revolution, "lead_m_per_revolution")
    return rotation * lead / math.tau


def thread_proxy_travel_for_degrees_m(
    rotation_degrees: Real, lead_m_per_revolution: Real
) -> float:
    """Degree-boundary helper; contract storage and arithmetic stay SI."""
    degrees = _finite(rotation_degrees, "rotation_degrees")
    return thread_proxy_travel_m(math.radians(degrees), lead_m_per_revolution)


@dataclass(frozen=True)
class AssemblyScope:
    start_state: str
    prepared_grip_required: bool
    pick_motion_included: bool
    runtime_integration: str
    default_execution_allowed: bool
    fail_closed: bool


@dataclass(frozen=True)
class AssemblyUnits:
    length: str
    angle: str
    time: str
    torque: str


@dataclass(frozen=True)
class FixedDatum:
    symbol: str
    prim_path: str
    feature: str
    position_world_m: tuple[float, float, float]
    axis_world: tuple[float, float, float]


@dataclass(frozen=True)
class PlugDatum:
    symbol: str
    prim_path: str
    feature: str
    position_source: str
    expected_axis_world: tuple[float, float, float]


@dataclass(frozen=True)
class AssemblyDatums:
    fixed: FixedDatum
    loose_plug: PlugDatum


@dataclass(frozen=True)
class AssemblyAlignment:
    axis_relation: str
    gap_definition: str
    positive_gap_direction: str


@dataclass(frozen=True)
class AssemblyAxialPlan:
    preinsert_gap_m: float
    entry_gap_m: float
    engage_gap_m: float
    insertion_travel_m: float
    remaining_screw_proxy_travel_m: float
    final_gap_m: float


@dataclass(frozen=True)
class AssemblyThreadProxy:
    model: str
    lead_m_per_revolution: float
    real_connector_pitch_claimed: bool
    thread_tooth_collision_modeled: bool
    target_rotation_rad: float
    expected_target_travel_m: float
    probe_rotation_rad: float
    probe_speed_rad_s: float
    expected_probe_duration_s: float
    expected_probe_travel_m: float

    def travel_m(self, rotation_rad: Real) -> float:
        return thread_proxy_travel_m(
            rotation_rad, self.lead_m_per_revolution
        )

    def travel_for_degrees_m(self, rotation_degrees: Real) -> float:
        return thread_proxy_travel_for_degrees_m(
            rotation_degrees, self.lead_m_per_revolution
        )


@dataclass(frozen=True)
class AssemblyQ7Direction:
    joint_name: str
    tightening_direction_candidate: int
    physical_direction_validated: bool
    candidate_use_requires_physical_validation: bool


@dataclass(frozen=True)
class AssemblySensing:
    source: str
    torque_joint_names: tuple[str, ...]
    operational_limit_nm: float
    hard_stop_nm: float
    require_all_channels_finite: bool
    fingertip_tactile_available: bool


@dataclass(frozen=True)
class AssemblyBoundaries:
    isaac_import_allowed: bool
    ros_import_allowed: bool
    object_pose_drive_allowed: bool
    autonomous_execution_allowed: bool
    physical_contact_validated: bool
    assembly_success_claimed: bool


@dataclass(frozen=True)
class D38999AssemblyBaseline:
    schema_version: str
    enabled: bool
    status: str
    scope: AssemblyScope
    units: AssemblyUnits
    datums: AssemblyDatums
    alignment: AssemblyAlignment
    axial_plan: AssemblyAxialPlan
    thread_proxy: AssemblyThreadProxy
    q7_direction: AssemblyQ7Direction
    sensing: AssemblySensing
    boundaries: AssemblyBoundaries

    def axial_gap_m(self, plug_position_world_m: Sequence[Real]) -> float:
        return signed_axial_gap_m(
            plug_position_world_m,
            self.datums.fixed.position_world_m,
            self.datums.fixed.axis_world,
        )

    def plug_position_for_gap_m(
        self, gap_m: Real
    ) -> tuple[float, float, float]:
        gap = _finite(gap_m, "gap_m")
        return tuple(
            origin + gap * axis
            for origin, axis in zip(
                self.datums.fixed.position_world_m,
                self.datums.fixed.axis_world,
            )
        )  # type: ignore[return-value]

    @property
    def candidate_target_q7_delta_rad(self) -> float:
        return (
            self.q7_direction.tightening_direction_candidate
            * self.thread_proxy.target_rotation_rad
        )

    @property
    def execution_permitted(self) -> bool:
        return bool(
            self.enabled
            and self.scope.default_execution_allowed
            and self.q7_direction.physical_direction_validated
            and self.boundaries.physical_contact_validated
            and self.boundaries.autonomous_execution_allowed
        )

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


def _load_scope(value: Any) -> AssemblyScope:
    document = _mapping(value, "scope")
    _exact_keys(document, AssemblyScope.__dataclass_fields__, "scope")
    result = AssemblyScope(
        start_state=_text(document["start_state"], "scope.start_state"),
        prepared_grip_required=_boolean(
            document["prepared_grip_required"],
            "scope.prepared_grip_required",
        ),
        pick_motion_included=_boolean(
            document["pick_motion_included"], "scope.pick_motion_included"
        ),
        runtime_integration=_text(
            document["runtime_integration"], "scope.runtime_integration"
        ),
        default_execution_allowed=_boolean(
            document["default_execution_allowed"],
            "scope.default_execution_allowed",
        ),
        fail_closed=_boolean(
            document["fail_closed"], "scope.fail_closed"
        ),
    )
    if result.start_state != "prepared_grip":
        raise ValueError("scope must start from prepared_grip")
    if not result.prepared_grip_required or result.pick_motion_included:
        raise ValueError("scope must remain prepared-grip only")
    if result.runtime_integration != "none":
        raise ValueError("runtime integration must remain disabled")
    if result.default_execution_allowed or not result.fail_closed:
        raise ValueError("scope must remain disabled and fail-closed")
    return result


def _load_units(value: Any) -> AssemblyUnits:
    document = _mapping(value, "units")
    _exact_keys(document, AssemblyUnits.__dataclass_fields__, "units")
    result = AssemblyUnits(
        length=_text(document["length"], "units.length"),
        angle=_text(document["angle"], "units.angle"),
        time=_text(document["time"], "units.time"),
        torque=_text(document["torque"], "units.torque"),
    )
    if result != AssemblyUnits("m", "rad", "s", "N*m"):
        raise ValueError("units must use the exact SI v1 convention")
    return result


def _load_fixed_datum(value: Any) -> FixedDatum:
    document = _mapping(value, "datums.fixed")
    _exact_keys(document, FixedDatum.__dataclass_fields__, "datums.fixed")
    result = FixedDatum(
        symbol=_text(document["symbol"], "datums.fixed.symbol"),
        prim_path=_text(document["prim_path"], "datums.fixed.prim_path"),
        feature=_text(document["feature"], "datums.fixed.feature"),
        position_world_m=_vector3(
            document["position_world_m"],
            "datums.fixed.position_world_m",
        ),
        axis_world=_vector3(
            document["axis_world"], "datums.fixed.axis_world"
        ),
    )
    if result.symbol != "F" or result.prim_path != (
        EXPECTED_FIXED_DATUM_PRIM_PATH
    ):
        raise ValueError("fixed datum identity is not the D38999 v1 datum")
    if result.feature != "root_contact_face_center":
        raise ValueError("fixed datum feature is unsupported")
    _require_vector(
        result.position_world_m,
        EXPECTED_FIXED_DATUM_WORLD_M,
        "datums.fixed.position_world_m",
    )
    _require_vector(
        result.axis_world,
        EXPECTED_INSERTION_AXIS_WORLD,
        "datums.fixed.axis_world",
    )
    return result


def _load_plug_datum(value: Any) -> PlugDatum:
    document = _mapping(value, "datums.loose_plug")
    _exact_keys(
        document, PlugDatum.__dataclass_fields__, "datums.loose_plug"
    )
    result = PlugDatum(
        symbol=_text(document["symbol"], "datums.loose_plug.symbol"),
        prim_path=_text(
            document["prim_path"], "datums.loose_plug.prim_path"
        ),
        feature=_text(document["feature"], "datums.loose_plug.feature"),
        position_source=_text(
            document["position_source"],
            "datums.loose_plug.position_source",
        ),
        expected_axis_world=_vector3(
            document["expected_axis_world"],
            "datums.loose_plug.expected_axis_world",
        ),
    )
    if result.symbol != "P" or result.prim_path != (
        EXPECTED_PLUG_DATUM_PRIM_PATH
    ):
        raise ValueError("plug datum identity is not the D38999 v1 datum")
    if result.feature != "socket_face_center":
        raise ValueError("plug datum feature is unsupported")
    if result.position_source != "runtime_observation":
        raise ValueError("plug datum must be observed at runtime")
    _require_vector(
        result.expected_axis_world,
        EXPECTED_INSERTION_AXIS_WORLD,
        "datums.loose_plug.expected_axis_world",
    )
    return result


def _load_datums(value: Any) -> AssemblyDatums:
    document = _mapping(value, "datums")
    _exact_keys(document, AssemblyDatums.__dataclass_fields__, "datums")
    return AssemblyDatums(
        fixed=_load_fixed_datum(document["fixed"]),
        loose_plug=_load_plug_datum(document["loose_plug"]),
    )


def _load_alignment(value: Any) -> AssemblyAlignment:
    document = _mapping(value, "alignment")
    _exact_keys(
        document, AssemblyAlignment.__dataclass_fields__, "alignment"
    )
    result = AssemblyAlignment(
        axis_relation=_text(
            document["axis_relation"], "alignment.axis_relation"
        ),
        gap_definition=_text(
            document["gap_definition"], "alignment.gap_definition"
        ),
        positive_gap_direction=_text(
            document["positive_gap_direction"],
            "alignment.positive_gap_direction",
        ),
    )
    expected = AssemblyAlignment(
        "parallel_same_direction", "dot(P-F,Fz)", "fixed_positive_z"
    )
    if result != expected:
        raise ValueError("alignment convention must match D38999 v1")
    return result


def _load_axial_plan(value: Any) -> AssemblyAxialPlan:
    document = _mapping(value, "axial_plan")
    _exact_keys(
        document, AssemblyAxialPlan.__dataclass_fields__, "axial_plan"
    )
    result = AssemblyAxialPlan(
        preinsert_gap_m=_positive(
            document["preinsert_gap_m"], "axial_plan.preinsert_gap_m"
        ),
        entry_gap_m=_positive(
            document["entry_gap_m"], "axial_plan.entry_gap_m"
        ),
        engage_gap_m=_positive(
            document["engage_gap_m"], "axial_plan.engage_gap_m"
        ),
        insertion_travel_m=_positive(
            document["insertion_travel_m"],
            "axial_plan.insertion_travel_m",
        ),
        remaining_screw_proxy_travel_m=_positive(
            document["remaining_screw_proxy_travel_m"],
            "axial_plan.remaining_screw_proxy_travel_m",
        ),
        final_gap_m=_nonnegative(
            document["final_gap_m"], "axial_plan.final_gap_m"
        ),
    )
    for actual, expected, label in (
        (result.preinsert_gap_m, 0.012, "preinsert gap"),
        (result.entry_gap_m, 0.010, "entry gap"),
        (result.engage_gap_m, 0.003, "engage gap"),
        (result.insertion_travel_m, 0.009, "insertion travel"),
        (
            result.remaining_screw_proxy_travel_m,
            0.003,
            "remaining screw proxy travel",
        ),
        (result.final_gap_m, 0.0, "final gap"),
    ):
        _require_close(actual, expected, f"axial_plan {label}")
    if not (
        result.preinsert_gap_m
        > result.entry_gap_m
        > result.engage_gap_m
        > result.final_gap_m
    ):
        raise ValueError("axial gaps must be strictly decreasing")
    _require_close(
        result.preinsert_gap_m - result.engage_gap_m,
        result.insertion_travel_m,
        "preinsert-to-engage travel",
    )
    _require_close(
        result.engage_gap_m - result.final_gap_m,
        result.remaining_screw_proxy_travel_m,
        "engage-to-final proxy travel",
    )
    return result


def _load_thread_proxy(value: Any) -> AssemblyThreadProxy:
    document = _mapping(value, "thread_proxy")
    _exact_keys(
        document, AssemblyThreadProxy.__dataclass_fields__, "thread_proxy"
    )
    result = AssemblyThreadProxy(
        model=_text(document["model"], "thread_proxy.model"),
        lead_m_per_revolution=_positive(
            document["lead_m_per_revolution"],
            "thread_proxy.lead_m_per_revolution",
        ),
        real_connector_pitch_claimed=_boolean(
            document["real_connector_pitch_claimed"],
            "thread_proxy.real_connector_pitch_claimed",
        ),
        thread_tooth_collision_modeled=_boolean(
            document["thread_tooth_collision_modeled"],
            "thread_proxy.thread_tooth_collision_modeled",
        ),
        target_rotation_rad=_positive(
            document["target_rotation_rad"],
            "thread_proxy.target_rotation_rad",
        ),
        expected_target_travel_m=_positive(
            document["expected_target_travel_m"],
            "thread_proxy.expected_target_travel_m",
        ),
        probe_rotation_rad=_positive(
            document["probe_rotation_rad"],
            "thread_proxy.probe_rotation_rad",
        ),
        probe_speed_rad_s=_positive(
            document["probe_speed_rad_s"],
            "thread_proxy.probe_speed_rad_s",
        ),
        expected_probe_duration_s=_positive(
            document["expected_probe_duration_s"],
            "thread_proxy.expected_probe_duration_s",
        ),
        expected_probe_travel_m=_positive(
            document["expected_probe_travel_m"],
            "thread_proxy.expected_probe_travel_m",
        ),
    )
    if result.model != "kinematic_lead_only":
        raise ValueError("thread proxy must remain kinematic-lead only")
    if (
        result.real_connector_pitch_claimed
        or result.thread_tooth_collision_modeled
    ):
        raise ValueError(
            "thread proxy cannot claim real pitch or tooth contact"
        )
    _require_close(
        result.lead_m_per_revolution, 0.003, "thread proxy lead"
    )
    _require_close(result.target_rotation_rad, math.tau, "target rotation")
    _require_close(
        result.probe_rotation_rad, math.radians(20.0), "probe rotation"
    )
    _require_close(
        result.probe_speed_rad_s,
        math.radians(5.0),
        "probe angular speed",
    )
    _require_close(
        result.expected_target_travel_m,
        result.travel_m(result.target_rotation_rad),
        "expected target travel",
    )
    _require_close(
        result.expected_probe_travel_m,
        result.travel_m(result.probe_rotation_rad),
        "expected probe travel",
    )
    _require_close(
        result.expected_probe_duration_s,
        result.probe_rotation_rad / result.probe_speed_rad_s,
        "expected probe duration",
    )
    return result


def _load_q7_direction(value: Any) -> AssemblyQ7Direction:
    document = _mapping(value, "q7_direction")
    _exact_keys(
        document, AssemblyQ7Direction.__dataclass_fields__, "q7_direction"
    )
    result = AssemblyQ7Direction(
        joint_name=_text(
            document["joint_name"], "q7_direction.joint_name"
        ),
        tightening_direction_candidate=_integer(
            document["tightening_direction_candidate"],
            "q7_direction.tightening_direction_candidate",
        ),
        physical_direction_validated=_boolean(
            document["physical_direction_validated"],
            "q7_direction.physical_direction_validated",
        ),
        candidate_use_requires_physical_validation=_boolean(
            document["candidate_use_requires_physical_validation"],
            "q7_direction.candidate_use_requires_physical_validation",
        ),
    )
    if result.joint_name != "iiwa_joint_7":
        raise ValueError("tightening actuator candidate must be iiwa_joint_7")
    if result.tightening_direction_candidate != -1:
        raise ValueError("q7 tightening direction candidate must be -1")
    if result.physical_direction_validated:
        raise ValueError("q7 direction has not been physically validated")
    if not result.candidate_use_requires_physical_validation:
        raise ValueError("q7 candidate must require physical validation")
    return result


def _load_sensing(value: Any) -> AssemblySensing:
    document = _mapping(value, "sensing")
    _exact_keys(document, AssemblySensing.__dataclass_fields__, "sensing")
    result = AssemblySensing(
        source=_text(document["source"], "sensing.source"),
        torque_joint_names=_names(
            document["torque_joint_names"],
            "sensing.torque_joint_names",
        ),
        operational_limit_nm=_positive(
            document["operational_limit_nm"],
            "sensing.operational_limit_nm",
        ),
        hard_stop_nm=_positive(
            document["hard_stop_nm"], "sensing.hard_stop_nm"
        ),
        require_all_channels_finite=_boolean(
            document["require_all_channels_finite"],
            "sensing.require_all_channels_finite",
        ),
        fingertip_tactile_available=_boolean(
            document["fingertip_tactile_available"],
            "sensing.fingertip_tactile_available",
        ),
    )
    if result.source != "finger_base_single_axis_torque":
        raise ValueError("only finger-base one-axis torque is available")
    if result.torque_joint_names != EXPECTED_TORQUE_JOINT_NAMES:
        raise ValueError("sensing must use exactly the three base channels")
    _require_close(
        result.operational_limit_nm, 1.8, "operational torque limit"
    )
    _require_close(result.hard_stop_nm, 2.0, "hard torque stop")
    if result.operational_limit_nm >= result.hard_stop_nm:
        raise ValueError("operational limit must remain below hard stop")
    if not result.require_all_channels_finite:
        raise ValueError("all torque channels must be finite")
    if result.fingertip_tactile_available:
        raise ValueError("fingertip tactile sensing is not available")
    return result


def _load_boundaries(value: Any) -> AssemblyBoundaries:
    document = _mapping(value, "boundaries")
    _exact_keys(
        document, AssemblyBoundaries.__dataclass_fields__, "boundaries"
    )
    result = AssemblyBoundaries(
        **{
            name: _boolean(document[name], f"boundaries.{name}")
            for name in AssemblyBoundaries.__dataclass_fields__
        }
    )
    if any(asdict(result).values()):
        raise ValueError("unvalidated runtime boundaries must remain false")
    return result


def load_d38999_assembly_baseline(
    path: str | Path = DEFAULT_D38999_ASSEMBLY_BASELINE_PATH,
) -> D38999AssemblyBaseline:
    """Load and validate the exact, disabled D38999 baseline contract."""
    config_path = Path(path)
    document = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "document",
    )
    _exact_keys(
        document,
        D38999AssemblyBaseline.__dataclass_fields__,
        "document",
    )
    result = D38999AssemblyBaseline(
        schema_version=_text(
            document["schema_version"], "schema_version"
        ),
        enabled=_boolean(document["enabled"], "enabled"),
        status=_text(document["status"], "status"),
        scope=_load_scope(document["scope"]),
        units=_load_units(document["units"]),
        datums=_load_datums(document["datums"]),
        alignment=_load_alignment(document["alignment"]),
        axial_plan=_load_axial_plan(document["axial_plan"]),
        thread_proxy=_load_thread_proxy(document["thread_proxy"]),
        q7_direction=_load_q7_direction(document["q7_direction"]),
        sensing=_load_sensing(document["sensing"]),
        boundaries=_load_boundaries(document["boundaries"]),
    )
    if result.schema_version != D38999_ASSEMBLY_BASELINE_SCHEMA_VERSION:
        raise ValueError("unsupported D38999 assembly baseline schema")
    if result.enabled:
        raise ValueError("unvalidated assembly baseline must remain disabled")
    if result.status != "contract_only_not_runtime_validated":
        raise ValueError("assembly baseline status cannot claim validation")
    _require_close(
        result.thread_proxy.expected_target_travel_m,
        result.axial_plan.remaining_screw_proxy_travel_m,
        "thread target versus remaining screw travel",
    )
    if result.execution_permitted:
        raise ValueError("disabled v1 contract cannot permit execution")
    return result


__all__ = (
    "D38999_ASSEMBLY_BASELINE_SCHEMA_VERSION",
    "DEFAULT_D38999_ASSEMBLY_BASELINE_PATH",
    "EXPECTED_FIXED_DATUM_PRIM_PATH",
    "EXPECTED_FIXED_DATUM_WORLD_M",
    "EXPECTED_INSERTION_AXIS_WORLD",
    "EXPECTED_PLUG_DATUM_PRIM_PATH",
    "EXPECTED_TORQUE_JOINT_NAMES",
    "D38999AssemblyBaseline",
    "load_d38999_assembly_baseline",
    "signed_axial_gap_m",
    "thread_proxy_travel_for_degrees_m",
    "thread_proxy_travel_m",
)
