"""Pure, fail-closed contract for the D38999 nut damping scan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


D38999_NUT_DAMPING_SCAN_SCHEMA_VERSION = (
    "kcg_d38999_nut_damping_scan_v1"
)
DEFAULT_D38999_NUT_DAMPING_SCAN_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_nut_damping_scan_v1.yaml"
)
EXPECTED_SCENE_SCHEMA_VERSION = "kcg_d38999_tabletop_scene_v1"
EXPECTED_ASSET_SHA256 = (
    "6f716b6e40129f98e5914b5597005c575617d5f55cd0f2c0c8df067ee6788740"
)
EXPECTED_TARGET_SUFFIX = (
    "/D38999Shell25JProxy/LoosePlug/CouplingNut"
)
BASELINE_CANDIDATE_ID = "baseline_schema_default"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: tuple[str, ...], label: str
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


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


@dataclass(frozen=True)
class DampingScanScope:
    scene_schema_version: str
    asset_sha256: str
    target_component: str
    target_prim_suffix: str
    source_asset_mutation_allowed: bool
    in_memory_stage_override_only: bool
    runtime_integration: str
    automatic_promotion_allowed: bool


@dataclass(frozen=True)
class DampingScanExperiment:
    physics_rate_hz: int
    settle_duration_s: float
    tail_duration_s: float
    repeats_per_candidate: int
    fresh_stage_per_repeat: bool
    object_pose_writes_after_start: int

    @property
    def settle_steps(self) -> int:
        return int(round(self.physics_rate_hz * self.settle_duration_s))

    @property
    def tail_steps(self) -> int:
        return int(round(self.physics_rate_hz * self.tail_duration_s))


@dataclass(frozen=True)
class PhysxDampingEvidence:
    physx_schema_version: str
    rigid_body_api: str
    authoring_call: str
    schema_default_angular_damping: float
    schema_default_source: str
    joint_friction_effect_claim: str


@dataclass(frozen=True)
class DampingCandidate:
    candidate_id: str
    run_order: int
    mechanism: str
    target_component: str
    angular_damping: float | None
    expected_resolved_angular_damping: float
    requires_articulation: bool
    efficacy_status: str

    @property
    def is_baseline(self) -> bool:
        return self.candidate_id == BASELINE_CANDIDATE_ID


@dataclass(frozen=True)
class ExcludedMechanism:
    mechanism: str
    target_component: str
    runnable_in_v1: bool
    effectiveness_claimed: bool
    reason: str


@dataclass(frozen=True)
class DampingAcceptance:
    baseline_reproduction_metric: str
    baseline_tail_scene_angular_speed_minimum_rad_s: float
    baseline_tail_scene_angular_speed_maximum_rad_s: float
    candidate_speed_reduction_metric: str
    maximum_tail_nut_angular_speed_rad_s: float
    maximum_tail_relative_axis_speed_rad_s: float
    minimum_speed_reduction_fraction: float
    require_every_repeat_scene_safety_pass: bool
    require_every_repeat_finite: bool
    selection_policy: str


@dataclass(frozen=True)
class DampingPromotionGates:
    automatic_promotion: bool
    require_separate_q7_twist_regression: bool
    require_regrasp_regression: bool
    require_contact_and_tabletop_regression: bool
    require_hardware_calibration_before_fidelity_claim: bool


@dataclass(frozen=True)
class D38999NutDampingScan:
    schema_version: str
    scope: DampingScanScope
    experiment: DampingScanExperiment
    physx_evidence: PhysxDampingEvidence
    candidates: tuple[DampingCandidate, ...]
    excluded_mechanisms: tuple[ExcludedMechanism, ...]
    acceptance: DampingAcceptance
    promotion_gates: DampingPromotionGates

    def candidate(self, candidate_id: str) -> DampingCandidate:
        matches = [
            item for item in self.candidates
            if item.candidate_id == candidate_id
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown damping candidate: {candidate_id!r}")
        return matches[0]

    @property
    def baseline(self) -> DampingCandidate:
        return self.candidate(BASELINE_CANDIDATE_ID)

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


@dataclass(frozen=True)
class CandidateSummary:
    candidate_id: str
    repeat_count: int
    every_repeat_finite: bool
    every_repeat_scene_safety_pass: bool
    maximum_tail_scene_angular_speed_rad_s: float
    maximum_tail_nut_angular_speed_rad_s: float
    maximum_tail_relative_axis_speed_rad_s: float


@dataclass(frozen=True)
class DampingSelection:
    baseline_valid: bool
    selected_candidate_id: str | None
    selected_angular_damping: float | None
    eligible_candidate_ids: tuple[str, ...]
    automatic_promotion_permitted: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


def _load_scope(value: Any) -> DampingScanScope:
    document = _mapping(value, "scope")
    keys = tuple(DampingScanScope.__dataclass_fields__)
    _exact_keys(document, keys, "scope")
    result = DampingScanScope(
        scene_schema_version=_text(
            document["scene_schema_version"],
            "scope.scene_schema_version",
        ),
        asset_sha256=_text(document["asset_sha256"], "scope.asset_sha256"),
        target_component=_text(
            document["target_component"], "scope.target_component"
        ),
        target_prim_suffix=_text(
            document["target_prim_suffix"], "scope.target_prim_suffix"
        ),
        source_asset_mutation_allowed=_boolean(
            document["source_asset_mutation_allowed"],
            "scope.source_asset_mutation_allowed",
        ),
        in_memory_stage_override_only=_boolean(
            document["in_memory_stage_override_only"],
            "scope.in_memory_stage_override_only",
        ),
        runtime_integration=_text(
            document["runtime_integration"], "scope.runtime_integration"
        ),
        automatic_promotion_allowed=_boolean(
            document["automatic_promotion_allowed"],
            "scope.automatic_promotion_allowed",
        ),
    )
    if result.scene_schema_version != EXPECTED_SCENE_SCHEMA_VERSION:
        raise ValueError("scope must use the D38999 tabletop v1 scene")
    if result.asset_sha256 != EXPECTED_ASSET_SHA256:
        raise ValueError("scope asset SHA-256 is not the validated proxy")
    if result.target_component != "coupling_nut_rigid_body":
        raise ValueError("scope target component is unsupported")
    if result.target_prim_suffix != EXPECTED_TARGET_SUFFIX:
        raise ValueError("scope target prim suffix is unsupported")
    if result.source_asset_mutation_allowed:
        raise ValueError("source asset mutation must remain forbidden")
    if not result.in_memory_stage_override_only:
        raise ValueError("scan overrides must remain in-memory only")
    if result.runtime_integration != "none":
        raise ValueError("scan must remain disconnected from runtime")
    if result.automatic_promotion_allowed:
        raise ValueError("automatic promotion must remain forbidden")
    return result


def _load_experiment(value: Any) -> DampingScanExperiment:
    document = _mapping(value, "experiment")
    keys = tuple(DampingScanExperiment.__dataclass_fields__)
    _exact_keys(document, keys, "experiment")
    result = DampingScanExperiment(
        physics_rate_hz=_positive_integer(
            document["physics_rate_hz"], "experiment.physics_rate_hz"
        ),
        settle_duration_s=_positive(
            document["settle_duration_s"],
            "experiment.settle_duration_s",
        ),
        tail_duration_s=_positive(
            document["tail_duration_s"], "experiment.tail_duration_s"
        ),
        repeats_per_candidate=_positive_integer(
            document["repeats_per_candidate"],
            "experiment.repeats_per_candidate",
        ),
        fresh_stage_per_repeat=_boolean(
            document["fresh_stage_per_repeat"],
            "experiment.fresh_stage_per_repeat",
        ),
        object_pose_writes_after_start=_positive_integer(
            document["object_pose_writes_after_start"] + 1,
            "experiment.object_pose_writes_after_start_plus_one",
        ) - 1,
    )
    if result.physics_rate_hz != 240:
        raise ValueError("experiment must use exactly 240 Hz")
    if result.settle_duration_s != 2.0 or result.tail_duration_s != 0.5:
        raise ValueError(
            "experiment must reuse the validated 2.0/0.5 s windows"
        )
    if not result.fresh_stage_per_repeat:
        raise ValueError("every damping repeat must use a fresh stage")
    if result.object_pose_writes_after_start != 0:
        raise ValueError("object pose writes after start must be zero")
    return result


def _load_evidence(value: Any) -> PhysxDampingEvidence:
    document = _mapping(value, "physx_evidence")
    keys = tuple(PhysxDampingEvidence.__dataclass_fields__)
    _exact_keys(document, keys, "physx_evidence")
    result = PhysxDampingEvidence(
        physx_schema_version=_text(
            document["physx_schema_version"],
            "physx_evidence.physx_schema_version",
        ),
        rigid_body_api=_text(
            document["rigid_body_api"], "physx_evidence.rigid_body_api"
        ),
        authoring_call=_text(
            document["authoring_call"],
            "physx_evidence.authoring_call",
        ),
        schema_default_angular_damping=_nonnegative(
            document["schema_default_angular_damping"],
            "physx_evidence.schema_default_angular_damping",
        ),
        schema_default_source=_text(
            document["schema_default_source"],
            "physx_evidence.schema_default_source",
        ),
        joint_friction_effect_claim=_text(
            document["joint_friction_effect_claim"],
            "physx_evidence.joint_friction_effect_claim",
        ),
    )
    if result.physx_schema_version != "110.1.13":
        raise ValueError("PhysX schema version is not the validated version")
    if result.rigid_body_api != "PhysxSchema.PhysxRigidBodyAPI":
        raise ValueError("rigid-body damping API is unsupported")
    if result.authoring_call != "CreateAngularDampingAttr().Set":
        raise ValueError("angular damping authoring call is unsupported")
    if not math.isclose(
        result.schema_default_angular_damping,
        0.05,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("PhysX angular damping schema default must be 0.05")
    if result.joint_friction_effect_claim != (
        "unverified_on_non_articulation_revolute"
    ):
        raise ValueError("joint-friction efficacy must remain unverified")
    return result


def _load_candidates(value: Any) -> tuple[DampingCandidate, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("candidates must be a sequence")
    results = []
    keys = tuple(DampingCandidate.__dataclass_fields__)
    for index, item in enumerate(value):
        label = f"candidates[{index}]"
        document = _mapping(item, label)
        _exact_keys(document, keys, label)
        damping_value = document["angular_damping"]
        damping = (
            None
            if damping_value is None
            else _positive(damping_value, f"{label}.angular_damping")
        )
        candidate = DampingCandidate(
            candidate_id=_text(
                document["candidate_id"], f"{label}.candidate_id"
            ),
            run_order=_positive_integer(
                document["run_order"] + 1, f"{label}.run_order_plus_one"
            ) - 1,
            mechanism=_text(document["mechanism"], f"{label}.mechanism"),
            target_component=_text(
                document["target_component"], f"{label}.target_component"
            ),
            angular_damping=damping,
            expected_resolved_angular_damping=_positive(
                document["expected_resolved_angular_damping"],
                f"{label}.expected_resolved_angular_damping",
            ),
            requires_articulation=_boolean(
                document["requires_articulation"],
                f"{label}.requires_articulation",
            ),
            efficacy_status=_text(
                document["efficacy_status"], f"{label}.efficacy_status"
            ),
        )
        results.append(candidate)
    if len(results) < 2:
        raise ValueError("scan requires baseline and at least one candidate")
    if len({item.candidate_id for item in results}) != len(results):
        raise ValueError("candidate IDs must be unique")
    if tuple(item.run_order for item in results) != tuple(range(len(results))):
        raise ValueError("candidate run_order must be contiguous and ordered")
    baseline = results[0]
    if not baseline.is_baseline:
        raise ValueError("baseline must run first")
    if (
        baseline.mechanism != "none"
        or baseline.angular_damping is not None
        or baseline.efficacy_status
        != "reference_baseline_must_be_measured"
    ):
        raise ValueError("baseline must not author a damping attribute")
    if not math.isclose(
        baseline.expected_resolved_angular_damping,
        0.05,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("baseline must retain schema default 0.05")
    previous = 0.05
    for candidate in results[1:]:
        if candidate.mechanism != "physx_rigid_body_angular_damping":
            raise ValueError("v1 candidates must use rigid-body damping")
        if candidate.target_component != "coupling_nut_rigid_body":
            raise ValueError("candidate target component is unsupported")
        if candidate.requires_articulation:
            raise ValueError(
                "rigid-body damping must not require articulation"
            )
        if candidate.efficacy_status != "must_be_measured":
            raise ValueError("candidate efficacy must remain unmeasured")
        if candidate.angular_damping is None:
            raise ValueError("non-baseline damping must be explicit")
        if not math.isclose(
            candidate.angular_damping,
            candidate.expected_resolved_angular_damping,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "candidate expected damping must match authored value"
            )
        if candidate.angular_damping <= previous:
            raise ValueError("candidate damping values must increase")
        if candidate.angular_damping > 2.0:
            raise ValueError("candidate damping exceeds v1 scan bound")
        previous = candidate.angular_damping
    return tuple(results)


def _load_excluded(value: Any) -> tuple[ExcludedMechanism, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("excluded_mechanisms must be a sequence")
    results = []
    keys = tuple(ExcludedMechanism.__dataclass_fields__)
    for index, item in enumerate(value):
        label = f"excluded_mechanisms[{index}]"
        document = _mapping(item, label)
        _exact_keys(document, keys, label)
        results.append(
            ExcludedMechanism(
                mechanism=_text(
                    document["mechanism"], f"{label}.mechanism"
                ),
                target_component=_text(
                    document["target_component"],
                    f"{label}.target_component",
                ),
                runnable_in_v1=_boolean(
                    document["runnable_in_v1"],
                    f"{label}.runnable_in_v1",
                ),
                effectiveness_claimed=_boolean(
                    document["effectiveness_claimed"],
                    f"{label}.effectiveness_claimed",
                ),
                reason=_text(document["reason"], f"{label}.reason"),
            )
        )
    matches = [
        item
        for item in results
        if item.mechanism == "physx_joint_friction"
    ]
    if len(matches) != 1:
        raise ValueError("physx_joint_friction must be explicitly excluded")
    joint_friction = matches[0]
    if joint_friction.runnable_in_v1 or joint_friction.effectiveness_claimed:
        raise ValueError("joint friction cannot be run or claimed in v1")
    if joint_friction.target_component != "non_articulation_revolute_joint":
        raise ValueError("joint friction exclusion target is incorrect")
    return tuple(results)


def _load_acceptance(value: Any) -> DampingAcceptance:
    document = _mapping(value, "acceptance")
    keys = tuple(DampingAcceptance.__dataclass_fields__)
    _exact_keys(document, keys, "acceptance")
    result = DampingAcceptance(
        baseline_reproduction_metric=_text(
            document["baseline_reproduction_metric"],
            "acceptance.baseline_reproduction_metric",
        ),
        baseline_tail_scene_angular_speed_minimum_rad_s=_positive(
            document["baseline_tail_scene_angular_speed_minimum_rad_s"],
            "acceptance.baseline_scene_minimum",
        ),
        baseline_tail_scene_angular_speed_maximum_rad_s=_positive(
            document["baseline_tail_scene_angular_speed_maximum_rad_s"],
            "acceptance.baseline_scene_maximum",
        ),
        candidate_speed_reduction_metric=_text(
            document["candidate_speed_reduction_metric"],
            "acceptance.candidate_speed_reduction_metric",
        ),
        maximum_tail_nut_angular_speed_rad_s=_positive(
            document["maximum_tail_nut_angular_speed_rad_s"],
            "acceptance.maximum_tail_nut",
        ),
        maximum_tail_relative_axis_speed_rad_s=_positive(
            document["maximum_tail_relative_axis_speed_rad_s"],
            "acceptance.maximum_tail_relative_axis",
        ),
        minimum_speed_reduction_fraction=_positive(
            document["minimum_speed_reduction_fraction"],
            "acceptance.minimum_speed_reduction_fraction",
        ),
        require_every_repeat_scene_safety_pass=_boolean(
            document["require_every_repeat_scene_safety_pass"],
            "acceptance.require_every_repeat_scene_safety_pass",
        ),
        require_every_repeat_finite=_boolean(
            document["require_every_repeat_finite"],
            "acceptance.require_every_repeat_finite",
        ),
        selection_policy=_text(
            document["selection_policy"], "acceptance.selection_policy"
        ),
    )
    if result.baseline_reproduction_metric != (
        "maximum_tail_any_dynamic_body_angular_speed_rad_s"
    ):
        raise ValueError("baseline reproduction metric must match old smoke")
    if result.candidate_speed_reduction_metric != (
        "maximum_tail_nut_angular_speed_rad_s"
    ):
        raise ValueError("candidate reduction metric must remain nut-specific")
    if not (
        result.maximum_tail_nut_angular_speed_rad_s
        < result.baseline_tail_scene_angular_speed_minimum_rad_s
        < result.baseline_tail_scene_angular_speed_maximum_rad_s
    ):
        raise ValueError("baseline and target speed gates do not separate")
    if not 0.0 < result.minimum_speed_reduction_fraction < 1.0:
        raise ValueError("speed reduction fraction must be in (0, 1)")
    if not (
        result.require_every_repeat_scene_safety_pass
        and result.require_every_repeat_finite
    ):
        raise ValueError("all repeats must remain finite and scene-safe")
    if result.selection_policy != (
        "lowest_positive_damping_then_candidate_id"
    ):
        raise ValueError("selection policy is unsupported")
    return result


def _load_promotion(value: Any) -> DampingPromotionGates:
    document = _mapping(value, "promotion_gates")
    keys = tuple(DampingPromotionGates.__dataclass_fields__)
    _exact_keys(document, keys, "promotion_gates")
    result = DampingPromotionGates(
        **{
            name: _boolean(document[name], f"promotion_gates.{name}")
            for name in keys
        }
    )
    if result.automatic_promotion:
        raise ValueError("automatic promotion must remain disabled")
    if not all(
        (
            result.require_separate_q7_twist_regression,
            result.require_regrasp_regression,
            result.require_contact_and_tabletop_regression,
            result.require_hardware_calibration_before_fidelity_claim,
        )
    ):
        raise ValueError(
            "every downstream promotion gate must remain required"
        )
    return result


def load_d38999_nut_damping_scan(
    config_path: Path | str = DEFAULT_D38999_NUT_DAMPING_SCAN_PATH,
) -> D38999NutDampingScan:
    """Load the versioned scan without importing a simulator."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    root = _mapping(document, "root")
    keys = (
        "schema_version",
        "scope",
        "experiment",
        "physx_evidence",
        "candidates",
        "excluded_mechanisms",
        "acceptance",
        "promotion_gates",
    )
    _exact_keys(root, keys, "root")
    if root["schema_version"] != D38999_NUT_DAMPING_SCAN_SCHEMA_VERSION:
        raise ValueError("unsupported D38999 nut damping scan schema")
    config = D38999NutDampingScan(
        schema_version=D38999_NUT_DAMPING_SCAN_SCHEMA_VERSION,
        scope=_load_scope(root["scope"]),
        experiment=_load_experiment(root["experiment"]),
        physx_evidence=_load_evidence(root["physx_evidence"]),
        candidates=_load_candidates(root["candidates"]),
        excluded_mechanisms=_load_excluded(root["excluded_mechanisms"]),
        acceptance=_load_acceptance(root["acceptance"]),
        promotion_gates=_load_promotion(root["promotion_gates"]),
    )
    config.as_dict()
    return config


def _summary_from_mapping(
    value: CandidateSummary | Mapping[str, Any],
) -> CandidateSummary:
    if isinstance(value, CandidateSummary):
        return value
    document = _mapping(value, "candidate summary")
    keys = tuple(CandidateSummary.__dataclass_fields__)
    _exact_keys(document, keys, "candidate summary")
    return CandidateSummary(
        candidate_id=_text(document["candidate_id"], "summary.candidate_id"),
        repeat_count=_positive_integer(
            document["repeat_count"], "summary.repeat_count"
        ),
        every_repeat_finite=_boolean(
            document["every_repeat_finite"],
            "summary.every_repeat_finite",
        ),
        every_repeat_scene_safety_pass=_boolean(
            document["every_repeat_scene_safety_pass"],
            "summary.every_repeat_scene_safety_pass",
        ),
        maximum_tail_scene_angular_speed_rad_s=_nonnegative(
            document["maximum_tail_scene_angular_speed_rad_s"],
            "summary.maximum_tail_scene_angular_speed_rad_s",
        ),
        maximum_tail_nut_angular_speed_rad_s=_nonnegative(
            document["maximum_tail_nut_angular_speed_rad_s"],
            "summary.maximum_tail_nut_angular_speed_rad_s",
        ),
        maximum_tail_relative_axis_speed_rad_s=_nonnegative(
            document["maximum_tail_relative_axis_speed_rad_s"],
            "summary.maximum_tail_relative_axis_speed_rad_s",
        ),
    )


def select_damping_candidate(
    config: D38999NutDampingScan,
    summaries: Sequence[CandidateSummary | Mapping[str, Any]],
) -> DampingSelection:
    """Select evidence only; never promote it into a runtime default."""
    parsed = tuple(_summary_from_mapping(value) for value in summaries)
    if len(parsed) != len(config.candidates):
        raise ValueError("candidate summaries must cover the complete scan")
    if len({item.candidate_id for item in parsed}) != len(parsed):
        raise ValueError("candidate summaries must be unique")
    by_id = {item.candidate_id: item for item in parsed}
    if set(by_id) != {item.candidate_id for item in config.candidates}:
        raise ValueError("candidate summary IDs do not match the scan")
    for summary in parsed:
        if summary.repeat_count != config.experiment.repeats_per_candidate:
            raise ValueError("candidate repeat count does not match contract")
    baseline = by_id[BASELINE_CANDIDATE_ID]
    baseline_scene_speed = baseline.maximum_tail_scene_angular_speed_rad_s
    baseline_nut_speed = baseline.maximum_tail_nut_angular_speed_rad_s
    acceptance = config.acceptance
    baseline_valid = bool(
        baseline.every_repeat_finite
        and baseline.every_repeat_scene_safety_pass
        and acceptance.baseline_tail_scene_angular_speed_minimum_rad_s
        <= baseline_scene_speed
        <= acceptance.baseline_tail_scene_angular_speed_maximum_rad_s
        and baseline_nut_speed > 0.0
    )
    eligible = []
    if baseline_valid:
        for candidate in config.candidates[1:]:
            summary = by_id[candidate.candidate_id]
            reduction = (
                1.0
                - summary.maximum_tail_nut_angular_speed_rad_s
                / baseline_nut_speed
            )
            if (
                summary.every_repeat_finite
                and summary.every_repeat_scene_safety_pass
                and summary.maximum_tail_nut_angular_speed_rad_s
                <= acceptance.maximum_tail_nut_angular_speed_rad_s
                and summary.maximum_tail_relative_axis_speed_rad_s
                <= acceptance.maximum_tail_relative_axis_speed_rad_s
                and reduction
                >= acceptance.minimum_speed_reduction_fraction
            ):
                eligible.append(candidate)
    eligible.sort(
        key=lambda item: (
            float(item.angular_damping),
            item.candidate_id,
        )
    )
    selected = eligible[0] if eligible else None
    if not baseline_valid:
        reason = "baseline_not_reproduced"
    elif selected is None:
        reason = "no_candidate_met_scan_acceptance"
    else:
        reason = "evidence_candidate_selected_not_promoted"
    return DampingSelection(
        baseline_valid=baseline_valid,
        selected_candidate_id=(
            None if selected is None else selected.candidate_id
        ),
        selected_angular_damping=(
            None if selected is None else selected.angular_damping
        ),
        eligible_candidate_ids=tuple(
            item.candidate_id for item in eligible
        ),
        automatic_promotion_permitted=False,
        reason=reason,
    )


__all__ = [
    "BASELINE_CANDIDATE_ID",
    "DEFAULT_D38999_NUT_DAMPING_SCAN_PATH",
    "D38999_NUT_DAMPING_SCAN_SCHEMA_VERSION",
    "CandidateSummary",
    "D38999NutDampingScan",
    "DampingCandidate",
    "DampingSelection",
    "load_d38999_nut_damping_scan",
    "select_damping_candidate",
]
