"""Strict loader for the tabletop physical-grasp experiment contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .finger_contact_detector import FingerContactDetectorConfig
from .grasp_stability_monitor import GraspStabilityConfig
from .lift_recovery import LiftRecoveryConfig
from .lift_x_force_admittance import (
    LiftXForceAdmittanceConfig,
    load_lift_x_force_admittance_config,
)
from .lift_xy_force_admittance import (
    LiftXYForceAdmittanceConfig,
    load_lift_xy_force_admittance_config,
)
from .lift_finger_load_balance import (
    LiftFingerLoadBalanceConfig,
    load_lift_finger_load_balance_config,
)
from .lift_finger_absolute_load_hold import (
    LiftFingerAbsoluteLoadHoldConfig,
    load_lift_finger_absolute_load_hold_config,
)
from .lift_finger_root_load_two_sample_suppression import (
    LiftFingerRootLoadTwoSampleSuppressionConfig,
    load_lift_finger_root_load_two_sample_suppression_config,
)
from .lift_finger_fixed_target_hold import (
    LiftFingerFixedTargetHoldConfig,
    load_lift_finger_fixed_target_hold_config,
)
from .lift_phase_arm_damping import (
    LiftPhaseArmDampingConfig,
    load_lift_phase_arm_damping_config,
)
from .lift_task_space_vertical_force_ramp import (
    LiftTaskSpaceVerticalForceRampConfig,
    load_lift_task_space_vertical_force_ramp_config,
)
from .lift_sensor_origin_vertical_force_ramp import (
    LiftSensorOriginVerticalForceRampConfig,
    load_lift_sensor_origin_vertical_force_ramp_config,
)
from .pre_lift_arm_drive_compliance import (
    PreLiftArmDriveComplianceConfig,
    load_pre_lift_arm_drive_compliance_config,
)
from .pre_lift_gravity_preload_transfer import (
    PreLiftGravityPreloadTransferConfig,
    load_pre_lift_gravity_preload_transfer_config,
)
from .pre_lift_arm_stiffness_restoration import (
    PreLiftArmStiffnessRestorationConfig,
    load_pre_lift_arm_stiffness_restoration_config,
)
from .post_contact_finger_damping import (
    PostContactFingerDampingConfig,
    load_post_contact_finger_damping_config,
)
from .pre_lift_wrench_centering import (
    PreLiftWrenchCenteringConfig,
    load_pre_lift_wrench_centering_config,
)
from .pre_lift_vertical_force_xy_admittance import (
    PreLiftVerticalForceXYAdmittanceConfig,
    load_pre_lift_vertical_force_xy_admittance_config,
)
from .pre_lift_xy_nyquist_suppression import (
    PreLiftXYNyquistSuppressionConfig,
    load_pre_lift_xy_nyquist_suppression_config,
)
from .pre_lift_quasistatic_lateral_preload_nulling import (
    PreLiftQuasistaticLateralPreloadNullingConfig,
    load_pre_lift_quasistatic_lateral_preload_nulling_config,
)
from .pre_lift_differential_finger_preload_diagnostic import (
    DifferentialFingerPreloadDiagnosticConfig,
    load_differential_finger_preload_diagnostic_config,
)
from .pre_lift_differential_finger_preload_correction import (
    DifferentialFingerPreloadCorrectionConfig,
    load_differential_finger_preload_correction_config,
)
from .pre_lift_realized_state_rebase import (
    PreLiftRealizedStateRebaseConfig,
    load_pre_lift_realized_state_rebase_config,
)
from .randomization import IntervalContract, RandomizationContract
from .realized_authoring import RandomizationValidationConfig
from .single_finger_contact_test import SingleFingerContactConfig
from .three_finger_sequential_grasp import SequentialGraspConfig


RANDOMIZATION_KEYS = (
    "plug_x_offset_m",
    "plug_y_offset_m",
    "plug_yaw_deg",
    "arm_center_error_x_m",
    "arm_center_error_y_m",
    "finger_start_delay_steps",
    "table_static_friction",
    "table_dynamic_friction",
    "fingertip_static_friction",
    "fingertip_dynamic_friction",
    "plug_mass_scale",
    "center_of_mass_offset_m",
    "lift_speed_scale",
)

RANDOMIZATION_VALIDATION_KEYS = (
    "threshold_label",
    "maximum_arm_joint_delta_rad",
    "maximum_fk_position_error_m",
    "maximum_fk_rotation_error_rad",
)

SINGLE_FINGER_KEYS = (
    "threshold_label",
    "soft_hold_steps",
    "minimum_release_travel_rad",
    "maximum_release_tracking_error_rad",
    "maximum_release_steps",
    "maximum_approach_steps",
    "approach_rate_rad_s",
    "release_rate_rad_s",
)

@dataclass(frozen=True)
class LiftStage:
    increment_m: float
    speed_m_s: float
    hold_s: float


@dataclass(frozen=True)
class PhysicalGraspExperimentConfig:
    schema_version: str
    pick_config: str
    wrist_ft_config: str
    hand_frame: str
    physics_rate_hz: int
    sequential: SequentialGraspConfig
    stability: GraspStabilityConfig
    recovery: LiftRecoveryConfig
    lift_stages: tuple[LiftStage, ...]
    synchronous_closure_duration_s: float
    synchronous_preload_duration_s: float
    reference_window_steps: int
    zero_lift_hold_duration_s: float
    zero_lift_hold_maximum_duration_s: float
    terminal_evaluator_lift_started_dz_m: float
    randomization: RandomizationContract
    randomization_validation: RandomizationValidationConfig
    single_finger: SingleFingerContactConfig
    pre_lift_centering: PreLiftWrenchCenteringConfig
    pre_lift_realized_state_rebase: PreLiftRealizedStateRebaseConfig
    pre_lift_arm_drive_compliance: PreLiftArmDriveComplianceConfig
    pre_lift_gravity_preload_transfer: PreLiftGravityPreloadTransferConfig
    pre_lift_arm_stiffness_restoration: PreLiftArmStiffnessRestorationConfig
    post_contact_finger_damping: PostContactFingerDampingConfig
    lift_x_force_admittance: LiftXForceAdmittanceConfig
    lift_xy_force_admittance: LiftXYForceAdmittanceConfig
    lift_finger_load_balance: LiftFingerLoadBalanceConfig
    lift_finger_absolute_load_hold: LiftFingerAbsoluteLoadHoldConfig
    lift_finger_root_load_two_sample_suppression: (
        LiftFingerRootLoadTwoSampleSuppressionConfig
    )
    lift_finger_fixed_target_hold: LiftFingerFixedTargetHoldConfig
    lift_phase_arm_damping: LiftPhaseArmDampingConfig
    lift_task_space_vertical_force_ramp: LiftTaskSpaceVerticalForceRampConfig
    lift_sensor_origin_vertical_force_ramp: LiftSensorOriginVerticalForceRampConfig
    pre_lift_vertical_force_xy_admittance: PreLiftVerticalForceXYAdmittanceConfig
    pre_lift_xy_nyquist_suppression: PreLiftXYNyquistSuppressionConfig
    pre_lift_quasistatic_lateral_preload_nulling: (
        PreLiftQuasistaticLateralPreloadNullingConfig
    )
    pre_lift_differential_finger_preload_diagnostic: (
        DifferentialFingerPreloadDiagnosticConfig
    )
    pre_lift_differential_finger_preload_correction: (
        DifferentialFingerPreloadCorrectionConfig
    )
    post_grasp_stabilization_proxy_enabled: bool


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _exact_keys(
    document: Mapping[str, Any], expected: tuple[str, ...], label: str
) -> None:
    # Reject both unknown and missing keys for a strict section.
    unknown = sorted(set(document) - set(expected))
    missing = sorted(set(expected) - set(document))
    if unknown or missing:
        raise ValueError(
            f"{label} has unknown keys {unknown} and/or missing keys {missing}"
        )


def _interval(value: Any, label: str) -> IntervalContract:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a [low, high] pair")
    if any(isinstance(entry, bool) for entry in value):
        raise ValueError(f"{label} bounds must not be booleans")
    return IntervalContract(float(value[0]), float(value[1]))


def _delay_set(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = []
    for entry in value:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, int)
            or entry < 0
        ):
            raise ValueError(
                f"{label} must contain non-negative integers"
            )
        result.append(int(entry))
    return tuple(result)


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be text")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must remain repository-relative")
    return value


def load_physical_grasp_experiment_config(
    path: str | Path,
) -> PhysicalGraspExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    document = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "physical grasp config",
    )
    if document.get("schema_version") != "kcg_d38999_tabletop_physical_grasp_v1":
        raise ValueError("unsupported physical grasp schema_version")
    base = _mapping(document.get("base"), "base")
    detector_doc = _mapping(document.get("detector"), "detector")
    sequential_doc = _mapping(document.get("sequential"), "sequential")
    synchronous_doc = _mapping(document.get("synchronous"), "synchronous")
    lift_doc = _mapping(document.get("lift"), "lift")
    recovery_doc = _mapping(document.get("recovery"), "recovery")
    boundaries = _mapping(document.get("boundaries"), "boundaries")
    forbidden_true = (
        "magnetic_force_allowed",
        "pregrasp_latch_allowed",
        "world_to_plug_joint_allowed",
        "object_actuator_allowed",
        "object_pose_write_after_reset_allowed",
        "contact_truth_in_controller_allowed",
        "full_skill_rl_enabled",
    )
    if any(boundaries.get(name) is not False for name in forbidden_true):
        raise ValueError("physical-grasp truth/safety boundary was weakened")
    if boundaries.get("post_grasp_stabilization_proxy_enabled") is not False:
        raise ValueError("post-grasp stabilization proxy must default disabled")

    reference_doc = _mapping(document.get("reference"), "reference")
    zero_lift_doc = _mapping(
        document.get("zero_lift_hold"), "zero_lift_hold"
    )
    terminal_doc = _mapping(
        document.get("terminal_evaluator"), "terminal_evaluator"
    )
    pre_lift_centering = load_pre_lift_wrench_centering_config(
        document.get("pre_lift_centering")
    )
    pre_lift_realized_state_rebase = (
        load_pre_lift_realized_state_rebase_config(
            document.get("pre_lift_realized_state_rebase")
        )
    )
    pre_lift_arm_drive_compliance = (
        load_pre_lift_arm_drive_compliance_config(
            document.get("pre_lift_arm_drive_compliance")
        )
    )
    pre_lift_gravity_preload_transfer = (
        load_pre_lift_gravity_preload_transfer_config(
            document.get("pre_lift_gravity_preload_transfer")
        )
    )
    pre_lift_arm_stiffness_restoration = (
        load_pre_lift_arm_stiffness_restoration_config(
            document.get("pre_lift_arm_stiffness_restoration")
        )
    )
    post_contact_finger_damping = (
        load_post_contact_finger_damping_config(
            document.get("post_contact_finger_damping")
        )
    )
    lift_x_force_admittance = load_lift_x_force_admittance_config(
        document.get("lift_x_force_admittance")
    )
    lift_xy_force_admittance = load_lift_xy_force_admittance_config(
        document.get("lift_xy_force_admittance")
    )
    lift_finger_load_balance = load_lift_finger_load_balance_config(
        document.get("lift_finger_load_balance")
    )
    lift_finger_absolute_load_hold = (
        load_lift_finger_absolute_load_hold_config(
            document.get("lift_finger_absolute_load_hold")
        )
    )
    lift_finger_root_load_two_sample_suppression = (
        load_lift_finger_root_load_two_sample_suppression_config(
            document.get("lift_finger_root_load_two_sample_suppression")
        )
    )
    lift_finger_fixed_target_hold = (
        load_lift_finger_fixed_target_hold_config(
            document.get("lift_finger_fixed_target_hold")
        )
    )
    lift_phase_arm_damping = load_lift_phase_arm_damping_config(
        document.get("lift_phase_arm_damping")
    )
    lift_task_space_vertical_force_ramp = (
        load_lift_task_space_vertical_force_ramp_config(
            document.get("lift_task_space_vertical_force_ramp")
        )
    )
    lift_sensor_origin_vertical_force_ramp = (
        load_lift_sensor_origin_vertical_force_ramp_config(
            document.get("lift_sensor_origin_vertical_force_ramp")
        )
    )
    pre_lift_vertical_force_xy_admittance = (
        load_pre_lift_vertical_force_xy_admittance_config(
            document.get("pre_lift_vertical_force_xy_admittance")
        )
    )
    pre_lift_xy_nyquist_suppression = (
        load_pre_lift_xy_nyquist_suppression_config(
            document.get("pre_lift_xy_nyquist_suppression")
        )
    )
    pre_lift_quasistatic_lateral_preload_nulling = (
        load_pre_lift_quasistatic_lateral_preload_nulling_config(
            document.get("pre_lift_quasistatic_lateral_preload_nulling")
        )
    )
    pre_lift_differential_finger_preload_diagnostic = (
        load_differential_finger_preload_diagnostic_config(
            document.get("pre_lift_differential_finger_preload_diagnostic")
        )
    )
    pre_lift_differential_finger_preload_correction = (
        load_differential_finger_preload_correction_config(
            document.get("pre_lift_differential_finger_preload_correction")
        )
    )
    if (
        lift_task_space_vertical_force_ramp.enabled
        and lift_sensor_origin_vertical_force_ramp.enabled
    ):
        raise ValueError("H18 and H19 vertical force mappings are mutually exclusive")
    if sum(
        bool(value)
        for value in (
            pre_lift_centering.enabled,
            pre_lift_realized_state_rebase.enabled,
            pre_lift_arm_drive_compliance.enabled,
        )
    ) > 1:
        raise ValueError(
            "pre-lift centering, realized-state rebase and arm-drive "
            "compliance are mutually exclusive"
        )
    if (
        post_contact_finger_damping.enabled
        and not pre_lift_arm_drive_compliance.enabled
    ):
        raise ValueError(
            "H7 post-contact finger damping requires the dynamically "
            "supported H6 arm-drive compliance transition"
        )
    if (
        pre_lift_gravity_preload_transfer.enabled
        and not pre_lift_arm_drive_compliance.enabled
    ):
        raise ValueError(
            "H15 gravity preload transfer requires the dynamically supported "
            "H6 arm-drive compliance transition"
        )
    if pre_lift_arm_stiffness_restoration.enabled and not (
        pre_lift_arm_drive_compliance.enabled
        and lift_phase_arm_damping.enabled
    ):
        raise ValueError(
            "H16 stiffness restoration requires the dynamically supported "
            "H6 compliance and H13 damping transitions"
        )
    if lift_x_force_admittance.enabled and not (
        pre_lift_arm_drive_compliance.enabled
        and post_contact_finger_damping.enabled
    ):
        raise ValueError(
            "H8 lift X force admittance requires the dynamically supported "
            "H6 arm-drive compliance and H7 finger damping"
        )
    if lift_xy_force_admittance.enabled and not (
        pre_lift_arm_drive_compliance.enabled
        and post_contact_finger_damping.enabled
    ):
        raise ValueError(
            "H14 lift XY force admittance requires the dynamically supported "
            "H6 arm-drive compliance and H7 finger damping"
        )
    if (
        lift_x_force_admittance.enabled
        and lift_xy_force_admittance.enabled
    ):
        raise ValueError("H8 X and H14 XY admittance are mutually exclusive")
    lateral_admittance_enabled = (
        lift_x_force_admittance.enabled
        or lift_xy_force_admittance.enabled
    )
    if lift_finger_load_balance.enabled and not (
        pre_lift_arm_drive_compliance.enabled
        and post_contact_finger_damping.enabled
        and lateral_admittance_enabled
    ):
        raise ValueError(
            "H11 lift finger load balance requires the dynamically supported "
            "H6 arm-drive compliance, H7 finger damping and one lateral "
            "admittance path"
        )
    if lift_phase_arm_damping.enabled and not (
        pre_lift_arm_drive_compliance.enabled
        and post_contact_finger_damping.enabled
        and lateral_admittance_enabled
        and lift_finger_load_balance.enabled
    ):
        raise ValueError(
            "H13 lift-phase arm damping requires the frozen H6, H7, one "
            "lateral admittance path and H11 control chain"
        )
    if lift_finger_absolute_load_hold.enabled and not (
        pre_lift_arm_drive_compliance.enabled
        and post_contact_finger_damping.enabled
        and lateral_admittance_enabled
        and lift_finger_load_balance.enabled
        and lift_phase_arm_damping.enabled
    ):
        raise ValueError(
            "H17 absolute root-load hold requires the frozen H6, H7, one "
            "lateral admittance path, H11 parameter source and H13 damping"
        )
    if lift_task_space_vertical_force_ramp.enabled and not (
        pre_lift_arm_drive_compliance.enabled
        and pre_lift_gravity_preload_transfer.enabled
        and lift_phase_arm_damping.enabled
        and lift_finger_absolute_load_hold.enabled
    ):
        raise ValueError(
            "H18 vertical force ramp requires the frozen H6/H13/H17 chain "
            "and the dynamically verified H15 joint-effort interface"
        )
    if lift_sensor_origin_vertical_force_ramp.enabled and not (
        pre_lift_arm_drive_compliance.enabled
        and pre_lift_gravity_preload_transfer.enabled
        and lift_phase_arm_damping.enabled
        and lift_finger_absolute_load_hold.enabled
    ):
        raise ValueError(
            "H19 sensor-origin vertical force ramp requires the frozen "
            "H6/H13/H17 chain and the verified H15 joint-effort interface"
        )
    if pre_lift_vertical_force_xy_admittance.enabled and not (
        lift_task_space_vertical_force_ramp.enabled
        and lift_xy_force_admittance.enabled
        and not lift_sensor_origin_vertical_force_ramp.enabled
    ):
        raise ValueError(
            "H20 requires the H18 grasp-TCP force ramp, the frozen H14 XY "
            "admittance and a disabled H19 mapping"
        )
    if (
        pre_lift_vertical_force_xy_admittance.enabled
        and pre_lift_xy_nyquist_suppression.enabled
    ):
        raise ValueError("H20 and H21 pre-lift XY paths are mutually exclusive")
    if pre_lift_xy_nyquist_suppression.enabled and not (
        lift_task_space_vertical_force_ramp.enabled
        and lift_xy_force_admittance.enabled
        and not lift_sensor_origin_vertical_force_ramp.enabled
        and not pre_lift_vertical_force_xy_admittance.enabled
    ):
        raise ValueError(
            "H21 requires the H18 grasp-TCP force ramp, frozen H14 XY "
            "admittance, disabled H19 mapping and disabled H20 path"
        )
    if pre_lift_quasistatic_lateral_preload_nulling.enabled and not (
        pre_lift_xy_nyquist_suppression.enabled
        and lift_task_space_vertical_force_ramp.enabled
        and lift_xy_force_admittance.enabled
        and not lift_sensor_origin_vertical_force_ramp.enabled
        and not pre_lift_vertical_force_xy_admittance.enabled
    ):
        raise ValueError(
            "H22 requires the enabled H21 two-sample path, H18 grasp-TCP "
            "force ramp, frozen H14 XY admittance, disabled H19 mapping "
            "and disabled H20 path"
        )
    if pre_lift_differential_finger_preload_diagnostic.enabled and not (
        pre_lift_quasistatic_lateral_preload_nulling.enabled
        and pre_lift_xy_nyquist_suppression.enabled
        and lift_task_space_vertical_force_ramp.enabled
        and lift_finger_absolute_load_hold.enabled
        and not pre_lift_centering.enabled
    ):
        raise ValueError(
            "H23 diagnostic requires the enabled H22/H21/H18/H17 chain "
            "and the rejected H4 controller must remain disabled"
        )
    if (
        pre_lift_differential_finger_preload_diagnostic.enabled
        and pre_lift_differential_finger_preload_correction.enabled
    ):
        raise ValueError("H23 diagnostic and fixed correction are mutually exclusive")
    if pre_lift_differential_finger_preload_correction.enabled and not (
        pre_lift_quasistatic_lateral_preload_nulling.enabled
        and pre_lift_xy_nyquist_suppression.enabled
        and lift_task_space_vertical_force_ramp.enabled
        and lift_finger_absolute_load_hold.enabled
        and not pre_lift_centering.enabled
        and not pre_lift_differential_finger_preload_diagnostic.enabled
    ):
        raise ValueError(
            "H23 correction requires the enabled H22/H21/H18/H17 chain, "
            "disabled H4 and consumed H23 diagnostic"
        )
    if lift_finger_root_load_two_sample_suppression.enabled and not (
        lift_finger_absolute_load_hold.enabled
        and lift_task_space_vertical_force_ramp.enabled
        and pre_lift_xy_nyquist_suppression.enabled
        and pre_lift_quasistatic_lateral_preload_nulling.enabled
        and pre_lift_differential_finger_preload_correction.enabled
        and not pre_lift_differential_finger_preload_diagnostic.enabled
    ):
        raise ValueError(
            "H24 requires the enabled H17/H18/H21/H22/H23-fixed chain "
            "and the H23 diagnostic must remain disabled"
        )
    if lift_finger_fixed_target_hold.enabled and not (
        lift_finger_absolute_load_hold.enabled
        and lift_task_space_vertical_force_ramp.enabled
        and pre_lift_xy_nyquist_suppression.enabled
        and pre_lift_quasistatic_lateral_preload_nulling.enabled
        and pre_lift_differential_finger_preload_correction.enabled
        and not pre_lift_differential_finger_preload_diagnostic.enabled
        and not lift_finger_root_load_two_sample_suppression.enabled
    ):
        raise ValueError(
            "H25 requires the enabled H17/H18/H21/H22/H23-fixed chain, "
            "disabled H23 diagnostic and disabled rejected H24 path"
        )
    if reference_doc.get("evidence_only") is not True:
        raise ValueError("reference diagnostics must stay evidence-only")
    if terminal_doc.get("posthoc_truth_evaluation_only") is not True:
        raise ValueError("terminal evaluator truth must stay log-only")
    reference_window_steps = _integer(
        reference_doc.get("window_steps"), "reference.window_steps"
    )
    synchronous_preload_duration = _number(
        synchronous_doc.get("preload_duration_s"),
        "synchronous.preload_duration_s",
    )
    zero_lift_duration = _number(
        zero_lift_doc.get("duration_s"), "zero_lift_hold.duration_s"
    )
    zero_lift_maximum = _number(
        zero_lift_doc.get("maximum_duration_s"),
        "zero_lift_hold.maximum_duration_s",
    )
    if zero_lift_duration > zero_lift_maximum:
        raise ValueError(
            "zero_lift_hold.duration_s must be bounded by maximum_duration_s"
        )
    lift_started_dz = _number(
        terminal_doc.get("lift_started_dz_m"),
        "terminal_evaluator.lift_started_dz_m",
    )

    randomization_doc = _mapping(
        document.get("randomization"), "randomization"
    )
    _exact_keys(randomization_doc, RANDOMIZATION_KEYS, "randomization")
    randomization = RandomizationContract(
        plug_x_offset_m=_interval(
            randomization_doc.get("plug_x_offset_m"),
            "randomization.plug_x_offset_m",
        ),
        plug_y_offset_m=_interval(
            randomization_doc.get("plug_y_offset_m"),
            "randomization.plug_y_offset_m",
        ),
        plug_yaw_deg=_interval(
            randomization_doc.get("plug_yaw_deg"),
            "randomization.plug_yaw_deg",
        ),
        arm_center_error_x_m=_interval(
            randomization_doc.get("arm_center_error_x_m"),
            "randomization.arm_center_error_x_m",
        ),
        arm_center_error_y_m=_interval(
            randomization_doc.get("arm_center_error_y_m"),
            "randomization.arm_center_error_y_m",
        ),
        finger_start_delay_steps=_delay_set(
            randomization_doc.get("finger_start_delay_steps"),
            "randomization.finger_start_delay_steps",
        ),
        table_static_friction=_interval(
            randomization_doc.get("table_static_friction"),
            "randomization.table_static_friction",
        ),
        table_dynamic_friction=_interval(
            randomization_doc.get("table_dynamic_friction"),
            "randomization.table_dynamic_friction",
        ),
        fingertip_static_friction=_interval(
            randomization_doc.get("fingertip_static_friction"),
            "randomization.fingertip_static_friction",
        ),
        fingertip_dynamic_friction=_interval(
            randomization_doc.get("fingertip_dynamic_friction"),
            "randomization.fingertip_dynamic_friction",
        ),
        plug_mass_scale=_interval(
            randomization_doc.get("plug_mass_scale"),
            "randomization.plug_mass_scale",
        ),
        center_of_mass_offset_m=_interval(
            randomization_doc.get("center_of_mass_offset_m"),
            "randomization.center_of_mass_offset_m",
        ),
        lift_speed_scale=_interval(
            randomization_doc.get("lift_speed_scale"),
            "randomization.lift_speed_scale",
        ),
    )

    validation_doc = _mapping(
        document.get("randomization_validation"),
        "randomization_validation",
    )
    _exact_keys(
        validation_doc, RANDOMIZATION_VALIDATION_KEYS, "randomization_validation"
    )
    randomization_validation = RandomizationValidationConfig(
        threshold_label=str(
            validation_doc.get("threshold_label")
        ),
        maximum_arm_joint_delta_rad=_number(
            validation_doc.get("maximum_arm_joint_delta_rad"),
            "randomization_validation.maximum_arm_joint_delta_rad",
        ),
        maximum_fk_position_error_m=_number(
            validation_doc.get("maximum_fk_position_error_m"),
            "randomization_validation.maximum_fk_position_error_m",
        ),
        maximum_fk_rotation_error_rad=_number(
            validation_doc.get("maximum_fk_rotation_error_rad"),
            "randomization_validation.maximum_fk_rotation_error_rad",
        ),
    )

    single_finger_doc = _mapping(
        document.get("single_finger"), "single_finger"
    )
    _exact_keys(single_finger_doc, SINGLE_FINGER_KEYS, "single_finger")
    single_finger = SingleFingerContactConfig(
        threshold_label=str(
            single_finger_doc.get("threshold_label")
        ),
        soft_hold_steps=_integer(
            single_finger_doc.get("soft_hold_steps"),
            "single_finger.soft_hold_steps",
        ),
        minimum_release_travel_rad=_number(
            single_finger_doc.get("minimum_release_travel_rad"),
            "single_finger.minimum_release_travel_rad",
        ),
        maximum_release_tracking_error_rad=_number(
            single_finger_doc.get("maximum_release_tracking_error_rad"),
            "single_finger.maximum_release_tracking_error_rad",
        ),
        maximum_release_steps=_integer(
            single_finger_doc.get("maximum_release_steps"),
            "single_finger.maximum_release_steps",
        ),
        maximum_approach_steps=_integer(
            single_finger_doc.get("maximum_approach_steps"),
            "single_finger.maximum_approach_steps",
        ),
        approach_rate_rad_s=_number(
            single_finger_doc.get("approach_rate_rad_s"),
            "single_finger.approach_rate_rad_s",
        ),
        release_rate_rad_s=_number(
            single_finger_doc.get("release_rate_rad_s"),
            "single_finger.release_rate_rad_s",
        ),
    )

    rate_hz = _integer(base.get("physics_rate_hz"), "base.physics_rate_hz")
    if reference_window_steps != round(
        synchronous_preload_duration * rate_hz
    ):
        raise ValueError(
            "reference.window_steps must match the configured synchronous "
            "preload duration at physics_rate_hz"
        )
    sample_period = 1.0 / rate_hz
    detector = FingerContactDetectorConfig(
        sample_period_s=sample_period,
        lowpass_alpha=_number(detector_doc.get("lowpass_alpha"), "detector.lowpass_alpha"),
        derivative_alpha=_number(detector_doc.get("derivative_alpha"), "detector.derivative_alpha"),
        contact_sigma_multiplier=_number(detector_doc.get("contact_sigma_multiplier"), "detector.contact_sigma_multiplier"),
        minimum_contact_delta_nm=_number(detector_doc.get("minimum_contact_delta_nm"), "detector.minimum_contact_delta_nm"),
        release_ratio=_number(detector_doc.get("release_ratio"), "detector.release_ratio"),
        minimum_release_delta_nm=_number(detector_doc.get("minimum_release_delta_nm"), "detector.minimum_release_delta_nm"),
        minimum_rise_rate_nm_s=_number(detector_doc.get("minimum_rise_rate_nm_s"), "detector.minimum_rise_rate_nm_s"),
        maximum_stall_velocity_rad_s=_number(detector_doc.get("maximum_stall_velocity_rad_s"), "detector.maximum_stall_velocity_rad_s"),
        minimum_tracking_error_rad=_number(detector_doc.get("minimum_tracking_error_rad"), "detector.minimum_tracking_error_rad"),
        confirm_steps=_integer(detector_doc.get("confirm_steps"), "detector.confirm_steps"),
        release_confirm_steps=_integer(detector_doc.get("release_confirm_steps"), "detector.release_confirm_steps"),
        maximum_sample_gap_s=_number(detector_doc.get("maximum_sample_gap_s"), "detector.maximum_sample_gap_s"),
        position_velocity_window_steps=_integer(
            detector_doc.get("position_velocity_window_steps"),
            "detector.position_velocity_window_steps",
        ),
        threshold_label=detector_doc.get("threshold_label"),
    )
    sequential = SequentialGraspConfig(
        detector=detector,
        sample_period_s=sample_period,
        approach_rate_rad_s=_number(sequential_doc.get("approach_rate_rad_s"), "sequential.approach_rate_rad_s"),
        soft_hold_preload_rad=_number(sequential_doc.get("soft_hold_preload_rad"), "sequential.soft_hold_preload_rad"),
        load_build_rate_rad_s=_number(sequential_doc.get("load_build_rate_rad_s"), "sequential.load_build_rate_rad_s"),
        balance_gain_rad_per_load=_number(sequential_doc.get("balance_gain_rad_per_load"), "sequential.balance_gain_rad_per_load"),
        maximum_balance_step_rad=_number(sequential_doc.get("maximum_balance_step_rad"), "sequential.maximum_balance_step_rad"),
        maximum_balance_total_rad=_number(sequential_doc.get("maximum_balance_total_rad"), "sequential.maximum_balance_total_rad"),
        probe_increment_rad=_number(sequential_doc.get("probe_increment_rad"), "sequential.probe_increment_rad"),
        probe_settle_steps=_integer(sequential_doc.get("probe_settle_steps"), "sequential.probe_settle_steps"),
        minimum_probe_response_nm=_number(sequential_doc.get("minimum_probe_response_nm"), "sequential.minimum_probe_response_nm"),
        maximum_probe_cross_coupling_ratio=_number(sequential_doc.get("maximum_probe_cross_coupling_ratio"), "sequential.maximum_probe_cross_coupling_ratio"),
        load_scale_nm=tuple(float(value) for value in sequential_doc.get("load_scale_nm", ())),
        stable_minimum_normalized_load=_number(sequential_doc.get("stable_minimum_normalized_load"), "sequential.stable_minimum_normalized_load"),
        maximum_normalized_load_imbalance=_number(sequential_doc.get("maximum_normalized_load_imbalance"), "sequential.maximum_normalized_load_imbalance"),
        stable_confirm_steps=_integer(sequential_doc.get("stable_confirm_steps"), "sequential.stable_confirm_steps"),
        maximum_approach_steps=_integer(sequential_doc.get("maximum_approach_steps"), "sequential.maximum_approach_steps"),
        maximum_load_build_steps=_integer(sequential_doc.get("maximum_load_build_steps"), "sequential.maximum_load_build_steps"),
        soft_hold_window_steps=_integer(sequential_doc.get("soft_hold_window_steps"), "sequential.soft_hold_window_steps"),
        soft_hold_stiffness_scale=_number(
            sequential_doc.get("soft_hold_stiffness_scale"),
            "sequential.soft_hold_stiffness_scale",
        ),
        consolidation_final_stiffness_scale=_number(
            sequential_doc.get("consolidation_final_stiffness_scale"),
            "sequential.consolidation_final_stiffness_scale",
        ),
        consolidation_ramp_steps=_integer(
            sequential_doc.get("consolidation_ramp_steps"),
            "sequential.consolidation_ramp_steps",
        ),
        consolidation_window_steps=_integer(
            sequential_doc.get("consolidation_window_steps"),
            "sequential.consolidation_window_steps",
        ),
        consolidation_threshold_label=str(
            sequential_doc.get("consolidation_threshold_label")
        ),
        probe_mode=str(
            sequential_doc.get("probe_mode", "per_finger")
        ),
    )
    stability = GraspStabilityConfig(
        maximum_root_torque_delta_nm=_number(lift_doc.get("maximum_root_torque_delta_nm"), "lift.maximum_root_torque_delta_nm"),
        minimum_retained_load_fraction=_number(lift_doc.get("minimum_retained_load_fraction"), "lift.minimum_retained_load_fraction"),
        maximum_normalized_load_imbalance=_number(lift_doc.get("maximum_normalized_load_imbalance"), "lift.maximum_normalized_load_imbalance"),
        maximum_load_rate_nm_s=_number(lift_doc.get("maximum_load_rate_nm_s"), "lift.maximum_load_rate_nm_s"),
        maximum_wrist_force_n=_number(lift_doc.get("maximum_wrist_force_n"), "lift.maximum_wrist_force_n"),
        maximum_wrist_moment_nm=_number(lift_doc.get("maximum_wrist_moment_nm"), "lift.maximum_wrist_moment_nm"),
        maximum_arm_tracking_error_rad=_number(lift_doc.get("maximum_arm_tracking_error_rad"), "lift.maximum_arm_tracking_error_rad"),
        maximum_finger_speed_rad_s=_number(lift_doc.get("maximum_finger_speed_rad_s"), "lift.maximum_finger_speed_rad_s"),
        loss_confirm_steps=_integer(lift_doc.get("loss_confirm_steps"), "lift.loss_confirm_steps"),
    )
    increments = tuple(lift_doc.get("stage_increment_m", ()))
    speeds = tuple(lift_doc.get("stage_speed_m_s", ()))
    holds = tuple(lift_doc.get("stage_hold_s", ()))
    if not (len(increments) == len(speeds) == len(holds) == 3):
        raise ValueError("lift requires exactly three increment/speed/hold stages")
    stages = tuple(
        LiftStage(
            _number(increment, f"lift.stage_increment_m[{index}]"),
            _number(speed, f"lift.stage_speed_m_s[{index}]"),
            _number(hold, f"lift.stage_hold_s[{index}]"),
        )
        for index, (increment, speed, hold) in enumerate(zip(increments, speeds, holds))
    )
    recovery = LiftRecoveryConfig(
        return_steps_per_waypoint=_integer(
            recovery_doc.get("return_steps_per_waypoint"),
            "recovery.return_steps_per_waypoint",
        ),
        settle_steps=_integer(
            recovery_doc.get("settle_steps"), "recovery.settle_steps"
        ),
        open_duration_s=_number(
            recovery_doc.get("open_duration_s"), "recovery.open_duration_s"
        ),
    )
    return PhysicalGraspExperimentConfig(
        schema_version=str(document["schema_version"]),
        pick_config=_relative_path(base.get("pick_config"), "base.pick_config"),
        wrist_ft_config=_relative_path(base.get("wrist_ft_config"), "base.wrist_ft_config"),
        hand_frame=str(base.get("hand_frame")),
        physics_rate_hz=rate_hz,
        sequential=sequential,
        stability=stability,
        recovery=recovery,
        lift_stages=stages,
        synchronous_closure_duration_s=_number(synchronous_doc.get("closure_duration_s"), "synchronous.closure_duration_s"),
        synchronous_preload_duration_s=synchronous_preload_duration,
        reference_window_steps=reference_window_steps,
        zero_lift_hold_duration_s=zero_lift_duration,
        zero_lift_hold_maximum_duration_s=zero_lift_maximum,
        terminal_evaluator_lift_started_dz_m=lift_started_dz,
        randomization=randomization,
        randomization_validation=randomization_validation,
        single_finger=single_finger,
        pre_lift_centering=pre_lift_centering,
        pre_lift_realized_state_rebase=pre_lift_realized_state_rebase,
        pre_lift_arm_drive_compliance=pre_lift_arm_drive_compliance,
        pre_lift_gravity_preload_transfer=(
            pre_lift_gravity_preload_transfer
        ),
        pre_lift_arm_stiffness_restoration=(
            pre_lift_arm_stiffness_restoration
        ),
        post_contact_finger_damping=post_contact_finger_damping,
        lift_x_force_admittance=lift_x_force_admittance,
        lift_xy_force_admittance=lift_xy_force_admittance,
        lift_finger_load_balance=lift_finger_load_balance,
        lift_finger_absolute_load_hold=lift_finger_absolute_load_hold,
        lift_finger_root_load_two_sample_suppression=(
            lift_finger_root_load_two_sample_suppression
        ),
        lift_finger_fixed_target_hold=lift_finger_fixed_target_hold,
        lift_phase_arm_damping=lift_phase_arm_damping,
        lift_task_space_vertical_force_ramp=(
            lift_task_space_vertical_force_ramp
        ),
        lift_sensor_origin_vertical_force_ramp=(
            lift_sensor_origin_vertical_force_ramp
        ),
        pre_lift_vertical_force_xy_admittance=(
            pre_lift_vertical_force_xy_admittance
        ),
        pre_lift_xy_nyquist_suppression=(
            pre_lift_xy_nyquist_suppression
        ),
        pre_lift_quasistatic_lateral_preload_nulling=(
            pre_lift_quasistatic_lateral_preload_nulling
        ),
        pre_lift_differential_finger_preload_diagnostic=(
            pre_lift_differential_finger_preload_diagnostic
        ),
        pre_lift_differential_finger_preload_correction=(
            pre_lift_differential_finger_preload_correction
        ),
        post_grasp_stabilization_proxy_enabled=False,
    )
