"""Three-finger grasp APIs without importing retired controllers eagerly."""

from importlib import import_module


_EXPORT_GROUPS = {
    "finger_contact_detector": (
        "FingerContactDetector",
        "FingerContactDetectorConfig",
        "FingerContactState",
    ),
    "grasp_stability_monitor": (
        "GraspStabilityConfig",
        "GraspStabilityMonitor",
        "wrist_payload_increment",
    ),
    "lift_recovery": (
        "LiftRecoveryConfig",
        "plan_recovery_open",
        "plan_recovery_return",
    ),
    "randomization": (
        "IntervalContract",
        "RandomizationContract",
        "RealizedRandomization",
        "active_fields",
        "realize_randomization",
        "validate_realized",
    ),
    "realized_authoring": (
        "RandomizationValidationConfig",
        "closure_onset_plan",
        "compose_loose_plug_transform",
        "float32_readback_evidence",
        "synchronous_contact_stability",
        "validate_offset_arm_targets",
    ),
    "single_finger_contact_test": (
        "ReleaseBudgetEvidence",
        "SingleFingerCommand",
        "SingleFingerContactConfig",
        "SingleFingerContactTest",
        "release_budget_feasibility",
    ),
    "three_finger_sequential_grasp": (
        "SequentialGraspConfig",
        "ThreeFingerSequentialGrasp",
    ),
}
_LAZY_EXPORTS = {
    name: module_name
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}


def __getattr__(name):
    """Load legacy public exports only when a caller requests one."""

    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = sorted(_LAZY_EXPORTS)
