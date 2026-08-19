import ast
from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp.lift_xy_force_admittance import (
    LiftXYForceAdmittanceConfig,
    MAXIMUM_STEP_CORRECTION_NORM_M,
    MAXIMUM_TOTAL_CORRECTION_NORM_M,
    SOURCE_TARGET_STIFFNESS_N_M,
    TASK_XY_COMPLIANCE_M_N,
    derive_lift_xy_force_admittance_step,
    load_lift_xy_force_admittance_config,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_multilayer_tabletop_physical_grasp_v2.yaml"
)
RUNNER = REPOSITORY / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"

RUN14_TERMINAL_ROTATION = (
    (-0.9272157959045737, 0.37451846313367265, -0.0026054936269319317),
    (0.3745175791969395, 0.9272194352684281, 0.0008376950288661639),
    (0.002729596584145056, -0.0001990791028561477, -0.999996254827986),
)
RUN14_TERMINAL_FORCE_XY_N = (0.9970551120697054, -0.6751753663533504)


def _config():
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return load_lift_xy_force_admittance_config(
        document["lift_xy_force_admittance"]
    )


def test_h14_is_the_single_run14_derived_parameter_set():
    config = _config()
    assert config.enabled is True
    assert config.source_run_id == "B-V2-GRASP-14"
    assert config.source_h8_run_id == "B-V2-GRASP-05"
    assert config.source_target_stiffness_n_m == pytest.approx(
        SOURCE_TARGET_STIFFNESS_N_M
    )
    assert config.task_xy_compliance_m_n == pytest.approx(
        TASK_XY_COMPLIANCE_M_N
    )
    assert config.maximum_total_correction_norm_m == pytest.approx(
        MAXIMUM_TOTAL_CORRECTION_NORM_M
    )
    assert config.maximum_step_correction_norm_m == pytest.approx(
        MAXIMUM_STEP_CORRECTION_NORM_M
    )


def test_h14_rotates_run14_xy_force_and_applies_vector_rate_bound():
    step = derive_lift_xy_force_admittance_step(
        RUN14_TERMINAL_FORCE_XY_N,
        RUN14_TERMINAL_ROTATION,
        (0.0, 0.0),
        _config(),
    )
    assert step["world_projected_force_xy_n"][0] < 0.0
    assert step["world_projected_force_xy_n"][1] < 0.0
    assert step["desired_unbounded_correction_norm_m"] == pytest.approx(
        29.3519e-6, abs=1.0e-10
    )
    assert step["total_bound_active"] is False
    assert step["rate_bound_active"] is True
    assert step["applied_delta_norm_m"] == pytest.approx(1.0e-6)
    assert step["applied_correction_norm_m"] == pytest.approx(1.0e-6)


def test_h14_applies_total_norm_bound_without_changing_direction():
    correction = (0.0, 0.0)
    for _ in range(40):
        step = derive_lift_xy_force_admittance_step(
            (8.0, 8.0),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            correction,
            _config(),
        )
        correction = tuple(step["applied_correction_xy_m"])
    assert step["total_bound_active"] is True
    assert step["applied_correction_norm_m"] == pytest.approx(30.0e-6)
    assert correction[0] == pytest.approx(correction[1])


def test_h14_rejects_parameter_rotation_and_force_drift():
    values = dict(_config().__dict__)
    values["maximum_total_correction_norm_m"] = 31.0e-6
    with pytest.raises(ValueError, match="frozen"):
        LiftXYForceAdmittanceConfig(**values)
    with pytest.raises(ValueError, match="orthonormal"):
        derive_lift_xy_force_admittance_step(
            (1.0, 0.0),
            ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0),
            _config(),
        )
    with pytest.raises(ValueError, match="force gate"):
        derive_lift_xy_force_admittance_step(
            (8.01, 0.0),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0),
            _config(),
        )


def test_h14_runner_keeps_truth_out_and_hard_gate_in_charge():
    source = RUNNER.read_text(encoding="utf-8")
    assert "derive_lift_xy_force_admittance_step" in source
    assert '"rotation_from_robot_fk_only": True' in source
    assert '"sensor_origin_hard_gate_unchanged": True' in source
    assert '"z_target_modified": False' in source
    assert '"orientation_target_modified": False' in source
    assert '"object_truth_used": False' in source
    assert '"contact_truth_used": False' in source
    assert '"event_truth_used": False' in source
    assert '"object_pose_written": False' in source


def test_h14_runner_adapts_all_runtime_arguments_to_tuples_at_every_call():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "derive_lift_xy_force_admittance_step"
    ]
    assert calls
    for call in calls:
        assert len(call.args) == 4
        for argument in call.args[:3]:
            assert isinstance(argument, ast.Call)
            assert isinstance(argument.func, ast.Name)
            assert argument.func.id == "tuple"
