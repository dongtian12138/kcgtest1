"""Pure tests for the disabled D38999 assembly baseline contract."""

from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from kcg_connector.d38999_assembly_baseline import (
    DEFAULT_D38999_ASSEMBLY_BASELINE_PATH,
    D38999_ASSEMBLY_BASELINE_SCHEMA_VERSION,
    EXPECTED_FIXED_DATUM_PRIM_PATH,
    EXPECTED_FIXED_DATUM_WORLD_M,
    EXPECTED_INSERTION_AXIS_WORLD,
    EXPECTED_PLUG_DATUM_PRIM_PATH,
    EXPECTED_TORQUE_JOINT_NAMES,
    load_d38999_assembly_baseline,
    signed_axial_gap_m,
    thread_proxy_travel_for_degrees_m,
    thread_proxy_travel_m,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _document():
    return yaml.safe_load(
        DEFAULT_D38999_ASSEMBLY_BASELINE_PATH.read_text(encoding="utf-8")
    )


def _write(tmp_path, document):
    path = tmp_path / "d38999_assembly.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def test_shipped_contract_is_prepared_grip_only_and_fail_closed():
    contract = load_d38999_assembly_baseline()
    assert contract.schema_version == (
        D38999_ASSEMBLY_BASELINE_SCHEMA_VERSION
    )
    assert contract.enabled is False
    assert contract.scope.start_state == "prepared_grip"
    assert contract.scope.prepared_grip_required is True
    assert contract.scope.pick_motion_included is False
    assert contract.scope.runtime_integration == "none"
    assert contract.scope.default_execution_allowed is False
    assert contract.scope.fail_closed is True
    assert contract.execution_permitted is False
    assert contract.boundaries.isaac_import_allowed is False
    assert contract.boundaries.ros_import_allowed is False
    assert contract.boundaries.autonomous_execution_allowed is False
    assert contract.boundaries.assembly_success_claimed is False
    json.dumps(contract.as_dict(), allow_nan=False, sort_keys=True)


def test_datum_identity_axis_and_signed_gap_math_are_exact():
    contract = load_d38999_assembly_baseline()
    fixed = contract.datums.fixed
    plug = contract.datums.loose_plug
    assert fixed.prim_path == EXPECTED_FIXED_DATUM_PRIM_PATH
    assert plug.prim_path == EXPECTED_PLUG_DATUM_PRIM_PATH
    assert fixed.feature == "root_contact_face_center"
    assert plug.feature == "socket_face_center"
    assert fixed.position_world_m == pytest.approx(
        EXPECTED_FIXED_DATUM_WORLD_M
    )
    assert fixed.axis_world == pytest.approx(
        EXPECTED_INSERTION_AXIS_WORLD
    )
    assert plug.expected_axis_world == pytest.approx(
        EXPECTED_INSERTION_AXIS_WORLD
    )
    preinsert = contract.plug_position_for_gap_m(0.012)
    assert preinsert == pytest.approx((0.550, 0.185, 0.2735))
    assert contract.axial_gap_m(preinsert) == pytest.approx(0.012)
    assert signed_axial_gap_m(
        preinsert, fixed.position_world_m, fixed.axis_world
    ) == pytest.approx(0.012)
    assert contract.axial_gap_m((0.550, 0.185, 0.2605)) == pytest.approx(
        -0.001
    )


def test_axial_waypoints_partition_insertion_and_remaining_screw_travel():
    plan = load_d38999_assembly_baseline().axial_plan
    assert plan.preinsert_gap_m == pytest.approx(0.012)
    assert plan.entry_gap_m == pytest.approx(0.010)
    assert plan.engage_gap_m == pytest.approx(0.003)
    assert plan.insertion_travel_m == pytest.approx(0.009)
    assert plan.remaining_screw_proxy_travel_m == pytest.approx(0.003)
    assert plan.final_gap_m == pytest.approx(0.0)
    assert plan.preinsert_gap_m - plan.engage_gap_m == pytest.approx(
        plan.insertion_travel_m
    )
    assert plan.engage_gap_m - plan.final_gap_m == pytest.approx(
        plan.remaining_screw_proxy_travel_m
    )


@pytest.mark.parametrize(
    ("rotation_degrees", "expected_travel_m"),
    (
        (20.0, 0.00016666666666666666),
        (120.0, 0.001),
        (360.0, 0.003),
    ),
)
def test_three_millimetre_lead_travel_at_20_120_and_360_degrees(
    rotation_degrees, expected_travel_m
):
    proxy = load_d38999_assembly_baseline().thread_proxy
    assert proxy.travel_for_degrees_m(rotation_degrees) == pytest.approx(
        expected_travel_m
    )
    assert thread_proxy_travel_for_degrees_m(
        rotation_degrees, 0.003
    ) == pytest.approx(expected_travel_m)
    assert thread_proxy_travel_m(
        math.radians(rotation_degrees), 0.003
    ) == pytest.approx(expected_travel_m)


def test_probe_target_and_q7_candidate_are_math_consistent_but_unverified():
    contract = load_d38999_assembly_baseline()
    proxy = contract.thread_proxy
    assert proxy.real_connector_pitch_claimed is False
    assert proxy.thread_tooth_collision_modeled is False
    assert proxy.target_rotation_rad == pytest.approx(math.tau)
    assert proxy.expected_target_travel_m == pytest.approx(0.003)
    assert proxy.probe_rotation_rad == pytest.approx(math.radians(20.0))
    assert proxy.probe_speed_rad_s == pytest.approx(math.radians(5.0))
    assert proxy.expected_probe_duration_s == pytest.approx(4.0)
    assert proxy.expected_probe_travel_m == pytest.approx(1.0 / 6000.0)
    assert contract.q7_direction.tightening_direction_candidate == -1
    assert contract.q7_direction.physical_direction_validated is False
    assert contract.q7_direction.candidate_use_requires_physical_validation
    assert contract.candidate_target_q7_delta_rad == pytest.approx(-math.tau)
    assert contract.execution_permitted is False


def test_three_finger_base_channels_have_operational_and_hard_limits():
    sensing = load_d38999_assembly_baseline().sensing
    assert sensing.torque_joint_names == EXPECTED_TORQUE_JOINT_NAMES
    assert sensing.operational_limit_nm == pytest.approx(1.8)
    assert sensing.hard_stop_nm == pytest.approx(2.0)
    assert sensing.operational_limit_nm < sensing.hard_stop_nm
    assert sensing.require_all_channels_finite is True
    assert sensing.fingertip_tactile_available is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda doc: doc.update(extra=True), "keys are invalid"),
        (lambda doc: doc["scope"].update(extra=True), "keys are invalid"),
        (
            lambda doc: doc.update(enabled=True),
            "must remain disabled",
        ),
        (
            lambda doc: doc["scope"].update(start_state="home"),
            "prepared_grip",
        ),
        (
            lambda doc: doc["units"].update(length="mm"),
            "exact SI",
        ),
        (
            lambda doc: doc["alignment"].update(
                gap_definition="dot(F-P,Fz)"
            ),
            "alignment convention",
        ),
        (
            lambda doc: doc["thread_proxy"].update(
                real_connector_pitch_claimed=True
            ),
            "real pitch",
        ),
        (
            lambda doc: doc["thread_proxy"].update(
                thread_tooth_collision_modeled=True
            ),
            "tooth contact",
        ),
        (
            lambda doc: doc["q7_direction"].update(
                physical_direction_validated=True
            ),
            "not been physically validated",
        ),
        (
            lambda doc: doc["sensing"].update(hard_stop_nm=1.8),
            "hard torque stop",
        ),
    ),
)
def test_exact_schema_and_fail_closed_claims_reject_drift(
    tmp_path, mutator, message
):
    document = deepcopy(_document())
    mutator(document)
    with pytest.raises(ValueError, match=message):
        load_d38999_assembly_baseline(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("datums.fixed", "position_world_m"),
        ("axial_plan", "preinsert_gap_m"),
        ("thread_proxy", "lead_m_per_revolution"),
        ("sensing", "operational_limit_nm"),
    ),
)
@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf")))
def test_nonfinite_contract_numbers_are_rejected(
    tmp_path, section, field, nonfinite
):
    document = deepcopy(_document())
    target = document
    for key in section.split("."):
        target = target[key]
    if field == "position_world_m":
        target[field][2] = nonfinite
    else:
        target[field] = nonfinite
    with pytest.raises(ValueError, match="finite"):
        load_d38999_assembly_baseline(_write(tmp_path, document))


@pytest.mark.parametrize(
    "arguments",
    (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((math.nan, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ),
)
def test_gap_math_rejects_invalid_or_nonfinite_axes(arguments):
    with pytest.raises(ValueError):
        signed_axial_gap_m(*arguments)


@pytest.mark.parametrize(
    "rotation,lead",
    ((math.nan, 0.003), (0.0, math.inf), (0.0, 0.0), (0.0, -0.003)),
)
def test_thread_math_rejects_nonfinite_or_nonpositive_inputs(rotation, lead):
    with pytest.raises(ValueError):
        thread_proxy_travel_m(rotation, lead)


def test_top_level_import_is_pure_and_does_not_load_isaac_or_ros():
    script = """
import importlib
import json
import sys
module = importlib.import_module(
    'kcg_connector.d38999_assembly_baseline'
)
module.load_d38999_assembly_baseline()
for name in ('isaacsim', 'omni', 'pxr', 'rclpy'):
    assert name not in sys.modules, name
print(json.dumps({'pure_import': True}))
"""
    environment = dict(__import__("os").environ)
    python_path = str(PACKAGE_ROOT)
    if environment.get("PYTHONPATH"):
        python_path += ":" + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == {"pure_import": True}
    module = importlib.import_module(
        "kcg_connector.d38999_assembly_baseline"
    )
    source_names = set(module.__dict__)
    assert not {"isaacsim", "omni", "pxr", "rclpy"} & source_names
