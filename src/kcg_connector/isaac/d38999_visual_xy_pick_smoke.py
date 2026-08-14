#!/usr/bin/env python3

"""Independent opt-in visual-XY pick of one non-nominal D38999 plug.

The loose endpoint is authored at +10 mm world X/Y before physics starts.
After settling in the same ``World``, a real masked RGB-D capture supplies only
its ray-plane XY estimate to the strict PoseProvider/adapter/planner chain.
Scene truth never enters target or IK computation; it is retained solely as a
post-hoc evaluation record.  Orientation remains registered nominal, and this
probe does not claim full 6D pose, production control, or collision planning.
An additional disabled flag may continue the same capture and ``World`` to the
registered 12 mm preinsert waypoint.  The only active insertion experiment is
wrist-FT-only: it consumes compensated wrist wrench and robot proprioception,
never PhysX contact identity/manifold or simulator object truth.  Historical
``--tactile-*`` spellings are retained only to fail old commands explicitly.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import traceback


RESULT_MARKER = "ISAAC D38999 VISUAL XY PICK PROBE V1"
PREINSERT_RESULT_MARKER = "ISAAC D38999 VISUAL XY PREINSERT PROBE V1"
WRIST_FT_GUARDED_RESULT_MARKER = (
    "ISAAC D38999 VISUAL WRIST FT GUARDED INSERTION V1"
)
TACTILE_LIP_RESULT_MARKER = "ISAAC D38999 TACTILE LIP CALIBRATION V1"
TACTILE_RETRACT_PREFLIGHT_RESULT_MARKER = (
    "ISAAC D38999 TACTILE RETRACT PREFLIGHT V1"
)
TACTILE_LIP_MANIFOLD_RESULT_MARKER = (
    "ISAAC D38999 TACTILE LIP MANIFOLD CAPTURE V1"
)

# This displacement is intentionally a runtime-calibration constant rather
# than a hidden pose correction.  Each direction starts from the immutable
# visual fixed-XY preinsert plan, so no simulator truth can steer the touches.
TACTILE_LIP_OFFSET_M = 0.0006
# The no-contact GPU preflight intentionally commands below its immutable
# actual-motion ceiling.  The first frozen run observed 0.141 um overshoot at
# a 0.5 mm descent and ~0.56--0.60 mm/s measured peaks at a 0.5 mm/s command;
# 0.45 mm and 0.35 mm/s preserve explicit headroom without weakening any gate.
TACTILE_RETRACT_PREFLIGHT_COMMAND_DESCENT_M = 0.00045
TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S = 0.00035
TACTILE_RETRACT_PREFLIGHT_DESCENT_M = 0.0005
# This is the exact maximum negative reversal observed by the frozen, audited
# no-contact GPU run ``gpu_20260812T203340Z``.  Later preflights must prove
# their own observed reversal stays inside this immutable ceiling; they may
# never replace it with a larger fresh observation.
TACTILE_FIXED_NEGATIVE_RETRACT_BOUND_M = 1.1928619392254092e-7
# Stage A deliberately captures only the first +X segmented-lip contact
# manifold.  It shares the proven no-contact command-speed headroom, then
# freezes the exact last applied drive target.  No preload or FT sign
# calibration is authorized by this diagnostic-only milestone.
TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S = 0.00035
# The axial approach is slower than the lateral offset and unload.  The first
# Stage-A GPU capture interrupted a 0.35 mm/s profile while measured motion
# was still descending; 0.05 mm/s is a seven-fold command reduction.  A new
# matched no-contact mid-slope reversal must still pass before contact runs.
TACTILE_LIP_MANIFOLD_APPROACH_SPEED_M_S = 0.00005
TACTILE_MATCHED_REVERSAL_WINDOW_SAMPLES = 6
TACTILE_LIP_MANIFOLD_DIRECTION = (
    "plus_x",
    (1.0, 0.0),
)
TACTILE_LIP_MANIFOLD_NORMAL_CONVENTION = (
    "physx_pxcontactpoint_normal_points_shape1_to_shape0"
)
TACTILE_PREFLIGHT_PEAK_FIELDS = (
    "absolute_axial_force_n",
    "lateral_force_n",
    "bending_torque_nm",
    "absolute_tightening_torque_nm",
    "absolute_finger_base_torque_nm",
)
TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS = (
    "loose_fixed",
    "intended_lip",
    "unexpected_loose_fixed",
    "loose_fixture",
    "loose_table",
)
RUNTIME_SOURCE_PATH = Path(__file__).resolve()
RUNTIME_SOURCE_IMPORT_SHA256 = hashlib.sha256(
    RUNTIME_SOURCE_PATH.read_bytes()
).hexdigest()
TACTILE_LIP_DIRECTIONS = (
    ("plus_x", (1.0, 0.0), "My", 4),
    ("minus_x", (-1.0, 0.0), "My", 4),
    ("plus_y", (0.0, 1.0), "Mx", 3),
    ("minus_y", (0.0, -1.0), "Mx", 3),
)


class TactileSafetyStop(RuntimeError):
    """Carry whether an unsafe sample forbids every further physics step."""

    def __init__(self, reason, *, zero_step_abort):
        super().__init__(str(reason))
        self.reason = str(reason)
        self.zero_step_abort = bool(zero_step_abort)


class TactilePreflightComplete(RuntimeError):
    """Internal control signal for a successful preflight-only terminal."""


class TactileManifoldComplete(RuntimeError):
    """Internal signal for a passed capture-and-terminal-retract probe."""


def bounded_endpoint_hold_path_indices(command_count, hold_tick_limit):
    """Return path indices for a bounded final-command release hold.

    The first ``command_count`` entries traverse the path once.  At most
    ``hold_tick_limit`` further entries repeat only its last index; callers can
    stop sooner when the measured retract plus release debounce is complete.
    """

    if type(command_count) is not int or command_count <= 0:
        raise ValueError("command_count must be a positive integer")
    if type(hold_tick_limit) is not int or hold_tick_limit < 0:
        raise ValueError("hold_tick_limit must be a nonnegative integer")
    return tuple(range(command_count)) + (
        (command_count - 1,) * hold_tick_limit
    )


def minimum_jerk_steps_for_peak_speed(
    distance_m, rate_hz, maximum_peak_speed_m_s
):
    """Return enough discrete ticks for a bounded quintic minimum jerk.

    The derivative of ``10s^3 - 15s^4 + 6s^5`` peaks at exactly 1.875.
    Ceiling the continuous-time requirement, rather than rounding it, keeps
    the nominal Cartesian peak at or below the requested limit.
    """

    import math

    values = tuple(
        float(value)
        for value in (distance_m, rate_hz, maximum_peak_speed_m_s)
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("minimum-jerk distance/rate/speed must be positive")
    return max(
        1,
        int(math.ceil(1.875 * values[0] * values[1] / values[2])),
    )


def minimum_jerk_steps_with_strict_headroom(
    distance_m, rate_hz, nominal_peak_speed_m_s
):
    """Add one discrete tick beyond the continuous minimum.

    A ceiling computed from the analytical continuous peak can land exactly
    on the requested limit (2.5 mm at 240 Hz and 0.05 mm/s is the concrete
    22,500-tick case).  State-equivalence preflight is safety evidence, so it
    must not rely on that floating-point knife edge.  The extra tick is an
    explicit planner rule; it does not relax the independent hard ceiling.
    """

    base = minimum_jerk_steps_for_peak_speed(
        distance_m, rate_hz, nominal_peak_speed_m_s
    )
    final = base + 1
    return {
        "semantics": "continuous_minimum_plus_one_discrete_headroom_tick",
        "base_required_command_count": base,
        "added_headroom_command_count": 1,
        "final_command_count": final,
        "nominal_peak_speed_ceiling_m_s": float(
            nominal_peak_speed_m_s
        ),
    }


def build_discrete_mid_slope_interrupt_plan(
    start_arm,
    target_arm,
    command_count,
    task_z_axis_world,
    sample_rate_hz,
    fk_transform,
    *,
    strict_headroom_plan,
):
    """Find the unique worst discrete downward command-FK step.

    The returned motion index uses the same zero-based convention as
    ``run_tactile_motion`` target mode: index zero commands blend(1/N), not the
    exact start command.  The complete unexecuted plan is inspected so the
    selected interrupt is a measured discrete argmax, never a nominal 50%
    assumption.  Only its compact three-point neighborhood is serialized.
    """

    import hashlib
    import json
    import math

    start = seven_float_arm_tuple(start_arm, "interrupt_start_arm")
    target = seven_float_arm_tuple(target_arm, "interrupt_target_arm")
    axis = three_float_position_tuple(
        task_z_axis_world, "interrupt_task_z_axis_world"
    )
    if type(command_count) is not int or command_count < 3:
        raise ValueError("interrupt command_count must be integer >= 3")
    if not isinstance(strict_headroom_plan, Mapping):
        raise ValueError("interrupt strict headroom plan is missing")
    expected_headroom_keys = {
        "semantics",
        "base_required_command_count",
        "added_headroom_command_count",
        "final_command_count",
        "nominal_peak_speed_ceiling_m_s",
    }
    if set(strict_headroom_plan) != expected_headroom_keys:
        raise ValueError("interrupt strict headroom plan schema changed")
    if (
        strict_headroom_plan.get("semantics")
        != "continuous_minimum_plus_one_discrete_headroom_tick"
        or type(strict_headroom_plan.get("base_required_command_count"))
        is not int
        or strict_headroom_plan.get("base_required_command_count") < 1
        or strict_headroom_plan.get("added_headroom_command_count") != 1
        or strict_headroom_plan.get("final_command_count") != command_count
        or command_count
        != strict_headroom_plan["base_required_command_count"] + 1
    ):
        raise ValueError("interrupt command count lacks strict headroom")
    rate = float(sample_rate_hz)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("interrupt sample rate must be positive")
    if abs(math.sqrt(sum(value * value for value in axis)) - 1.0) > 1e-12:
        raise ValueError("interrupt task Z axis must be unit length")

    def fk_position(command, label):
        rows = tuple(
            tuple(float(value) for value in row)
            for row in fk_transform(command)
        )
        if (
            len(rows) != 4
            or any(len(row) != 4 for row in rows)
            or not all(math.isfinite(value) for row in rows for value in row)
        ):
            raise ValueError(f"{label} FK must be finite 4x4")
        return tuple(rows[index][3] for index in range(3))

    previous_position = fk_position(start, "interrupt start")
    step_down = []
    commands = []
    positions = []
    for index in range(command_count):
        fraction = float(index + 1) / float(command_count)
        # Keep this helper import-lazy: the production planner is imported
        # only after SimulationApp starts.  This algebra is exactly the same
        # quintic used by that planner and is independently unit tested.
        blend = (
            10.0 * fraction**3
            - 15.0 * fraction**4
            + 6.0 * fraction**5
        )
        command = tuple(
            initial + blend * (final - initial)
            for initial, final in zip(start, target)
        )
        position = fk_position(command, f"interrupt command[{index}]")
        downward = -sum(
            (position[axis_index] - previous_position[axis_index])
            * axis[axis_index]
            for axis_index in range(3)
        )
        commands.append(command)
        positions.append(position)
        step_down.append(downward)
        previous_position = position
    maximum = max(step_down)
    maxima = tuple(
        index for index, value in enumerate(step_down) if value == maximum
    )
    if len(maxima) != 1:
        raise ValueError("interrupt command-FK downward argmax is not unique")
    interrupt = maxima[0]
    if interrupt <= 0 or interrupt >= command_count - 1:
        raise ValueError("interrupt command-FK argmax is not interior")
    neighborhood = []
    for index in range(interrupt - 1, interrupt + 2):
        neighborhood.append(
            {
                "motion_index": index,
                "command_arm_rad": list(commands[index]),
                "command_fk_tcp_world_m": list(positions[index]),
                "downward_step_m": step_down[index],
                "downward_speed_m_s": step_down[index] * rate,
            }
        )
    digest_payload = {
        "start_arm_rad": list(start),
        "target_arm_rad": list(target),
        "command_count": command_count,
        "task_z_axis_world": list(axis),
        "sample_rate_hz": rate,
        "step_down_m": step_down,
    }
    digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "semantics": "unique_discrete_command_fk_downward_speed_argmax",
        "start_arm_rad": start,
        "target_arm_rad": target,
        "command_count": command_count,
        "sample_rate_hz": rate,
        "task_z_axis_world": axis,
        "unique_argmax_motion_index": interrupt,
        "unique_argmax_count": len(maxima),
        "interior_argmax_gate": True,
        "maximum_downward_step_m": maximum,
        "maximum_downward_speed_m_s": maximum * rate,
        "planned_argmax_neighborhood": tuple(neighborhood),
        "full_step_down_sha256": digest,
        "strict_headroom_plan": dict(strict_headroom_plan),
    }


def tactile_reversal_state_features(samples, task_rotation_world, rate_hz):
    """Derive offset-invariant loaded-drive dynamics from six raw ticks."""

    import math

    if not isinstance(samples, (list, tuple)) or len(samples) != (
        TACTILE_MATCHED_REVERSAL_WINDOW_SAMPLES
    ):
        raise ValueError("state-equivalence window must contain exactly 6")
    rotation = tuple(
        tuple(float(value) for value in row) for row in task_rotation_world
    )
    if (
        len(rotation) != 3
        or any(len(row) != 3 for row in rotation)
        or not all(math.isfinite(value) for row in rotation for value in row)
    ):
        raise ValueError("state-equivalence rotation must be finite 3x3")
    rate = float(rate_hz)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("state-equivalence rate must be positive")

    def vector(sample, name, length):
        value = sample.get(name) if isinstance(sample, Mapping) else None
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise ValueError(f"state-equivalence {name} shape changed")
        result = tuple(float(component) for component in value)
        if not all(math.isfinite(component) for component in result):
            raise ValueError(f"state-equivalence {name} is non-finite")
        return result

    def task_vector(world_vector):
        # ``rotation`` stores task-frame axes as columns expressed in world.
        # Therefore world -> task is R.T @ v, not R @ v.  Identity fixtures
        # hide this distinction, so a non-identity regression locks it down.
        return tuple(
            sum(rotation[world_index][task_index] * world_vector[world_index]
                for world_index in range(3))
            for task_index in range(3)
        )

    rows = []
    for index, sample in enumerate(samples):
        command = vector(sample, "command_arm_rad", 7)
        measured = vector(sample, "measured_arm_rad", 7)
        velocity = vector(sample, "measured_arm_velocity_rad_s", 7)
        command_fk = vector(sample, "command_fk_tcp_world_m", 3)
        measured_fk = vector(sample, "measured_arm_fk_tcp_world_m", 3)
        measured_tcp = vector(sample, "measured_tcp_prim_world_m", 3)
        pre_tcp = vector(sample, "pre_measured_tcp_prim_world_m", 3)
        tcp_velocity = task_vector(
            tuple((measured_tcp[i] - pre_tcp[i]) * rate for i in range(3))
        )
        rows.append(
            {
                "sample_index": index,
                "motion_index": sample.get("motion_index"),
                "measured_arm_velocity_rad_s": velocity,
                "command_tracking_error_arm_rad": tuple(
                    measured[i] - command[i] for i in range(7)
                ),
                "command_fk_minus_measured_fk_task_m": task_vector(
                    tuple(command_fk[i] - measured_fk[i] for i in range(3))
                ),
                "command_fk_minus_measured_tcp_task_m": task_vector(
                    tuple(command_fk[i] - measured_tcp[i] for i in range(3))
                ),
                "measured_tcp_velocity_task_m_s": tcp_velocity,
                "measured_tcp_speed_m_s": math.sqrt(
                    sum(value * value for value in tcp_velocity)
                ),
            }
        )
    origin_tcp = vector(samples[0], "measured_tcp_prim_world_m", 3)
    negative_drift = []
    for sample in samples[1:]:
        tcp = vector(sample, "measured_tcp_prim_world_m", 3)
        task_delta = task_vector(
            tuple(tcp[index] - origin_tcp[index] for index in range(3))
        )
        negative_drift.append(max(0.0, -task_delta[2]))
    vector_fields = {
        "measured_arm_velocity_rad_s": 7,
        "command_tracking_error_arm_rad": 7,
        "command_fk_minus_measured_fk_task_m": 3,
        "command_fk_minus_measured_tcp_task_m": 3,
        "measured_tcp_velocity_task_m_s": 3,
    }

    # Signed component ranges retain direction; maximum-absolute summaries
    # alone could accept a candidate moving the opposite way.  These bounds
    # are observations, not user-tuned margins.
    signed_component_ranges = {}
    for name, length in vector_fields.items():
        signed_component_ranges[name] = {
            "minimum": [
                min(row[name][component] for row in rows)
                for component in range(length)
            ],
            "maximum": [
                max(row[name][component] for row in rows)
                for component in range(length)
            ],
        }

    return {
        "semantics": "moving_interrupt_tick_plus_five_frozen_ticks",
        "sample_count": len(rows),
        "samples": rows,
        "signed_component_ranges": signed_component_ranges,
        "maximum_abs_joint_velocity_rad_s": [
            max(abs(row["measured_arm_velocity_rad_s"][index]) for row in rows)
            for index in range(7)
        ],
        "maximum_joint_velocity_norm_rad_s": max(
            math.sqrt(sum(value * value for value in row[
                "measured_arm_velocity_rad_s"
            ]))
            for row in rows
        ),
        "maximum_abs_command_tracking_error_arm_rad": [
            max(abs(row["command_tracking_error_arm_rad"][index]) for row in rows)
            for index in range(7)
        ],
        "maximum_abs_command_fk_minus_measured_fk_task_m": [
            max(abs(row["command_fk_minus_measured_fk_task_m"][index])
                for row in rows)
            for index in range(3)
        ],
        "maximum_abs_command_fk_minus_measured_tcp_task_m": [
            max(abs(row["command_fk_minus_measured_tcp_task_m"][index])
                for row in rows)
            for index in range(3)
        ],
        "maximum_abs_measured_tcp_velocity_task_m_s": [
            max(abs(row["measured_tcp_velocity_task_m_s"][index]) for row in rows)
            for index in range(3)
        ],
        "maximum_measured_tcp_speed_m_s": max(
            row["measured_tcp_speed_m_s"] for row in rows
        ),
        "negative_task_z_drift_after_frozen_tick_m": negative_drift,
    }


def compare_tactile_reversal_state_equivalence(
    candidate_samples,
    reference_limits,
    task_rotation_world,
    rate_hz,
):
    """Fail closed unless Stage-A dynamics stay within preflight limits."""

    import math

    candidate = tactile_reversal_state_features(
        candidate_samples, task_rotation_world, rate_hz
    )
    if not isinstance(reference_limits, Mapping):
        raise ValueError("state-equivalence reference limits are missing")
    comparisons = {}
    # This tolerance only covers float serialization/representation.  It is
    # intentionally not a dynamic margin and is many orders below motion
    # limits used by either runtime.
    representation_tolerance = 1.0e-12

    reference_ranges = reference_limits.get("signed_component_ranges")
    if not isinstance(reference_ranges, Mapping):
        raise ValueError("state-equivalence signed ranges are missing")

    signed_fields = {
        "measured_arm_velocity_rad_s": 7,
        "command_tracking_error_arm_rad": 7,
        "command_fk_minus_measured_fk_task_m": 3,
        "command_fk_minus_measured_tcp_task_m": 3,
        "measured_tcp_velocity_task_m_s": 3,
    }
    for name, length in signed_fields.items():
        bounds = reference_ranges.get(name)
        if not isinstance(bounds, Mapping):
            raise ValueError(f"state-equivalence signed range {name} missing")
        minimum = bounds.get("minimum")
        maximum = bounds.get("maximum")
        if (
            not isinstance(minimum, (list, tuple))
            or not isinstance(maximum, (list, tuple))
            or len(minimum) != length
            or len(maximum) != length
        ):
            raise ValueError(f"state-equivalence signed range {name} changed")
        lower = tuple(float(value) for value in minimum)
        upper = tuple(float(value) for value in maximum)
        if not all(
            math.isfinite(low) and math.isfinite(high) and low <= high
            for low, high in zip(lower, upper)
        ):
            raise ValueError(f"state-equivalence signed range {name} invalid")
        sample_gates = []
        for row in candidate["samples"]:
            sample_gates.append(
                all(
                    lower[index] - representation_tolerance
                    <= row[name][index]
                    <= upper[index] + representation_tolerance
                    for index in range(length)
                )
            )
        comparisons[f"signed_range:{name}"] = {
            "minimum": list(lower),
            "maximum": list(upper),
            "sample_gates": sample_gates,
            "gate": all(sample_gates),
        }

    def gate_vector(name, length):
        actual = candidate[name]
        limit = reference_limits.get(name)
        if not isinstance(limit, (list, tuple)) or len(limit) != length:
            raise ValueError(f"state-equivalence limit {name} changed")
        limits = tuple(float(value) for value in limit)
        if not all(math.isfinite(value) and value >= 0.0 for value in limits):
            raise ValueError(f"state-equivalence limit {name} is invalid")
        gates = tuple(
            actual[index] <= limits[index] + representation_tolerance
            for index in range(length)
        )
        comparisons[name] = {
            "actual": list(actual),
            "limit": list(limits),
            "gate": all(gates),
        }
        return all(gates)

    gates = [
        comparisons[f"signed_range:{name}"]["gate"]
        for name in signed_fields
    ] + [
        gate_vector("maximum_abs_joint_velocity_rad_s", 7),
        gate_vector("maximum_abs_command_tracking_error_arm_rad", 7),
        gate_vector("maximum_abs_command_fk_minus_measured_fk_task_m", 3),
        gate_vector("maximum_abs_command_fk_minus_measured_tcp_task_m", 3),
        gate_vector("maximum_abs_measured_tcp_velocity_task_m_s", 3),
        gate_vector("negative_task_z_drift_after_frozen_tick_m", 5),
    ]
    for name in (
        "maximum_joint_velocity_norm_rad_s",
        "maximum_measured_tcp_speed_m_s",
    ):
        actual = float(candidate[name])
        limit = float(reference_limits.get(name))
        if not all(math.isfinite(value) and value >= 0.0 for value in (
            actual, limit
        )):
            raise ValueError(f"state-equivalence scalar {name} is invalid")
        gate = actual <= limit + representation_tolerance
        comparisons[name] = {"actual": actual, "limit": limit, "gate": gate}
        gates.append(gate)
    # Retraction must never be authorized when Stage A is descending faster
    # in task Z than the worst signed value observed in the matched preflight.
    task_z_lower_bound = float(
        reference_ranges["measured_tcp_velocity_task_m_s"]["minimum"][2]
    )
    task_z_samples = [
        row["measured_tcp_velocity_task_m_s"][2]
        for row in candidate["samples"]
    ]
    task_z_gate = all(
        value >= task_z_lower_bound - representation_tolerance
        for value in task_z_samples
    )
    comparisons["task_z_velocity_one_sided_lower_bound"] = {
        "actual_samples_m_s": task_z_samples,
        "minimum_allowed_m_s": task_z_lower_bound,
        "gate": task_z_gate,
    }
    gates.append(task_z_gate)
    return {
        "passed": all(gates),
        "compared_without_world_step": True,
        "candidate_features": candidate,
        "reference_limits": dict(reference_limits),
        "comparisons": comparisons,
        "representation_tolerance": representation_tolerance,
        "expected_differences_not_compared": [
            "absolute_arm_configuration_due_to_plus_x_offset",
            "absolute_world_tcp_due_to_plus_x_offset",
            "contact_manifold",
            "contact_wrench",
        ],
        "terminal_retract_authorized": all(gates),
    }


def compare_applied_arm_command_float32(
    applied_joint_positions,
    controlled_joint_indices,
    arm_joint_indices,
    expected_arm_command,
    robot_num_dof,
):
    """Compare controller readback to the prior command after float32 cast.

    Isaac's articulation drive stores the applied target as float32.  The
    planner retains float64 joints, so bit-for-bit float64 comparison would
    reject a correct target.  Equality after independently casting both sides
    to float32 is the strict representation-aware gate; no tolerance is used.
    """

    import numpy as np

    if type(robot_num_dof) is not int or robot_num_dof <= 0:
        raise ValueError("robot_num_dof must be a positive integer")
    expected = np.asarray(
        seven_float_arm_tuple(
            expected_arm_command, "expected_applied_arm_command"
        ),
        dtype=np.float64,
    )
    controlled = np.asarray(controlled_joint_indices)
    arm = np.asarray(arm_joint_indices)
    if controlled.ndim != 1 or arm.ndim != 1 or arm.shape != (7,):
        raise ValueError(
            "applied-action joint indices must be flat arm/control"
        )
    if (
        not np.issubdtype(controlled.dtype, np.integer)
        or not np.issubdtype(arm.dtype, np.integer)
    ):
        raise ValueError("applied-action joint indices must be integers")
    controlled = controlled.astype(np.int64, copy=False)
    arm = arm.astype(np.int64, copy=False)
    if (
        len(set(int(value) for value in controlled)) != len(controlled)
        or len(set(int(value) for value in arm)) != len(arm)
        or np.any(controlled < 0)
        or np.any(controlled >= robot_num_dof)
        or np.any(arm < 0)
        or np.any(arm >= robot_num_dof)
        or not set(int(value) for value in arm).issubset(
            set(int(value) for value in controlled)
        )
    ):
        raise ValueError("applied-action joint indices are invalid")
    applied = np.asarray(applied_joint_positions, dtype=np.float64)
    if applied.ndim != 1 or not np.all(np.isfinite(applied)):
        raise ValueError(
            "applied-action readback must be a finite flat vector"
        )
    if applied.shape == (robot_num_dof,):
        controlled_readback = applied[controlled]
        arm_readback = applied[arm]
        storage = "full_articulation"
    elif applied.shape == (len(controlled),):
        controlled_readback = applied
        controlled_lookup = {
            int(joint): index for index, joint in enumerate(controlled)
        }
        arm_readback = np.asarray(
            [
                controlled_readback[controlled_lookup[int(joint)]]
                for joint in arm
            ],
            dtype=np.float64,
        )
        storage = "controlled_subset"
    else:
        raise ValueError("applied-action readback shape changed")
    actual_float32 = arm_readback.astype(np.float32)
    expected_float32 = expected.astype(np.float32)
    if not np.all(np.isfinite(actual_float32)) or not np.all(
        np.isfinite(expected_float32)
    ):
        raise ValueError("applied-action float32 conversion is non-finite")
    error = actual_float32.astype(np.float64) - expected_float32.astype(
        np.float64
    )
    gate = bool(np.array_equal(actual_float32, expected_float32))
    return {
        "storage": storage,
        "controlled_position_readback": tuple(
            float(value) for value in controlled_readback
        ),
        "arm_position_readback_float32": tuple(
            float(value) for value in actual_float32
        ),
        "expected_arm_command_float32": tuple(
            float(value) for value in expected_float32
        ),
        "arm_error_float32_rad": tuple(float(value) for value in error),
        "maximum_abs_arm_error_float32_rad": float(np.max(np.abs(error))),
        "float32_equivalent_gate": gate,
    }


def build_command_continuous_retract_path(
    last_applied_command,
    task_z_axis_world,
    retract_distance_m,
    rate_hz,
    maximum_speed_m_s,
    fk_transform,
    solve_target,
    *,
    maximum_fk_position_error_m,
    maximum_fk_orientation_error_rad,
    q7_tolerance_rad=1.0e-12,
    strict_discrete_headroom=False,
):
    """Build and fully validate one position-drive-continuous +Z path.

    A loaded stiff position drive needs its previous *command* to preserve the
    bias that is supporting gravity/contact.  Substituting measured joints for
    that command removes the bias in one action and can move the TCP opposite
    the requested retreat.  Therefore the first returned action is exactly the
    last applied command; measured state is deliberately not an input.

    ``solve_target`` receives ``(start_q, target_position, start_rotation)``.
    Keeping the start command's orientation makes every interpolated FK check
    meaningful even when the interrupted approach was between IK waypoints.
    """

    import math

    start = seven_float_arm_tuple(
        last_applied_command, "last_applied_command"
    )
    axis = three_float_position_tuple(
        task_z_axis_world, "task_z_axis_world"
    )
    scalar_values = (
        retract_distance_m,
        rate_hz,
        maximum_speed_m_s,
        maximum_fk_position_error_m,
        maximum_fk_orientation_error_rad,
        q7_tolerance_rad,
    )
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise ValueError("retract path scalars must be finite")
    if type(strict_discrete_headroom) is not bool:
        raise ValueError("strict_discrete_headroom must be boolean")
    if (
        retract_distance_m <= 0.0
        or rate_hz <= 0.0
        or maximum_speed_m_s <= 0.0
        or maximum_fk_position_error_m <= 0.0
        or maximum_fk_orientation_error_rad <= 0.0
        or q7_tolerance_rad < 0.0
    ):
        raise ValueError("retract path bounds must be positive")
    axis_norm = math.sqrt(sum(value * value for value in axis))
    if abs(axis_norm - 1.0) > 1.0e-12:
        raise ValueError("task_z_axis_world must be unit length")

    def checked_transform(command, label):
        rows = tuple(tuple(float(value) for value in row) for row in (
            fk_transform(command)
        ))
        if len(rows) != 4 or any(len(row) != 4 for row in rows):
            raise ValueError(f"{label} FK must be a 4x4 transform")
        if not all(math.isfinite(value) for row in rows for value in row):
            raise ValueError(f"{label} FK must be finite")
        return rows

    def rotation_error(first, second):
        trace = sum(
            first[row][column] * second[row][column]
            for row in range(3)
            for column in range(3)
        )
        cosine = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
        return math.acos(cosine)

    start_transform = checked_transform(start, "start")
    start_position = tuple(start_transform[index][3] for index in range(3))
    start_rotation = tuple(
        tuple(start_transform[row][column] for column in range(3))
        for row in range(3)
    )
    target_position = tuple(
        start_position[index] + retract_distance_m * axis[index]
        for index in range(3)
    )
    target = seven_float_arm_tuple(
        solve_target(start, target_position, start_rotation),
        "retract_target",
    )
    base_steps = minimum_jerk_steps_for_peak_speed(
        retract_distance_m, rate_hz, maximum_speed_m_s
    )
    steps = base_steps + int(strict_discrete_headroom)
    commands = [start]
    for index in range(1, steps + 1):
        fraction = float(index) / float(steps)
        blend = fraction * fraction * fraction * (
            10.0 + fraction * (-15.0 + 6.0 * fraction)
        )
        commands.append(
            tuple(
                initial + blend * (final - initial)
                for initial, final in zip(start, target)
            )
        )

    command_positions = []
    axial_progress = []
    lateral_error = []
    orientation_error = []
    previous_axial = None
    for index, command in enumerate(commands):
        if abs(command[6] - start[6]) > q7_tolerance_rad:
            raise ValueError("retract path changed fixed q7")
        transform = checked_transform(command, f"command[{index}]")
        position = tuple(transform[axis_index][3] for axis_index in range(3))
        delta = tuple(
            position[axis_index] - start_position[axis_index]
            for axis_index in range(3)
        )
        axial = sum(delta[i] * axis[i] for i in range(3))
        lateral = math.sqrt(
            sum((delta[i] - axial * axis[i]) ** 2 for i in range(3))
        )
        orientation = rotation_error(start_rotation, transform)
        if previous_axial is not None and axial < previous_axial - 1.0e-12:
            raise ValueError("retract command FK is not monotonic +task-Z")
        if lateral > maximum_fk_position_error_m:
            raise ValueError("retract command FK left lateral bound")
        if orientation > maximum_fk_orientation_error_rad:
            raise ValueError("retract command FK left orientation bound")
        command_positions.append(position)
        axial_progress.append(axial)
        lateral_error.append(lateral)
        orientation_error.append(orientation)
        previous_axial = axial

    target_error = math.sqrt(
        sum(
            (command_positions[-1][index] - target_position[index]) ** 2
            for index in range(3)
        )
    )
    if target_error > maximum_fk_position_error_m:
        raise ValueError("retract target FK position error exceeded")
    peak_speed = max(
        abs(axial_progress[index] - axial_progress[index - 1]) * rate_hz
        for index in range(1, len(axial_progress))
    )
    if peak_speed > maximum_speed_m_s + 1.0e-12:
        raise ValueError("retract command FK speed exceeded")
    if abs(axial_progress[-1] - retract_distance_m) > (
        maximum_fk_position_error_m
    ):
        raise ValueError("retract command FK distance error exceeded")
    return {
        "commands": tuple(commands),
        "start_arm_rad": start,
        "target_arm_rad": target,
        "start_fk_position_m": start_position,
        "target_fk_position_m": command_positions[-1],
        "requested_target_position_m": target_position,
        "command_fk_axial_progress_m": tuple(axial_progress),
        "command_fk_lateral_error_m": tuple(lateral_error),
        "command_fk_orientation_error_rad": tuple(orientation_error),
        "peak_command_fk_axial_speed_m_s": peak_speed,
        "first_command_exact": commands[0] == start,
        "steps_excluding_exact_start_hold": steps,
        "base_required_steps": base_steps,
        "added_discrete_headroom_steps": int(strict_discrete_headroom),
    }


def _exact_mapping(value, expected_keys, label):
    """Return one mapping only when its runtime schema is exact."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    actual_keys = set(value)
    expected_keys = set(expected_keys)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{label} keys differ; "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    return value


def _zero_integer(value, label):
    """Reject booleans and nonzero/loosely typed side-effect counters."""

    if type(value) is not int or value != 0:
        raise ValueError(f"{label} must be integer zero")
    return value


def validate_matched_reversal_preflight_evidence(
    section,
    *,
    task_rotation_world,
    sample_rate_hz,
    minimum_gap_m,
    maximum_measured_speed_m_s,
    maximum_approach_command_speed_m_s,
    maximum_other_command_speed_m_s,
    experimental_abort_ceilings,
    fk_transform,
):
    """Recompute the no-contact mid-slope reversal authorization.

    This validator intentionally consumes raw per-tick evidence instead of
    trusting PASS summaries.  It binds the discrete argmax plan, exact moving
    tick plus five frozen ticks, action readback, no-contact scope, three-axis
    speed/gap/envelope gates, and the immutable negative-progress ceiling.
    """

    import json
    import math
    import struct

    if not isinstance(section, Mapping):
        raise ValueError("matched reversal evidence is missing")
    if section.get("passed") is not True or section.get("status") != (
        "PASSED_MATCHED_MID_SLOPE_REVERSAL"
    ):
        raise ValueError("matched reversal did not pass")
    fixed = TACTILE_FIXED_NEGATIVE_RETRACT_BOUND_M
    for name in (
        "fixed_negative_progress_ceiling_m",
        "effective_negative_progress_ceiling_m",
    ):
        if not math.isclose(
            float(section.get(name)), fixed, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("matched reversal fixed negative bound changed")

    plan = section.get("discrete_interrupt_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("matched reversal discrete plan is missing")
    strict_plan = plan.get("strict_headroom_plan")
    recomputed_plan = build_discrete_mid_slope_interrupt_plan(
        plan.get("start_arm_rad"),
        plan.get("target_arm_rad"),
        plan.get("command_count"),
        plan.get("task_z_axis_world"),
        plan.get("sample_rate_hz"),
        fk_transform,
        strict_headroom_plan=strict_plan,
    )
    plan_fields = (
        "command_count",
        "unique_argmax_motion_index",
        "unique_argmax_count",
        "interior_argmax_gate",
        "maximum_downward_step_m",
        "maximum_downward_speed_m_s",
        "planned_argmax_neighborhood",
        "full_step_down_sha256",
        "strict_headroom_plan",
    )
    if any(
        json.dumps(plan.get(name), sort_keys=True, allow_nan=False)
        != json.dumps(
            recomputed_plan.get(name), sort_keys=True, allow_nan=False
        )
        for name in plan_fields
    ):
        raise ValueError("matched reversal discrete plan differs from FK")
    if recomputed_plan["maximum_downward_speed_m_s"] > float(
        maximum_approach_command_speed_m_s
    ):
        raise ValueError("matched reversal approach command speed changed")

    rate = float(sample_rate_hz)
    gap_floor = float(minimum_gap_m)
    measured_speed_ceiling = float(maximum_measured_speed_m_s)
    rotation = tuple(
        tuple(float(value) for value in row)
        for row in task_rotation_world
    )
    if (
        rate <= 0.0
        or gap_floor < 0.0
        or measured_speed_ceiling <= 0.0
        or len(rotation) != 3
        or any(len(row) != 3 for row in rotation)
    ):
        raise ValueError("matched reversal validation inputs are invalid")

    expected_phases = (
        (
            "moving_interrupt_evidence",
            float(maximum_approach_command_speed_m_s),
        ),
        (
            "frozen_hold_evidence",
            float(maximum_approach_command_speed_m_s),
        ),
        (
            "terminal_retract_evidence",
            float(maximum_other_command_speed_m_s),
        ),
        (
            "recovery_evidence",
            float(maximum_other_command_speed_m_s),
        ),
    )
    ceilings = {
        name: float(experimental_abort_ceilings[name])
        for name in TACTILE_PREFLIGHT_PEAK_FIELDS
    }

    def float32(value):
        return struct.unpack("!f", struct.pack("!f", float(value)))[0]

    raw_phases = []
    prior_global_step = None
    prior_command_float32 = None
    for phase_name, command_ceiling in expected_phases:
        evidence = section.get(phase_name)
        if not isinstance(evidence, Mapping):
            raise ValueError(f"matched reversal {phase_name} is missing")
        raw = evidence.get("guarded_tick_samples")
        if (
            not isinstance(raw, list)
            or not raw
            or evidence.get("checked_sample_count") != len(raw)
            or evidence.get("finite_sample_count") != len(raw)
            or evidence.get("all_samples_finite") is not True
            or evidence.get("minimum_body_contact_finger_count") != 3
            or evidence.get("applied_action_precheck_count") != len(raw)
            or evidence.get("applied_action_postcheck_count") != len(raw)
            or evidence.get("applied_action_all_float32_equivalent") is not True
            or evidence.get(
                "applied_action_maximum_abs_arm_error_float32_rad"
            )
            != 0.0
            or any(evidence.get("contact_record_totals", {}).values())
        ):
            raise ValueError(f"matched reversal {phase_name} summary changed")
        measured_start = tuple(
            float(value)
            for value in evidence["measured_start_tcp_prim_world_m"]
        )
        command_start = tuple(
            float(value)
            for value in evidence["command_start_fk_tcp_world_m"]
        )
        previous_measured = measured_start
        previous_command = command_start
        derived_measured_peak = 0.0
        derived_command_peak = 0.0
        for index, tick in enumerate(raw):
            global_step = tick.get("global_step")
            if (
                type(global_step) is not int
                or (index and global_step != raw[index - 1]["global_step"] + 1)
                or (
                    index == 0
                    and prior_global_step is not None
                    and global_step != prior_global_step + 1
                )
            ):
                raise ValueError("matched reversal global steps are detached")
            command = tuple(float(value) for value in tick["command_arm_rad"])
            expected_float32 = tuple(float32(value) for value in command)
            pre_expected = tuple(tick["applied_pre_expected_float32"])
            pre_readback = tuple(tick["applied_pre_readback_float32"])
            post_expected = tuple(tick["applied_post_expected_float32"])
            post_readback = tuple(tick["applied_post_readback_float32"])
            if (
                len(command) != 7
                or tick.get("finite") is not True
                or tick.get("body_contact_fingers") != ["finger1", "finger2", "finger3"]
                or tick.get("applied_pre_float32_gate") is not True
                or tick.get("applied_post_float32_gate") is not True
                or tick.get("applied_pre_max_error_float32_rad") != 0.0
                or tick.get("applied_post_max_error_float32_rad") != 0.0
                or pre_expected != pre_readback
                or post_expected != post_readback
                or post_expected != expected_float32
                or (
                    prior_command_float32 is not None
                    and index == 0
                    and pre_expected != prior_command_float32
                )
                or (index and pre_expected != tuple(
                    raw[index - 1]["applied_post_expected_float32"]
                ))
                or tick.get("loose_fixed_contact_records") != 0
                or tick.get("intended_lip_contact_pairs") != []
                or tick.get("unexpected_loose_fixed_contact_pairs") != []
                or tick.get("loose_fixture_contact_pairs") != []
                or tick.get("loose_table_contact_pairs") != []
            ):
                raise ValueError("matched reversal raw guard changed")
            measured = tuple(
                float(value)
                for value in tick["measured_tcp_prim_world_m"]
            )
            command_fk = tuple(
                float(value) for value in tick["command_fk_tcp_world_m"]
            )
            if (
                len(measured) != 3
                or len(command_fk) != 3
                or not all(math.isfinite(value) for value in (*measured, *command_fk))
                or float(tick["estimated_gap_m"]) < gap_floor
            ):
                raise ValueError("matched reversal raw pose/gap changed")
            measured_speed = math.sqrt(sum(
                (measured[axis] - previous_measured[axis]) ** 2
                for axis in range(3)
            )) * rate
            command_speed = math.sqrt(sum(
                (command_fk[axis] - previous_command[axis]) ** 2
                for axis in range(3)
            )) * rate
            derived_measured_peak = max(derived_measured_peak, measured_speed)
            derived_command_peak = max(derived_command_peak, command_speed)
            if (
                measured_speed > measured_speed_ceiling
                or command_speed > command_ceiling
            ):
                raise ValueError("matched reversal raw speed exceeded ceiling")
            wrench = tuple(float(value) for value in tick[
                "compensated_wrench_task"
            ])
            fingers = tuple(float(value) for value in tick[
                "finger_base_torque_delta_nm"
            ])
            raw_peaks = {
                "absolute_axial_force_n": abs(wrench[2]),
                "lateral_force_n": math.hypot(wrench[0], wrench[1]),
                "bending_torque_nm": math.hypot(wrench[3], wrench[4]),
                "absolute_tightening_torque_nm": abs(wrench[5]),
                "absolute_finger_base_torque_nm": max(abs(value) for value in fingers),
            }
            if (
                len(wrench) != 6
                or len(fingers) != 3
                or not all(math.isfinite(value) for value in (*wrench, *fingers))
                or any(raw_peaks[name] > ceilings[name] for name in ceilings)
            ):
                raise ValueError("matched reversal raw safety envelope changed")
            previous_measured = measured
            previous_command = command_fk
            prior_global_step = global_step
            prior_command_float32 = post_expected
        if (
            not math.isclose(
                derived_measured_peak,
                float(evidence["peak_abs_tcp_speed_m_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                derived_command_peak,
                float(evidence["peak_abs_command_fk_tcp_speed_m_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("matched reversal reported speed differs")
        raw_phases.append(raw)

    moving, frozen, retract, _recovery = raw_phases
    window = moving[-1:] + frozen
    reported_window = section.get("equivalence_window_raw")
    if (
        len(window) != TACTILE_MATCHED_REVERSAL_WINDOW_SAMPLES
        or json.dumps(window, sort_keys=True, allow_nan=False)
        != json.dumps(reported_window, sort_keys=True, allow_nan=False)
        or any(
            sample["command_arm_rad"] != window[0]["command_arm_rad"]
            for sample in window
        )
        or section.get("equivalence_window_semantics")
        != "moving_argmax_tick_plus_five_frozen_ticks"
    ):
        raise ValueError("matched reversal exact-six window changed")
    recomputed_features = tactile_reversal_state_features(
        window, rotation, rate
    )
    if json.dumps(
        recomputed_features, sort_keys=True, allow_nan=False
    ) != json.dumps(
        section.get("state_equivalence_limits"),
        sort_keys=True,
        allow_nan=False,
    ):
        raise ValueError("matched reversal state limits differ from raw")
    retract_start = tuple(float(value) for value in retract[0][
        "pre_measured_tcp_prim_world_m"
    ])
    task_z = tuple(rotation[index][2] for index in range(3))
    retract_progress = [
        sum(
            (
                float(tick["measured_tcp_prim_world_m"][axis])
                - retract_start[axis]
            )
            * task_z[axis]
            for axis in range(3)
        )
        for tick in retract
    ]
    observed = max(0.0, -min(retract_progress))
    if (
        observed > fixed
        or not math.isclose(
            observed,
            float(section.get("observed_negative_reversal_progress_m")),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or section.get("safety_gate") is not True
    ):
        raise ValueError("matched reversal negative progress changed")
    terminal = section.get("terminal_retract")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("attempted") is not True
        or terminal.get("started") is not True
        or terminal.get("completed") is not True
        or terminal.get("hard_stop") is not False
        or terminal.get("zero_step_abort") is not False
        or terminal.get("resume_attempted") is not False
        or terminal.get("terminal_state") != "RECOVERED_PREINSERT"
        or terminal.get("world_steps_after_original_failure") != 0
    ):
        raise ValueError("matched reversal terminal evidence changed")
    return {
        "fixed_negative_progress_bound_m": fixed,
        "observed_negative_progress_bound_m": observed,
        "effective_negative_progress_bound_m": fixed,
        "state_equivalence_limits": recomputed_features,
        "discrete_interrupt_plan": recomputed_plan,
    }


def _nonnegative_integer(value, label):
    """Return one strict nonnegative evidence counter (never a bool)."""

    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def seven_float_arm_tuple(values, label="arm_rad"):
    """Adapt a flat NumPy work vector to the strict seven-joint FK API."""

    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a flat numeric sequence") from error
    if len(result) != 7:
        raise ValueError(f"{label} must contain exactly 7 numbers")
    return result


def three_float_position_tuple(values, label="position_m"):
    """Adapt a flat NumPy work vector to the strict three-value IK API."""

    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a flat numeric sequence") from error
    if len(result) != 3:
        raise ValueError(f"{label} must contain exactly 3 numbers")
    return result


def validate_rgbd_capture_side_effects(capture_metrics):
    """Fail closed on capture lifecycle effects without consulting truth.

    Deliberately excluded inputs include the aggregate capture ``passed`` bit,
    per-endpoint truth/error gates, and camera projection truth diagnostics.
    Only the actual RGB-D runtime lifecycle schema is accepted here.
    """

    metrics = capture_metrics
    if not isinstance(metrics, Mapping):
        raise ValueError("capture_metrics must be a mapping")
    reset_or_clear = _zero_integer(
        metrics.get("world_reset_or_clear_calls"),
        "capture_metrics.world_reset_or_clear_calls",
    )
    pose_writes = _zero_integer(
        metrics.get("object_pose_writes_after_start"),
        "capture_metrics.object_pose_writes_after_start",
    )
    cleanup = _exact_mapping(
        metrics.get("resource_cleanup"),
        {
            "annotator_detach_count",
            "camera_destroyed",
            "errors",
            "render_product_destroyed",
            "resources_released",
            "scene_cleared",
            "stage_prims_removed",
            "world_reset",
        },
        "capture_metrics.resource_cleanup",
    )
    if type(cleanup["annotator_detach_count"]) is not int:
        raise ValueError("annotator_detach_count must be an integer")
    if cleanup["annotator_detach_count"] != 3:
        raise ValueError("all three RGB-D annotators must detach")
    if cleanup["camera_destroyed"] is not True:
        raise ValueError("RGB-D Camera wrapper was not destroyed")
    if cleanup["render_product_destroyed"] is not True:
        raise ValueError("RGB-D render product was not destroyed")
    if cleanup["resources_released"] is not True:
        raise ValueError("RGB-D runtime resources were not released")
    if cleanup["errors"] != []:
        raise ValueError("RGB-D runtime cleanup reported errors")
    if cleanup["scene_cleared"] is not False:
        raise ValueError("RGB-D capture cleared the caller scene")
    _zero_integer(cleanup["stage_prims_removed"], "stage_prims_removed")
    if cleanup["world_reset"] is not False:
        raise ValueError("RGB-D capture reset the caller World")

    timeline = _exact_mapping(
        metrics.get("timeline_state"),
        {
            "playing_after_cleanup",
            "playing_after_restore",
            "playing_before_capture",
            "restore_attempted",
            "restored",
        },
        "capture_metrics.timeline_state",
    )
    for name in timeline:
        if type(timeline[name]) is not bool:
            raise ValueError(f"timeline_state.{name} must be boolean")
    before = timeline["playing_before_capture"]
    after_cleanup = timeline["playing_after_cleanup"]
    after_restore = timeline["playing_after_restore"]
    if timeline["restore_attempted"] is not (after_cleanup is not before):
        raise ValueError("timeline restore-attempt semantics are inconsistent")
    if timeline["restored"] is not True or after_restore is not before:
        raise ValueError("RGB-D capture did not restore the caller timeline")
    return {
        # The current runtime exposes one combined reset/clear counter.  Keep
        # its exact name instead of pretending it distinguishes the two calls.
        "world_reset_or_clear_calls": reset_or_clear,
        "endpoint_pose_writes_after_physics": pose_writes,
        "resource_cleanup_verified": True,
        "timeline_state_restored": True,
        "playing_before_capture": before,
        "playing_after_restore": after_restore,
        "truth_or_error_gate_consulted": False,
    }


def create_exclusive_output_directory(path):
    """Atomically reserve a new run directory; never reuse old evidence."""

    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    return output


def contact_pair_crosses_prim_roots(paths, first_root, second_root):
    """Return whether one reported pair crosses two disjoint prim subtrees."""

    def below(path, root):
        return path == root or path.startswith(root + "/")

    first_seen = any(below(path, first_root) for path in paths)
    second_seen = any(below(path, second_root) for path in paths)
    return first_seen and second_seen


def classify_tactile_lip_contact_pair(paths, body_root, nut_root, fixed_root):
    """Classify one loose/fixed contact for the segmented lip experiment.

    Only a BodyAssembly/MatingShell segment against a
    FixedReceptacle/EntryShell segment is the intended proxy lip contact.
    CouplingNut contact and every other loose/fixed pairing fail closed.
    """

    paths = tuple(str(path) for path in paths)

    def below(path, root):
        return path == root or path.startswith(root + "/")

    if not (
        any(below(path, body_root) or below(path, nut_root) for path in paths)
        and any(below(path, fixed_root) for path in paths)
    ):
        return None
    intended_loose_prefix = body_root + "/MatingShell/Segment_"
    intended_fixed_prefix = fixed_root + "/EntryShell/Segment_"
    if (
        any(path.startswith(intended_loose_prefix) for path in paths)
        and any(path.startswith(intended_fixed_prefix) for path in paths)
        and not any(below(path, nut_root) for path in paths)
    ):
        return "intended_segmented_lip"
    return "unexpected_loose_fixed"


def build_tactile_manifold_pair_evidence(
    *,
    actor_paths,
    collider_paths,
    event_type,
    contact_points,
    body_root,
    fixed_root,
    task_rotation_world,
    task_origin_world,
    physics_dt_s,
):
    """Copy and validate one intended PhysX lip-contact report slice.

    PhysX reports the contact normal in world coordinates in the direction
    shape/collider 0 must move to resolve shape/collider 1.  The raw actor and
    collider ordering is therefore evidence, not presentation detail.  We
    preserve it byte-for-value and derive ``loose_resolution_normal`` only
    after identifying which collider side belongs to the loose mating shell.

    Impulse divided by ``dt`` is diagnostic only.  A zero impulse is valid for
    a contact-offset manifold and does not fail this Stage-A capture.
    """

    import math

    actors = tuple(str(value) for value in actor_paths)
    colliders = tuple(str(value) for value in collider_paths)
    if len(actors) != 2 or len(colliders) != 2:
        raise ValueError("contact actor/collider ordering must have two sides")
    loose_prefix = str(body_root) + "/MatingShell/Segment_"
    fixed_prefix = str(fixed_root) + "/EntryShell/Segment_"
    loose_sides = tuple(
        index
        for index, path in enumerate(colliders)
        if path.startswith(loose_prefix)
    )
    fixed_sides = tuple(
        index
        for index, path in enumerate(colliders)
        if path.startswith(fixed_prefix)
    )
    if (
        len(loose_sides) != 1
        or len(fixed_sides) != 1
        or loose_sides[0] == fixed_sides[0]
    ):
        raise ValueError("manifold pair is not one exact segmented lip pair")

    def at_or_below(path, root):
        return path == root or path.startswith(root + "/")

    if (
        not at_or_below(actors[loose_sides[0]], str(body_root))
        or not at_or_below(actors[fixed_sides[0]], str(fixed_root))
    ):
        raise ValueError(
            "manifold actor ordering does not match collider ordering"
        )

    try:
        dt = float(physics_dt_s)
        rotation = tuple(
            tuple(float(value) for value in row)
            for row in task_rotation_world
        )
        origin = tuple(float(value) for value in task_origin_world)
    except (TypeError, ValueError) as error:
        raise ValueError("manifold task transform must be numeric") from error
    if (
        not math.isfinite(dt)
        or dt <= 0.0
        or len(rotation) != 3
        or any(len(row) != 3 for row in rotation)
        or len(origin) != 3
        or not all(math.isfinite(value) for row in rotation for value in row)
        or not all(math.isfinite(value) for value in origin)
    ):
        raise ValueError("manifold task transform/dt is invalid")
    columns = tuple(
        tuple(rotation[row][column] for row in range(3))
        for column in range(3)
    )
    if any(
        abs(
            sum(columns[first][i] * columns[second][i] for i in range(3))
            - (1.0 if first == second else 0.0)
        )
        > 1.0e-9
        for first in range(3)
        for second in range(3)
    ):
        raise ValueError("manifold task rotation is not orthonormal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1.0e-9:
        raise ValueError("manifold task rotation must be right handed")

    def world_to_task(vector):
        return tuple(
            sum(columns[index][axis] * vector[axis] for axis in range(3))
            for index in range(3)
        )

    copied_points = []
    for point_index, point in enumerate(tuple(contact_points)):
        if not isinstance(point, Mapping):
            raise ValueError("manifold contact point must be a mapping")
        try:
            normal = tuple(float(value) for value in point["normal"])
            impulse = tuple(float(value) for value in point["impulse"])
            position = tuple(float(value) for value in point["position"])
            separation = float(point["separation"])
            face_indices = (
                int(point["face_index0"]),
                int(point["face_index1"]),
            )
            materials = (
                str(point["material0"]),
                str(point["material1"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "manifold contact point schema is invalid"
            ) from error
        scalars = (*normal, *impulse, *position, separation)
        if (
            len(normal) != 3
            or len(impulse) != 3
            or len(position) != 3
            or not all(math.isfinite(value) for value in scalars)
        ):
            raise ValueError("manifold contact point must be finite 3D data")
        normal_norm = math.sqrt(sum(value * value for value in normal))
        if abs(normal_norm - 1.0) > 1.0e-5:
            raise ValueError("manifold reported contact normal is not unit")
        normal_task = world_to_task(normal)
        impulse_task = world_to_task(impulse)
        loose_sign = 1.0 if loose_sides[0] == 0 else -1.0
        loose_normal_world = tuple(loose_sign * value for value in normal)
        loose_normal_task = world_to_task(loose_normal_world)
        copied_points.append(
            {
                "point_index": int(point_index),
                "normal_world_reported": list(normal),
                "normal_task_reported": list(normal_task),
                "normal_norm": normal_norm,
                "loose_resolution_normal_world": list(loose_normal_world),
                "loose_resolution_normal_task": list(loose_normal_task),
                "impulse_world_reported_n_s": list(impulse),
                "impulse_task_reported_n_s": list(impulse_task),
                "impulse_norm_n_s": math.sqrt(
                    sum(value * value for value in impulse)
                ),
                "impulse_over_dt_task_diagnostic_n": [
                    value / dt for value in impulse_task
                ],
                "position_world_m": list(position),
                # Positions are projected only after subtracting the frozen
                # measured preinsert TCP origin; treating a world point as a
                # free vector would produce a meaningless task coordinate.
                "position_task_from_preinsert_origin_m": list(
                    world_to_task(
                        tuple(
                            position[index] - origin[index]
                            for index in range(3)
                        )
                    )
                ),
                "separation_m": separation,
                "face_index0": face_indices[0],
                "face_index1": face_indices[1],
                "material0": materials[0],
                "material1": materials[1],
            }
        )
    if not copied_points:
        raise ValueError("manifold contact pair has no point data")
    return {
        "actor0": actors[0],
        "actor1": actors[1],
        "collider0": colliders[0],
        "collider1": colliders[1],
        "event_type": str(event_type),
        "loose_side": int(loose_sides[0]),
        "fixed_side": int(fixed_sides[0]),
        "normal_convention": TACTILE_LIP_MANIFOLD_NORMAL_CONVENTION,
        "task_origin_world_m": list(origin),
        "physics_dt_s": dt,
        "contact_point_count": len(copied_points),
        "contact_points": copied_points,
    }


def validate_tactile_manifold_capture_evidence(
    value,
    *,
    body_root,
    fixed_root,
    task_rotation_world,
    task_origin_world,
    physics_dt_s,
    sample_rate_hz,
    expected_frame_count,
    expected_offset_task_xy_m,
    maximum_command_speed_m_s,
    maximum_measured_speed_m_s,
    minimum_gap_m,
    expected_preinsert_gap_m,
    required_retract_distance_m,
    maximum_negative_retract_progress_m,
    expected_release_compression_ceiling_n,
    expected_release_bending_ceiling_nm,
    compressive_axial_force_sign_candidate,
    expected_experimental_abort_ceilings,
    expected_body_contact_finger_count=3,
):
    """Re-derive every Stage-A PASS gate from serialized raw evidence.

    The online guards remain authoritative for stopping motion.  This pure
    validator is an independent report-integrity boundary: a persisted PASS
    cannot depend only on mutable summary booleans or precomputed peaks.  It
    reconstructs contact-point transforms from the raw PhysX ordering and
    recomputes all three phase trajectories from their TCP sample arrays.
    """

    import math
    import struct

    if not isinstance(value, Mapping):
        raise ValueError("manifold capture evidence must be a mapping")

    def number(item, label):
        if isinstance(item, bool):
            raise ValueError(f"{label} must be numeric, not boolean")
        try:
            result = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be numeric") from error
        if not math.isfinite(result):
            raise ValueError(f"{label} must be finite")
        return result

    def vector(item, length, label):
        if not isinstance(item, list) or len(item) != length:
            raise ValueError(f"{label} must be a {length}-value list")
        return tuple(
            number(component, f"{label}[{index}]")
            for index, component in enumerate(item)
        )

    def close(first, second, label, tolerance=1.0e-12):
        if not math.isclose(
            number(first, label),
            number(second, f"expected {label}"),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(f"{label} differs from raw evidence")

    rotation = tuple(
        tuple(number(component, "task rotation") for component in row)
        for row in task_rotation_world
    )
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError("manifold task rotation must be 3x3")
    origin = tuple(
        number(component, "task origin") for component in task_origin_world
    )
    if len(origin) != 3:
        raise ValueError("manifold task origin must have three values")
    rate = number(sample_rate_hz, "sample rate")
    dt = number(physics_dt_s, "physics dt")
    command_speed_limit = number(
        maximum_command_speed_m_s, "command speed limit"
    )
    measured_speed_limit = number(
        maximum_measured_speed_m_s, "measured speed limit"
    )
    gap_floor = number(minimum_gap_m, "entry gap floor")
    preinsert_gap = number(
        expected_preinsert_gap_m, "registered preinsert gap"
    )
    retract_distance = number(
        required_retract_distance_m, "required retract distance"
    )
    negative_retract_bound = number(
        maximum_negative_retract_progress_m,
        "maximum negative retract progress",
    )
    compression_sign = number(
        compressive_axial_force_sign_candidate,
        "manifold compression sign",
    )
    if (
        rate <= 0.0
        or dt <= 0.0
        or not math.isclose(dt * rate, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        or command_speed_limit <= 0.0
        or measured_speed_limit <= 0.0
        or command_speed_limit > measured_speed_limit
        or gap_floor < 0.0
        or preinsert_gap <= gap_floor
        or retract_distance <= 0.0
        or negative_retract_bound < 0.0
        or compression_sign not in (-1.0, 1.0)
        or type(expected_frame_count) is not int
        or expected_frame_count <= 0
        or type(expected_body_contact_finger_count) is not int
        or expected_body_contact_finger_count != 3
    ):
        raise ValueError("manifold capture scalar contract is invalid")

    expected_axes = {
        "x": [rotation[index][0] for index in range(3)],
        "y": [rotation[index][1] for index in range(3)],
        "z": [rotation[index][2] for index in range(3)],
    }
    reported_axes = _exact_mapping(
        value.get("task_frame_axes_world"),
        ("x", "y", "z", "determinant"),
        "manifold task axes",
    )
    for name in ("x", "y", "z"):
        if vector(reported_axes[name], 3, f"task axis {name}") != tuple(
            expected_axes[name]
        ):
            raise ValueError("manifold task axes differ from runtime axes")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    close(reported_axes["determinant"], determinant, "task determinant")
    if abs(determinant - 1.0) > 1.0e-9:
        raise ValueError("manifold task axes are not right handed")
    if vector(
        value.get("task_frame_origin_world_m"),
        3,
        "manifold task origin",
    ) != origin:
        raise ValueError("manifold task origin differs from runtime origin")

    required_literals = {
        "manifold_capture_only": True,
        "ft_sign_calibrated": False,
        "stage_b_authorized": False,
        "direction": "plus_x",
        "truth_pose_used_for_touch_control": False,
        "engage_executed": False,
        "insertion_executed": False,
        "twist_executed": False,
        "home_return_executed": False,
        "production_control_authorized": False,
        "hardware_safety_calibration_claimed": False,
        "assembly_success_claimed": False,
    }
    for name, expected in required_literals.items():
        actual = value.get(name)
        if actual != expected or (
            isinstance(expected, bool) and actual is not expected
        ):
            raise ValueError(f"manifold capture {name} is outside Stage A")
    if vector(
        value.get("known_offset_task_xy_m"), 2, "manifold task offset"
    ) != tuple(float(item) for item in expected_offset_task_xy_m):
        raise ValueError("manifold capture offset is not registered +X")
    close(
        value.get("maximum_commanded_tcp_speed_m_s"),
        command_speed_limit,
        "reported command speed limit",
        tolerance=1.0e-15,
    )
    close(
        value.get("maximum_measured_tcp_speed_m_s"),
        measured_speed_limit,
        "reported measured speed limit",
        tolerance=1.0e-15,
    )
    close(
        value.get("entry_gap_floor_m"),
        gap_floor,
        "reported entry gap floor",
        tolerance=1.0e-15,
    )
    if value.get("expected_manifold_frame_count") != expected_frame_count:
        raise ValueError("manifold expected frame count changed")

    runtime_identity = _exact_mapping(
        value.get("contact_runtime_identity"),
        (
            "contact_query_callable_module",
            "contact_query_callable_name",
            "contact_query_owner_type_module",
            "contact_query_owner_type_name",
            "normal_convention",
            "simulation_app_type_module",
            "simulation_app_type_name",
        ),
        "manifold contact runtime identity",
    )
    if (
        any(
            not isinstance(runtime_identity[name], str)
            or not runtime_identity[name]
            for name in runtime_identity
        )
        or runtime_identity["normal_convention"]
        != TACTILE_LIP_MANIFOLD_NORMAL_CONVENTION
    ):
        raise ValueError("manifold contact runtime identity is incomplete")

    frames = value.get("manifold_frames")
    if not isinstance(frames, list) or len(frames) != expected_frame_count:
        raise ValueError("manifold capture does not contain exact frame count")
    first_command = None
    first_command_fk = None
    first_global_step = None
    frame_contact_record_count = 0
    contact_point_count = 0
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError("manifold frame must be a mapping")
        if frame.get("frame_index") != frame_index:
            raise ValueError("manifold frame index is not consecutive")
        global_step = frame.get("global_step")
        if type(global_step) is not int:
            raise ValueError("manifold global step must be an integer")
        if first_global_step is None:
            first_global_step = global_step
        elif global_step != first_global_step + frame_index:
            raise ValueError("manifold frames are not consecutive ticks")
        command = vector(
            frame.get("command_arm_rad"), 7, "manifold frozen command"
        )
        command_fk = vector(
            frame.get("command_fk_tcp_world_m"),
            3,
            "manifold frozen command FK",
        )
        vector(frame.get("measured_arm_rad"), 7, "manifold measured arm")
        applied_readback = vector(
            frame.get("applied_action_arm_readback_float32"),
            7,
            "manifold applied readback",
        )
        expected_readback = vector(
            frame.get("expected_command_float32"),
            7,
            "manifold expected float32 command",
        )
        applied_error = vector(
            frame.get("applied_action_arm_error_float32_rad"),
            7,
            "manifold applied float32 error",
        )
        command_float32 = tuple(
            float(struct.unpack("!f", struct.pack("!f", value))[0])
            for value in command
        )
        if (
            frame.get("applied_action_float32_equivalent_gate") is not True
            or applied_readback != expected_readback
            or expected_readback != command_float32
            or any(error != 0.0 for error in applied_error)
        ):
            raise ValueError("manifold frozen drive readback does not match")
        vector(
            frame.get("measured_tcp_prim_world_m"),
            3,
            "manifold measured TCP",
        )
        for wrench_name in (
            "raw_wrench",
            "canonical_wrench_sensor",
            "compensated_wrench_sensor",
            "compensated_wrench_task",
        ):
            vector(frame.get(wrench_name), 6, f"manifold {wrench_name}")
        number(frame.get("estimated_gap_m"), "manifold estimated gap")
        if first_command is None:
            first_command = command
            first_command_fk = command_fk
        elif command != first_command or command_fk != first_command_fk:
            raise ValueError("manifold drive target changed after contact")
        if (
            frame.get("unexpected_loose_fixed_contact_pairs") != []
            or frame.get("loose_fixture_contact_pairs") != []
            or frame.get("loose_table_contact_pairs") != []
        ):
            raise ValueError("manifold frame contains a forbidden contact")
        pairs = frame.get("intended_lip_contact_pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("manifold frame lost intended lip contact")
        positive_frame_records = 0
        for pair in pairs:
            if not isinstance(pair, Mapping):
                raise ValueError("manifold pair evidence must be a mapping")
            records = _nonnegative_integer(
                pair.get("contact_records"), "manifold pair record count"
            )
            if records == 0:
                if "contact_manifold" in pair:
                    raise ValueError(
                        "zero-record manifold header has point evidence"
                    )
                continue
            manifold = pair.get("contact_manifold")
            if not isinstance(manifold, Mapping):
                raise ValueError("manifold raw contact slice is missing")
            points = manifold.get("contact_points")
            if not isinstance(points, list) or len(points) != records:
                raise ValueError("manifold point and record counts differ")
            raw_points = []
            for point_index, point in enumerate(points):
                if not isinstance(point, Mapping):
                    raise ValueError(
                        "manifold point evidence must be a mapping"
                    )
                if point.get("point_index") != point_index:
                    raise ValueError("manifold point index changed")
                raw_points.append(
                    {
                        "normal": point.get("normal_world_reported"),
                        "impulse": point.get(
                            "impulse_world_reported_n_s"
                        ),
                        "position": point.get("position_world_m"),
                        "separation": point.get("separation_m"),
                        "material0": point.get("material0"),
                        "material1": point.get("material1"),
                        "face_index0": point.get("face_index0"),
                        "face_index1": point.get("face_index1"),
                    }
                )
            rebuilt = build_tactile_manifold_pair_evidence(
                actor_paths=(manifold.get("actor0"), manifold.get("actor1")),
                collider_paths=(
                    manifold.get("collider0"),
                    manifold.get("collider1"),
                ),
                event_type=manifold.get("event_type"),
                contact_points=raw_points,
                body_root=body_root,
                fixed_root=fixed_root,
                task_rotation_world=rotation,
                task_origin_world=origin,
                physics_dt_s=dt,
            )
            if rebuilt != manifold:
                raise ValueError(
                    "manifold derived contact evidence differs from raw data"
                )
            expected_paths = [
                rebuilt["actor0"],
                rebuilt["actor1"],
                rebuilt["collider0"],
                rebuilt["collider1"],
            ]
            if pair.get("paths") != expected_paths:
                raise ValueError("manifold raw actor/collider order changed")
            frame_contact_record_count += records
            contact_point_count += records
            positive_frame_records += records
        if positive_frame_records <= 0:
            raise ValueError("manifold frame has no positive contact records")

    expected_ceilings = _exact_mapping(
        expected_experimental_abort_ceilings,
        TACTILE_PREFLIGHT_PEAK_FIELDS,
        "expected manifold abort ceilings",
    )
    expected_ceilings = {
        name: number(expected_ceilings[name], f"expected ceiling {name}")
        for name in TACTILE_PREFLIGHT_PEAK_FIELDS
    }

    def validate_phase(evidence, phase_name):
        if not isinstance(evidence, Mapping):
            raise ValueError(f"manifold {phase_name} evidence is missing")
        checked = _nonnegative_integer(
            evidence.get("checked_sample_count"),
            f"manifold {phase_name} checked count",
        )
        finite = _nonnegative_integer(
            evidence.get("finite_sample_count"),
            f"manifold {phase_name} finite count",
        )
        if checked <= 0 or finite != checked:
            raise ValueError(f"manifold {phase_name} sample counts differ")
        if evidence.get("all_samples_finite") is not True:
            raise ValueError(f"manifold {phase_name} finite gate is false")
        if _nonnegative_integer(
            evidence.get("minimum_body_contact_finger_count"),
            f"manifold {phase_name} minimum fingers",
        ) != expected_body_contact_finger_count:
            raise ValueError(f"manifold {phase_name} lost three-finger grasp")
        for count_name in (
            "applied_action_precheck_count",
            "applied_action_postcheck_count",
        ):
            if _nonnegative_integer(
                evidence.get(count_name), f"manifold {phase_name} {count_name}"
            ) != checked:
                raise ValueError(
                    f"manifold {phase_name} applied evidence is incomplete"
                )
        if (
            evidence.get("applied_action_all_float32_equivalent") is not True
            or number(
                evidence.get(
                    "applied_action_maximum_abs_arm_error_float32_rad"
                ),
                f"manifold {phase_name} applied error",
            )
            != 0.0
        ):
            raise ValueError(
                f"manifold {phase_name} applied command did not match"
            )
        start_tcp = vector(
            evidence.get("measured_start_tcp_prim_world_m"),
            3,
            f"manifold {phase_name} measured start TCP",
        )
        command_start = vector(
            evidence.get("command_start_fk_tcp_world_m"),
            3,
            f"manifold {phase_name} command start FK",
        )
        command_start_arm = vector(
            evidence.get("command_start_arm_rad"),
            7,
            f"manifold {phase_name} command start arm",
        )
        command_start_float32 = tuple(
            float(struct.unpack("!f", struct.pack("!f", value))[0])
            for value in command_start_arm
        )
        start_gap = number(
            evidence.get("estimated_start_gap_m"),
            f"manifold {phase_name} start gap",
        )
        derived_start_gap = preinsert_gap + sum(
            (start_tcp[axis] - origin[axis]) * expected_axes["z"][axis]
            for axis in range(3)
        )
        if not math.isclose(
            start_gap,
            derived_start_gap,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"manifold {phase_name} start gap differs from raw TCP"
            )
        list_fields = (
            "measured_tcp_prim_world_samples_m",
            "command_fk_tcp_world_samples_m",
            "task_z_progress_samples_m",
            "command_fk_task_z_progress_samples_m",
            "estimated_gap_samples_m",
            "guarded_tick_samples",
        )
        if any(
            not isinstance(evidence.get(name), list)
            for name in list_fields
        ):
            raise ValueError(
                f"manifold {phase_name} raw trajectory is missing"
            )
        measured_tcp = tuple(
            vector(sample, 3, f"manifold {phase_name} measured TCP")
            for sample in evidence["measured_tcp_prim_world_samples_m"]
        )
        command_tcp = tuple(
            vector(sample, 3, f"manifold {phase_name} command TCP")
            for sample in evidence["command_fk_tcp_world_samples_m"]
        )
        actual_progress = tuple(
            number(sample, f"manifold {phase_name} actual progress")
            for sample in evidence["task_z_progress_samples_m"]
        )
        command_progress = tuple(
            number(sample, f"manifold {phase_name} command progress")
            for sample in evidence["command_fk_task_z_progress_samples_m"]
        )
        gaps = tuple(
            number(sample, f"manifold {phase_name} gap")
            for sample in evidence["estimated_gap_samples_m"]
        )
        guarded_ticks = evidence["guarded_tick_samples"]
        if any(
            len(samples) != checked
            for samples in (
                measured_tcp,
                command_tcp,
                actual_progress,
                command_progress,
                gaps,
                guarded_ticks,
            )
        ):
            raise ValueError(f"manifold {phase_name} raw sample counts differ")
        task_z = expected_axes["z"]
        for index in range(checked):
            derived_actual = sum(
                (measured_tcp[index][axis] - start_tcp[axis]) * task_z[axis]
                for axis in range(3)
            )
            derived_command = sum(
                (command_tcp[index][axis] - command_start[axis]) * task_z[axis]
                for axis in range(3)
            )
            if (
                not math.isclose(
                    actual_progress[index],
                    derived_actual,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    command_progress[index],
                    derived_command,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    gaps[index],
                    preinsert_gap
                    + sum(
                        (measured_tcp[index][axis] - origin[axis])
                        * task_z[axis]
                        for axis in range(3)
                    ),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or gaps[index] < gap_floor
            ):
                raise ValueError(
                    f"manifold {phase_name} raw progress/gap is invalid"
                )

        raw_contacts = {
            name: 0 for name in TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS
        }
        raw_phase_peaks = {
            name: 0.0 for name in TACTILE_PREFLIGHT_PEAK_FIELDS
        }
        previous_global_step = None
        previous_command_float32 = None
        raw_commands = []
        for index, tick in enumerate(guarded_ticks):
            if not isinstance(tick, Mapping):
                raise ValueError(
                    f"manifold {phase_name} guarded tick is not a mapping"
                )
            global_step = tick.get("global_step")
            if type(global_step) is not int or (
                previous_global_step is not None
                and global_step != previous_global_step + 1
            ):
                raise ValueError(
                    f"manifold {phase_name} guarded ticks are not consecutive"
                )
            previous_global_step = global_step
            if (
                tick.get("finite") is not True
                or tick.get("body_contact_fingers")
                != ["f1", "f2", "f3"]
                or tick.get("applied_pre_float32_gate") is not True
                or tick.get("applied_post_float32_gate") is not True
                or number(
                    tick.get("applied_pre_max_error_float32_rad"),
                    f"manifold {phase_name} tick pre-action error",
                )
                != 0.0
                or number(
                    tick.get("applied_post_max_error_float32_rad"),
                    f"manifold {phase_name} tick post-action error",
                )
                != 0.0
            ):
                raise ValueError(
                    f"manifold {phase_name} guarded tick gate failed"
                )
            command = vector(
                tick.get("command_arm_rad"),
                7,
                f"manifold {phase_name} tick command",
            )
            command_float32 = tuple(
                float(struct.unpack("!f", struct.pack("!f", value))[0])
                for value in command
            )
            raw_commands.append(command)
            pre_applied = vector(
                tick.get("applied_pre_readback_float32"),
                7,
                f"manifold {phase_name} tick pre readback",
            )
            pre_expected = vector(
                tick.get("applied_pre_expected_float32"),
                7,
                f"manifold {phase_name} tick pre expected",
            )
            post_applied = vector(
                tick.get("applied_post_readback_float32"),
                7,
                f"manifold {phase_name} tick post readback",
            )
            post_expected = vector(
                tick.get("applied_post_expected_float32"),
                7,
                f"manifold {phase_name} tick post expected",
            )
            if (
                post_applied != post_expected
                or post_expected != command_float32
                or (
                    pre_applied != pre_expected
                    or pre_expected
                    != (
                        command_start_float32
                        if index == 0
                        else previous_command_float32
                    )
                )
            ):
                raise ValueError(
                    f"manifold {phase_name} tick raw readback differs"
                )
            previous_command_float32 = command_float32
            if (
                vector(
                    tick.get("command_fk_tcp_world_m"),
                    3,
                    f"manifold {phase_name} tick command FK",
                )
                != command_tcp[index]
                or vector(
                    tick.get("measured_tcp_prim_world_m"),
                    3,
                    f"manifold {phase_name} tick measured TCP",
                )
                != measured_tcp[index]
            ):
                raise ValueError(
                    f"manifold {phase_name} tick trajectory differs"
                )
            close(
                tick.get("estimated_gap_m"),
                gaps[index],
                f"manifold {phase_name} tick gap",
                tolerance=0.0,
            )
            for wrench_name in (
                "raw_wrench",
                "canonical_wrench_sensor",
                "compensated_wrench_sensor",
                "compensated_wrench_task",
            ):
                vector(
                    tick.get(wrench_name),
                    6,
                    f"manifold {phase_name} tick {wrench_name}",
                )
            wrench = vector(
                tick.get("compensated_wrench_task"),
                6,
                f"manifold {phase_name} tick task wrench",
            )
            finger_delta = vector(
                tick.get("finger_base_torque_delta_nm"),
                3,
                f"manifold {phase_name} tick finger effort",
            )
            scalar_peaks = {
                "absolute_axial_force_n": abs(wrench[2]),
                "lateral_force_n": math.hypot(wrench[0], wrench[1]),
                "bending_torque_nm": math.hypot(wrench[3], wrench[4]),
                "absolute_tightening_torque_nm": abs(wrench[5]),
                "absolute_finger_base_torque_nm": max(
                    abs(value) for value in finger_delta
                ),
            }
            for name, scalar in scalar_peaks.items():
                raw_phase_peaks[name] = max(
                    raw_phase_peaks[name], scalar
                )
            tick_loose_fixed_records = _nonnegative_integer(
                tick.get("loose_fixed_contact_records"),
                f"manifold {phase_name} tick loose-fixed records",
            )
            raw_contacts["loose_fixed"] += tick_loose_fixed_records
            tick_pair_records = {}
            for name, key in (
                ("intended_lip", "intended_lip_contact_pairs"),
                (
                    "unexpected_loose_fixed",
                    "unexpected_loose_fixed_contact_pairs",
                ),
                ("loose_fixture", "loose_fixture_contact_pairs"),
                ("loose_table", "loose_table_contact_pairs"),
            ):
                pairs = tick.get(key)
                if not isinstance(pairs, list):
                    raise ValueError(
                        f"manifold {phase_name} tick {key} is not a list"
                    )
                tick_pair_records[name] = sum(
                    _nonnegative_integer(
                        pair.get("contact_records")
                        if isinstance(pair, Mapping)
                        else None,
                        f"manifold {phase_name} tick {name} records",
                    )
                    for pair in pairs
                )
                raw_contacts[name] += tick_pair_records[name]
            # The query header's loose/fixed point count and its classified
            # pair lists must describe this same physics tick.  Aggregate-only
            # equality would allow records to be shifted between adjacent
            # ticks and would no longer prove six consecutive lip frames.
            if (
                tick_loose_fixed_records
                != tick_pair_records["intended_lip"]
                + tick_pair_records["unexpected_loose_fixed"]
                or tick_pair_records["unexpected_loose_fixed"] != 0
                or tick_pair_records["loose_fixture"] != 0
                or tick_pair_records["loose_table"] != 0
            ):
                raise ValueError(
                    f"manifold {phase_name} per-tick contact scope differs"
                )

        def peak_speed(samples, start):
            previous = start
            peak = 0.0
            for sample in samples:
                peak = max(
                    peak,
                    math.sqrt(
                        sum(
                            (sample[axis] - previous[axis]) ** 2
                            for axis in range(3)
                        )
                    )
                    * rate,
                )
                previous = sample
            return peak

        actual_peak = peak_speed(measured_tcp, start_tcp)
        command_peak = peak_speed(command_tcp, command_start)
        actual_axial_peak = max(
            abs(
                actual_progress[index]
                - (0.0 if index == 0 else actual_progress[index - 1])
            )
            * rate
            for index in range(checked)
        )
        command_axial_peak = max(
            abs(
                command_progress[index]
                - (0.0 if index == 0 else command_progress[index - 1])
            )
            * rate
            for index in range(checked)
        )
        for name, derived in (
            ("peak_abs_tcp_speed_m_s", actual_peak),
            ("peak_abs_command_fk_tcp_speed_m_s", command_peak),
            ("peak_abs_task_z_speed_m_s", actual_axial_peak),
            (
                "peak_abs_command_fk_task_z_speed_m_s",
                command_axial_peak,
            ),
            ("minimum_task_z_progress_m", min(actual_progress)),
            ("maximum_task_z_progress_m", max(actual_progress)),
            ("final_task_z_progress_m", actual_progress[-1]),
        ):
            close(
                evidence.get(name),
                derived,
                f"manifold {phase_name} {name}",
            )
        if (
            actual_peak > measured_speed_limit
            or command_peak > command_speed_limit + 1.0e-12
        ):
            raise ValueError(f"manifold {phase_name} speed ceiling exceeded")
        contacts = _exact_mapping(
            evidence.get("contact_record_totals"),
            TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS,
            f"manifold {phase_name} contact totals",
        )
        contacts = {
            name: _nonnegative_integer(
                contacts[name], f"manifold {phase_name} {name} contacts"
            )
            for name in TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS
        }
        if contacts != raw_contacts:
            raise ValueError(
                f"manifold {phase_name} raw contact totals differ"
            )
        peaks = _exact_mapping(
            evidence.get("peak_experimental_observations"),
            TACTILE_PREFLIGHT_PEAK_FIELDS,
            f"manifold {phase_name} peaks",
        )
        peaks = {
            name: number(peaks[name], f"manifold {phase_name} peak {name}")
            for name in TACTILE_PREFLIGHT_PEAK_FIELDS
        }
        if any(
            peaks[name] < 0.0 or peaks[name] > expected_ceilings[name]
            for name in TACTILE_PREFLIGHT_PEAK_FIELDS
        ):
            raise ValueError(f"manifold {phase_name} abort ceiling exceeded")
        if any(
            peaks[name] + 1.0e-15 < raw_phase_peaks[name]
            for name in TACTILE_PREFLIGHT_PEAK_FIELDS
        ):
            raise ValueError(
                f"manifold {phase_name} raw peak exceeds its summary"
            )
        return {
            "checked": checked,
            "finite": finite,
            "minimum_fingers": expected_body_contact_finger_count,
            "actual_peak": actual_peak,
            "command_peak": command_peak,
            "minimum_gap": min(gaps),
            "contacts": contacts,
            "peaks": peaks,
            "raw_peaks": raw_phase_peaks,
            "measured_tcp": measured_tcp,
            "command_tcp": command_tcp,
            "gaps": gaps,
            "actual_progress": actual_progress,
            "command_progress": command_progress,
            "measured_start_tcp": start_tcp,
            "command_start_tcp": command_start,
            "command_start_arm": command_start_arm,
            "first_global_step": guarded_ticks[0]["global_step"],
            "last_global_step": guarded_ticks[-1]["global_step"],
            "commands": tuple(raw_commands),
            "guarded_ticks": tuple(guarded_ticks),
        }

    phase_names = (
        "plus_x_offset",
        "guarded_approach_and_frozen_hold",
        "terminal_retract",
    )
    terminal = value.get("terminal_retract")
    if not isinstance(terminal, Mapping):
        raise ValueError("manifold terminal retract evidence is missing")
    phases = (
        validate_phase(value.get("lateral_motion_evidence"), phase_names[0]),
        validate_phase(
            value.get("approach_and_hold_motion_evidence"), phase_names[1]
        ),
        validate_phase(terminal.get("motion_evidence"), phase_names[2]),
    )
    if (
        phases[1]["first_global_step"]
        != phases[0]["last_global_step"] + 1
        or phases[2]["first_global_step"]
        != phases[1]["last_global_step"] + 1
        or phases[1]["measured_start_tcp"]
        != phases[0]["measured_tcp"][-1]
        or phases[1]["command_start_tcp"]
        != phases[0]["command_tcp"][-1]
        or phases[1]["command_start_arm"] != phases[0]["commands"][-1]
        or phases[2]["measured_start_tcp"]
        != phases[1]["measured_tcp"][-1]
        or phases[2]["command_start_tcp"]
        != phases[1]["command_tcp"][-1]
        or phases[2]["command_start_arm"] != phases[1]["commands"][-1]
    ):
        raise ValueError("manifold phase boundaries are not continuous")
    if any(phases[0]["contacts"].values()):
        raise ValueError("manifold lateral offset contacted the environment")
    for phase_index in (1, 2):
        contacts = phases[phase_index]["contacts"]
        if (
            contacts["loose_fixed"] != contacts["intended_lip"]
            or contacts["unexpected_loose_fixed"] != 0
            or contacts["loose_fixture"] != 0
            or contacts["loose_table"] != 0
        ):
            raise ValueError("manifold phase contact scope is not exact")
    if phases[1]["contacts"]["intended_lip"] != frame_contact_record_count:
        raise ValueError("manifold frame records differ from approach totals")
    frame_tail = len(frames)
    for index, frame in enumerate(frames):
        tail_index = len(phases[1]["measured_tcp"]) - frame_tail + index
        if tail_index < 0 or (
            tuple(frame["measured_tcp_prim_world_m"])
            != phases[1]["measured_tcp"][tail_index]
            or frame["global_step"]
            != phases[1]["guarded_ticks"][tail_index]["global_step"]
            or tuple(frame["command_arm_rad"])
            != phases[1]["commands"][tail_index]
            or tuple(frame["command_fk_tcp_world_m"])
            != phases[1]["command_tcp"][tail_index]
            or not math.isclose(
                float(frame["estimated_gap_m"]),
                phases[1]["gaps"][tail_index],
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or frame["intended_lip_contact_pairs"]
            != phases[1]["guarded_ticks"][tail_index][
                "intended_lip_contact_pairs"
            ]
            or frame["unexpected_loose_fixed_contact_pairs"]
            != phases[1]["guarded_ticks"][tail_index][
                "unexpected_loose_fixed_contact_pairs"
            ]
            or frame["loose_fixture_contact_pairs"]
            != phases[1]["guarded_ticks"][tail_index][
                "loose_fixture_contact_pairs"
            ]
            or frame["loose_table_contact_pairs"]
            != phases[1]["guarded_ticks"][tail_index][
                "loose_table_contact_pairs"
            ]
        ):
            raise ValueError("manifold frames are not the approach tail")

    derived_actual_peaks = {
        name: phases[index]["actual_peak"]
        for index, name in enumerate(phase_names)
    }
    derived_command_peaks = {
        name: phases[index]["command_peak"]
        for index, name in enumerate(phase_names)
    }
    for reported_name, derived in (
        ("actual_phase_peak_speeds_m_s", derived_actual_peaks),
        ("command_phase_peak_speeds_m_s", derived_command_peaks),
    ):
        reported = _exact_mapping(
            value.get(reported_name), phase_names, f"manifold {reported_name}"
        )
        for name in phase_names:
            close(
                reported[name],
                derived[name],
                f"manifold {reported_name}.{name}",
            )
    derived_minimum_gap = min(phase["minimum_gap"] for phase in phases)
    close(
        value.get("minimum_measured_gap_m"),
        derived_minimum_gap,
        "manifold aggregate minimum gap",
    )
    if derived_minimum_gap < gap_floor:
        raise ValueError("manifold aggregate gap crossed entry")
    total_checked = sum(phase["checked"] for phase in phases)
    if (
        value.get("total_checked_sample_count") != total_checked
        or value.get("total_finite_sample_count") != total_checked
        or value.get("minimum_body_contact_finger_count")
        != expected_body_contact_finger_count
        or value.get("applied_action_precheck_count") != total_checked
        or value.get("applied_action_postcheck_count") != total_checked
        or number(
            value.get(
                "applied_action_maximum_abs_arm_error_float32_rad"
            ),
            "manifold aggregate applied error",
        )
        != 0.0
    ):
        raise ValueError("manifold aggregate sample evidence differs")
    derived_contacts = {
        name: sum(phase["contacts"][name] for phase in phases)
        for name in TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS
    }
    if value.get("contact_record_totals") != derived_contacts:
        raise ValueError("manifold aggregate contact totals differ")
    reported_ceilings = _exact_mapping(
        value.get("experimental_abort_ceilings"),
        TACTILE_PREFLIGHT_PEAK_FIELDS,
        "manifold reported abort ceilings",
    )
    reported_peaks = _exact_mapping(
        value.get("peak_experimental_observations"),
        TACTILE_PREFLIGHT_PEAK_FIELDS,
        "manifold aggregate peaks",
    )
    expected_phase_summary_peaks = (
        phases[0]["raw_peaks"],
        {
            name: max(
                phases[0]["raw_peaks"][name],
                phases[1]["raw_peaks"][name],
            )
            for name in TACTILE_PREFLIGHT_PEAK_FIELDS
        },
        phases[2]["raw_peaks"],
    )
    if any(
        not math.isclose(
            phases[index]["peaks"][name],
            expected_phase_summary_peaks[index][name],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        for index in range(3)
        for name in TACTILE_PREFLIGHT_PEAK_FIELDS
    ):
        raise ValueError("manifold phase peak accumulation differs from raw")
    derived_peaks = {
        name: max(
            expected_phase_summary_peaks[index][name]
            for index in range(3)
        )
        for name in TACTILE_PREFLIGHT_PEAK_FIELDS
    }
    for name in TACTILE_PREFLIGHT_PEAK_FIELDS:
        close(
            reported_ceilings[name],
            expected_ceilings[name],
            f"manifold ceiling {name}",
            tolerance=1.0e-15,
        )
        close(
            reported_peaks[name],
            derived_peaks[name],
            f"manifold aggregate peak {name}",
            tolerance=1.0e-15,
        )

    required_true_gates = (
        "frame_count_gate",
        "consecutive_step_gate",
        "frozen_command_gate",
        "exact_contact_manifold_gate",
        "actual_speed_gate",
        "command_speed_gate",
        "outside_entry_gate",
        "cumulative_finite_gate",
        "three_finger_body_contact_gate",
        "applied_action_gate",
        "contact_scope_gate",
        "experimental_abort_envelope_gate",
        "retract_gate",
    )
    if any(value.get(name) is not True for name in required_true_gates):
        raise ValueError("manifold runtime summary gate is not true")
    if (
        terminal.get("attempted") is not True
        or terminal.get("release_debounced") is not True
        or terminal.get("resume_attempted") is not False
        or terminal.get("terminal_state") != "TERMINAL_MANIFOLD_CAPTURE"
        or terminal.get("motion_evidence", {}).get(
            "required_retract_distance_reached"
        )
        is not True
    ):
        raise ValueError("manifold terminal retract did not complete")
    close(
        terminal.get("commanded_retract_m"),
        retract_distance,
        "manifold commanded retract",
        tolerance=1.0e-15,
    )
    close(
        terminal.get("measured_tcp_prim_retract_m"),
        phases[2]["actual_progress"][-1],
        "manifold measured retract",
    )
    if (
        phases[2]["actual_progress"][-1] < retract_distance
        or min(phases[2]["actual_progress"]) < -negative_retract_bound
    ):
        raise ValueError("manifold terminal retract motion is insufficient")
    release_samples = terminal.get("release_samples")
    if (
        release_samples
        != terminal.get("motion_evidence", {}).get("release_samples")
        or not isinstance(release_samples, list)
        or len(release_samples) < expected_frame_count
    ):
        raise ValueError("manifold terminal release evidence is incomplete")
    release_tail = release_samples[-expected_frame_count:]
    expected_release_compression_limit = number(
        expected_release_compression_ceiling_n,
        "expected manifold release compression ceiling",
    )
    expected_release_bending_limit = number(
        expected_release_bending_ceiling_nm,
        "expected manifold release bending ceiling",
    )
    release_compression_limit = number(
        value.get("release_compression_ceiling_n"),
        "manifold release compression ceiling",
    )
    release_bending_limit = number(
        value.get("release_bending_ceiling_nm"),
        "manifold release bending ceiling",
    )
    if (
        release_compression_limit != expected_release_compression_limit
        or release_bending_limit != expected_release_bending_limit
        or release_compression_limit < 0.0
        or release_bending_limit < 0.0
    ):
        raise ValueError("manifold release ceilings differ from contract")
    if terminal.get("motion_evidence", {}).get(
        "post_retract_release_ticks"
    ) != expected_frame_count:
        raise ValueError("manifold release hold is not exact bounded count")
    reached_retract_indices = [
        index
        for index, progress in enumerate(phases[2]["actual_progress"])
        if progress >= retract_distance
    ]
    expected_retract_tail = list(
        range(
            len(phases[2]["actual_progress"]) - expected_frame_count,
            len(phases[2]["actual_progress"]),
        )
    )
    if reached_retract_indices != expected_retract_tail:
        raise ValueError(
            "manifold raw retract crossing is not the exact release tail"
        )
    terminal_tick_tail = phases[2]["guarded_ticks"][-expected_frame_count:]
    if len(terminal_tick_tail) != expected_frame_count:
        raise ValueError("manifold terminal guarded tail is incomplete")
    for release_index, (sample, tick) in enumerate(
        zip(release_tail, terminal_tick_tail)
    ):
        if (
            not isinstance(sample, Mapping)
            or sample.get("physical_contact") is not False
            or sample.get("release_candidate") is not True
        ):
            raise ValueError("manifold terminal release was not debounced")
        progress = number(
            sample.get("task_z_progress_m"), "release progress"
        )
        compression = number(
            sample.get("signed_compression_n"), "release compression"
        )
        bending = number(
            sample.get("bending_torque_nm"), "release bending"
        )
        raw_wrench = vector(
            tick.get("compensated_wrench_task"),
            6,
            "manifold terminal release raw task wrench",
        )
        raw_compression = compression_sign * raw_wrench[2]
        raw_bending = math.hypot(raw_wrench[3], raw_wrench[4])
        raw_physical_contact = bool(
            tick.get("loose_fixed_contact_records", 0) > 0
            and tick.get("intended_lip_contact_pairs")
            and not tick.get("unexpected_loose_fixed_contact_pairs")
            and not tick.get("loose_fixture_contact_pairs")
            and not tick.get("loose_table_contact_pairs")
        )
        if (
            progress < retract_distance
            or compression > release_compression_limit
            or bending > release_bending_limit
            or not math.isclose(
                progress,
                phases[2]["actual_progress"][
                    len(phases[2]["actual_progress"])
                    - expected_frame_count
                    + release_index
                ],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or sample.get("physical_contact") is not raw_physical_contact
            or not math.isclose(
                compression,
                raw_compression,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                bending,
                raw_bending,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or sample.get("intended_lip_contact_pairs")
            != tick.get("intended_lip_contact_pairs")
            or sample.get("release_candidate")
            is not bool(
                not raw_physical_contact
                and raw_compression <= release_compression_limit
                and raw_bending <= release_bending_limit
            )
        ):
            raise ValueError("manifold release raw threshold gate failed")
    diagnostics = terminal.get("start_and_target_diagnostics")
    if (
        not isinstance(diagnostics, Mapping)
        or diagnostics.get("first_command_exact") is not True
        or diagnostics.get("measured_state_used_for_command") is not False
    ):
        raise ValueError("manifold retract was not command-continuous")
    close(
        diagnostics.get("maximum_commanded_tcp_speed_m_s"),
        command_speed_limit,
        "manifold retract command speed",
        tolerance=1.0e-15,
    )
    return {
        "validated": True,
        "recomputed_manifold_frame_count": len(frames),
        "recomputed_contact_point_count": contact_point_count,
        "recomputed_phase_actual_peak_speeds_m_s": derived_actual_peaks,
        "recomputed_phase_command_peak_speeds_m_s": derived_command_peaks,
        "recomputed_minimum_gap_m": derived_minimum_gap,
        "recomputed_total_checked_sample_count": total_checked,
        "recomputed_contact_record_totals": derived_contacts,
        "recomputed_peak_experimental_observations": derived_peaks,
        "normal_convention": TACTILE_LIP_MANIFOLD_NORMAL_CONVENTION,
        "ft_sign_calibrated": False,
        "stage_b_authorized": False,
    }


def classify_loose_environment_contact(
    paths,
    plug_root,
    fixed_root,
    fixture_root,
    table_root,
):
    """Classify connector/environment pairs that contain no robot path."""

    paths = tuple(str(path) for path in paths)

    def contains(root):
        return any(
            path == root or path.startswith(root + "/") for path in paths
        )

    if not contains(plug_root):
        return None
    for category, root in (
        ("loose_fixed", fixed_root),
        ("loose_fixture", fixture_root),
        ("loose_table", table_root),
    ):
        if contains(root):
            return category
    return None


def _arguments(repository: Path):
    parser = argparse.ArgumentParser(
        description="Run the independent opt-in D38999 visual-XY pick"
    )
    parser.add_argument(
        "--config",
        default=str(
            repository
            / "src/kcg_connector/config/"
            "d38999_visual_xy_pick_probe_v1.yaml"
        ),
        help="strict disabled-by-default independent probe contract",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="optional repository-relative override for probe artifacts",
    )
    parser.add_argument(
        "--preinsert-probe",
        action="store_true",
        help=(
            "continue the passed visual pick in the same World through visual "
            "fixed-XY transport, axis-high and 12 mm preinsert only"
        ),
    )
    parser.add_argument(
        "--preinsert-config",
        default=str(
            repository
            / "src/kcg_connector/config/"
            "d38999_visual_xy_preinsert_probe_v1.yaml"
        ),
        help="strict disabled-by-default preinsert continuation contract",
    )
    parser.add_argument(
        "--wrist-ft-guarded-insertion",
        action="store_true",
        help=(
            "after a passed visual preinsert, continue using only the "
            "compensated wrist 6D wrench and robot proprioception; simulator "
            "contact truth is excluded from this controller"
        ),
    )
    parser.add_argument(
        "--wrist-ft-guarded-config",
        default=str(
            repository
            / "src/kcg_connector/config/"
            "d38999_wrist_ft_guarded_insertion_v1.yaml"
        ),
        help="wrist-FT-only guarded insertion contract",
    )
    parser.add_argument(
        "--show-pose5d",
        action="store_true",
        help=(
            "estimate and render truth-free XYZ/axis/C2 hypotheses at the "
            "initial capture and, with --preinsert-probe, after grasp"
        ),
    )
    parser.add_argument(
        "--pose5d-config",
        default=str(
            repository
            / "src/kcg_connector/config/d38999_pose5d_v1.yaml"
        ),
    )
    parser.add_argument(
        "--wrist-camera-search-report",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_visual_ft_e2e_v1/"
            "wrist_camera_search_20260813T0730Z/camera_search.json"
        ),
        help="generated wrist-camera mounting/posture search result",
    )
    parser.add_argument(
        "--active-wrist-multiview",
        action="store_true",
        help=(
            "after stopped preinsert, acquire generated wrist VIEW_1/VIEW_2 "
            "with local FK/IK motions and return to FINAL_PREINSERT_VIEW"
        ),
    )
    parser.add_argument(
        "--insert-tolerance-report",
        default=str(
            repository
            / "artifacts/kcg_connector/d38999_insert_proxy_v2/"
            "tolerance_sweep.json"
        ),
    )
    parser.add_argument(
        "--tactile-lip-calibration",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tactile-retract-preflight",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tactile-lip-manifold-capture",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tactile-engage-config",
        default=str(
            repository
            / "src/kcg_connector/config/"
            "d38999_tactile_engage_probe_v1.yaml"
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tactile-retract-preflight-report",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="required explicit opt-in; without it the script fails closed",
    )
    arguments = parser.parse_args()
    # Retired 2026-08-13: the physical hand has no fingertip tactile sensor,
    # and the old experiment consumed PhysX contact identities/manifolds.  Keep
    # the parser spellings only long enough to fail old command lines clearly;
    # none of these modes may enter the control path.  The replacement is the
    # wrist-FT-only guarded insertion interface.
    retired_tactile_modes = tuple(
        name
        for name, selected in (
            (
                "--tactile-lip-calibration",
                arguments.tactile_lip_calibration,
            ),
            (
                "--tactile-retract-preflight",
                arguments.tactile_retract_preflight,
            ),
            (
                "--tactile-lip-manifold-capture",
                arguments.tactile_lip_manifold_capture,
            ),
        )
        if selected
    )
    if retired_tactile_modes:
        parser.error(
            ", ".join(retired_tactile_modes)
            + " retired: the hand has no fingertip tactile sensor and PhysX "
            "contact truth is forbidden for control; use the wrist-FT-only "
            "guarded insertion path"
        )
    if not arguments.run:
        parser.error("independent visual XY pick requires explicit --run")
    if (
        arguments.wrist_ft_guarded_insertion
        and not arguments.preinsert_probe
    ):
        parser.error(
            "--wrist-ft-guarded-insertion requires --preinsert-probe"
        )
    tactile_runtime_requested = bool(
        arguments.tactile_lip_calibration
        or arguments.tactile_retract_preflight
        or arguments.tactile_lip_manifold_capture
    )
    if tactile_runtime_requested and not arguments.preinsert_probe:
        parser.error(
            "tactile runtime requires explicit --preinsert-probe"
        )
    tactile_mode_count = sum(
        int(value)
        for value in (
            arguments.tactile_lip_calibration,
            arguments.tactile_retract_preflight,
            arguments.tactile_lip_manifold_capture,
        )
    )
    if tactile_mode_count > 1:
        parser.error(
            "tactile lip, manifold capture, and retract preflight are "
            "mutually exclusive"
        )
    if (
        (
            arguments.tactile_lip_calibration
            or arguments.tactile_lip_manifold_capture
        )
        and not arguments.tactile_retract_preflight_report
    ):
        parser.error(
            "tactile lip calibration requires a passed retract preflight "
            "report; manifold capture has the same prerequisite"
        )
    if (
        not (
            arguments.tactile_lip_calibration
            or arguments.tactile_lip_manifold_capture
        )
        and arguments.tactile_retract_preflight_report
    ):
        parser.error(
            "retract preflight report is accepted only for a lip contact "
            "mode"
        )
    return arguments


def _repository_output_path(repository: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("output directory must be repository-relative")
    result = (repository / relative).resolve()
    if result == repository or repository not in result.parents:
        raise ValueError("output directory must remain below repository")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_tactile_retract_preflight_report(
    value,
    *,
    expected_visual_config_sha256,
    expected_preinsert_config_sha256,
    expected_tactile_config_sha256,
    expected_runtime_source_import_sha256,
    expected_runtime_source_start_sha256,
    expected_trial_id,
    expected_authored_before_physics,
    maximum_negative_progress_m,
    minimum_allowed_gap_m,
    sample_rate_hz,
    expected_commanded_descent_m,
    expected_maximum_commanded_speed_m_s,
    maximum_measured_speed_m_s,
    expected_experimental_abort_ceilings,
    expected_body_contact_finger_count,
):
    """Validate and return the exact no-contact reversal evidence bound.

    A passed report is deliberately not portable across visual variants,
    authored poses, contracts, or runtime revisions.  In particular, a +10 mm
    scene cannot authorize contact in a +20 mm scene merely because both used
    the same high-level tactile YAML.
    """

    import math

    if not isinstance(value, Mapping):
        raise ValueError("tactile retract preflight report must be a mapping")
    section = value.get("tactile_retract_preflight")
    if not isinstance(section, Mapping):
        raise ValueError("tactile retract preflight section is missing")
    required_top_level = {
        "passed": True,
        "preinsert_probe_requested": True,
        "tactile_retract_preflight_requested": True,
        "tactile_lip_calibration_requested": False,
        "tactile_lip_manifold_capture_requested": False,
        "truth_xy_used_for_target": False,
        "config_sha256": expected_visual_config_sha256,
        "preinsert_config_sha256": expected_preinsert_config_sha256,
        "tactile_engage_config_sha256": expected_tactile_config_sha256,
        "runtime_source_import_sha256": (
            expected_runtime_source_import_sha256
        ),
        "runtime_source_start_sha256": expected_runtime_source_start_sha256,
        "runtime_source_finalize_sha256": (
            expected_runtime_source_start_sha256
        ),
        "runtime_source_unchanged": True,
        "trial_id": expected_trial_id,
        "authored_before_physics": expected_authored_before_physics,
    }
    for name, expected in required_top_level.items():
        if value.get(name) != expected or (
            isinstance(expected, bool) and value.get(name) is not expected
        ):
            raise ValueError(
                f"tactile retract preflight {name} does not match"
            )
    _zero_integer(
        value.get("object_pose_writes_after_physics"),
        "tactile retract preflight object pose writes",
    )
    required_section = {
        "passed": True,
        "status": "PASSED_NO_CONTACT_COMMAND_REVERSAL_AT_PREINSERT",
        "lip_contact_executed": False,
        "touch_trials_executed": 0,
        "engage_executed": False,
        "insertion_executed": False,
        "twist_executed": False,
        "home_return_executed": False,
        "assembly_success_claimed": False,
        "all_trajectory_samples_outside_entry_gate": True,
        "measured_descent_bound_gate": True,
        "reversal_negative_progress_bound_gate": True,
        "all_phase_measured_speed_gates": True,
        "all_phase_command_speed_gates": True,
        "phase_sample_count_gates": True,
        "all_samples_finite_gate": True,
        "three_finger_body_contact_gate": True,
        "applied_action_all_float32_equivalent": True,
        "applied_action_evidence_gate": True,
        "all_contact_record_totals_zero_gate": True,
        "experimental_abort_envelope_gate": True,
    }
    for name, expected in required_section.items():
        actual = section.get(name)
        if actual != expected or (
            isinstance(expected, bool) and actual is not expected
        ):
            raise ValueError(
                f"tactile retract preflight section {name} does not match"
            )
    if type(section.get("touch_trials_executed")) is not int:
        raise ValueError(
            "tactile retract preflight touch trial count must be integer zero"
        )
    bound_value = section.get(
        "observed_negative_reversal_progress_bound_m"
    )
    if isinstance(bound_value, bool):
        raise ValueError("tactile retract preflight bound must be numeric")
    try:
        bound = float(bound_value)
        maximum_bound = float(maximum_negative_progress_m)
        minimum_gap = float(section.get("minimum_measured_gap_m"))
        minimum_allowed_gap = float(minimum_allowed_gap_m)
        reported_gap_floor = float(section.get("entry_gap_floor_m"))
        measured_descent = float(
            section.get("measured_descent_tcp_prim_m")
        )
        reported_actual_descent_ceiling = float(
            section.get("maximum_actual_descent_m")
        )
        commanded_descent = float(section.get("commanded_descent_m"))
        reported_command_speed = float(
            section.get("maximum_commanded_tcp_speed_m_s")
        )
        expected_descent = float(expected_commanded_descent_m)
        expected_command_speed = float(
            expected_maximum_commanded_speed_m_s
        )
        measured_speed_ceiling = float(maximum_measured_speed_m_s)
        sample_rate = float(sample_rate_hz)
        reported_measured_speed_ceiling = float(
            section.get("maximum_measured_tcp_speed_m_s")
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "tactile retract preflight trajectory bounds must be numeric"
        ) from error

    axes = _exact_mapping(
        section.get("task_frame_axes_world"),
        ("x", "y", "z", "determinant"),
        "tactile retract preflight task frame axes",
    )
    try:
        task_axes = tuple(
            tuple(float(component) for component in axes[name])
            for name in ("x", "y", "z")
        )
        reported_determinant = float(axes["determinant"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "tactile retract preflight task frame must be numeric"
        ) from error
    if (
        any(len(axis) != 3 for axis in task_axes)
        or not all(
            math.isfinite(component)
            for axis in task_axes
            for component in axis
        )
        or not math.isfinite(reported_determinant)
    ):
        raise ValueError(
            "tactile retract preflight task frame must be finite 3-vectors"
        )
    task_x, task_y, task_z = task_axes
    derived_determinant = sum(
        task_x[index]
        * (
            task_y[(index + 1) % 3] * task_z[(index + 2) % 3]
            - task_y[(index + 2) % 3] * task_z[(index + 1) % 3]
        )
        for index in range(3)
    )
    if (
        any(
            not math.isclose(
                sum(component * component for component in axis),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for axis in task_axes
        )
        or any(
            not math.isclose(
                sum(first[i] * second[i] for i in range(3)),
                0.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for first, second in (
                (task_x, task_y),
                (task_x, task_z),
                (task_y, task_z),
            )
        )
        or not math.isclose(
            derived_determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-9
        )
        or not math.isclose(
            reported_determinant,
            derived_determinant,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "tactile retract preflight task frame is not right-handed"
        )

    expected_ceilings = _exact_mapping(
        expected_experimental_abort_ceilings,
        TACTILE_PREFLIGHT_PEAK_FIELDS,
        "expected tactile preflight experimental abort ceilings",
    )
    reported_ceilings = _exact_mapping(
        section.get("experimental_abort_ceilings"),
        TACTILE_PREFLIGHT_PEAK_FIELDS,
        "tactile preflight experimental abort ceilings",
    )
    reported_peaks = _exact_mapping(
        section.get("peak_experimental_observations"),
        TACTILE_PREFLIGHT_PEAK_FIELDS,
        "tactile preflight experimental peaks",
    )
    try:
        expected_ceilings = {
            name: float(expected_ceilings[name])
            for name in TACTILE_PREFLIGHT_PEAK_FIELDS
        }
        reported_ceilings = {
            name: float(reported_ceilings[name])
            for name in TACTILE_PREFLIGHT_PEAK_FIELDS
        }
        reported_peaks = {
            name: float(reported_peaks[name])
            for name in TACTILE_PREFLIGHT_PEAK_FIELDS
        }
    except (TypeError, ValueError) as error:
        raise ValueError(
            "tactile preflight experimental envelope must be numeric"
        ) from error
    if any(
        not math.isfinite(expected_ceilings[name])
        or expected_ceilings[name] <= 0.0
        or not math.isfinite(reported_ceilings[name])
        or not math.isclose(
            reported_ceilings[name],
            expected_ceilings[name],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isfinite(reported_peaks[name])
        or reported_peaks[name] < 0.0
        or reported_peaks[name] > expected_ceilings[name]
        for name in TACTILE_PREFLIGHT_PEAK_FIELDS
    ):
        raise ValueError(
            "tactile preflight experimental envelope evidence is invalid"
        )

    phase_names = ("descent", "reversal", "recovery")
    phase_evidence = []
    for phase_name in phase_names:
        evidence = section.get(f"{phase_name}_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(
                f"tactile retract preflight {phase_name} evidence is missing"
            )
        gap_samples = evidence.get("estimated_gap_samples_m")
        tcp_samples = evidence.get("measured_tcp_prim_world_samples_m")
        progress_samples = evidence.get("task_z_progress_samples_m")
        command_progress_samples = evidence.get(
            "command_fk_task_z_progress_samples_m"
        )
        command_tcp_samples = evidence.get(
            "command_fk_tcp_world_samples_m"
        )
        measured_start_tcp_sample = evidence.get(
            "measured_start_tcp_prim_world_m"
        )
        command_start_tcp_sample = evidence.get(
            "command_start_fk_tcp_world_m"
        )
        checked_count = _nonnegative_integer(
            evidence.get("checked_sample_count"),
            f"tactile preflight {phase_name} checked sample count",
        )
        finite_count = _nonnegative_integer(
            evidence.get("finite_sample_count"),
            f"tactile preflight {phase_name} finite sample count",
        )
        minimum_fingers = _nonnegative_integer(
            evidence.get("minimum_body_contact_finger_count"),
            f"tactile preflight {phase_name} minimum finger count",
        )
        applied_precheck_count = _nonnegative_integer(
            evidence.get("applied_action_precheck_count"),
            f"tactile preflight {phase_name} applied precheck count",
        )
        applied_postcheck_count = _nonnegative_integer(
            evidence.get("applied_action_postcheck_count"),
            f"tactile preflight {phase_name} applied postcheck count",
        )
        contact_totals = _exact_mapping(
            evidence.get("contact_record_totals"),
            TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS,
            f"tactile preflight {phase_name} contact totals",
        )
        contact_totals = {
            name: _nonnegative_integer(
                contact_totals[name],
                f"tactile preflight {phase_name} {name} contacts",
            )
            for name in TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS
        }
        phase_peaks = _exact_mapping(
            evidence.get("peak_experimental_observations"),
            TACTILE_PREFLIGHT_PEAK_FIELDS,
            f"tactile preflight {phase_name} experimental peaks",
        )
        try:
            phase_peaks = {
                name: float(phase_peaks[name])
                for name in TACTILE_PREFLIGHT_PEAK_FIELDS
            }
            applied_max_error = float(
                evidence.get(
                    "applied_action_maximum_abs_arm_error_float32_rad"
                )
            )
            measured_start_tcp = tuple(
                float(component) for component in measured_start_tcp_sample
            )
            command_start_tcp = tuple(
                float(component) for component in command_start_tcp_sample
            )
            estimated_start_gap = float(
                evidence.get("estimated_start_gap_m")
            )
            gaps = tuple(float(sample) for sample in gap_samples)
            tcp = tuple(
                tuple(float(component) for component in sample)
                for sample in tcp_samples
            )
            progress = tuple(float(sample) for sample in progress_samples)
            command_progress = tuple(
                float(sample) for sample in command_progress_samples
            )
            command_tcp = tuple(
                tuple(float(component) for component in sample)
                for sample in command_tcp_samples
            )
            reported_actual_peak = float(
                evidence.get("peak_abs_task_z_speed_m_s")
            )
            reported_command_peak = float(
                evidence.get("peak_abs_command_fk_task_z_speed_m_s")
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"tactile retract preflight {phase_name} evidence must be "
                "numeric"
            ) from error
        if (
            not isinstance(gap_samples, list)
            or not isinstance(tcp_samples, list)
            or not isinstance(progress_samples, list)
            or not isinstance(command_progress_samples, list)
            or not isinstance(command_tcp_samples, list)
            or not isinstance(measured_start_tcp_sample, list)
            or not isinstance(command_start_tcp_sample, list)
            or checked_count <= 0
            or any(
                sample_count != checked_count
                for sample_count in (
                    len(gaps),
                    len(tcp),
                    len(progress),
                    len(command_progress),
                    len(command_tcp),
                )
            )
            or finite_count != checked_count
            or evidence.get("all_samples_finite") is not True
            or minimum_fingers != expected_body_contact_finger_count
            or applied_precheck_count != checked_count
            or applied_postcheck_count != checked_count
            or evidence.get(
                "applied_action_all_float32_equivalent"
            )
            is not True
            or not math.isfinite(applied_max_error)
            or applied_max_error != 0.0
            or any(contact_totals.values())
            or len(measured_start_tcp) != 3
            or len(command_start_tcp) != 3
            or any(len(sample) != 3 for sample in tcp + command_tcp)
            or not all(
                math.isfinite(component)
                for sample in (
                    measured_start_tcp,
                    command_start_tcp,
                    *tcp,
                    *command_tcp,
                )
                for component in sample
            )
            or not math.isfinite(estimated_start_gap)
            or not all(
                math.isfinite(sample)
                for sample in gaps + progress + command_progress
            )
            or not math.isfinite(reported_actual_peak)
            or not math.isfinite(reported_command_peak)
            or any(
                not math.isfinite(phase_peaks[name])
                or phase_peaks[name] < 0.0
                or phase_peaks[name] > expected_ceilings[name]
                for name in TACTILE_PREFLIGHT_PEAK_FIELDS
            )
        ):
            raise ValueError(
                f"tactile retract preflight {phase_name} cumulative evidence "
                "is incomplete"
            )
        for index in range(checked_count):
            previous_tcp = (
                measured_start_tcp if index == 0 else tcp[index - 1]
            )
            previous_command_tcp = (
                command_start_tcp if index == 0 else command_tcp[index - 1]
            )
            previous_progress = 0.0 if index == 0 else progress[index - 1]
            previous_command_progress = (
                0.0 if index == 0 else command_progress[index - 1]
            )
            previous_gap = (
                estimated_start_gap if index == 0 else gaps[index - 1]
            )
            actual_tcp_delta = sum(
                (tcp[index][axis] - previous_tcp[axis]) * task_z[axis]
                for axis in range(3)
            )
            command_tcp_delta = sum(
                (
                    command_tcp[index][axis]
                    - previous_command_tcp[axis]
                )
                * task_z[axis]
                for axis in range(3)
            )
            if (
                not math.isclose(
                    progress[index] - previous_progress,
                    actual_tcp_delta,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    command_progress[index] - previous_command_progress,
                    command_tcp_delta,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    gaps[index] - previous_gap,
                    progress[index] - previous_progress,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    f"tactile retract preflight {phase_name} raw TCP and "
                    "progress evidence differ"
                )
        derived_actual_peak = max(
            (
                abs(
                    progress[index]
                    - (0.0 if index == 0 else progress[index - 1])
                )
                * sample_rate
                for index in range(checked_count)
            ),
            default=0.0,
        )
        derived_command_peak = max(
            (
                abs(
                    command_progress[index]
                    - (
                        0.0
                        if index == 0
                        else command_progress[index - 1]
                    )
                )
                * sample_rate
                for index in range(checked_count)
            ),
            default=0.0,
        )
        if (
            derived_actual_peak > measured_speed_ceiling
            or derived_command_peak > expected_command_speed
            or not math.isclose(
                reported_actual_peak,
                derived_actual_peak,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                reported_command_peak,
                derived_command_peak,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                f"tactile retract preflight {phase_name} speed evidence "
                "exceeded its immutable ceiling"
            )
        phase_evidence.append(
            {
                "gaps": gaps,
                "progress": progress,
                "actual_peak": derived_actual_peak,
                "command_peak": derived_command_peak,
                "checked_count": checked_count,
                "finite_count": finite_count,
                "minimum_fingers": minimum_fingers,
                "applied_precheck_count": applied_precheck_count,
                "applied_postcheck_count": applied_postcheck_count,
                "applied_max_error": applied_max_error,
                "contact_totals": contact_totals,
                "peaks": phase_peaks,
            }
        )

    all_gaps = tuple(
        gap for evidence in phase_evidence for gap in evidence["gaps"]
    )
    derived_minimum_gap = min(all_gaps)
    derived_descent = max(0.0, -min(phase_evidence[0]["progress"]))
    derived_negative_bound = max(
        0.0, -min(phase_evidence[1]["progress"])
    )
    reported_actual_phase_peaks = _exact_mapping(
        section.get("measured_phase_peak_speeds_m_s"),
        phase_names,
        "tactile preflight measured phase peak speeds",
    )
    reported_command_phase_peaks = _exact_mapping(
        section.get("command_fk_phase_peak_speeds_m_s"),
        phase_names,
        "tactile preflight command phase peak speeds",
    )
    reported_sample_counts = _exact_mapping(
        section.get("trajectory_sample_counts"),
        phase_names,
        "tactile preflight trajectory sample counts",
    )
    try:
        reported_actual_phase_peaks = {
            name: float(reported_actual_phase_peaks[name])
            for name in phase_names
        }
        reported_command_phase_peaks = {
            name: float(reported_command_phase_peaks[name])
            for name in phase_names
        }
    except (TypeError, ValueError) as error:
        raise ValueError(
            "tactile preflight reported phase speeds must be numeric"
        ) from error
    total_checked = _nonnegative_integer(
        section.get("total_checked_sample_count"),
        "tactile preflight total checked sample count",
    )
    total_finite = _nonnegative_integer(
        section.get("total_finite_sample_count"),
        "tactile preflight total finite sample count",
    )
    aggregate_minimum_fingers = _nonnegative_integer(
        section.get("minimum_body_contact_finger_count"),
        "tactile preflight aggregate minimum finger count",
    )
    aggregate_applied_prechecks = _nonnegative_integer(
        section.get("applied_action_precheck_count"),
        "tactile preflight aggregate applied precheck count",
    )
    aggregate_applied_postchecks = _nonnegative_integer(
        section.get("applied_action_postcheck_count"),
        "tactile preflight aggregate applied postcheck count",
    )
    aggregate_contacts = _exact_mapping(
        section.get("contact_record_totals"),
        TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS,
        "tactile preflight aggregate contact totals",
    )
    aggregate_contacts = {
        name: _nonnegative_integer(
            aggregate_contacts[name],
            f"tactile preflight aggregate {name} contacts",
        )
        for name in TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS
    }
    try:
        aggregate_applied_max_error = float(
            section.get(
                "applied_action_maximum_abs_arm_error_float32_rad"
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "tactile preflight aggregate applied error must be numeric"
        ) from error
    if (
        not all(
            math.isfinite(value)
            for value in (
                bound,
                maximum_bound,
                minimum_gap,
                minimum_allowed_gap,
                reported_gap_floor,
                measured_descent,
                commanded_descent,
                reported_actual_descent_ceiling,
                reported_command_speed,
                expected_descent,
                expected_command_speed,
                measured_speed_ceiling,
                reported_measured_speed_ceiling,
                sample_rate,
                aggregate_applied_max_error,
                *reported_actual_phase_peaks.values(),
                *reported_command_phase_peaks.values(),
            )
        )
        or type(expected_body_contact_finger_count) is not int
        or expected_body_contact_finger_count != 3
        or maximum_bound < 0.0
        or minimum_allowed_gap < 0.0
        or bound < 0.0
        or bound > maximum_bound
        or measured_descent < 0.0
        or measured_descent > maximum_bound
        or commanded_descent <= 0.0
        or commanded_descent >= maximum_bound
        or expected_descent <= 0.0
        or expected_descent >= maximum_bound
        or reported_command_speed <= 0.0
        or expected_command_speed <= 0.0
        or reported_command_speed > measured_speed_ceiling
        or measured_speed_ceiling <= 0.0
        or sample_rate <= 0.0
        or minimum_gap < minimum_allowed_gap
        or total_checked <= 0
        or total_checked
        != sum(evidence["checked_count"] for evidence in phase_evidence)
        or total_finite != total_checked
        or total_finite
        != sum(evidence["finite_count"] for evidence in phase_evidence)
        or aggregate_minimum_fingers != expected_body_contact_finger_count
        or aggregate_minimum_fingers
        != min(
            evidence["minimum_fingers"] for evidence in phase_evidence
        )
        or aggregate_applied_prechecks != total_checked
        or aggregate_applied_prechecks
        != sum(
            evidence["applied_precheck_count"]
            for evidence in phase_evidence
        )
        or aggregate_applied_postchecks != total_checked
        or aggregate_applied_postchecks
        != sum(
            evidence["applied_postcheck_count"]
            for evidence in phase_evidence
        )
        or aggregate_applied_max_error != 0.0
        or aggregate_applied_max_error
        != max(
            evidence["applied_max_error"] for evidence in phase_evidence
        )
        or section.get("applied_action_all_float32_equivalent") is not True
        or any(aggregate_contacts.values())
        or any(
            aggregate_contacts[name]
            != sum(
                evidence["contact_totals"][name]
                for evidence in phase_evidence
            )
            for name in TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS
        )
        or any(
            not math.isclose(
                reported_peaks[name],
                phase_evidence[-1]["peaks"][name],
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            for name in TACTILE_PREFLIGHT_PEAK_FIELDS
        )
        or any(
            phase_evidence[index]["peaks"][name]
            > phase_evidence[index + 1]["peaks"][name]
            for name in TACTILE_PREFLIGHT_PEAK_FIELDS
            for index in range(len(phase_evidence) - 1)
        )
        or any(
            type(reported_sample_counts[name]) is not int
            or reported_sample_counts[name]
            != phase_evidence[index]["checked_count"]
            for index, name in enumerate(phase_names)
        )
        or any(
            not math.isclose(
                reported_actual_phase_peaks[name],
                phase_evidence[index]["actual_peak"],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                reported_command_phase_peaks[name],
                phase_evidence[index]["command_peak"],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for index, name in enumerate(phase_names)
        )
        or not math.isclose(
            reported_gap_floor,
            minimum_allowed_gap,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            minimum_gap,
            derived_minimum_gap,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            measured_descent,
            derived_descent,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            bound,
            derived_negative_bound,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            commanded_descent,
            expected_descent,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            reported_actual_descent_ceiling,
            maximum_bound,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            reported_command_speed,
            expected_command_speed,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            reported_measured_speed_ceiling,
            measured_speed_ceiling,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError(
            "tactile retract preflight cumulative evidence is outside the "
            "no-contact scope"
        )
    return bound


def _scene_for_probe(tabletop, probe):
    """Return the one immutable non-nominal scene authored pre-physics."""

    loose = replace(
        tabletop.loose_endpoint,
        initial_origin_m=(
            probe.authored_loose_xy_m[0],
            probe.authored_loose_xy_m[1],
            tabletop.loose_endpoint.initial_origin_m[2],
        ),
    )
    fixed_delta = (
        probe.authored_fixed_xy_m[0]
        - tabletop.fixed_endpoint.receptacle_origin_m[0],
        probe.authored_fixed_xy_m[1]
        - tabletop.fixed_endpoint.receptacle_origin_m[1],
    )
    fixed = replace(
        tabletop.fixed_endpoint,
        fixture_center_m=(
            tabletop.fixed_endpoint.fixture_center_m[0] + fixed_delta[0],
            tabletop.fixed_endpoint.fixture_center_m[1] + fixed_delta[1],
            tabletop.fixed_endpoint.fixture_center_m[2],
        ),
        receptacle_origin_m=(
            probe.authored_fixed_xy_m[0],
            probe.authored_fixed_xy_m[1],
            tabletop.fixed_endpoint.receptacle_origin_m[2],
        ),
    )
    return replace(tabletop, loose_endpoint=loose, fixed_endpoint=fixed)


def main():
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository)
    wrist_ft_guarded_requested = bool(
        arguments.wrist_ft_guarded_insertion
    )
    tactile_runtime_requested = bool(
        arguments.tactile_lip_calibration
        or arguments.tactile_retract_preflight
        or arguments.tactile_lip_manifold_capture
    )
    if arguments.tactile_retract_preflight:
        tactile_report_key = "tactile_retract_preflight"
    elif arguments.tactile_lip_manifold_capture:
        tactile_report_key = "tactile_lip_manifold_capture"
    else:
        tactile_report_key = "tactile_lip_calibration"

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not arguments.gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    passed = False
    world = None
    tactile_touch_motion_started = False
    tactile_abort_retract = None
    tactile_zero_step_abort_reason = None
    tactile_transition_ring = []
    tactile_abort_runtime_evidence = {}
    tactile_terminal_retract_started = False
    tactile_negative_progress_bound_m = None
    first_tactile_forbidden_contact = None
    runtime_source_start_sha256 = _sha256(RUNTIME_SOURCE_PATH)
    report = {
        "schema_version": "kcg_d38999_visual_xy_pick_report_v1",
        "passed": False,
        "gui": arguments.gui,
        "explicit_opt_in": True,
        "object_pose_writes_after_physics": 0,
        "truth_xy_used_for_target": False,
        "orientation_source": "registered_nominal",
        "full_6d": False,
        "production_control_authorized": False,
        "collision_planned": False,
    }
    if arguments.preinsert_probe:
        report.update(
            {
                "preinsert_probe_requested": True,
                "engage_executed": False,
                "insertion_executed": False,
                "twist_executed": False,
                "home_return_executed": False,
            }
        )
    if wrist_ft_guarded_requested:
        report.update(
            {
                "wrist_ft_guarded_insertion_requested": True,
                "wrist_ft_only_control": True,
                "fingertip_tactile_sensor_used": False,
                "physx_contact_truth_used_for_insertion_control": False,
                "simulator_truth_used_for_insertion_control": False,
                "engage_executed": False,
                "insertion_executed": False,
                "twist_executed": False,
                "home_return_executed": False,
            }
        )
    if tactile_runtime_requested:
        report.update(
            {
                # Bind a successful no-contact preflight to the exact tactile
                # runtime that later consumes it.  This field is opt-in only,
                # preserving the original visual/default report byte schema.
                "runtime_source_import_sha256": (
                    RUNTIME_SOURCE_IMPORT_SHA256
                ),
                "runtime_source_start_sha256": (
                    runtime_source_start_sha256
                ),
                "tactile_lip_calibration_requested": bool(
                    arguments.tactile_lip_calibration
                ),
                "tactile_retract_preflight_requested": bool(
                    arguments.tactile_retract_preflight
                ),
                "tactile_lip_manifold_capture_requested": bool(
                    arguments.tactile_lip_manifold_capture
                ),
                "tactile_contact_characterization_only": bool(
                    arguments.tactile_lip_calibration
                    or arguments.tactile_lip_manifold_capture
                ),
                "virtual_ft_is_calibrated_safety_gate": False,
                "engage_executed": False,
                "insertion_executed": False,
                "twist_executed": False,
                "home_return_executed": False,
                "assembly_success_claimed": False,
            }
        )
    try:
        import numpy as np
        from PIL import Image

        from isaacsim.core.api import World
        from isaacsim.core.experimental.utils.semantics import (
            add_labels,
            get_labels,
        )
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
        )
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep
        from omni.physx import get_physx_simulation_interface
        from omni.physx.scripts import physicsUtils
        from pxr import (
            Gf,
            PhysxSchema,
            PhysicsSchemaTools,
            Sdf,
            Usd,
            UsdGeom,
            UsdLux,
            UsdPhysics,
            UsdShade,
        )

        from d38999_tabletop_pick_smoke import (
            EXPECTED_DOF_NAMES,
            _all_fingers_have_body_contact,
            _array_quaternion_error_radians,
            _axis_error_radians,
            _classify_robot_external_contact,
            _d38999_loose_collider_group,
            _finger_loose_contact_group,
            _gf_quaternion_error_radians,
            _is_finger_plug_contact,
            _is_plug_table_contact,
            _quaternion_world_z_axis,
            _world_pose,
        )
        from kcg_connector.d38999_tabletop_pick import (
            interpolate_arm,
            minimum_jerk_blend,
            verify_d38999_pick_dependencies,
        )
        from kcg_connector.d38999_tabletop_scene import (
            author_d38999_tabletop_scene,
        )
        from kcg_connector.d38999_visual_xy_pick_probe import (
            build_visual_xy_pick_plan,
            evaluate_visual_xy_truth_only,
            load_visual_xy_pick_probe_contract,
            pose_provider_sample_from_rgbd_metrics,
        )
        from kcg_connector.isaac_d38999_rgbd_runtime import (
            capture_d38999_rgbd_runtime,
        )
        if arguments.preinsert_probe:
            # The default visual-pick path does not import or load continuation
            # code.  This branch is an explicit independent experiment only.
            from kcg_connector.d38999_physical_insertion import (
                measure_alignment,
            )
            from kcg_connector.d38999_visual_xy_preinsert_probe import (
                build_visual_xy_preinsert_plan,
                load_visual_xy_preinsert_probe_contract,
            )
        if wrist_ft_guarded_requested:
            from kcg_connector.d38999_wrist_ft_guarded_insertion import (
                GuardedInsertionPhase,
                initial_guarded_insertion_state,
                load_guarded_insertion_contract,
                parse_guarded_insertion_observation,
                step_guarded_insertion,
                verify_guarded_insertion_inputs,
            )
        if (
            wrist_ft_guarded_requested
            or tactile_runtime_requested
            or arguments.show_pose5d
        ):
            from kcg_connector.d38999_physical_insertion import (
                solve_fixed_q7_tcp_pose,
            )
            from kcg_connector.d38999_tabletop_pick import (
                iiwa14_grasp_tcp_transform,
            )
            from kcg_connector.virtual_wrist_ft_runtime import (
                VirtualWristFtMonitor,
                column_rotation_from_gf_matrix3d,
                load_virtual_wrist_ft_monitor_config,
                reaction_row_index,
                verify_virtual_wrist_ft_monitor_inputs,
            )
        if tactile_runtime_requested:
            # Keep every calibration dependency out of the default visual-pick
            # and preinsert-only import paths.  This branch is both explicit
            # and dependent on the already-explicit preinsert branch above.
            from kcg_connector.d38999_tactile_engage_probe import (
                EngageObservation,
                EngageState,
                contact_candidate,
                contact_release_candidate,
                decide_engage_transition,
                load_tactile_engage_contract,
            )
            from kcg_connector.d38999_proxy_collision_filter import (
                apply_proxy_collision_filter,
                build_proxy_collision_filter_plan,
            )

        config_path = Path(arguments.config).expanduser().resolve()
        probe = load_visual_xy_pick_probe_contract(
            config_path, repository=repository
        )
        pose5d_config = None
        pose5d_gates = None
        wrist_camera_search = None
        if arguments.show_pose5d:
            from kcg_connector.d38999_pose5d import load_pose5d_config

            pose5d_config = load_pose5d_config(arguments.pose5d_config)
            tolerance_document = json.loads(
                Path(arguments.insert_tolerance_report).read_text(
                    encoding="utf-8"
                )
            )
            boundaries = tolerance_document["measured_boundaries"]
            pose5d_gates = {
                "lateral_position_m": min(
                    boundaries["x_offset_m"]["authorization_gate_abs"],
                    boundaries["y_offset_m"]["authorization_gate_abs"],
                ),
                "axis_angle_rad": min(
                    boundaries["tilt_x_rad"]["authorization_gate_abs"],
                    boundaries["tilt_y_rad"]["authorization_gate_abs"],
                ),
            }
            wrist_camera_search_path = Path(
                arguments.wrist_camera_search_report
            ).expanduser().resolve()
            if wrist_camera_search_path.is_file():
                wrist_camera_search = json.loads(
                    wrist_camera_search_path.read_text(encoding="utf-8")
                )
        preinsert_probe = None
        preinsert_config_path = None
        if arguments.preinsert_probe:
            preinsert_config_path = (
                Path(arguments.preinsert_config).expanduser().resolve()
            )
            preinsert_probe = load_visual_xy_preinsert_probe_contract(
                preinsert_config_path, repository=repository
            )
        tactile_probe = None
        tactile_config_path = None
        tactile_retract_preflight_report_path = None
        tactile_retract_preflight_report = None
        wrist_ft_config = None
        wrist_ft_inputs = None
        wrist_ft_guarded_contract = None
        wrist_ft_guarded_config_path = None
        if wrist_ft_guarded_requested:
            wrist_ft_guarded_config_path = Path(
                arguments.wrist_ft_guarded_config
            ).expanduser().resolve()
            wrist_ft_guarded_contract = load_guarded_insertion_contract(
                wrist_ft_guarded_config_path
            )
            verify_guarded_insertion_inputs(
                wrist_ft_guarded_contract, repository
            )
            guarded_inputs = {
                label: (repository / relative_path).resolve()
                for label, relative_path, _ in (
                    wrist_ft_guarded_contract.input_files
                )
            }
            if guarded_inputs["visual_preinsert"] != preinsert_config_path:
                raise RuntimeError(
                    "wrist-FT guarded insertion and selected preinsert "
                    "contracts differ"
                )
            wrist_ft_config = load_virtual_wrist_ft_monitor_config(
                guarded_inputs["wrist_ft_monitor"]
            )
            wrist_ft_inputs = verify_virtual_wrist_ft_monitor_inputs(
                wrist_ft_config, repository
            )
        if tactile_runtime_requested:
            tactile_config_path = (
                Path(arguments.tactile_engage_config).expanduser().resolve()
            )
            tactile_probe = load_tactile_engage_contract(
                tactile_config_path, repository=repository
            )
            if (
                tactile_probe.input_paths["visual_preinsert_contract"]
                != preinsert_config_path
            ):
                raise RuntimeError(
                    "tactile and selected preinsert contracts differ"
                )
            wrist_ft_config = load_virtual_wrist_ft_monitor_config(
                tactile_probe.input_paths["virtual_wrist_ft_monitor"]
            )
            wrist_ft_inputs = verify_virtual_wrist_ft_monitor_inputs(
                wrist_ft_config, repository
            )
            if (
                tactile_probe.sensor.control_rate_hz
                != wrist_ft_config.physics_rate_hz
                or tactile_probe.sensor.local_reference_samples
                != wrist_ft_config.payload_baseline_window_steps
            ):
                raise RuntimeError(
                    "tactile and virtual wrist FT sample contracts differ"
                )
            if (
                arguments.tactile_lip_calibration
                or arguments.tactile_lip_manifold_capture
            ):
                tactile_retract_preflight_report_path = Path(
                    arguments.tactile_retract_preflight_report
                ).expanduser().resolve()
                tactile_retract_preflight_report = json.loads(
                    tactile_retract_preflight_report_path.read_text(
                        encoding="utf-8"
                    )
                )
                tactile_negative_progress_bound_m = (
                    validate_tactile_retract_preflight_report(
                        tactile_retract_preflight_report,
                        expected_visual_config_sha256=_sha256(config_path),
                        expected_preinsert_config_sha256=_sha256(
                            preinsert_config_path
                        ),
                        expected_tactile_config_sha256=_sha256(
                            tactile_config_path
                        ),
                        expected_runtime_source_import_sha256=(
                            RUNTIME_SOURCE_IMPORT_SHA256
                        ),
                        expected_runtime_source_start_sha256=(
                            runtime_source_start_sha256
                        ),
                        expected_trial_id=probe.trial_id,
                        expected_authored_before_physics={
                            "loose_plug_xy_m": list(
                                probe.authored_loose_xy_m
                            ),
                            "fixed_receptacle_xy_m": list(
                                probe.authored_fixed_xy_m
                            ),
                            "loose_yaw_rad": probe.loose_yaw_rad,
                            "fixed_yaw_rad": probe.fixed_yaw_rad,
                        },
                        maximum_negative_progress_m=(
                            TACTILE_RETRACT_PREFLIGHT_DESCENT_M
                        ),
                        minimum_allowed_gap_m=(
                            tactile_probe.motion.entry_gap_m
                        ),
                        sample_rate_hz=tactile_probe.sensor.control_rate_hz,
                        expected_commanded_descent_m=(
                            TACTILE_RETRACT_PREFLIGHT_COMMAND_DESCENT_M
                        ),
                        expected_maximum_commanded_speed_m_s=(
                            TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S
                        ),
                        maximum_measured_speed_m_s=(
                            tactile_probe.motion.unload_retract_speed_m_s
                        ),
                        expected_experimental_abort_ceilings={
                            "absolute_axial_force_n": (
                                tactile_probe.abort
                                .maximum_absolute_axial_force_n
                            ),
                            "lateral_force_n": (
                                tactile_probe.abort.maximum_lateral_force_n
                            ),
                            "bending_torque_nm": (
                                tactile_probe.abort.maximum_bending_torque_nm
                            ),
                            "absolute_tightening_torque_nm": (
                                tactile_probe.abort
                                .maximum_tightening_torque_nm
                            ),
                            "absolute_finger_base_torque_nm": (
                                tactile_probe.abort
                                .maximum_finger_base_torque_nm
                            ),
                        },
                        expected_body_contact_finger_count=3,
                    )
                )
        pick = probe.pick
        dependencies = verify_d38999_pick_dependencies(
            pick,
            probe.input_paths["nominal_pick"],
            repository,
        )
        if dependencies["tabletop"] != probe.tabletop:
            raise RuntimeError("probe and nominal pick tabletop differ")
        tabletop = _scene_for_probe(probe.tabletop, probe)
        asset_path = dependencies["d38999_asset"]
        robot_asset = dependencies["robot_asset"]
        if (
            wrist_ft_inputs is not None
            and wrist_ft_inputs["robot_asset"] != robot_asset
        ):
            raise RuntimeError(
                "virtual wrist FT and visual pick robot assets differ"
            )
        rate_hz = tabletop.physics.rate_hz
        output_dir = _repository_output_path(
            repository,
            arguments.output_dir or probe.output_directory,
        )
        create_exclusive_output_directory(output_dir)
        report.update(
            {
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path),
                "output_directory": str(output_dir),
                "trial_id": probe.trial_id,
                "authored_before_physics": {
                    "loose_plug_xy_m": list(probe.authored_loose_xy_m),
                    "fixed_receptacle_xy_m": list(
                        probe.authored_fixed_xy_m
                    ),
                    "loose_yaw_rad": probe.loose_yaw_rad,
                    "fixed_yaw_rad": probe.fixed_yaw_rad,
                },
            }
        )
        if preinsert_probe is not None:
            report.update(
                {
                    "preinsert_config_path": str(preinsert_config_path),
                    "preinsert_config_sha256": _sha256(
                        preinsert_config_path
                    ),
                }
            )
        if wrist_ft_guarded_contract is not None:
            report.update(
                {
                    "wrist_ft_guarded_config_path": str(
                        wrist_ft_guarded_config_path
                    ),
                    "wrist_ft_guarded_config_sha256": _sha256(
                        wrist_ft_guarded_config_path
                    ),
                    "wrist_ft_guarded_insertion": {
                        "status": "WAITING_FOR_VISUAL_PREINSERT",
                        "passed": False,
                        "controller_inputs": [
                            "rgbd_visual_preinsert",
                            "compensated_wrist_6d_wrench",
                            "robot_joint_state",
                            "robot_tcp_fk",
                        ],
                        "fingertip_tactile_sensor_used": False,
                        "physx_contact_truth_used_for_control": False,
                        "simulator_truth_used_for_control": False,
                        "hardware_safety_certified": False,
                        "production_control_authorized": False,
                        "twist_executed": False,
                        "home_return_executed": False,
                    },
                }
            )
        if tactile_probe is not None:
            report.update(
                {
                    "tactile_engage_config_path": str(tactile_config_path),
                    "tactile_engage_config_sha256": _sha256(
                        tactile_config_path
                    ),
                    tactile_report_key: {
                        "status": "WAITING_FOR_PREINSERT_PASS",
                        "passed": False,
                        "mode": (
                            "NO_CONTACT_COMMAND_REVERSAL_PREFLIGHT"
                            if arguments.tactile_retract_preflight
                            else (
                                "PLUS_X_LIP_MANIFOLD_CAPTURE_ONLY"
                                if arguments.tactile_lip_manifold_capture
                                else "SIGNED_LIP_CONTACT_CALIBRATION"
                            )
                        ),
                        "known_offset_m": TACTILE_LIP_OFFSET_M,
                        "direction_order": [
                            name
                            for name, _, _, _ in (
                                TACTILE_LIP_DIRECTIONS
                                if arguments.tactile_lip_calibration
                                else ()
                            )
                        ]
                        or (
                            [TACTILE_LIP_MANIFOLD_DIRECTION[0]]
                            if arguments.tactile_lip_manifold_capture
                            else []
                        ),
                        "manifold_capture_only": bool(
                            arguments.tactile_lip_manifold_capture
                        ),
                        "ft_sign_calibrated": False,
                        "stage_b_authorized": False,
                        "engage_executed": False,
                        "insertion_executed": False,
                        "twist_executed": False,
                        "home_return_executed": False,
                        "production_control_authorized": False,
                        "hardware_safety_calibration_claimed": False,
                        "assembly_success_claimed": False,
                    },
                }
            )
            if tactile_retract_preflight_report_path is not None:
                report.update(
                    {
                        "tactile_retract_preflight_report_path": str(
                            tactile_retract_preflight_report_path
                        ),
                        "tactile_retract_preflight_report_sha256": _sha256(
                            tactile_retract_preflight_report_path
                        ),
                    }
                )

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / rate_hz,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        stage = get_current_stage()
        report["d38999_authoring"] = author_d38999_tabletop_scene(
            stage,
            tabletop,
            asset_path,
            add_reference_to_stage=add_reference_to_stage,
            Gf=Gf,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
            physics_utils=physicsUtils,
        )
        add_reference_to_stage(
            str(robot_asset), pick.scene.robot_root_prim_path
        )
        tcp_prim = stage.GetPrimAtPath(pick.scene.grasp_tcp_prim_path)
        fixed_prim = stage.GetPrimAtPath(
            tabletop.asset.fixed_receptacle_prim_path
        )
        loose_prim = stage.GetPrimAtPath(tabletop.asset.loose_plug_prim_path)
        for path, prim in (
            (pick.scene.grasp_tcp_prim_path, tcp_prim),
            (tabletop.asset.fixed_receptacle_prim_path, fixed_prim),
            (tabletop.asset.loose_plug_prim_path, loose_prim),
        ):
            if not prim.IsValid():
                raise RuntimeError(f"required scene prim is missing: {path}")

        grip_material_path = "/World/D38999VisualPickGripMaterial"
        grip_material = UsdShade.Material.Define(stage, grip_material_path)
        grip_api = UsdPhysics.MaterialAPI.Apply(grip_material.GetPrim())
        grip_api.CreateStaticFrictionAttr(pick.motion.grip_static_friction)
        grip_api.CreateDynamicFrictionAttr(pick.motion.grip_dynamic_friction)
        grip_api.CreateRestitutionAttr(pick.motion.grip_restitution)
        robot_root = pick.scene.robot_root_prim_path
        plug_root = tabletop.asset.loose_plug_prim_path
        body_root = tabletop.asset.body_prim_path
        nut_root = tabletop.asset.nut_prim_path
        finger_collision_anchors = []
        plug_collision_counts = {"body": 0, "nut": 0}
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            finger_anchor = bool(
                prim_path.startswith(robot_root + "/")
                and prim.GetName().endswith("_convex")
                and any(
                    name in prim_path
                    for name in ("/f1Link", "/f2Link", "/f3Link")
                )
            )
            plug_collision = bool(
                prim.HasAPI(UsdPhysics.CollisionAPI)
                and prim_path.startswith(plug_root + "/")
            )
            if not (finger_anchor or plug_collision):
                continue
            physicsUtils.add_physics_material_to_prim(
                stage, prim, Sdf.Path(grip_material_path)
            )
            if finger_anchor:
                finger_collision_anchors.append(prim_path)
            else:
                group = _d38999_loose_collider_group(
                    prim_path, body_root, nut_root
                )
                if group is None:
                    raise RuntimeError(
                        f"unclassified loose collider: {prim_path}"
                    )
                plug_collision_counts[group] += 1
        if len(finger_collision_anchors) != 8 or plug_collision_counts != {
            "body": 21,
            "nut": 24,
        }:
            raise RuntimeError(
                "visual pick collision/material topology changed"
            )

        proxy_collision_filter = {
            "body_mating_segment_count": 0,
            "enabled": False,
            "fixed_entry_segment_count": 0,
            "mode": "none",
            "nut_segment_count": 0,
            "pair_count": 0,
        }
        if tactile_probe is not None:
            # Disable only the 480 nut-entry and 20 same-angle mating-entry
            # proxy false contacts proven by the existing physical insertion
            # contract.  Other segment pairs remain physical candidates for
            # the signed lip touches.  Topology is authored before reset.
            filter_contract = (
                tactile_probe.input_paths["physical_insertion_contract"]
            )
            if filter_contract != preinsert_probe.input_paths[
                "nominal_insertion"
            ]:
                raise RuntimeError(
                    "tactile and preinsert physical insertion inputs differ"
                )
            filter_policy = preinsert_probe.insertion.proxy_collision_filter
            filter_plan = build_proxy_collision_filter_plan(
                body_root,
                nut_root,
                tabletop.asset.fixed_receptacle_prim_path,
                body_mating_segment_count=(
                    filter_policy.expected_body_mating_segment_count
                ),
                nut_segment_count=(
                    filter_policy.expected_nut_segment_count
                ),
                fixed_entry_segment_count=(
                    filter_policy.expected_fixed_entry_segment_count
                ),
            )
            proxy_collision_filter = apply_proxy_collision_filter(
                stage, UsdPhysics, Sdf, filter_plan
            )
            if (
                proxy_collision_filter["pair_count"]
                != filter_policy.expected_filtered_pair_count
                or proxy_collision_filter["pair_count"]
                != tactile_probe.proxy_boundaries[
                    "filtered_proxy_collision_pair_count"
                ]
            ):
                raise RuntimeError(
                    "D38999 tactile proxy collision filter count changed"
                )
            report[tactile_report_key].update(
                {"proxy_collision_filter": proxy_collision_filter}
            )

        contact_report_body_count = 0
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            is_robot_body = bool(
                prim_path.startswith(robot_root + "/")
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            is_loose_body = prim_path in (body_root, nut_root)
            if is_robot_body or is_loose_body:
                report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                report_api.CreateThresholdAttr().Set(0.0)
                contact_report_body_count += int(is_robot_body)
        if contact_report_body_count < 17:
            raise RuntimeError("robot contact reporting is incomplete")

        robot = world.scene.add(
            SingleArticulation(
                prim_path=pick.scene.articulation_prim_path,
                name="d38999_visual_xy_pick_handarm",
            )
        )
        body = world.scene.add(
            SingleRigidPrim(
                prim_path=body_root,
                name="d38999_visual_xy_pick_body",
            )
        )
        nut = world.scene.add(
            SingleRigidPrim(
                prim_path=nut_root,
                name="d38999_visual_xy_pick_nut",
            )
        )

        # Physics starts here.  No endpoint transform is authored below.
        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError(
                "robot articulation handles were not initialized"
            )
        dof_names = tuple(robot.dof_names)
        if set(dof_names) != set(EXPECTED_DOF_NAMES) or len(dof_names) != 15:
            raise RuntimeError("unexpected articulation DOF layout")
        name_to_index = {name: index for index, name in enumerate(dof_names)}
        arm_indices = np.asarray(
            [name_to_index[name] for name in pick.robot.arm_joint_names],
            dtype=np.int32,
        )
        hand_indices = np.asarray(
            [
                name_to_index[name]
                for name in pick.robot.active_hand_joint_names
            ],
            dtype=np.int32,
        )
        sensor_indices = np.asarray(
            [name_to_index[name] for name in pick.sensing.torque_joint_names],
            dtype=np.int32,
        )
        controlled_indices = np.concatenate((arm_indices, hand_indices))
        zeros = np.zeros(robot.num_dof, dtype=np.float32)
        robot.set_joint_positions(zeros)
        robot.set_joint_velocities(zeros)
        controller = robot.get_articulation_controller()
        kps = np.zeros(robot.num_dof, dtype=np.float32)
        kds = np.zeros(robot.num_dof, dtype=np.float32)
        kps[arm_indices] = pick.robot.arm_stiffness
        kds[arm_indices] = pick.robot.arm_damping
        kps[hand_indices] = pick.robot.hand_stiffness
        kds[hand_indices] = pick.robot.hand_damping
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        world.get_physics_context().set_gravity(tabletop.physics.gravity_m_s2)

        home_arm = np.asarray(pick.robot.home_arm_rad, dtype=np.float64)
        open_hand = np.asarray(pick.robot.open_hand_rad, dtype=np.float64)
        grasp_hand = np.asarray(pick.motion.grasp_hand_rad, dtype=np.float64)
        current_arm = home_arm.copy()
        current_hand = np.zeros(4, dtype=np.float64)
        dof_properties = robot.dof_properties
        phase = "initial_settle"
        global_step = 0
        finite_throughout = True
        maximum_joint_speed = 0.0
        maximum_arm_tracking_error = 0.0
        maximum_joint_limit_violation = 0.0
        maximum_post_tare_delta = np.zeros(3, dtype=np.float64)
        preinsert_checked_steps = 0
        preinsert_minimum_body_contact_fingers = 3
        preinsert_loose_fixed_contact_records = 0
        latest_preinsert_body_contact_fingers = frozenset()
        latest_loose_fixed_contact_records = 0
        latest_intended_lip_contact_pairs = ()
        latest_unexpected_loose_fixed_contact_pairs = ()
        latest_loose_fixture_contact_pairs = ()
        latest_loose_table_contact_pairs = ()
        tactile_contact_rotation = None
        tactile_contact_origin = None
        wrist_ft_monitor = None
        wrist_ft_reaction_row = None
        wrist_ft_sensor_prim = None
        latest_wrist_ft_sample = None
        external_contacts = {
            "table": 0,
            "fixture": 0,
            "fixed_endpoint": 0,
            "loose_plug_preclosure": 0,
            "loose_plug_allowed": 0,
            "loose_plug_unexpected_robot_link": 0,
        }
        grip_material_contact_records = 0
        fixed_initial_position, fixed_initial_orientation = _world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )

        def body_in_tcp_frame(body_position):
            tcp_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
                tcp_prim
            )
            point = tcp_matrix.GetInverse().Transform(
                Gf.Vec3d(*(float(value) for value in body_position))
            )
            return np.asarray(point, dtype=np.float64)

        def contact_snapshot():
            headers, contacts, _ = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            snapshot = {
                "finger_body_group_records": {
                    finger: {"body": 0, "nut": 0}
                    for finger in ("f1", "f2", "f3")
                },
                "grip_material_records": 0,
                "finger_loose_plug_records": 0,
                "plug_table_records": 0,
                "robot_loose_plug_records": 0,
                "unexpected_robot_link_records": 0,
            }
            for header in headers:
                paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                if _is_plug_table_contact(
                    paths, plug_root, tabletop.table.prim_path
                ):
                    snapshot["plug_table_records"] += int(
                        header.num_contact_data
                    )
                category = _classify_robot_external_contact(
                    paths,
                    robot_root,
                    tabletop.table.prim_path,
                    tabletop.fixed_endpoint.fixture_prim_path,
                    tabletop.asset.fixed_receptacle_prim_path,
                    plug_root,
                )
                if category != "loose_plug":
                    continue
                snapshot["robot_loose_plug_records"] += int(
                    header.num_contact_data
                )
                if not _is_finger_plug_contact(
                    paths, robot_root, plug_root
                ):
                    snapshot["unexpected_robot_link_records"] += int(
                        header.num_contact_data
                    )
                    continue
                snapshot["finger_loose_plug_records"] += int(
                    header.num_contact_data
                )
                group = _finger_loose_contact_group(
                    paths, robot_root, body_root, nut_root
                )
                if group is None:
                    raise RuntimeError(
                        "finger contact cannot be assigned to one loose body"
                    )
                finger, loose_group = group
                snapshot["finger_body_group_records"][finger][
                    loose_group
                ] += int(header.num_contact_data)
                for index in range(
                    header.contact_data_offset,
                    header.contact_data_offset + header.num_contact_data,
                ):
                    contact = contacts[index]
                    materials = (
                        str(
                            PhysicsSchemaTools.intToSdfPath(contact.material0)
                        ),
                        str(
                            PhysicsSchemaTools.intToSdfPath(contact.material1)
                        ),
                    )
                    if materials == (
                        grip_material_path,
                        grip_material_path,
                    ):
                        snapshot["grip_material_records"] += 1
            return snapshot

        def loose_fixed_contact_count():
            """Count loose/fixed records for the pre-entry zero gate."""

            headers, _, _ = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            records = 0
            for header in headers:
                paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                if contact_pair_crosses_prim_roots(
                    paths,
                    plug_root,
                    tabletop.asset.fixed_receptacle_prim_path,
                ):
                    records += int(header.num_contact_data)
            return records

        def observe_and_step(
            arm_target,
            hand_target,
            allow_loose_contact,
            enforce_preinsert_gates=False,
            allow_loose_fixed_contact=False,
        ):
            nonlocal finite_throughout
            nonlocal global_step
            nonlocal grip_material_contact_records
            nonlocal maximum_arm_tracking_error
            nonlocal maximum_joint_limit_violation
            nonlocal maximum_joint_speed
            nonlocal preinsert_checked_steps
            nonlocal preinsert_loose_fixed_contact_records
            nonlocal preinsert_minimum_body_contact_fingers
            nonlocal latest_loose_fixed_contact_records
            nonlocal latest_preinsert_body_contact_fingers
            nonlocal latest_intended_lip_contact_pairs
            nonlocal latest_unexpected_loose_fixed_contact_pairs
            nonlocal latest_loose_fixture_contact_pairs
            nonlocal latest_loose_table_contact_pairs
            nonlocal latest_wrist_ft_sample
            nonlocal first_tactile_forbidden_contact
            target = np.concatenate((arm_target, hand_target)).astype(
                np.float32
            )
            robot.apply_action(
                ArticulationAction(
                    joint_positions=target,
                    joint_indices=controlled_indices,
                )
            )
            world.step(render=arguments.gui)
            global_step += 1
            if wrist_ft_monitor is not None:
                raw_wrench = np.asarray(
                    robot.get_measured_joint_forces(
                        joint_indices=np.asarray(
                            [wrist_ft_reaction_row], dtype=np.int32
                        )
                    ),
                    dtype=np.float64,
                )
                if raw_wrench.shape != (1, 6):
                    raise RuntimeError(
                        "unexpected hand2arm reaction wrench shape: "
                        f"{raw_wrench.shape}"
                    )
                sensor_matrix = UsdGeom.Xformable(
                    wrist_ft_sensor_prim
                ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                sensor_transform = Gf.Transform(sensor_matrix)
                # Gf transforms row vectors while the pure wrench helper uses
                # NumPy column vectors.  The shared adapter performs the one
                # required transpose and validates a right-handed rotation.
                sensor_rotation = column_rotation_from_gf_matrix3d(
                    np.asarray(
                        Gf.Matrix3d(sensor_transform.GetRotation()),
                        dtype=np.float64,
                    )
                )
                latest_wrist_ft_sample = wrist_ft_monitor.observe(
                    raw_wrench[0],
                    global_step=global_step,
                    runtime_phase=phase,
                    sensor_position_world=np.asarray(
                        sensor_transform.GetTranslation(),
                        dtype=np.float64,
                    ),
                    sensor_rotation_world=sensor_rotation,
                )
            positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            body_position, body_orientation = body.get_world_pose()
            nut_position, nut_orientation = nut.get_world_pose()
            sample = np.concatenate(
                (
                    positions,
                    velocities,
                    np.asarray(body_position, dtype=np.float64),
                    np.asarray(body_orientation, dtype=np.float64),
                    np.asarray(nut_position, dtype=np.float64),
                    np.asarray(nut_orientation, dtype=np.float64),
                    np.asarray(body.get_linear_velocity(), dtype=np.float64),
                    np.asarray(body.get_angular_velocity(), dtype=np.float64),
                )
            )
            finite = bool(np.all(np.isfinite(sample)))
            finite_throughout = bool(finite_throughout and finite)
            if not finite:
                raise RuntimeError(
                    f"non-finite state in {phase} at step {global_step}"
                )
            maximum_joint_speed = max(
                maximum_joint_speed, float(np.max(np.abs(velocities)))
            )
            maximum_arm_tracking_error = max(
                maximum_arm_tracking_error,
                float(np.max(np.abs(positions[arm_indices] - arm_target))),
            )
            for index in range(robot.num_dof):
                if bool(dof_properties[index]["hasLimits"]):
                    maximum_joint_limit_violation = max(
                        maximum_joint_limit_violation,
                        float(dof_properties[index]["lower"])
                        - float(positions[index]),
                        float(positions[index])
                        - float(dof_properties[index]["upper"]),
                    )
            headers, contacts, _ = (
                get_physx_simulation_interface().get_full_contact_report()
            )
            body_contact_fingers = set()
            step_loose_fixed_records = 0
            step_intended_lip_contact_pairs = []
            step_unexpected_loose_fixed_contact_pairs = []
            step_loose_fixture_contact_pairs = []
            step_loose_table_contact_pairs = []
            for header in headers:
                paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                tactile_pair_class = None
                environment_category = None
                if enforce_preinsert_gates:
                    environment_category = classify_loose_environment_contact(
                        paths,
                        plug_root,
                        tabletop.asset.fixed_receptacle_prim_path,
                        tabletop.fixed_endpoint.fixture_prim_path,
                        tabletop.table.prim_path,
                    )
                    tactile_pair_class = classify_tactile_lip_contact_pair(
                        paths,
                        body_root,
                        nut_root,
                        tabletop.asset.fixed_receptacle_prim_path,
                    )
                    if (
                        environment_category == "loose_fixed"
                        and tactile_pair_class is None
                    ):
                        tactile_pair_class = "unexpected_loose_fixed"
                    if environment_category in {
                        "loose_fixture",
                        "loose_table",
                    }:
                        pair_evidence = {
                            "paths": list(paths),
                            "contact_records": int(header.num_contact_data),
                        }
                        target = (
                            step_loose_fixture_contact_pairs
                            if environment_category == "loose_fixture"
                            else step_loose_table_contact_pairs
                        )
                        target.append(pair_evidence)
                        if first_tactile_forbidden_contact is None:
                            first_tactile_forbidden_contact = {
                                "category": environment_category,
                                "contact_record_count": int(
                                    header.num_contact_data
                                ),
                                "global_step": global_step,
                                "paths": list(paths),
                                "phase": phase,
                            }
                        # This explicit loose/environment check complements
                        # the robot-only external-contact classifier below.
                        raise RuntimeError(
                            "forbidden loose/environment contact in "
                            f"{phase}: category={environment_category}, "
                            f"paths={paths}"
                        )
                if enforce_preinsert_gates and tactile_pair_class is not None:
                    records = int(header.num_contact_data)
                    step_loose_fixed_records += records
                    preinsert_loose_fixed_contact_records += records
                    pair_evidence = {
                        "paths": list(paths),
                        "contact_records": records,
                    }
                    if tactile_pair_class == "intended_segmented_lip":
                        if (
                            arguments.tactile_lip_manifold_capture
                            and records > 0
                        ):
                            if (
                                tactile_contact_rotation is None
                                or tactile_contact_origin is None
                            ):
                                raise TactileSafetyStop(
                                    "manifold contact lacks task-frame "
                                    "provenance",
                                    zero_step_abort=True,
                                )
                            # Copy this header's exact contact-data slice in
                            # the same physics-report query that classified
                            # actor/collider paths.  The PhysX buffers are
                            # invalidated by the next simulation step.
                            try:
                                point_records = []
                                for contact_index in range(
                                    header.contact_data_offset,
                                    header.contact_data_offset
                                    + header.num_contact_data,
                                ):
                                    contact = contacts[contact_index]
                                    point_records.append(
                                        {
                                            "impulse": [
                                                float(value)
                                                for value in contact.impulse
                                            ],
                                            "normal": [
                                                float(value)
                                                for value in contact.normal
                                            ],
                                            "position": [
                                                float(value)
                                                for value in contact.position
                                            ],
                                            "separation": float(
                                                contact.separation
                                            ),
                                            "material0": str(
                                                PhysicsSchemaTools
                                                .intToSdfPath(
                                                    contact.material0
                                                )
                                            ),
                                            "material1": str(
                                                PhysicsSchemaTools
                                                .intToSdfPath(
                                                    contact.material1
                                                )
                                            ),
                                            "face_index0": int(
                                                contact.face_index0
                                            ),
                                            "face_index1": int(
                                                contact.face_index1
                                            ),
                                        }
                                    )
                                pair_evidence["contact_manifold"] = (
                                    build_tactile_manifold_pair_evidence(
                                        actor_paths=paths[:2],
                                        collider_paths=paths[2:],
                                        event_type=header.type,
                                        contact_points=point_records,
                                        body_root=body_root,
                                        fixed_root=(
                                            tabletop.asset
                                            .fixed_receptacle_prim_path
                                        ),
                                        task_rotation_world=(
                                            tactile_contact_rotation
                                        ),
                                        task_origin_world=(
                                            tactile_contact_origin
                                        ),
                                        physics_dt_s=1.0 / float(rate_hz),
                                    )
                                )
                            except (
                                IndexError,
                                TypeError,
                                ValueError,
                            ) as error:
                                raise TactileSafetyStop(
                                    "malformed/non-finite manifold contact "
                                    f"evidence: {error}",
                                    zero_step_abort=True,
                                )
                        step_intended_lip_contact_pairs.append(pair_evidence)
                    else:
                        step_unexpected_loose_fixed_contact_pairs.append(
                            pair_evidence
                        )
                    if (
                        not allow_loose_fixed_contact
                        or tactile_pair_class
                        != "intended_segmented_lip"
                    ):
                        if first_tactile_forbidden_contact is None:
                            first_tactile_forbidden_contact = {
                                "category": tactile_pair_class,
                                "contact_record_count": records,
                                "global_step": global_step,
                                "paths": list(paths),
                                "phase": phase,
                            }
                        raise RuntimeError(
                            "disallowed loose/fixed contact in "
                            f"{phase}: class={tactile_pair_class}, "
                            f"paths={paths}"
                        )
                category = _classify_robot_external_contact(
                    paths,
                    robot_root,
                    tabletop.table.prim_path,
                    tabletop.fixed_endpoint.fixture_prim_path,
                    tabletop.asset.fixed_receptacle_prim_path,
                    plug_root,
                )
                if category is None:
                    continue
                key = category
                allowed = False
                if category == "loose_plug":
                    if not allow_loose_contact:
                        key += "_preclosure"
                    elif _is_finger_plug_contact(
                        paths, robot_root, plug_root
                    ):
                        key += "_allowed"
                        allowed = True
                    else:
                        key += "_unexpected_robot_link"
                external_contacts[key] += int(header.num_contact_data)
                if not allowed:
                    raise RuntimeError(
                        f"forbidden {key} contact in {phase}: {paths}"
                    )
                if enforce_preinsert_gates:
                    group = _finger_loose_contact_group(
                        paths, robot_root, body_root, nut_root
                    )
                    if group is None:
                        raise RuntimeError(
                            "preinsert finger contact has no loose-body group"
                        )
                    finger, loose_group = group
                    if loose_group == "body":
                        body_contact_fingers.add(finger)
                for index in range(
                    header.contact_data_offset,
                    header.contact_data_offset + header.num_contact_data,
                ):
                    contact = contacts[index]
                    materials = (
                        str(
                            PhysicsSchemaTools.intToSdfPath(contact.material0)
                        ),
                        str(
                            PhysicsSchemaTools.intToSdfPath(contact.material1)
                        ),
                    )
                    if materials == (
                        grip_material_path,
                        grip_material_path,
                    ):
                        grip_material_contact_records += 1
            if enforce_preinsert_gates:
                preinsert_checked_steps += 1
                latest_loose_fixed_contact_records = (
                    step_loose_fixed_records
                )
                latest_intended_lip_contact_pairs = tuple(
                    step_intended_lip_contact_pairs
                )
                latest_unexpected_loose_fixed_contact_pairs = tuple(
                    step_unexpected_loose_fixed_contact_pairs
                )
                latest_loose_fixture_contact_pairs = tuple(
                    step_loose_fixture_contact_pairs
                )
                latest_loose_table_contact_pairs = tuple(
                    step_loose_table_contact_pairs
                )
                latest_preinsert_body_contact_fingers = frozenset(
                    body_contact_fingers
                )
                preinsert_minimum_body_contact_fingers = min(
                    preinsert_minimum_body_contact_fingers,
                    len(body_contact_fingers),
                )
                if len(body_contact_fingers) != 3:
                    contact_loss_reason = (
                        "all three fingers must retain BodyAssembly contact "
                        "through preinsert; observed="
                        f"{sorted(body_contact_fingers)}"
                    )
                    if tactile_runtime_requested:
                        # A lost grasp is already an unsafe observation.  Do
                        # not take even one recovery physics step from an
                        # unknown connector pose.
                        raise TactileSafetyStop(
                            contact_loss_reason, zero_step_abort=True
                        )
                    raise RuntimeError(contact_loss_reason)
            return positions, velocities

        def run_arm_motion(
            target,
            duration_s,
            allow_loose_contact,
            sample_torque=False,
            enforce_preinsert_gates=False,
        ):
            nonlocal current_arm
            start = current_arm.copy()
            steps = round(duration_s * rate_hz)
            for index in range(steps):
                current_arm = np.asarray(
                    interpolate_arm(
                        tuple(float(value) for value in start),
                        tuple(float(value) for value in target),
                        float(index + 1) / float(steps),
                    ),
                    dtype=np.float64,
                )
                observe_and_step(
                    current_arm,
                    current_hand,
                    allow_loose_contact,
                    enforce_preinsert_gates,
                )
                if sample_torque:
                    sample_efforts()
            current_arm = np.asarray(target, dtype=np.float64)

        for _ in range(tabletop.physics.settle_steps):
            observe_and_step(current_arm, current_hand, False)
        settled_body_position, _ = body.get_world_pose()
        settled_nut_position, _ = nut.get_world_pose()
        settled_bottom = min(
            float(settled_body_position[2])
            + tabletop.loose_endpoint.body_bottom_offset_m,
            float(settled_nut_position[2])
            + tabletop.loose_endpoint.nut_bottom_offset_m,
        )
        settled_on_table = bool(
            -tabletop.physics.maximum_transient_table_penetration_m
            <= settled_bottom - tabletop.table.top_z_m
            <= tabletop.physics.maximum_final_surface_gap_m
        )

        phase = "visual_xy_preflight"
        capture_world_reference = world
        capture_dir = output_dir / "rgbd_capture"
        capture = capture_d38999_rgbd_runtime(
            bindings={
                "Camera": Camera,
                "Gf": Gf,
                "Image": Image,
                "Usd": Usd,
                "UsdGeom": UsdGeom,
                "UsdLux": UsdLux,
                "add_labels": add_labels,
                "get_labels": get_labels,
                "rep": rep,
            },
            simulation_app=simulation_app,
            world=world,
            stage=stage,
            tabletop=tabletop,
            rgbd=probe.rgbd,
            loose_prim=loose_prim,
            fixed_prim=fixed_prim,
            body=body,
            output_dir=capture_dir,
            pose5d_config=pose5d_config,
            pose5d_capture_id=(
                f"{probe.trial_id}-pose5d-before-grasp"
                if pose5d_config is not None
                else None
            ),
            pose5d_axis_priors=(
                {
                    "loose_plug": (0.0, 0.0, 1.0),
                    "fixed_receptacle": (0.0, 0.0, 1.0),
                }
                if pose5d_config is not None
                else None
            ),
            pose5d_authorization_gates=pose5d_gates,
        )
        runtime_side_effects = validate_rgbd_capture_side_effects(
            capture.metrics
        )
        report["runtime_side_effects"] = runtime_side_effects
        capture_timestamp = float(global_step) / float(rate_hz)
        provider_sample = pose_provider_sample_from_rgbd_metrics(
            probe,
            capture.metrics,
            timestamp_s=capture_timestamp,
            capture_id=f"{probe.trial_id}-capture-001",
        )
        plan = build_visual_xy_pick_plan(
            probe,
            provider_sample,
            now_s=capture_timestamp,
            explicit_probe_opt_in=True,
        )
        cpu_plan = plan.to_mapping()
        cpu_plan_path = output_dir / probe.cpu_plan_filename
        cpu_plan_path.write_text(
            json.dumps(
                cpu_plan, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        preinsert_plan = None
        if preinsert_probe is not None:
            # Build once from the immutable PoseProvider-derived pick plan.
            # The continuation API has no simulator-pose arguments, and the
            # resulting targets are never corrected from later truth reads.
            preinsert_plan = build_visual_xy_preinsert_plan(
                preinsert_probe,
                probe,
                plan,
                explicit_probe_opt_in=True,
            )
            preinsert_cpu_plan = preinsert_plan.to_mapping()
            preinsert_cpu_plan_path = (
                output_dir / preinsert_probe.cpu_plan_filename
            )
            preinsert_cpu_plan_path.write_text(
                json.dumps(
                    preinsert_cpu_plan,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            report.update(
                {
                    "preinsert_cpu_plan": preinsert_cpu_plan,
                    "preinsert_cpu_plan_path": str(
                        preinsert_cpu_plan_path
                    ),
                }
            )
        # Truth is copied only after the plan is immutable and is never passed
        # to the adapter, fixed-q7 solver, or motion functions.
        truth_evaluation = evaluate_visual_xy_truth_only(
            plan,
            loose_truth_xy_m=capture.loose_position_world_m[:2],
            fixed_truth_xy_m=capture.fixed_position_world_m[:2],
        )
        report.update(
            {
                "rgbd_capture": capture.metrics,
                "pose_provider": {
                    "provider_id": provider_sample.provider_id,
                    "purpose": provider_sample.purpose.value,
                    "uses_truth_position": False,
                    "uses_truth_orientation": False,
                    "full_6d": False,
                    "control_authorized": False,
                    "diagnostics": provider_sample.diagnostics,
                },
                "cpu_plan": cpu_plan,
                "cpu_plan_path": str(cpu_plan_path),
                "truth_evaluation": truth_evaluation,
            }
        )

        if wrist_ft_config is not None:
            # Bind the exact hand2arm reaction row proven by the existing
            # virtual-wrist monitor.  Isaac exposes fixed-joint reactions at
            # metadata index + 1, so both the selected and full arrays are
            # compared before any sample can be accepted.
            metadata_joint_indices = dict(
                robot._articulation_view._metadata.joint_indices
            )
            wrist_ft_reaction_row = reaction_row_index(
                metadata_joint_indices, wrist_ft_config
            )
            selected_wrench = np.asarray(
                robot.get_measured_joint_forces(
                    joint_indices=np.asarray(
                        [wrist_ft_reaction_row], dtype=np.int32
                    )
                ),
                dtype=np.float64,
            )
            all_wrenches = np.asarray(
                robot.get_measured_joint_forces(), dtype=np.float64
            )
            if (
                selected_wrench.shape != (1, 6)
                or wrist_ft_reaction_row >= all_wrenches.shape[0]
                or not np.array_equal(
                    selected_wrench[0],
                    all_wrenches[wrist_ft_reaction_row],
                )
            ):
                raise RuntimeError(
                    "hand2arm reaction row failed joint-index-plus-one check"
                )
            hand2arm_joints = [
                prim
                for prim in stage.Traverse()
                if prim.GetName() == wrist_ft_config.measurement_joint
                and prim.IsA(UsdPhysics.FixedJoint)
            ]
            if len(hand2arm_joints) != 1:
                raise RuntimeError(
                    "expected exactly one fixed hand2arm measurement joint"
                )
            body1_targets = [
                str(path)
                for path in UsdPhysics.FixedJoint(
                    hand2arm_joints[0]
                ).GetBody1Rel().GetTargets()
            ]
            if (
                len(body1_targets) != 1
                or not body1_targets[0].endswith(
                    "/" + wrist_ft_config.raw_frame
                )
            ):
                raise RuntimeError(
                    "hand2arm child does not match configured raw frame"
                )
            wrist_ft_sensor_prim = stage.GetPrimAtPath(body1_targets[0])
            if not wrist_ft_sensor_prim.IsValid():
                raise RuntimeError("wrist FT sensor-frame prim is missing")

            registered_fixed = np.asarray(
                preinsert_probe.assembly.datums.fixed.position_world_m,
                dtype=np.float64,
            )
            visual_task_origin = registered_fixed.copy()
            visual_task_origin[:2] += np.asarray(
                preinsert_plan.fixed_translation_xy_m, dtype=np.float64
            )
            wrist_ft_monitor = VirtualWristFtMonitor(
                wrist_ft_config,
                reaction_row=wrist_ft_reaction_row,
                task_origin_world=visual_task_origin,
                task_z_axis_world=(
                    preinsert_probe.assembly.datums.fixed.axis_world
                ),
            )
            wrist_runtime_report = (
                report["wrist_ft_guarded_insertion"]
                if wrist_ft_guarded_requested
                else report[tactile_report_key]
            )
            wrist_runtime_report.update(
                {
                    "status": "CAPTURING_HOME_EMPTY_TARE",
                    "measurement_joint": wrist_ft_config.measurement_joint,
                    "metadata_joint_index": int(
                        metadata_joint_indices[
                            wrist_ft_config.measurement_joint
                        ]
                    ),
                    "reaction_row_index": wrist_ft_reaction_row,
                    "joint_index_plus_one_verified": True,
                    "task_origin_world_m": [
                        float(value) for value in visual_task_origin
                    ],
                    "task_origin_source": (
                        "registered_fixed_datum_plus_visual_fixed_xy"
                    ),
                }
            )
            phase = "initial_settle"
            for _ in range(wrist_ft_config.home_tare_window_steps):
                observe_and_step(current_arm, current_hand, False)
            home_tare = wrist_ft_monitor.capture_home_tare()
            wrist_runtime_report.update(
                {
                    "status": "HOME_EMPTY_TARE_READY",
                    "home_empty_tare_canonical": home_tare,
                    "home_tare_samples": (
                        wrist_ft_config.home_tare_window_steps
                    ),
                }
            )

        phase = "home_hand_open"
        hand_start = current_hand.copy()
        hand_steps = round(pick.motion.hand_open_duration_s * rate_hz)
        for index in range(hand_steps):
            blend = minimum_jerk_blend(
                float(index + 1) / float(hand_steps)
            )
            current_hand = hand_start + blend * (open_hand - hand_start)
            observe_and_step(current_arm, current_hand, False)
        current_hand = open_hand.copy()

        # Reuse the two broad nominal safe segments.  Only the final nearby
        # pregrasp/grasp IK targets come from visual XY.
        for segment in pick.motion.approach_segments[:-1]:
            phase = segment.name
            run_arm_motion(segment.target_arm_rad, segment.duration_s, False)
        phase = "visual_xy_high_approach_to_pregrasp"
        run_arm_motion(
            plan.arm_targets_rad["pregrasp"],
            pick.motion.approach_segments[-1].duration_s,
            False,
        )
        phase = "visual_xy_pregrasp_hold"
        for _ in range(
            round(pick.motion.pregrasp_hold_duration_s * rate_hz)
        ):
            observe_and_step(current_arm, current_hand, False)
        phase = "visual_xy_open_hand_descent"
        run_arm_motion(
            plan.arm_targets_rad["closure_clearance"],
            pick.motion.descent_duration_s,
            False,
        )

        kps[hand_indices] = pick.motion.grip_hand_stiffness
        kds[hand_indices] = pick.motion.grip_hand_damping
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        phase = "visual_xy_open_grasp_tare"
        tare_samples = []
        for _ in range(round(pick.motion.open_tare_duration_s * rate_hz)):
            observe_and_step(current_arm, current_hand, False)
            tare_samples.append(
                np.asarray(
                    robot.get_measured_joint_efforts(
                        joint_indices=sensor_indices
                    ),
                    dtype=np.float64,
                )
            )
        tare_efforts = np.mean(np.stack(tare_samples), axis=0)

        def sample_efforts():
            measured = np.asarray(
                robot.get_measured_joint_efforts(
                    joint_indices=sensor_indices
                ),
                dtype=np.float64,
            )
            delta = measured - tare_efforts
            if not np.all(np.isfinite(delta)):
                raise RuntimeError("non-finite finger torque delta")
            np.maximum(
                maximum_post_tare_delta,
                np.abs(delta),
                out=maximum_post_tare_delta,
            )
            if np.any(
                np.abs(delta)
                > pick.sensing.maximum_absolute_torque_delta_nm
            ):
                raise RuntimeError("2 Nm finger torque hard stop exceeded")
            return measured

        closure_tcp_position, closure_tcp_orientation = _world_pose(
            Gf, Usd, UsdGeom, tcp_prim
        )
        closure_tcp_position_error = float(
            np.linalg.norm(
                np.asarray(closure_tcp_position, dtype=np.float64)
                - np.asarray(
                    plan.tcp_targets_world_m["closure_clearance_tcp"],
                    dtype=np.float64,
                )
            )
        )
        closure_axis_error = _axis_error_radians(
            _quaternion_world_z_axis(closure_tcp_orientation),
            pick.motion.grasp_tcp_down_axis_world,
        )
        phase = "visual_xy_physical_hand_closure"
        hand_start = current_hand.copy()
        closure_steps = round(pick.motion.closure_duration_s * rate_hz)
        for index in range(closure_steps):
            blend = minimum_jerk_blend(
                float(index + 1) / float(closure_steps)
            )
            current_hand = hand_start + blend * (grasp_hand - hand_start)
            observe_and_step(current_arm, current_hand, True)
            sample_efforts()
        current_hand = grasp_hand.copy()
        phase = "visual_xy_closed_hand_seating"
        run_arm_motion(
            plan.arm_targets_rad["grasp"],
            pick.motion.closed_seating_duration_s,
            True,
            sample_torque=True,
        )
        grasp_tcp_position, grasp_tcp_orientation = _world_pose(
            Gf, Usd, UsdGeom, tcp_prim
        )
        grasp_tcp_position_error = float(
            np.linalg.norm(
                np.asarray(grasp_tcp_position, dtype=np.float64)
                - np.asarray(
                    plan.tcp_targets_world_m["grasp_tcp"],
                    dtype=np.float64,
                )
            )
        )
        grasp_axis_error = _axis_error_radians(
            _quaternion_world_z_axis(grasp_tcp_orientation),
            pick.motion.grasp_tcp_down_axis_world,
        )
        phase = "visual_xy_grip_preload"
        preload_samples = []
        for _ in range(round(pick.motion.preload_duration_s * rate_hz)):
            observe_and_step(current_arm, current_hand, True)
            preload_samples.append(sample_efforts())
        contact_efforts = np.mean(np.stack(preload_samples), axis=0)
        contact_torque_deltas = contact_efforts - tare_efforts
        loaded_channels = int(
            np.count_nonzero(
                np.abs(contact_torque_deltas)
                >= pick.sensing.loaded_torque_threshold_nm
            )
        )
        postclosure_body_position, _ = body.get_world_pose()
        postclosure_nut_position, _ = nut.get_world_pose()
        postclosure_body_in_tcp = body_in_tcp_frame(
            postclosure_body_position
        )
        postclosure_separation = float(
            np.linalg.norm(
                postclosure_nut_position - postclosure_body_position
            )
        )
        postclosure_contacts = contact_snapshot()

        phase = "visual_xy_grip_lift"
        run_arm_motion(
            plan.arm_targets_rad["pregrasp"],
            pick.motion.lift_duration_s,
            True,
            sample_torque=True,
        )
        phase = "visual_xy_unsupported_hold"
        hold_start_body, _ = body.get_world_pose()
        maximum_hold_displacement = 0.0
        tail_positions = []
        tail_velocities = []
        tail_body_positions = []
        tail_body_orientations = []
        final_effort_samples = []
        final_hold_steps = round(pick.motion.final_hold_duration_s * rate_hz)
        tail_steps = min(120, final_hold_steps)
        effort_steps = round(pick.motion.effort_sample_duration_s * rate_hz)
        for index in range(final_hold_steps):
            positions, velocities = observe_and_step(
                current_arm, current_hand, True
            )
            measured_effort = sample_efforts()
            current_body_position, current_body_orientation = (
                body.get_world_pose()
            )
            maximum_hold_displacement = max(
                maximum_hold_displacement,
                float(
                    np.linalg.norm(
                        current_body_position - hold_start_body
                    )
                ),
            )
            if index >= final_hold_steps - tail_steps:
                tail_positions.append(positions.copy())
                tail_velocities.append(velocities.copy())
                tail_body_positions.append(
                    np.asarray(current_body_position, dtype=np.float64)
                )
                tail_body_orientations.append(
                    np.asarray(current_body_orientation, dtype=np.float64)
                )
            if index >= final_hold_steps - effort_steps:
                final_effort_samples.append(measured_effort.copy())

        final_positions = np.asarray(
            robot.get_joint_positions(), dtype=np.float64
        )
        final_velocities = np.asarray(
            robot.get_joint_velocities(), dtype=np.float64
        )
        final_body_position, final_body_orientation = body.get_world_pose()
        final_nut_position, final_nut_orientation = nut.get_world_pose()
        final_contacts = contact_snapshot()
        final_body_in_tcp = body_in_tcp_frame(final_body_position)
        body_tcp_slip = float(
            np.linalg.norm(final_body_in_tcp - postclosure_body_in_tcp)
        )
        body_nut_separation_change = abs(
            float(
                np.linalg.norm(final_nut_position - final_body_position)
                - postclosure_separation
            )
        )
        body_lift = float(
            final_body_position[2] - settled_body_position[2]
        )
        final_bottom = min(
            float(final_body_position[2])
            + tabletop.loose_endpoint.body_bottom_offset_m,
            float(final_nut_position[2])
            + tabletop.loose_endpoint.nut_bottom_offset_m,
        )
        final_bottom_clearance = final_bottom - tabletop.table.top_z_m
        final_efforts = np.mean(np.stack(final_effort_samples), axis=0)
        final_torque_deltas = final_efforts - tare_efforts
        final_loaded_channels = int(
            np.count_nonzero(
                np.abs(final_torque_deltas)
                >= pick.sensing.loaded_torque_threshold_nm
            )
        )
        tail_positions_array = np.stack(tail_positions)
        tail_velocities_array = np.stack(tail_velocities)
        pose_joint_speeds = np.diff(tail_positions_array, axis=0) * rate_hz
        body_linear_speeds = np.linalg.norm(
            np.diff(np.stack(tail_body_positions), axis=0), axis=1
        ) * rate_hz
        body_angular_speeds = np.asarray(
            [
                _array_quaternion_error_radians(first, second) * rate_hz
                for first, second in zip(
                    tail_body_orientations[:-1],
                    tail_body_orientations[1:],
                )
            ]
        )
        final_observable_joint_speed = float(
            np.max(np.abs(pose_joint_speeds))
        )
        final_solver_joint_speed = float(
            np.max(np.abs(tail_velocities_array))
        )
        final_body_observable_linear_speed = float(
            np.max(body_linear_speeds)
        )
        final_body_observable_angular_speed = float(
            np.max(body_angular_speeds)
        )
        fixed_final_position, fixed_final_orientation = _world_pose(
            Gf, Usd, UsdGeom, fixed_prim
        )
        fixed_translation_drift = float(
            np.linalg.norm(
                np.asarray(fixed_final_position, dtype=np.float64)
                - np.asarray(fixed_initial_position, dtype=np.float64)
            )
        )
        fixed_rotation_drift = _gf_quaternion_error_radians(
            fixed_initial_orientation, fixed_final_orientation
        )
        finite_final = bool(
            np.all(np.isfinite(final_positions))
            and np.all(np.isfinite(final_velocities))
            and np.all(np.isfinite(final_body_position))
            and np.all(np.isfinite(final_body_orientation))
            and np.all(np.isfinite(final_nut_position))
            and np.all(np.isfinite(final_nut_orientation))
            and np.all(np.isfinite(tare_efforts))
            and np.all(np.isfinite(contact_efforts))
            and np.all(np.isfinite(final_efforts))
        )
        acceptance = pick.acceptance
        final_tracking_error = float(
            np.max(np.abs(final_positions[arm_indices] - current_arm))
        )
        zero_forbidden_contacts = bool(
            external_contacts["table"] == 0
            and external_contacts["fixture"] == 0
            and external_contacts["fixed_endpoint"] == 0
            and external_contacts["loose_plug_preclosure"] == 0
            and external_contacts["loose_plug_unexpected_robot_link"] == 0
        )
        torque_gate = bool(
            loaded_channels >= pick.sensing.minimum_loaded_channels
            and final_loaded_channels >= pick.sensing.minimum_loaded_channels
            and np.max(np.abs(contact_torque_deltas))
            <= pick.sensing.maximum_absolute_torque_delta_nm
            and np.max(np.abs(final_torque_deltas))
            <= pick.sensing.maximum_absolute_torque_delta_nm
            and np.max(maximum_post_tare_delta)
            <= pick.sensing.maximum_absolute_torque_delta_nm
        )
        contact_gate = bool(
            grip_material_contact_records > 0
            and postclosure_contacts["grip_material_records"] > 0
            and final_contacts["grip_material_records"] > 0
            and final_contacts["unexpected_robot_link_records"] == 0
            and _all_fingers_have_body_contact(postclosure_contacts)
            and _all_fingers_have_body_contact(final_contacts)
        )
        unsupported_gate = bool(
            final_contacts["plug_table_records"] == 0
            and body_lift >= acceptance.minimum_body_lift_m
            and final_bottom_clearance
            >= acceptance.minimum_final_bottom_clearance_m
        )
        truth_xy_evaluation_gate = bool(
            truth_evaluation["loose_xy_error_m"]
            <= probe.per_capture_xy_error_bound_m
            and truth_evaluation["fixed_xy_error_m"]
            <= probe.per_capture_xy_error_bound_m
        )
        # This prerequisite contains only the visual plan, robot
        # proprioception/FK and gripper effort.  It deliberately excludes
        # object truth and PhysX contact identities so the wrist-FT insertion
        # controller cannot inherit simulator-only authority.
        visual_pick_control_ready = bool(
            finite_throughout
            and finite_final
            and plan.adapter_result.eligible_for_independent_probe
            and maximum_joint_limit_violation
            <= acceptance.maximum_joint_limit_violation_rad
            and maximum_joint_speed
            <= acceptance.maximum_observed_joint_speed_rad_s
            and maximum_arm_tracking_error
            <= acceptance.maximum_arm_tracking_error_rad
            and final_tracking_error
            <= acceptance.maximum_endpoint_arm_tracking_error_rad
            and closure_tcp_position_error
            <= acceptance.maximum_grasp_tcp_position_error_m
            and closure_axis_error
            <= acceptance.maximum_grasp_tcp_axis_error_rad
            and grasp_tcp_position_error
            <= acceptance.maximum_grasp_tcp_position_error_m
            and grasp_axis_error
            <= acceptance.maximum_grasp_tcp_axis_error_rad
            and torque_gate
        )
        passed = bool(
            finite_throughout
            and finite_final
            and settled_on_table
            and plan.adapter_result.eligible_for_independent_probe
            and truth_xy_evaluation_gate
            and maximum_joint_limit_violation
            <= acceptance.maximum_joint_limit_violation_rad
            and maximum_joint_speed
            <= acceptance.maximum_observed_joint_speed_rad_s
            and maximum_arm_tracking_error
            <= acceptance.maximum_arm_tracking_error_rad
            and final_tracking_error
            <= acceptance.maximum_endpoint_arm_tracking_error_rad
            and final_observable_joint_speed
            <= acceptance.maximum_final_observable_joint_speed_rad_s
            and final_solver_joint_speed
            <= acceptance.maximum_final_post_solver_joint_speed_rad_s
            and closure_tcp_position_error
            <= acceptance.maximum_grasp_tcp_position_error_m
            and closure_axis_error
            <= acceptance.maximum_grasp_tcp_axis_error_rad
            and grasp_tcp_position_error
            <= acceptance.maximum_grasp_tcp_position_error_m
            and grasp_axis_error
            <= acceptance.maximum_grasp_tcp_axis_error_rad
            and torque_gate
            and contact_gate
            and zero_forbidden_contacts
            and unsupported_gate
            and body_tcp_slip <= acceptance.maximum_body_tcp_slip_m
            and body_nut_separation_change
            <= acceptance.maximum_body_nut_separation_change_m
            and maximum_hold_displacement
            <= acceptance.maximum_final_hold_displacement_m
            and final_body_observable_linear_speed
            <= acceptance.maximum_final_body_observable_linear_speed_m_s
            and final_body_observable_angular_speed
            <= acceptance.maximum_final_body_observable_angular_speed_rad_s
            and float(np.linalg.norm(body.get_linear_velocity()))
            <= acceptance.maximum_final_body_post_solver_linear_speed_m_s
            and float(np.linalg.norm(body.get_angular_velocity()))
            <= acceptance.maximum_final_body_post_solver_angular_speed_rad_s
            and fixed_translation_drift
            <= acceptance.maximum_fixed_translation_drift_m
            and fixed_rotation_drift
            <= acceptance.maximum_fixed_rotation_drift_rad
        )
        report.update(
            {
                "passed": passed,
                "finite_throughout": finite_throughout,
                "finite_final": finite_final,
                "settled_on_table": settled_on_table,
                "truth_xy_evaluation_gate": truth_xy_evaluation_gate,
                "visual_pick_control_ready_without_truth_or_physx_contact": (
                    visual_pick_control_ready
                ),
                "maximum_joint_speed_rad_s": maximum_joint_speed,
                "maximum_arm_tracking_error_rad": (
                    maximum_arm_tracking_error
                ),
                "maximum_joint_limit_violation_rad": (
                    maximum_joint_limit_violation
                ),
                "closure_tcp_position_error_m": (
                    closure_tcp_position_error
                ),
                "closure_tcp_axis_error_rad": closure_axis_error,
                "grasp_tcp_position_error_m": grasp_tcp_position_error,
                "grasp_tcp_axis_error_rad": grasp_axis_error,
                "contact_torque_deltas_nm": {
                    name: float(value)
                    for name, value in zip(
                        pick.sensing.torque_joint_names,
                        contact_torque_deltas,
                    )
                },
                "final_torque_deltas_nm": {
                    name: float(value)
                    for name, value in zip(
                        pick.sensing.torque_joint_names,
                        final_torque_deltas,
                    )
                },
                "maximum_post_tare_absolute_delta_nm": float(
                    np.max(maximum_post_tare_delta)
                ),
                "loaded_torque_channels": loaded_channels,
                "final_loaded_torque_channels": final_loaded_channels,
                "torque_gate": torque_gate,
                "contact_gate": contact_gate,
                "zero_forbidden_contacts": zero_forbidden_contacts,
                "external_contact_records": external_contacts,
                "postclosure_contacts": postclosure_contacts,
                "final_contacts": final_contacts,
                "body_lift_m": body_lift,
                "final_bottom_clearance_m": final_bottom_clearance,
                "body_tcp_slip_m": body_tcp_slip,
                "body_nut_separation_change_m": (
                    body_nut_separation_change
                ),
                "final_hold_displacement_m": maximum_hold_displacement,
                "final_observable_joint_speed_rad_s": (
                    final_observable_joint_speed
                ),
                "final_solver_joint_speed_rad_s": (
                    final_solver_joint_speed
                ),
                "final_body_observable_linear_speed_m_s": (
                    final_body_observable_linear_speed
                ),
                "final_body_observable_angular_speed_rad_s": (
                    final_body_observable_angular_speed
                ),
                "fixed_translation_drift_m": fixed_translation_drift,
                "fixed_rotation_drift_rad": fixed_rotation_drift,
                "unsupported_gate": unsupported_gate,
            }
        )
        if preinsert_plan is not None:
            # The continuation is allowed to start only after the complete
            # visual-pick finite/contact/torque/lift/hold gate has passed.
            # Simulator pose is never used to change any of the three immutable
            # visual targets; contact and torque may only abort the motion.
            prior_visual_pick_passed = bool(passed)
            prior_visual_control_ready = bool(visual_pick_control_ready)
            preinsert_prerequisite_ready = bool(
                prior_visual_control_ready
                if wrist_ft_guarded_requested
                else prior_visual_pick_passed
            )
            report["preinsert_probe"] = {
                "status": "RUNNING",
                "passed": False,
                "prior_visual_pick_passed": prior_visual_pick_passed,
                "prior_visual_control_ready_without_truth_or_physx_contact": (
                    prior_visual_control_ready
                ),
                "engage_executed": False,
                "insertion_executed": False,
                "twist_executed": False,
                "home_return_executed": False,
            }
            if not preinsert_prerequisite_ready:
                raise RuntimeError(
                    "visual XY preinsert prerequisite was not control-ready"
                )

            preinsert_start_step = global_step
            insertion_motion = preinsert_probe.insertion.motion
            continuation_targets = (
                (
                    "visual_xy_transport_to_fixed_safe",
                    "transport_safe",
                    insertion_motion.transport_duration_s,
                    0.0,
                ),
                (
                    "visual_xy_align_above_entry",
                    "axis_high",
                    insertion_motion.axis_high_duration_s,
                    insertion_motion.axis_high_hold_s,
                ),
                (
                    "visual_xy_preinsert",
                    "preinsert",
                    insertion_motion.preinsert_duration_s,
                    insertion_motion.preinsert_hold_s,
                ),
            )
            for phase_name, target_name, duration_s, hold_s in (
                continuation_targets
            ):
                phase = phase_name
                run_arm_motion(
                    preinsert_plan.arm_targets_rad[target_name],
                    duration_s,
                    True,
                    sample_torque=True,
                    enforce_preinsert_gates=True,
                )
                if hold_s > 0.0:
                    phase = f"{phase_name}_hold"
                    for _ in range(round(hold_s * rate_hz)):
                        observe_and_step(
                            current_arm,
                            current_hand,
                            True,
                            True,
                        )
                        sample_efforts()

            # Reobserve both mating endpoints after the plug is grasped and
            # stopped at the 12 mm waypoint.  The plug prior comes only from
            # TCP FK plus the registered opposite-axis hand/object relation;
            # no object pose is passed to the estimator.
            pose5d_reobserve_authorized = True
            post_grasp_capture = None
            if pose5d_config is not None:
                phase = "post_grasp_pose5d_reobserve"
                tcp_transform = np.asarray(
                    iiwa14_grasp_tcp_transform(
                        tuple(float(value) for value in current_arm)
                    ),
                    dtype=np.float64,
                )
                plug_axis_prior = -tcp_transform[:3, 2]
                post_grasp_capture = capture_d38999_rgbd_runtime(
                    bindings={
                        "Camera": Camera,
                        "Gf": Gf,
                        "Image": Image,
                        "Usd": Usd,
                        "UsdGeom": UsdGeom,
                        "UsdLux": UsdLux,
                        "add_labels": add_labels,
                        "get_labels": get_labels,
                        "rep": rep,
                    },
                    simulation_app=simulation_app,
                    world=world,
                    stage=stage,
                    tabletop=tabletop,
                    rgbd=probe.rgbd,
                    loose_prim=loose_prim,
                    fixed_prim=fixed_prim,
                    body=body,
                    output_dir=output_dir / "rgbd_reobserve_preinsert",
                    pose5d_config=pose5d_config,
                    pose5d_capture_id=(
                        f"{probe.trial_id}-pose5d-after-grasp-preinsert"
                    ),
                    pose5d_axis_priors={
                        "loose_plug": tuple(
                            float(value) for value in plug_axis_prior
                        ),
                        "fixed_receptacle": (0.0, 0.0, 1.0),
                    },
                    pose5d_authorization_gates=pose5d_gates,
                )
                fixed_post_grasp_capture = post_grasp_capture
                relative_pose5d = post_grasp_capture.metrics.get(
                    "pose5d", {}
                ).get("relative_receptacle_plug", {})
                pose5d_reobserve_authorized = bool(
                    post_grasp_capture.passed
                    and relative_pose5d.get("control_authorized") is True
                    and float(relative_pose5d.get("lateral_error_m", math.inf))
                    <= pose5d_gates["lateral_position_m"]
                    and float(relative_pose5d.get("axis_error_rad", math.inf))
                    <= pose5d_gates["axis_angle_rad"]
                )
                camera_selection = "fixed_rgbd"
                wrist_camera_record = None
                if not pose5d_reobserve_authorized:
                    # The fixed camera has now failed the measured V2 gate.
                    # Add a real child camera under assembly_tcp and feed its
                    # higher-resolution registered RGB-D into the same
                    # estimator during this stopped, synchronized state.
                    wrist_camera_path = (
                        str(tcp_prim.GetPath()) + "/WristRgbdCamera"
                    )
                    wrist_camera = UsdGeom.Camera.Define(
                        stage, wrist_camera_path
                    )
                    if wrist_camera_search is None:
                        raise RuntimeError(
                            "generated wrist-camera search report is required"
                        )
                    selected_wrist_camera = wrist_camera_search["selected"]
                    candidate_eye_assembly = np.asarray(
                        selected_wrist_camera[
                            "mount_eye_assembly_tcp_m"
                        ], dtype=np.float64
                    )
                    candidate_target_assembly = np.asarray(
                        selected_wrist_camera[
                            "mount_target_assembly_tcp_m"
                        ], dtype=np.float64
                    )
                    # Registered grasp_tcp -> connector mating-frame relation:
                    # at the 12 mm waypoint TCP z=321.98 mm while the mating
                    # face is z=249.50 mm.  assembly +Z is opposite grasp_tcp
                    # +Z.  This converts the generated assembly-frame mount to
                    # the actual parent prim without consulting object truth.
                    assembly_to_grasp_rotation = np.diag((1.0, -1.0, -1.0))
                    assembly_origin_in_grasp = np.asarray(
                        (0.0, 0.0, 0.07248), dtype=np.float64
                    )
                    local_eye = (
                        assembly_to_grasp_rotation @ candidate_eye_assembly
                        + assembly_origin_in_grasp
                    )
                    local_target = (
                        assembly_to_grasp_rotation @ candidate_target_assembly
                        + assembly_origin_in_grasp
                    )
                    local_direction = local_target - local_eye
                    local_direction /= np.linalg.norm(local_direction)
                    local_rotation = Gf.Rotation(
                        Gf.Vec3d(0.0, 0.0, -1.0),
                        Gf.Vec3d(*local_direction),
                    )
                    local_matrix = Gf.Matrix4d(1.0)
                    local_matrix.SetRotate(local_rotation)
                    local_matrix.SetTranslateOnly(Gf.Vec3d(*local_eye))
                    UsdGeom.Xformable(wrist_camera).AddTransformOp().Set(
                        local_matrix
                    )
                    wrist_camera.CreateFocalLengthAttr(24.0)
                    wrist_camera.CreateHorizontalApertureAttr(20.955)
                    wrist_camera.CreateVerticalApertureAttr(
                        20.955 * 720.0 / 1280.0
                    )
                    wrist_camera.CreateClippingRangeAttr(
                        Gf.Vec2f(0.1, 10.0)
                    )
                    tcp_world_matrix = UsdGeom.Xformable(
                        tcp_prim
                    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    wrist_eye_world = np.asarray(
                        tcp_world_matrix.Transform(Gf.Vec3d(*local_eye)),
                        dtype=np.float64,
                    )
                    wrist_target_world = np.asarray(
                        tcp_world_matrix.Transform(
                            Gf.Vec3d(*local_target)
                        ),
                        dtype=np.float64,
                    )
                    wrist_rgbd = replace(
                        probe.rgbd,
                        camera=replace(
                            probe.rgbd.camera,
                            prim_path=wrist_camera_path,
                            frame_id="wrist_rgbd_camera_optical",
                            eye_m=tuple(
                                float(value) for value in wrist_eye_world
                            ),
                            target_m=tuple(
                                float(value) for value in wrist_target_world
                            ),
                            resolution=(1280, 720),
                        ),
                    )
                    def current_wrist_rgbd_contract():
                        current_tcp_matrix = UsdGeom.Xformable(
                            tcp_prim
                        ).ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default()
                        )
                        current_eye_world = np.asarray(
                            current_tcp_matrix.Transform(
                                Gf.Vec3d(*local_eye)
                            ), dtype=np.float64
                        )
                        current_target_world = np.asarray(
                            current_tcp_matrix.Transform(
                                Gf.Vec3d(*local_target)
                            ), dtype=np.float64
                        )
                        return replace(
                            wrist_rgbd,
                            camera=replace(
                                wrist_rgbd.camera,
                                eye_m=tuple(
                                    float(value)
                                    for value in current_eye_world
                                ),
                                target_m=tuple(
                                    float(value)
                                    for value in current_target_world
                                ),
                            ),
                        )
                    post_grasp_capture = capture_d38999_rgbd_runtime(
                        bindings={
                            "Camera": Camera,
                            "Gf": Gf,
                            "Image": Image,
                            "Usd": Usd,
                            "UsdGeom": UsdGeom,
                            "UsdLux": UsdLux,
                            "add_labels": add_labels,
                            "get_labels": get_labels,
                            "rep": rep,
                        },
                        simulation_app=simulation_app,
                        world=world,
                        stage=stage,
                        tabletop=tabletop,
                        rgbd=wrist_rgbd,
                        loose_prim=loose_prim,
                        fixed_prim=fixed_prim,
                        body=body,
                        output_dir=(
                            output_dir / "rgbd_reobserve_wrist_preinsert"
                        ),
                        pose5d_config=pose5d_config,
                        pose5d_capture_id=(
                            f"{probe.trial_id}-pose5d-wrist-after-grasp"
                        ),
                        pose5d_axis_priors={
                            "loose_plug": tuple(
                                float(value) for value in plug_axis_prior
                            ),
                            "fixed_receptacle": (0.0, 0.0, 1.0),
                        },
                        pose5d_authorization_gates=pose5d_gates,
                    )
                    relative_pose5d = post_grasp_capture.metrics.get(
                        "pose5d", {}
                    ).get("relative_receptacle_plug", {})
                    pose5d_reobserve_authorized = bool(
                        post_grasp_capture.passed
                        and relative_pose5d.get("control_authorized") is True
                        and float(
                            relative_pose5d.get(
                                "lateral_error_m", math.inf
                            )
                        )
                        <= pose5d_gates["lateral_position_m"]
                        and float(
                            relative_pose5d.get(
                                "axis_error_rad", math.inf
                            )
                        )
                        <= pose5d_gates["axis_angle_rad"]
                    )
                    camera_selection = "wrist_rgbd"
                    wrist_camera_record = {
                        "prim_path": wrist_camera_path,
                        "parent_link": str(tcp_prim.GetPath()),
                        "optical_frame": "wrist_rgbd_camera_optical",
                        "local_eye_m": local_eye.tolist(),
                        "local_target_m": local_target.tolist(),
                        "candidate_eye_assembly_tcp_m": (
                            candidate_eye_assembly.tolist()
                        ),
                        "candidate_target_assembly_tcp_m": (
                            candidate_target_assembly.tolist()
                        ),
                        "registered_assembly_origin_in_grasp_tcp_m": (
                            assembly_origin_in_grasp.tolist()
                        ),
                        "resolution": [1280, 720],
                        "registered_depth": True,
                        "collision_geometry_present": False,
                        "generated_candidate_search": True,
                        "candidate_score": float(
                            selected_wrist_camera["score"]
                        ),
                        "candidate_collision_clearance_m": float(
                            selected_wrist_camera["collision_clearance_m"]
                        ),
                    }
                    active_wrist_multiview = []
                    if arguments.active_wrist_multiview:
                        from scipy.spatial.transform import (
                            Rotation as SciPyRotation,
                        )

                        final_preinsert_arm = current_arm.copy()
                        final_preinsert_tcp = np.asarray(
                            iiwa14_grasp_tcp_transform(
                                tuple(float(value) for value in current_arm)
                            ), dtype=np.float64
                        )
                        postures = wrist_camera_search[
                            "postures_xyz_rpy"
                        ]
                        final_posture = np.asarray(
                            postures["FINAL_PREINSERT_VIEW"],
                            dtype=np.float64,
                        )
                        for view_name in (
                            "MULTIVIEW_VIEW_1",
                            "MULTIVIEW_VIEW_2",
                        ):
                            view_record = {"name": view_name, "passed": False}
                            try:
                                view_posture = np.asarray(
                                    postures[view_name], dtype=np.float64
                                )
                                translation_delta = (
                                    view_posture[:3] - final_posture[:3]
                                )
                                rotation_delta = SciPyRotation.from_euler(
                                    "xyz",
                                    view_posture[3:] - final_posture[3:],
                                ).as_matrix()
                                final_assembly_rotation_world = (
                                    final_preinsert_tcp[:3, :3]
                                    @ assembly_to_grasp_rotation
                                )
                                target_position = (
                                    final_preinsert_tcp[:3, 3]
                                    + final_assembly_rotation_world
                                    @ translation_delta
                                )
                                target_rotation = (
                                    final_assembly_rotation_world
                                    @ rotation_delta
                                    @ assembly_to_grasp_rotation.T
                                )
                                target_arm = np.asarray(
                                    solve_fixed_q7_tcp_pose(
                                        tuple(
                                            float(value)
                                            for value in current_arm
                                        ),
                                        tuple(
                                            float(value)
                                            for value in target_position
                                        ),
                                        target_rotation=target_rotation,
                                        maximum_iterations=(
                                            preinsert_probe.ik.maximum_iterations
                                        ),
                                        damping=preinsert_probe.ik.damping,
                                    ), dtype=np.float64
                                )
                                phase = (
                                    "post_grasp_" + view_name.lower()
                                )
                                run_arm_motion(
                                    target_arm,
                                    2.0,
                                    True,
                                    sample_torque=True,
                                    enforce_preinsert_gates=True,
                                )
                                view_capture = capture_d38999_rgbd_runtime(
                                    bindings={
                                        "Camera": Camera,
                                        "Gf": Gf,
                                        "Image": Image,
                                        "Usd": Usd,
                                        "UsdGeom": UsdGeom,
                                        "UsdLux": UsdLux,
                                        "add_labels": add_labels,
                                        "get_labels": get_labels,
                                        "rep": rep,
                                    },
                                    simulation_app=simulation_app,
                                    world=world,
                                    stage=stage,
                                    tabletop=tabletop,
                                    rgbd=current_wrist_rgbd_contract(),
                                    loose_prim=loose_prim,
                                    fixed_prim=fixed_prim,
                                    body=body,
                                    output_dir=(
                                        output_dir
                                        / ("rgbd_wrist_" + view_name.lower())
                                    ),
                                    pose5d_config=None,
                                )
                                view_record.update(
                                    {
                                        "passed": bool(view_capture.passed),
                                        "target_arm_rad": target_arm.tolist(),
                                        "target_tcp_world_m": (
                                            target_position.tolist()
                                        ),
                                        "capture": view_capture.metrics,
                                    }
                                )
                            except BaseException as exception:
                                view_record["error"] = (
                                    f"{type(exception).__name__}: {exception}"
                                )
                            active_wrist_multiview.append(view_record)
                        phase = "return_final_preinsert_view"
                        run_arm_motion(
                            final_preinsert_arm,
                            2.0,
                            True,
                            sample_torque=True,
                            enforce_preinsert_gates=True,
                        )
                        final_view_capture = capture_d38999_rgbd_runtime(
                            bindings={
                                "Camera": Camera,
                                "Gf": Gf,
                                "Image": Image,
                                "Usd": Usd,
                                "UsdGeom": UsdGeom,
                                "UsdLux": UsdLux,
                                "add_labels": add_labels,
                                "get_labels": get_labels,
                                "rep": rep,
                            },
                            simulation_app=simulation_app,
                            world=world,
                            stage=stage,
                            tabletop=tabletop,
                            rgbd=current_wrist_rgbd_contract(),
                            loose_prim=loose_prim,
                            fixed_prim=fixed_prim,
                            body=body,
                            output_dir=(
                                output_dir
                                / "rgbd_wrist_final_preinsert_view"
                            ),
                            pose5d_config=None,
                        )
                        active_wrist_multiview.append(
                            {
                                "name": "FINAL_PREINSERT_VIEW",
                                "passed": bool(final_view_capture.passed),
                                "capture": final_view_capture.metrics,
                                "returned_to_original_preinsert_arm": bool(
                                    np.max(
                                        np.abs(
                                            current_arm
                                            - final_preinsert_arm
                                        )
                                    ) <= 1.0e-12
                                ),
                            }
                        )
                    wrist_camera_record["active_multiview_captures"] = (
                        active_wrist_multiview
                    )
                report["post_grasp_reobservation"] = {
                    "same_world": world is capture_world_reference,
                    "same_synchronized_capture_for_both_endpoints": True,
                    "capture": post_grasp_capture.metrics,
                    "fixed_camera_capture": (
                        fixed_post_grasp_capture.metrics
                    ),
                    "selected_camera": camera_selection,
                    "wrist_camera": wrist_camera_record,
                    "control_authorized": pose5d_reobserve_authorized,
                    "truth_pose_used_for_estimation": False,
                    "plug_axis_prior_source": (
                        "robot_tcp_fk_and_registered_opposite_axis"
                    ),
                    "fixed_axis_prior_source": "calibrated_fixture_plus_z",
                }

            # Truth is read only after target generation and all motion are
            # complete.  It may reject this experiment, but it cannot trigger a
            # correction or alter any target that was sent to the robot.
            final_tcp_position, final_tcp_orientation = _world_pose(
                Gf, Usd, UsdGeom, tcp_prim
            )
            preinsert_body_position, preinsert_body_orientation = (
                body.get_world_pose()
            )
            preinsert_fixed_position, preinsert_fixed_orientation = (
                _world_pose(Gf, Usd, UsdGeom, fixed_prim)
            )
            body_quaternion = Gf.Quatd(
                float(preinsert_body_orientation[0]),
                Gf.Vec3d(
                    float(preinsert_body_orientation[1]),
                    float(preinsert_body_orientation[2]),
                    float(preinsert_body_orientation[3]),
                ),
            )
            actual_alignment = measure_alignment(
                tuple(float(value) for value in preinsert_body_position),
                _quaternion_world_z_axis(body_quaternion),
                tuple(float(value) for value in preinsert_fixed_position),
                _quaternion_world_z_axis(preinsert_fixed_orientation),
                preinsert_probe.insertion.acceptance.entry_evaluation_length_m,
            )
            preinsert_tcp_target = np.asarray(
                preinsert_plan.tcp_targets_world_m["preinsert"],
                dtype=np.float64,
            )
            final_tcp_position_error = float(
                np.linalg.norm(
                    np.asarray(final_tcp_position, dtype=np.float64)
                    - preinsert_tcp_target
                )
            )
            final_tcp_axis_error = _axis_error_radians(
                _quaternion_world_z_axis(final_tcp_orientation),
                pick.motion.grasp_tcp_down_axis_world,
            )
            preinsert_positions = np.asarray(
                robot.get_joint_positions(), dtype=np.float64
            )
            preinsert_velocities = np.asarray(
                robot.get_joint_velocities(), dtype=np.float64
            )
            final_target_tracking_error = float(
                np.max(
                    np.abs(
                        preinsert_positions[arm_indices]
                        - np.asarray(
                            preinsert_plan.arm_targets_rad["preinsert"],
                            dtype=np.float64,
                        )
                    )
                )
            )
            preinsert_final_snapshot = contact_snapshot()
            current_loose_fixed_records = loose_fixed_contact_count()

            same_world_capture_gate = bool(
                world is capture_world_reference
                and preinsert_plan.capture_id == plan.capture_id
                and runtime_side_effects["world_reset_or_clear_calls"] == 0
            )
            object_pose_write_gate = bool(
                report["object_pose_writes_after_physics"] == 0
                and runtime_side_effects[
                    "endpoint_pose_writes_after_physics"
                ]
                == 0
                and report["d38999_authoring"][
                    "object_pose_writes_after_start"
                ]
                == 0
            )
            zero_preentry_contact_gate = bool(
                external_contacts["table"] == 0
                and external_contacts["fixture"] == 0
                and external_contacts["fixed_endpoint"] == 0
                and external_contacts["loose_plug_unexpected_robot_link"] == 0
                and preinsert_loose_fixed_contact_records == 0
                and current_loose_fixed_records == 0
            )
            body_contact_retention_gate = bool(
                preinsert_checked_steps > 0
                and preinsert_minimum_body_contact_fingers == 3
                and _all_fingers_have_body_contact(
                    preinsert_final_snapshot
                )
            )
            torque_hard_stop_gate = bool(
                float(np.max(maximum_post_tare_delta))
                <= pick.sensing.maximum_absolute_torque_delta_nm
            )
            finite_preinsert_gate = bool(
                finite_throughout
                and np.all(np.isfinite(preinsert_positions))
                and np.all(np.isfinite(preinsert_velocities))
                and np.all(np.isfinite(preinsert_body_position))
                and np.all(np.isfinite(preinsert_body_orientation))
                and np.all(np.isfinite(preinsert_fixed_position))
            )
            tracking_and_speed_gate = bool(
                maximum_joint_speed
                <= min(
                    pick.acceptance.maximum_observed_joint_speed_rad_s,
                    preinsert_probe.insertion.acceptance
                    .maximum_joint_speed_rad_s,
                )
                and maximum_arm_tracking_error
                <= min(
                    pick.acceptance.maximum_arm_tracking_error_rad,
                    preinsert_probe.insertion.acceptance
                    .maximum_arm_tracking_error_rad,
                )
                and final_target_tracking_error
                <= (
                    preinsert_probe.insertion.acceptance
                    .maximum_arm_tracking_error_rad
                )
                and maximum_joint_limit_violation
                <= (
                    preinsert_probe.insertion.acceptance
                    .maximum_joint_limit_violation_rad
                )
            )
            tcp_target_gate = bool(
                final_tcp_position_error
                <= pick.acceptance.maximum_grasp_tcp_position_error_m
                and final_tcp_axis_error
                <= pick.acceptance.maximum_grasp_tcp_axis_error_rad
            )
            outside_entry_gate = bool(
                actual_alignment.gap_m >= preinsert_probe.entry_gap_m
            )
            preinsert_passed = bool(
                prior_visual_pick_passed
                and same_world_capture_gate
                and object_pose_write_gate
                and zero_preentry_contact_gate
                and body_contact_retention_gate
                and torque_hard_stop_gate
                and finite_preinsert_gate
                and tracking_and_speed_gate
                and tcp_target_gate
                and outside_entry_gate
            )
            preinsert_control_ready = bool(
                prior_visual_control_ready
                and same_world_capture_gate
                and object_pose_write_gate
                and torque_hard_stop_gate
                and finite_preinsert_gate
                and tracking_and_speed_gate
                and tcp_target_gate
                and pose5d_reobserve_authorized
            )
            report["preinsert_probe"] = {
                "status": (
                    "PASSED_AT_PREINSERT_OUTSIDE_ENTRY"
                    if preinsert_passed
                    else "REJECTED_FAIL_CLOSED"
                ),
                "passed": preinsert_passed,
                "prior_visual_pick_passed": prior_visual_pick_passed,
                "prior_visual_control_ready_without_truth_or_physx_contact": (
                    prior_visual_control_ready
                ),
                "control_ready_without_truth_or_physx_contact": (
                    preinsert_control_ready
                ),
                "same_world_capture_gate": same_world_capture_gate,
                "object_pose_write_gate": object_pose_write_gate,
                "zero_preentry_contact_gate": zero_preentry_contact_gate,
                "body_contact_retention_gate": body_contact_retention_gate,
                "torque_hard_stop_gate": torque_hard_stop_gate,
                "finite_preinsert_gate": finite_preinsert_gate,
                "tracking_and_speed_gate": tracking_and_speed_gate,
                "tcp_target_gate": tcp_target_gate,
                "outside_entry_gate": outside_entry_gate,
                "pose5d_reobserve_required": pose5d_config is not None,
                "pose5d_reobserve_authorized": (
                    pose5d_reobserve_authorized
                ),
                "checked_physics_steps": preinsert_checked_steps,
                "continuation_global_steps": (
                    global_step - preinsert_start_step
                ),
                "minimum_body_contact_finger_count": (
                    preinsert_minimum_body_contact_fingers
                ),
                "loose_fixed_contact_records": (
                    preinsert_loose_fixed_contact_records
                    + current_loose_fixed_records
                ),
                "maximum_post_tare_absolute_delta_nm": float(
                    np.max(maximum_post_tare_delta)
                ),
                "maximum_joint_speed_rad_s": maximum_joint_speed,
                "maximum_arm_tracking_error_rad": (
                    maximum_arm_tracking_error
                ),
                "maximum_joint_limit_violation_rad": (
                    maximum_joint_limit_violation
                ),
                "final_target_tracking_error_rad": (
                    final_target_tracking_error
                ),
                "final_tcp_position_error_m": final_tcp_position_error,
                "final_tcp_axis_error_rad": final_tcp_axis_error,
                "final_contacts": preinsert_final_snapshot,
                "post_hoc_actual_alignment": {
                    "scope": (
                        "truth_evaluation_after_motion_never_target_or_"
                        "correction"
                    ),
                    "gap_m": actual_alignment.gap_m,
                    "lateral_error_m": actual_alignment.lateral_error_m,
                    "axis_error_rad": actual_alignment.axis_error_rad,
                    "combined_entry_error_m": (
                        actual_alignment.combined_entry_error_m
                    ),
                    "entry_gap_m": preinsert_probe.entry_gap_m,
                    "commanded_preinsert_gap_m": (
                        preinsert_probe.preinsert_gap_m
                    ),
                },
                "translation_source": "visual_fixed_receptacle_xy",
                "orientation_source": "registered_nominal_fk",
                "truth_xy_used_for_target": False,
                "truth_pose_feedback_used_for_target": False,
                "engage_executed": False,
                "insertion_executed": False,
                "twist_executed": False,
                "home_return_executed": False,
                "production_control_authorized": False,
                "assembly_success_claimed": False,
            }
            passed = bool(passed and preinsert_passed)
            report["passed"] = passed

            if wrist_ft_guarded_contract is not None:
                guarded_report = report["wrist_ft_guarded_insertion"]
                guarded_report.update(
                    {
                        "status": "CAPTURING_STOPPED_PREINSERT_PAYLOAD_BASELINE",
                        "preinsert_control_ready_without_truth_or_physx_contact": (
                            preinsert_control_ready
                        ),
                        "same_world_and_capture_id": bool(
                            world is capture_world_reference
                            and preinsert_plan.capture_id == plan.capture_id
                        ),
                    }
                )
                if not preinsert_control_ready:
                    raise RuntimeError(
                        "wrist-FT guarded insertion preinsert prerequisite "
                        "was not control-ready"
                    )

                task_rotation = np.asarray(
                    wrist_ft_monitor.task_rotation_world,
                    dtype=np.float64,
                )
                task_origin = np.asarray(
                    wrist_ft_monitor.task_origin_world,
                    dtype=np.float64,
                )
                target_rotation = np.asarray(
                    iiwa14_grasp_tcp_transform(
                        tuple(float(value) for value in current_arm)
                    ),
                    dtype=np.float64,
                )[:3, :3]
                guarded_trace = []
                guarded_peaks = {
                    "absolute_axial_force_n": 0.0,
                    "lateral_force_n": 0.0,
                    "bending_torque_nm": 0.0,
                    "absolute_tightening_torque_nm": 0.0,
                    "arm_tracking_error_rad": 0.0,
                    "joint_speed_rad_s": 0.0,
                    "gripper_position_drift_from_preinsert_rad": 0.0,
                }
                guarded_start_step = global_step
                preinsert_hand_position_baseline = None

                def wrist_ft_only_step(arm_command):
                    """One synchronous step with no PhysX contact query."""

                    nonlocal current_arm
                    nonlocal global_step
                    command = np.asarray(arm_command, dtype=np.float64)
                    if command.shape != (7,) or not np.all(
                        np.isfinite(command)
                    ):
                        raise RuntimeError(
                            "wrist-FT guarded arm command is invalid"
                        )
                    target = np.concatenate(
                        (command, current_hand)
                    ).astype(np.float32)
                    robot.apply_action(
                        ArticulationAction(
                            joint_positions=target,
                            joint_indices=controlled_indices,
                        )
                    )
                    world.step(render=arguments.gui)
                    global_step += 1
                    current_arm = command.copy()

                    raw_wrench = np.asarray(
                        robot.get_measured_joint_forces(
                            joint_indices=np.asarray(
                                [wrist_ft_reaction_row], dtype=np.int32
                            )
                        ),
                        dtype=np.float64,
                    )
                    if raw_wrench.shape != (1, 6):
                        raise RuntimeError(
                            "unexpected hand2arm reaction wrench shape in "
                            "wrist-FT guarded insertion"
                        )
                    sensor_matrix = UsdGeom.Xformable(
                        wrist_ft_sensor_prim
                    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    sensor_transform = Gf.Transform(sensor_matrix)
                    sensor_rotation = column_rotation_from_gf_matrix3d(
                        np.asarray(
                            Gf.Matrix3d(sensor_transform.GetRotation()),
                            dtype=np.float64,
                        )
                    )
                    ft_sample = wrist_ft_monitor.observe(
                        raw_wrench[0],
                        global_step=global_step,
                        runtime_phase=phase,
                        sensor_position_world=np.asarray(
                            sensor_transform.GetTranslation(),
                            dtype=np.float64,
                        ),
                        sensor_rotation_world=sensor_rotation,
                    )
                    positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    velocities = np.asarray(
                        robot.get_joint_velocities(), dtype=np.float64
                    )
                    measured_tcp_world, _ = _world_pose(
                        Gf, Usd, UsdGeom, tcp_prim
                    )
                    measured_tcp_task = task_rotation.T @ (
                        np.asarray(measured_tcp_world, dtype=np.float64)
                        - task_origin
                    )
                    command_transform = np.asarray(
                        iiwa14_grasp_tcp_transform(
                            tuple(float(value) for value in command)
                        ),
                        dtype=np.float64,
                    )
                    commanded_tcp_task = task_rotation.T @ (
                        command_transform[:3, 3] - task_origin
                    )
                    robot_state_finite = bool(
                        np.all(np.isfinite(positions))
                        and np.all(np.isfinite(velocities))
                        and np.all(np.isfinite(measured_tcp_task))
                    )
                    observation = parse_guarded_insertion_observation(
                        {
                            "timestamp_s": float(global_step) / rate_hz,
                            "sample_age_s": 0.0,
                            "compensated_wrench_task": ft_sample.get(
                                "compensated_wrench_task", [0.0] * 6
                            ),
                            "measured_tcp_position_task_m": [
                                float(value) for value in measured_tcp_task
                            ],
                            "commanded_tcp_position_task_m": [
                                float(value) for value in commanded_tcp_task
                            ],
                            "arm_tracking_error_rad": float(
                                np.max(
                                    np.abs(
                                        positions[arm_indices] - command
                                    )
                                )
                            ),
                            "maximum_joint_speed_rad_s": float(
                                np.max(np.abs(velocities[arm_indices]))
                            ),
                            "gripper_position_drift_from_preinsert_rad": (
                                0.0
                                if preinsert_hand_position_baseline is None
                                else float(
                                    np.max(
                                        np.abs(
                                            positions[hand_indices]
                                            - preinsert_hand_position_baseline
                                        )
                                    )
                                )
                            ),
                            "robot_state_finite": robot_state_finite,
                            "vision_preinsert_id": preinsert_plan.capture_id,
                        }
                    )
                    return observation, ft_sample

                # Capture the grasped payload baseline while stopped at the
                # visual 12 mm preinsert target.  No contact identity is read.
                phase = "unsupported_final_hold"
                for _ in range(
                    wrist_ft_config.payload_baseline_window_steps
                ):
                    wrist_ft_only_step(current_arm)
                payload_baseline = (
                    wrist_ft_monitor.capture_payload_baseline()
                )
                preinsert_positions = np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                )
                preinsert_hand_position_baseline = np.asarray(
                    preinsert_positions[hand_indices], dtype=np.float64
                ).copy()
                if (
                    preinsert_hand_position_baseline.shape
                    != (len(hand_indices),)
                    or not np.all(
                        np.isfinite(preinsert_hand_position_baseline)
                    )
                ):
                    raise RuntimeError(
                        "wrist-FT guarded gripper preinsert baseline is invalid"
                    )
                guarded_report.update(
                    {
                        "status": "RUNNING_WRIST_FT_ONLY_GUARDED_INSERTION",
                        "payload_baseline_canonical": payload_baseline,
                        "payload_baseline_samples": (
                            wrist_ft_config.payload_baseline_window_steps
                        ),
                        "gripper_preinsert_position_baseline_rad": [
                            float(value)
                            for value in preinsert_hand_position_baseline
                        ],
                        "task_frame_rotation_world": [
                            [float(value) for value in row]
                            for row in task_rotation
                        ],
                        "task_frame_origin_world_m": [
                            float(value) for value in task_origin
                        ],
                    }
                )

                phase = "mixed_grip_preinsert_wrist_ft_guarded_insertion"
                guarded_observation, _ = wrist_ft_only_step(current_arm)
                guarded_state = initial_guarded_insertion_state(
                    guarded_observation
                )
                guarded_passed = False
                last_logged_phase = None
                while True:
                    guarded_command = step_guarded_insertion(
                        wrist_ft_guarded_contract,
                        guarded_state,
                        guarded_observation,
                    )
                    next_state = guarded_command.next_state
                    wrench = np.asarray(
                        guarded_observation.compensated_wrench_task,
                        dtype=np.float64,
                    )
                    guarded_peaks["absolute_axial_force_n"] = max(
                        guarded_peaks["absolute_axial_force_n"],
                        abs(float(wrench[2])),
                    )
                    guarded_peaks["lateral_force_n"] = max(
                        guarded_peaks["lateral_force_n"],
                        float(np.linalg.norm(wrench[:2])),
                    )
                    guarded_peaks["bending_torque_nm"] = max(
                        guarded_peaks["bending_torque_nm"],
                        float(np.linalg.norm(wrench[3:5])),
                    )
                    guarded_peaks[
                        "absolute_tightening_torque_nm"
                    ] = max(
                        guarded_peaks[
                            "absolute_tightening_torque_nm"
                        ],
                        abs(float(wrench[5])),
                    )
                    guarded_peaks["arm_tracking_error_rad"] = max(
                        guarded_peaks["arm_tracking_error_rad"],
                        guarded_observation.arm_tracking_error_rad,
                    )
                    guarded_peaks["joint_speed_rad_s"] = max(
                        guarded_peaks["joint_speed_rad_s"],
                        guarded_observation.maximum_joint_speed_rad_s,
                    )
                    guarded_peaks[
                        "gripper_position_drift_from_preinsert_rad"
                    ] = max(
                        guarded_peaks[
                            "gripper_position_drift_from_preinsert_rad"
                        ],
                        guarded_observation
                        .gripper_position_drift_from_preinsert_rad,
                    )
                    if (
                        next_state.phase.value != last_logged_phase
                        or next_state.step_count % 240 == 0
                        or guarded_command.stop_motion
                    ):
                        guarded_trace.append(
                            {
                                "global_step": global_step,
                                "controller_step": next_state.step_count,
                                "phase": next_state.phase.value,
                                "status": guarded_command.status,
                                "contact_retry_count": (
                                    next_state.contact_retry_count
                                ),
                                "xy_offset_task_m": list(
                                    next_state.xy_offset_task_m
                                ),
                                "delta_tcp_task_m": list(
                                    guarded_command.delta_tcp_task_m
                                ),
                                "measured_tcp_position_task_m": list(
                                    guarded_observation
                                    .measured_tcp_position_task_m
                                ),
                                "wrench_task": list(
                                    guarded_observation
                                    .compensated_wrench_task
                                ),
                            }
                        )
                        last_logged_phase = next_state.phase.value
                    guarded_state = next_state
                    if guarded_command.stop_motion:
                        guarded_passed = bool(
                            guarded_state.phase
                            is GuardedInsertionPhase.COMPLETE
                        )
                        break

                    current_command_transform = np.asarray(
                        iiwa14_grasp_tcp_transform(
                            tuple(float(value) for value in current_arm)
                        ),
                        dtype=np.float64,
                    )
                    current_command_task = task_rotation.T @ (
                        current_command_transform[:3, 3] - task_origin
                    )
                    next_command_task = current_command_task + np.asarray(
                        guarded_command.delta_tcp_task_m,
                        dtype=np.float64,
                    )
                    next_command_world = (
                        task_origin + task_rotation @ next_command_task
                    )
                    next_arm = np.asarray(
                        solve_fixed_q7_tcp_pose(
                            tuple(float(value) for value in current_arm),
                            tuple(
                                float(value) for value in next_command_world
                            ),
                            target_rotation=target_rotation,
                            maximum_iterations=(
                                preinsert_probe.ik.maximum_iterations
                            ),
                            damping=preinsert_probe.ik.damping,
                        ),
                        dtype=np.float64,
                    )
                    if (
                        next_arm.shape != (7,)
                        or not np.all(np.isfinite(next_arm))
                        or not math.isclose(
                            float(next_arm[6]),
                            float(
                                preinsert_plan.arm_targets_rad[
                                    "preinsert"
                                ][6]
                            ),
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                    ):
                        raise RuntimeError(
                            "wrist-FT guarded fixed-q7 IK failed closed"
                        )
                    guarded_observation, _ = wrist_ft_only_step(next_arm)

                guarded_report.update(
                    {
                        "status": (
                            "PASSED_WRIST_FT_ONLY_GUARDED_ENGAGE"
                            if guarded_passed
                            else "FAILED_WRIST_FT_ONLY_GUARDED_INSERTION"
                        ),
                        "passed": guarded_passed,
                        "controller_steps": guarded_state.step_count,
                        "terminal_phase": guarded_state.phase.value,
                        "abort_reason": guarded_state.abort_reason,
                        "global_steps": global_step - guarded_start_step,
                        "peak_observations": guarded_peaks,
                        "trace": guarded_trace,
                        "physx_contact_queries_during_control": 0,
                        "fingertip_tactile_sensor_used": False,
                        "physx_contact_truth_used_for_control": False,
                        "simulator_truth_used_for_control": False,
                        "engage_executed": guarded_passed,
                        "insertion_executed": guarded_passed,
                        "twist_executed": False,
                        "home_return_executed": False,
                    }
                )
                report["engage_executed"] = guarded_passed
                report["insertion_executed"] = guarded_passed
                passed = bool(passed and guarded_passed)
                report["passed"] = passed
                if not guarded_passed:
                    raise RuntimeError(
                        "wrist-FT-only guarded insertion failed closed: "
                        f"{guarded_state.abort_reason}"
                    )

            if tactile_probe is not None:
                tactile_report = report[tactile_report_key]
                required_status = (
                    tactile_probe.eligibility.required_preinsert_status
                )
                if (
                    not preinsert_passed
                    or report["preinsert_probe"]["status"]
                    != required_status
                ):
                    raise RuntimeError(
                        "tactile lip calibration prerequisite did not pass"
                    )
                tactile_start_step = global_step
                tactile_report.update(
                    {
                        "status": "CAPTURING_STOPPED_PREINSERT_REFERENCE",
                        "preinsert_prerequisite_status": required_status,
                        "same_world_and_capture_id": bool(
                            world is capture_world_reference
                            and preinsert_plan.capture_id == plan.capture_id
                        ),
                    }
                )

                # The shared monitor's payload baseline is captured here, at
                # the stopped 12 mm preinsert pose.  It therefore also is the
                # tactile contract's local reference.  We do not recapture or
                # mutate a baseline after any lip contact has started.
                phase = "unsupported_final_hold"
                for _ in range(tactile_probe.sensor.local_reference_samples):
                    observe_and_step(
                        current_arm,
                        current_hand,
                        True,
                        True,
                        False,
                    )
                    sample_efforts()
                if latest_loose_fixed_contact_records != 0:
                    raise RuntimeError(
                        "stopped preinsert reference contains fixed contact"
                    )
                payload_baseline = (
                    wrist_ft_monitor.capture_payload_baseline()
                )
                preinsert_reference_tcp, _ = _world_pose(
                    Gf, Usd, UsdGeom, tcp_prim
                )
                preinsert_reference_tcp = np.asarray(
                    preinsert_reference_tcp, dtype=np.float64
                )
                preinsert_reference_body_in_tcp = body_in_tcp_frame(
                    body.get_world_pose()[0]
                )
                tactile_report.update(
                    {
                        "status": "RUNNING_SIGNED_LIP_TOUCHES",
                        "local_reference_samples": (
                            tactile_probe.sensor.local_reference_samples
                        ),
                        "payload_local_reference_canonical": (
                            payload_baseline
                        ),
                        "local_reference_is_safety_tare": False,
                        "trials": [],
                    }
                )

                task_rotation = np.asarray(
                    wrist_ft_monitor.task_rotation_world,
                    dtype=np.float64,
                )
                tactile_contact_rotation = task_rotation.copy()
                tactile_contact_origin = preinsert_reference_tcp.copy()
                task_x_world = task_rotation[:, 0]
                task_y_world = task_rotation[:, 1]
                task_z_world = task_rotation[:, 2]
                tactile_report["task_frame_axes_world"] = {
                    "x": [float(value) for value in task_x_world],
                    "y": [float(value) for value in task_y_world],
                    "z": [float(value) for value in task_z_world],
                    "determinant": float(np.linalg.det(task_rotation)),
                }
                if arguments.tactile_lip_manifold_capture:
                    tactile_report["task_frame_origin_world_m"] = [
                        float(value) for value in tactile_contact_origin
                    ]
                center_arm = np.asarray(
                    preinsert_plan.arm_targets_rad["preinsert"],
                    dtype=np.float64,
                )
                center_tcp_target = np.asarray(
                    preinsert_plan.tcp_targets_world_m["preinsert"],
                    dtype=np.float64,
                )
                target_rotation = np.asarray(
                    iiwa14_grasp_tcp_transform(
                        seven_float_arm_tuple(center_arm, "center_arm")
                    ),
                    dtype=np.float64,
                )[:3, :3]
                approach_travel_m = (
                    tactile_probe.motion.preinsert_gap_m
                    - tactile_probe.motion.entry_confirmation_gap_m
                )
                if approach_travel_m <= 0.0:
                    raise RuntimeError(
                        "tactile guarded approach interval is empty"
                    )

                def solve_tactile_arm(seed, tcp_target):
                    """Solve one registered-orientation target; q7 is fixed."""

                    solved = np.asarray(
                        solve_fixed_q7_tcp_pose(
                            seven_float_arm_tuple(seed, "tactile_ik_seed"),
                            three_float_position_tuple(
                                tcp_target, "tactile_ik_tcp_target"
                            ),
                            target_rotation=target_rotation,
                            maximum_iterations=(
                                preinsert_probe.ik.maximum_iterations
                            ),
                            damping=preinsert_probe.ik.damping,
                        ),
                        dtype=np.float64,
                    )
                    solved_transform = np.asarray(
                        iiwa14_grasp_tcp_transform(
                            seven_float_arm_tuple(solved, "solved_arm")
                        ),
                        dtype=np.float64,
                    )
                    if (
                        not np.all(np.isfinite(solved))
                        or not np.isclose(
                            solved[6], center_arm[6], atol=1.0e-12
                        )
                        or np.linalg.norm(
                            solved_transform[:3, 3] - tcp_target
                        )
                        > preinsert_probe.ik.maximum_fk_position_error_m
                        or np.max(np.abs(solved - center_arm))
                        > (
                            preinsert_probe.ik
                            .maximum_abs_joint_delta_from_nominal_rad
                        )
                    ):
                        raise RuntimeError(
                            "tactile fixed-q7 target left local IK bounds"
                        )
                    return solved

                def measured_tactile_gap_m(tcp_position):
                    """Estimate axial progress from FK, never object truth."""

                    delta = np.asarray(
                        tcp_position, dtype=np.float64
                    ) - preinsert_reference_tcp
                    return float(
                        tactile_probe.motion.preinsert_gap_m
                        + np.dot(delta, task_z_world)
                    )

                def guarded_tactile_sample(
                    search_offset_xy_m,
                    contact_attempts,
                    trial_peaks,
                ):
                    """Apply every abort gate to one fresh sensor tick."""

                    if (
                        latest_wrist_ft_sample is None
                        or latest_wrist_ft_sample.get("policy_phase")
                        != "INSERT"
                    ):
                        raise RuntimeError(
                            "tactile step lacks a protected INSERT wrench"
                        )
                    wrench = np.asarray(
                        latest_wrist_ft_sample.get(
                            "compensated_wrench_task"
                        ),
                        dtype=np.float64,
                    )
                    if wrench.shape != (6,) or not np.all(
                        np.isfinite(wrench)
                    ):
                        raise RuntimeError(
                            "tactile task-frame wrench is non-finite"
                        )
                    measured_effort = sample_efforts()
                    finger_delta = measured_effort - tare_efforts
                    tcp_position, _ = _world_pose(
                        Gf, Usd, UsdGeom, tcp_prim
                    )
                    gap_m = measured_tactile_gap_m(tcp_position)
                    observation = EngageObservation(
                        sample_age_s=0.0,
                        axial_force_n=float(wrench[2]),
                        lateral_force_xy_n=(
                            float(wrench[0]),
                            float(wrench[1]),
                        ),
                        bending_torque_xy_nm=(
                            float(wrench[3]),
                            float(wrench[4]),
                        ),
                        tightening_torque_nm=float(wrench[5]),
                        finger_base_torques_nm=tuple(
                            float(value) for value in finger_delta
                        ),
                        estimated_gap_m=gap_m,
                        search_offset_xy_m=tuple(
                            float(value) for value in search_offset_xy_m
                        ),
                        contact_attempts=contact_attempts,
                        elapsed_search_s=(
                            float(global_step - tactile_start_step)
                            / float(rate_hz)
                        ),
                        finite=True,
                        three_finger_body_contact=(
                            latest_preinsert_body_contact_fingers
                            == frozenset(("f1", "f2", "f3"))
                        ),
                        forbidden_contact=False,
                    )
                    decision = decide_engage_transition(
                        EngageState.GUARDED_APPROACH,
                        observation,
                        tactile_probe,
                    )
                    if decision.requires_abort_retract:
                        zero_step_reasons = {
                            "malformed_observation",
                            "nonfinite_observation",
                            "invalid_observation_range",
                            "stale_wrench",
                            "forbidden_contact",
                            "grasp_contact_lost",
                            "experimental_axial_force_ceiling",
                            "experimental_lateral_force_ceiling",
                            "experimental_bending_torque_ceiling",
                            "experimental_tightening_torque_ceiling",
                            "finger_base_torque_hard_stop",
                        }
                        raise TactileSafetyStop(
                            "tactile experimental abort envelope: "
                            f"{decision.reason}",
                            zero_step_abort=(
                                decision.reason in zero_step_reasons
                            ),
                        )
                    scalars = {
                        "absolute_axial_force_n": abs(float(wrench[2])),
                        "lateral_force_n": float(
                            np.linalg.norm(wrench[:2])
                        ),
                        "bending_torque_nm": float(
                            np.linalg.norm(wrench[3:5])
                        ),
                        "absolute_tightening_torque_nm": abs(
                            float(wrench[5])
                        ),
                        "absolute_finger_base_torque_nm": float(
                            np.max(np.abs(finger_delta))
                        ),
                    }
                    for name, value in scalars.items():
                        trial_peaks[name] = max(
                            trial_peaks.get(name, 0.0), value
                        )
                    return {
                        "observation": observation,
                        "wrench": wrench,
                        "wrist_ft_sample": dict(latest_wrist_ft_sample),
                        "tcp_position": np.asarray(
                            tcp_position, dtype=np.float64
                        ),
                        "loose_fixed_contact_records": (
                            latest_loose_fixed_contact_records
                        ),
                        "intended_lip_contact_pairs": (
                            latest_intended_lip_contact_pairs
                        ),
                        "unexpected_loose_fixed_contact_pairs": (
                            latest_unexpected_loose_fixed_contact_pairs
                        ),
                        "loose_fixture_contact_pairs": (
                            latest_loose_fixture_contact_pairs
                        ),
                        "loose_table_contact_pairs": (
                            latest_loose_table_contact_pairs
                        ),
                    }

                def tactile_state_snapshot(expected_arm_command):
                    """Read state and float32-check the last drive target."""

                    all_positions = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    all_velocities = np.asarray(
                        robot.get_joint_velocities(), dtype=np.float64
                    )
                    measured_arm = all_positions[arm_indices].copy()
                    measured_velocity = all_velocities[arm_indices].copy()
                    if (
                        measured_arm.shape != (7,)
                        or measured_velocity.shape != (7,)
                        or not np.all(np.isfinite(measured_arm))
                        or not np.all(np.isfinite(measured_velocity))
                    ):
                        raise TactileSafetyStop(
                            "tactile transition state is non-finite",
                            zero_step_abort=True,
                        )
                    measured_fk = np.asarray(
                        iiwa14_grasp_tcp_transform(
                            seven_float_arm_tuple(
                                measured_arm, "transition_measured_arm"
                            )
                        ),
                        dtype=np.float64,
                    )[:3, 3]
                    tcp_position, _ = _world_pose(
                        Gf, Usd, UsdGeom, tcp_prim
                    )
                    applied = controller.get_applied_action()
                    try:
                        applied_gate = compare_applied_arm_command_float32(
                            applied.joint_positions,
                            controlled_indices,
                            arm_indices,
                            expected_arm_command,
                            int(robot.num_dof),
                        )
                    except (AttributeError, TypeError, ValueError) as error:
                        applied_gate = {
                            "storage": None,
                            "controlled_position_readback": (),
                            "arm_position_readback_float32": (),
                            "expected_arm_command_float32": tuple(
                                float(value)
                                for value in np.asarray(
                                    expected_arm_command,
                                    dtype=np.float32,
                                )
                            ),
                            "arm_error_float32_rad": (),
                            "maximum_abs_arm_error_float32_rad": None,
                            "float32_equivalent_gate": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    wrench_sample = latest_wrist_ft_sample or {}
                    return {
                        "global_step": global_step,
                        "measured_arm_rad": [
                            float(value) for value in measured_arm
                        ],
                        "measured_arm_velocity_rad_s": [
                            float(value) for value in measured_velocity
                        ],
                        "measured_arm_fk_tcp_world_m": [
                            float(value) for value in measured_fk
                        ],
                        "measured_tcp_prim_world_m": [
                            float(value) for value in tcp_position
                        ],
                        "applied_action_joint_indices": [
                            int(value) for value in controlled_indices
                        ],
                        "applied_action_storage": applied_gate["storage"],
                        "applied_action_position_readback": [
                            float(value)
                            for value in applied_gate[
                                "controlled_position_readback"
                            ]
                        ],
                        "applied_action_arm_readback_float32": list(
                            applied_gate["arm_position_readback_float32"]
                        ),
                        "expected_previous_command_float32": list(
                            applied_gate["expected_arm_command_float32"]
                        ),
                        "applied_action_arm_error_float32_rad": list(
                            applied_gate["arm_error_float32_rad"]
                        ),
                        "applied_action_maximum_abs_arm_error_float32_rad": (
                            applied_gate[
                                "maximum_abs_arm_error_float32_rad"
                            ]
                        ),
                        "applied_action_float32_equivalent_gate": (
                            applied_gate["float32_equivalent_gate"]
                        ),
                        "applied_action_readback_error": applied_gate.get(
                            "error"
                        ),
                        "raw_wrench": wrench_sample.get("raw_wrench"),
                        "canonical_wrench_sensor": wrench_sample.get(
                            "canonical_wrench_sensor"
                        ),
                        "compensated_wrench_task": wrench_sample.get(
                            "compensated_wrench_task"
                        ),
                        "loose_fixed_contact_records": (
                            latest_loose_fixed_contact_records
                        ),
                        "intended_lip_contact_pairs": list(
                            latest_intended_lip_contact_pairs
                        ),
                        "unexpected_loose_fixed_contact_pairs": list(
                            latest_unexpected_loose_fixed_contact_pairs
                        ),
                        "loose_fixture_contact_pairs": list(
                            latest_loose_fixture_contact_pairs
                        ),
                        "loose_table_contact_pairs": list(
                            latest_loose_table_contact_pairs
                        ),
                        "first_forbidden_contact": (
                            first_tactile_forbidden_contact
                        ),
                    }

                def record_tactile_transition(
                    previous_command,
                    planned_command,
                    pre_state,
                    post_state,
                    *,
                    motion_index,
                    observe_error=None,
                    evidence_sink=None,
                ):
                    """Persist bounded transition evidence before any guard."""

                    previous_fk = np.asarray(
                        iiwa14_grasp_tcp_transform(
                            seven_float_arm_tuple(
                                previous_command,
                                "transition_previous_command",
                            )
                        ),
                        dtype=np.float64,
                    )[:3, 3]
                    planned_fk = np.asarray(
                        iiwa14_grasp_tcp_transform(
                            seven_float_arm_tuple(
                                planned_command,
                                "transition_planned_command",
                            )
                        ),
                        dtype=np.float64,
                    )[:3, 3]
                    command_fk_delta = planned_fk - previous_fk
                    tactile_transition_ring.append(
                        {
                            "phase": phase,
                            "motion_index": int(motion_index),
                            "previous_command_arm_rad": [
                                float(value) for value in previous_command
                            ],
                            "planned_command_arm_rad": [
                                float(value) for value in planned_command
                            ],
                            "command_delta_arm_rad": [
                                float(value)
                                for value in (
                                    planned_command - previous_command
                                )
                            ],
                            "previous_command_fk_tcp_world_m": [
                                float(value) for value in previous_fk
                            ],
                            "planned_command_fk_tcp_world_m": [
                                float(value) for value in planned_fk
                            ],
                            "command_fk_jump_world_m": [
                                float(value) for value in command_fk_delta
                            ],
                            "command_fk_jump_norm_m": float(
                                np.linalg.norm(command_fk_delta)
                            ),
                            "pre_step": pre_state,
                            "post_step": post_state,
                            "observe_error": observe_error,
                        }
                    )
                    del tactile_transition_ring[:-12]
                    if evidence_sink is not None:
                        evidence_sink["transition_ring"] = list(
                            tactile_transition_ring
                        )

                def run_tactile_motion(
                    target_arm,
                    duration_s,
                    *,
                    search_offset_xy_m,
                    contact_attempts,
                    trial_peaks,
                    allow_fixed_contact,
                    detect_contact=False,
                    evaluate_release=False,
                    command_path=None,
                    evidence_sink=None,
                    record_full_progress=False,
                    record_raw_guarded_samples=False,
                    capture_manifold=False,
                    release_after_retract=False,
                    minimum_gap_m=None,
                    maximum_command_speed_m_s=None,
                    maximum_measured_speed_m_s=None,
                    command_count_override=None,
                    stop_after_motion_index=None,
                    maximum_negative_progress_m=None,
                    raw_window_role=None,
                ):
                    """Run one bounded minimum-jerk segment and debounce."""

                    nonlocal current_arm
                    if record_raw_guarded_samples and not record_full_progress:
                        raise RuntimeError(
                            "raw tactile samples require full progress "
                            "evidence"
                        )
                    start_arm = current_arm.copy()
                    if command_path is None:
                        if command_count_override is None:
                            steps = max(1, round(duration_s * rate_hz))
                        elif (
                            type(command_count_override) is not int
                            or command_count_override <= 0
                        ):
                            raise RuntimeError(
                                "tactile command count override is invalid"
                            )
                        else:
                            steps = command_count_override
                        path_commands = None
                    else:
                        path_commands = tuple(
                            np.asarray(command, dtype=np.float64)
                            for command in command_path
                        )
                        if (
                            not path_commands
                            or any(
                                command.shape != (7,)
                                or not np.all(np.isfinite(command))
                                for command in path_commands
                            )
                            or not np.array_equal(
                                path_commands[0], start_arm
                            )
                        ):
                            raise RuntimeError(
                                "tactile command path is not continuous"
                            )
                        steps = len(path_commands)
                    if stop_after_motion_index is not None:
                        if (
                            path_commands is not None
                            or detect_contact
                            or evaluate_release
                            or capture_manifold
                            or type(stop_after_motion_index) is not int
                            or stop_after_motion_index <= 0
                            or stop_after_motion_index >= steps - 1
                        ):
                            raise RuntimeError(
                                "tactile interrupt stop contract is invalid"
                            )
                    contact_window = []
                    rejected_physical_contact_samples = []
                    release_window = []
                    release_evidence = []
                    task_z_progress_samples_m = []
                    measured_tcp_prim_world_samples_m = []
                    estimated_gap_samples_m = []
                    command_fk_task_z_progress_samples_m = []
                    command_fk_tcp_world_samples_m = []
                    guarded_tick_samples = []
                    checked_sample_count = 0
                    finite_sample_count = 0
                    minimum_body_contact_finger_count = 3
                    applied_precheck_count = 0
                    applied_postcheck_count = 0
                    applied_all_float32_equivalent = True
                    applied_maximum_abs_error_float32_rad = 0.0
                    contact_record_totals = {
                        name: 0
                        for name in TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS
                    }
                    last_sample = None
                    retract_distance_reached = False
                    post_retract_release_ticks = 0
                    manifold_contact_lost = False
                    contact_on_compression_n = (
                        tactile_probe.contact
                        .contact_on_compressive_axial_force_n
                    )
                    motion_start_tcp, _ = _world_pose(
                        Gf, Usd, UsdGeom, tcp_prim
                    )
                    motion_start_tcp = np.asarray(
                        motion_start_tcp, dtype=np.float64
                    )
                    motion_start_gap_m = measured_tactile_gap_m(
                        motion_start_tcp
                    )
                    start_command_fk = None
                    if record_full_progress:
                        start_command_fk = np.asarray(
                            iiwa14_grasp_tcp_transform(
                                seven_float_arm_tuple(
                                    start_arm, "preflight_start_command"
                                )
                            ),
                            dtype=np.float64,
                        )[:3, 3]

                    def motion_evidence_payload():
                        progress = task_z_progress_samples_m
                        peak_speed = (
                            0.0
                            if not progress
                            else float(
                                np.max(
                                    np.abs(
                                        np.diff(
                                            np.asarray(
                                                (0.0, *progress),
                                                dtype=np.float64,
                                            )
                                        )
                                    )
                                    * float(rate_hz)
                                )
                            )
                        )
                        result = {
                            "minimum_task_z_progress_m": (
                                None if not progress else float(min(progress))
                            ),
                            "maximum_task_z_progress_m": (
                                None if not progress else float(max(progress))
                            ),
                            "final_task_z_progress_m": (
                                None if not progress else float(progress[-1])
                            ),
                            "peak_abs_task_z_speed_m_s": peak_speed,
                            "required_retract_distance_reached": bool(
                                retract_distance_reached
                            ),
                            "post_retract_release_ticks": int(
                                post_retract_release_ticks
                            ),
                            "release_samples": release_evidence,
                            "rejected_physical_contact_samples": (
                                rejected_physical_contact_samples
                            ),
                            "checked_sample_count": int(
                                checked_sample_count
                            ),
                            "finite_sample_count": int(
                                finite_sample_count
                            ),
                            "all_samples_finite": bool(
                                checked_sample_count == finite_sample_count
                            ),
                            "minimum_body_contact_finger_count": int(
                                minimum_body_contact_finger_count
                            ),
                            "applied_action_precheck_count": int(
                                applied_precheck_count
                            ),
                            "applied_action_postcheck_count": int(
                                applied_postcheck_count
                            ),
                            "applied_action_all_float32_equivalent": bool(
                                applied_all_float32_equivalent
                            ),
                            (
                                "applied_action_maximum_abs_arm_error_"
                                "float32_rad"
                            ): float(
                                applied_maximum_abs_error_float32_rad
                            ),
                            "contact_record_totals": dict(
                                contact_record_totals
                            ),
                            # The same dictionary is updated on every guarded
                            # tick.  Persisting a snapshot per phase plus the
                            # final aggregate makes the 900+ tick envelope
                            # auditable without relying on the 12-entry causal
                            # transition ring.
                            "peak_experimental_observations": {
                                name: float(trial_peaks.get(name, 0.0))
                                for name in TACTILE_PREFLIGHT_PEAK_FIELDS
                            },
                        }
                        if record_full_progress:
                            command_progress = (
                                command_fk_task_z_progress_samples_m
                            )
                            command_peak_speed = (
                                0.0
                                if not command_progress
                                else float(
                                    np.max(
                                        np.abs(
                                            np.diff(
                                                np.asarray(
                                                    (
                                                        0.0,
                                                        *command_progress,
                                                    ),
                                                    dtype=np.float64,
                                                )
                                            )
                                        )
                                        * float(rate_hz)
                                    )
                                )
                            )
                            result[
                                "measured_start_tcp_prim_world_m"
                            ] = [
                                float(value) for value in motion_start_tcp
                            ]
                            result["estimated_start_gap_m"] = float(
                                motion_start_gap_m
                            )
                            result["command_start_fk_tcp_world_m"] = [
                                float(value) for value in start_command_fk
                            ]
                            result["task_z_progress_samples_m"] = [
                                float(value) for value in progress
                            ]
                            result[
                                "measured_tcp_prim_world_samples_m"
                            ] = list(measured_tcp_prim_world_samples_m)
                            result["estimated_gap_samples_m"] = list(
                                estimated_gap_samples_m
                            )
                            result[
                                "command_fk_task_z_progress_samples_m"
                            ] = [
                                float(value)
                                for value in command_progress
                            ]
                            result[
                                "command_fk_tcp_world_samples_m"
                            ] = list(command_fk_tcp_world_samples_m)
                            result[
                                "peak_abs_command_fk_task_z_speed_m_s"
                            ] = command_peak_speed
                            if (
                                maximum_command_speed_m_s is not None
                                or maximum_measured_speed_m_s is not None
                            ):
                                measured_positions = np.asarray(
                                    (
                                        motion_start_tcp.tolist(),
                                        *measured_tcp_prim_world_samples_m,
                                    ),
                                    dtype=np.float64,
                                )
                                command_positions = np.asarray(
                                    (
                                        start_command_fk.tolist(),
                                        *command_fk_tcp_world_samples_m,
                                    ),
                                    dtype=np.float64,
                                )
                                result["peak_abs_tcp_speed_m_s"] = float(
                                    np.max(
                                        np.linalg.norm(
                                            np.diff(
                                                measured_positions, axis=0
                                            ),
                                            axis=1,
                                        )
                                    )
                                    * float(rate_hz)
                                )
                                result[
                                    "peak_abs_command_fk_tcp_speed_m_s"
                                ] = float(
                                    np.max(
                                        np.linalg.norm(
                                            np.diff(
                                                command_positions, axis=0
                                            ),
                                            axis=1,
                                        )
                                    )
                                    * float(rate_hz)
                                )
                        if record_raw_guarded_samples:
                            # The first pre-step readback belongs to the
                            # command that was active at phase entry.  Keep
                            # that seven-joint command so the offline
                            # validator can prove continuity at tick zero,
                            # including across all three Stage-A phases.
                            result["command_start_arm_rad"] = [
                                float(value) for value in start_arm
                            ]
                            result["guarded_tick_samples"] = list(
                                guarded_tick_samples
                            )
                        if capture_manifold:
                            result.update(
                                {
                                    "manifold_capture_only": True,
                                    "manifold_contact_lost_after_first": (
                                        manifold_contact_lost
                                    ),
                                    "frozen_contact_hold_steps": int(
                                        physical_contact_hold_steps
                                    ),
                                }
                            )
                        return result

                    drive_frozen = False
                    physical_contact_hold_steps = 0
                    contact_hold_limit = (
                        tactile_probe.sensor.contact_debounce_samples
                        if capture_manifold
                        else tactile_probe.sensor.entry_confirmation_samples
                    )
                    endpoint_hold_limit = (
                        tactile_probe.sensor.release_debounce_samples
                        if evaluate_release
                        else 0
                    )
                    motion_path_indices = bounded_endpoint_hold_path_indices(
                        steps, endpoint_hold_limit
                    )
                    if stop_after_motion_index is not None:
                        # Keep the original full-path denominator and stop only
                        # after the selected command has been applied/observed.
                        # Recomputing ``steps`` would change the min-jerk shape.
                        motion_path_indices = tuple(
                            range(stop_after_motion_index + 1)
                        )
                    loop_cursor = 0
                    while loop_cursor < len(motion_path_indices) or (
                        detect_contact
                        and drive_frozen
                        and physical_contact_hold_steps
                        < contact_hold_limit
                    ):
                        index = motion_path_indices[
                            min(loop_cursor, len(motion_path_indices) - 1)
                        ]
                        previous_command = current_arm.copy()
                        if drive_frozen:
                            planned_command = current_arm.copy()
                        elif path_commands is not None:
                            planned_command = path_commands[
                                min(index, steps - 1)
                            ].copy()
                        else:
                            blend = minimum_jerk_blend(
                                float(index + 1) / float(steps)
                            )
                            planned_command = start_arm + blend * (
                                np.asarray(target_arm, dtype=np.float64)
                                - start_arm
                            )
                        if record_full_progress:
                            planned_command_fk = np.asarray(
                                iiwa14_grasp_tcp_transform(
                                    seven_float_arm_tuple(
                                        planned_command,
                                        "preflight_planned_command",
                                    )
                                ),
                                dtype=np.float64,
                            )[:3, 3]
                            command_fk_tcp_world_samples_m.append(
                                [
                                    float(value)
                                    for value in planned_command_fk
                                ]
                            )
                            command_fk_task_z_progress_samples_m.append(
                                float(
                                    np.dot(
                                        planned_command_fk - start_command_fk,
                                        task_z_world,
                                    )
                                )
                            )
                            if maximum_command_speed_m_s is not None:
                                command_previous = (
                                    start_command_fk
                                    if len(
                                        command_fk_tcp_world_samples_m
                                    )
                                    == 1
                                    else np.asarray(
                                        command_fk_tcp_world_samples_m[-2],
                                        dtype=np.float64,
                                    )
                                )
                                command_speed_m_s = float(
                                    np.linalg.norm(
                                        planned_command_fk
                                        - command_previous
                                    )
                                    * float(rate_hz)
                                )
                                if command_speed_m_s > float(
                                    maximum_command_speed_m_s
                                ) + 1.0e-12:
                                    raise TactileSafetyStop(
                                        "tactile command FK speed exceeded "
                                        "the manifold ceiling",
                                        zero_step_abort=True,
                                    )
                        pre_state = tactile_state_snapshot(previous_command)
                        applied_precheck_count += 1
                        applied_all_float32_equivalent = bool(
                            applied_all_float32_equivalent
                            and pre_state[
                                "applied_action_float32_equivalent_gate"
                            ]
                        )
                        pre_error = pre_state[
                            "applied_action_maximum_abs_arm_error_float32_rad"
                        ]
                        if pre_error is not None:
                            applied_maximum_abs_error_float32_rad = max(
                                applied_maximum_abs_error_float32_rad,
                                float(pre_error),
                            )
                        if not pre_state[
                            "applied_action_float32_equivalent_gate"
                        ]:
                            readback_error = (
                                "applied action readback does not equal the "
                                "previous arm command after float32 conversion"
                            )
                            record_tactile_transition(
                                previous_command,
                                planned_command,
                                pre_state,
                                None,
                                motion_index=index,
                                observe_error=readback_error,
                                evidence_sink=evidence_sink,
                            )
                            raise TactileSafetyStop(
                                readback_error, zero_step_abort=True
                            )
                        current_arm = planned_command
                        try:
                            observe_and_step(
                                current_arm,
                                current_hand,
                                True,
                                True,
                                allow_fixed_contact,
                            )
                        except BaseException as observe_error:
                            post_state = tactile_state_snapshot(
                                planned_command
                            )
                            applied_postcheck_count += 1
                            applied_all_float32_equivalent = bool(
                                applied_all_float32_equivalent
                                and post_state[
                                    "applied_action_float32_equivalent_gate"
                                ]
                            )
                            post_error = post_state[
                                "applied_action_maximum_abs_arm_error_"
                                "float32_rad"
                            ]
                            if post_error is not None:
                                applied_maximum_abs_error_float32_rad = max(
                                    applied_maximum_abs_error_float32_rad,
                                    float(post_error),
                                )
                            record_tactile_transition(
                                previous_command,
                                planned_command,
                                pre_state,
                                post_state,
                                motion_index=index,
                                observe_error=(
                                    f"{type(observe_error).__name__}: "
                                    f"{observe_error}"
                                ),
                                evidence_sink=evidence_sink,
                            )
                            if not post_state[
                                "applied_action_float32_equivalent_gate"
                            ]:
                                raise TactileSafetyStop(
                                    "applied action readback does not equal "
                                    "the planned arm command after float32 "
                                    "conversion",
                                    zero_step_abort=True,
                                ) from observe_error
                            raise
                        post_state = tactile_state_snapshot(planned_command)
                        applied_postcheck_count += 1
                        applied_all_float32_equivalent = bool(
                            applied_all_float32_equivalent
                            and post_state[
                                "applied_action_float32_equivalent_gate"
                            ]
                        )
                        post_error = post_state[
                            "applied_action_maximum_abs_arm_error_float32_rad"
                        ]
                        if post_error is not None:
                            applied_maximum_abs_error_float32_rad = max(
                                applied_maximum_abs_error_float32_rad,
                                float(post_error),
                            )
                        record_tactile_transition(
                            previous_command,
                            planned_command,
                            pre_state,
                            post_state,
                            motion_index=index,
                            evidence_sink=evidence_sink,
                        )
                        if not post_state[
                            "applied_action_float32_equivalent_gate"
                        ]:
                            raise TactileSafetyStop(
                                "applied action readback does not equal the "
                                "planned arm command after float32 conversion",
                                zero_step_abort=True,
                            )
                        last_sample = guarded_tactile_sample(
                            search_offset_xy_m,
                            contact_attempts,
                            trial_peaks,
                        )
                        if capture_manifold:
                            command_fk = np.asarray(
                                iiwa14_grasp_tcp_transform(
                                    seven_float_arm_tuple(
                                        current_arm,
                                        "manifold_frozen_command",
                                    )
                                ),
                                dtype=np.float64,
                            )[:3, 3]
                            last_sample["command_arm_rad"] = (
                                current_arm.copy()
                            )
                            last_sample["command_fk_tcp_world_m"] = (
                                command_fk
                            )
                            last_sample["transition_post_state"] = dict(
                                post_state
                            )
                        task_z_progress_samples_m.append(
                            float(
                                np.dot(
                                    last_sample["tcp_position"]
                                    - motion_start_tcp,
                                    task_z_world,
                                )
                            )
                        )
                        observation = last_sample["observation"]
                        checked_sample_count += 1
                        sample_finite = bool(
                            observation.finite
                            and np.all(np.isfinite(last_sample["wrench"]))
                            and np.all(
                                np.isfinite(last_sample["tcp_position"])
                            )
                        )
                        finite_sample_count += int(sample_finite)
                        minimum_body_contact_finger_count = min(
                            minimum_body_contact_finger_count,
                            len(latest_preinsert_body_contact_fingers),
                        )
                        contact_record_totals["loose_fixed"] += int(
                            last_sample["loose_fixed_contact_records"]
                        )
                        for name, sample_key in (
                            ("intended_lip", "intended_lip_contact_pairs"),
                            (
                                "unexpected_loose_fixed",
                                "unexpected_loose_fixed_contact_pairs",
                            ),
                            ("loose_fixture", "loose_fixture_contact_pairs"),
                            ("loose_table", "loose_table_contact_pairs"),
                        ):
                            contact_record_totals[name] += sum(
                                int(pair["contact_records"])
                                for pair in last_sample[sample_key]
                            )
                        measured_tcp_prim_world_samples_m.append(
                            [
                                float(value)
                                for value in last_sample["tcp_position"]
                            ]
                        )
                        estimated_gap_samples_m.append(
                            float(observation.estimated_gap_m)
                        )
                        if record_raw_guarded_samples:
                            ft_sample = last_sample["wrist_ft_sample"]
                            guarded_tick_samples.append(
                                {
                                    "global_step": int(
                                        ft_sample["global_step"]
                                    ),
                                    "phase": phase,
                                    "motion_index": int(index),
                                    "window_role": raw_window_role,
                                    "command_arm_rad": [
                                        float(value)
                                        for value in current_arm
                                    ],
                                    "command_fk_tcp_world_m": [
                                        float(value)
                                        for value in planned_command_fk
                                    ],
                                    "measured_tcp_prim_world_m": [
                                        float(value)
                                        for value in last_sample[
                                            "tcp_position"
                                        ]
                                    ],
                                    "pre_measured_tcp_prim_world_m": list(
                                        pre_state[
                                            "measured_tcp_prim_world_m"
                                        ]
                                    ),
                                    "measured_arm_rad": list(
                                        post_state["measured_arm_rad"]
                                    ),
                                    "measured_arm_velocity_rad_s": list(
                                        post_state[
                                            "measured_arm_velocity_rad_s"
                                        ]
                                    ),
                                    "measured_arm_fk_tcp_world_m": list(
                                        post_state[
                                            "measured_arm_fk_tcp_world_m"
                                        ]
                                    ),
                                    "pre_global_step": int(
                                        pre_state["global_step"]
                                    ),
                                    "post_global_step": int(
                                        post_state["global_step"]
                                    ),
                                    "estimated_gap_m": float(
                                        observation.estimated_gap_m
                                    ),
                                    "finite": bool(sample_finite),
                                    "body_contact_fingers": sorted(
                                        latest_preinsert_body_contact_fingers
                                    ),
                                    "finger_base_torque_delta_nm": [
                                        float(value)
                                        for value in (
                                            observation
                                            .finger_base_torques_nm
                                        )
                                    ],
                                    "raw_wrench": ft_sample["raw_wrench"],
                                    "canonical_wrench_sensor": ft_sample[
                                        "canonical_wrench_sensor"
                                    ],
                                    "compensated_wrench_sensor": ft_sample[
                                        "compensated_wrench_sensor"
                                    ],
                                    "compensated_wrench_task": [
                                        float(value)
                                        for value in last_sample["wrench"]
                                    ],
                                    "applied_pre_float32_gate": pre_state[
                                        "applied_action_float32_"
                                        "equivalent_gate"
                                    ],
                                    "applied_post_float32_gate": post_state[
                                        "applied_action_float32_"
                                        "equivalent_gate"
                                    ],
                                    "applied_pre_max_error_float32_rad": (
                                        pre_state[
                                            "applied_action_maximum_abs_"
                                            "arm_error_float32_rad"
                                        ]
                                    ),
                                    "applied_post_max_error_float32_rad": (
                                        post_state[
                                            "applied_action_maximum_abs_"
                                            "arm_error_float32_rad"
                                        ]
                                    ),
                                    "applied_pre_readback_float32": list(
                                        pre_state[
                                            "applied_action_arm_readback_"
                                            "float32"
                                        ]
                                    ),
                                    "applied_pre_expected_float32": list(
                                        pre_state[
                                            "expected_previous_command_"
                                            "float32"
                                        ]
                                    ),
                                    "applied_post_readback_float32": list(
                                        post_state[
                                            "applied_action_arm_readback_"
                                            "float32"
                                        ]
                                    ),
                                    "applied_post_expected_float32": list(
                                        post_state[
                                            "expected_previous_command_"
                                            "float32"
                                        ]
                                    ),
                                    "loose_fixed_contact_records": int(
                                        last_sample[
                                            "loose_fixed_contact_records"
                                        ]
                                    ),
                                    "intended_lip_contact_pairs": list(
                                        last_sample[
                                            "intended_lip_contact_pairs"
                                        ]
                                    ),
                                    (
                                        "unexpected_loose_fixed_"
                                        "contact_pairs"
                                    ): list(
                                        last_sample[
                                            "unexpected_loose_fixed_"
                                            "contact_pairs"
                                        ]
                                    ),
                                    "loose_fixture_contact_pairs": list(
                                        last_sample[
                                            "loose_fixture_contact_pairs"
                                        ]
                                    ),
                                    "loose_table_contact_pairs": list(
                                        last_sample[
                                            "loose_table_contact_pairs"
                                        ]
                                    ),
                                }
                            )
                        # Update a complete bounded partial payload before any
                        # post-observation hard gate.  A failure report can then
                        # identify the violating tick without another step.
                        if evidence_sink is not None:
                            evidence_sink["partial_motion_evidence"] = (
                                motion_evidence_payload()
                            )
                        if minimum_gap_m is not None and (
                            observation.estimated_gap_m
                            < float(minimum_gap_m)
                        ):
                            raise TactileSafetyStop(
                                "tactile manifold approach crossed the "
                                "entry-gap floor",
                                zero_step_abort=True,
                            )
                        if maximum_measured_speed_m_s is not None:
                            measured_previous = (
                                motion_start_tcp
                                if len(
                                    measured_tcp_prim_world_samples_m
                                )
                                == 1
                                else np.asarray(
                                    measured_tcp_prim_world_samples_m[-2],
                                    dtype=np.float64,
                                )
                            )
                            measured_speed_m_s = float(
                                np.linalg.norm(
                                    last_sample["tcp_position"]
                                    - measured_previous
                                )
                                * float(rate_hz)
                            )
                            if measured_speed_m_s > float(
                                maximum_measured_speed_m_s
                            ):
                                raise TactileSafetyStop(
                                    "tactile measured TCP speed exceeded "
                                    "the hard manifold ceiling",
                                    zero_step_abort=True,
                                )
                        physical_contact = bool(
                            last_sample["loose_fixed_contact_records"] > 0
                            and last_sample["intended_lip_contact_pairs"]
                            and not last_sample[
                                "unexpected_loose_fixed_contact_pairs"
                            ]
                            and not last_sample[
                                "loose_fixture_contact_pairs"
                            ]
                            and not last_sample[
                                "loose_table_contact_pairs"
                            ]
                        )
                        effective_negative_bound = (
                            maximum_negative_progress_m
                            if maximum_negative_progress_m is not None
                            else (
                                tactile_negative_progress_bound_m
                                if evaluate_release
                                else None
                            )
                        )
                        if (
                            effective_negative_bound is not None
                            and task_z_progress_samples_m[-1]
                            < -float(effective_negative_bound)
                        ):
                            raise TactileSafetyStop(
                                "contact retract exceeded the GPU preflight "
                                "negative-progress bound",
                                zero_step_abort=True,
                            )
                        if detect_contact:
                            if physical_contact and not drive_frozen:
                                # Stop the downward position target on the
                                # first intended physical lip record.  The six
                                # force-qualified samples are collected while
                                # holding this exact drive target, never by
                                # continuing the guarded approach trajectory.
                                drive_frozen = True
                                physical_contact_hold_steps = 1
                            elif drive_frozen:
                                physical_contact_hold_steps += 1
                            if capture_manifold and drive_frozen:
                                if physical_contact and not (
                                    manifold_contact_lost
                                ):
                                    contact_window.append(last_sample)
                                else:
                                    manifold_contact_lost = True
                                if (
                                    physical_contact_hold_steps
                                    >= contact_hold_limit
                                ):
                                    return (
                                        last_sample,
                                        tuple(contact_window),
                                        False,
                                        motion_evidence_payload(),
                                    )
                                loop_cursor += 1
                                continue
                            signed_compression = (
                                tactile_probe.motion
                                .compressive_axial_force_sign_candidate
                                * observation.axial_force_n
                            )
                            wrench_contact_candidate = contact_candidate(
                                observation, tactile_probe
                            )
                            if (
                                physical_contact
                                and wrench_contact_candidate
                                and signed_compression
                                >= contact_on_compression_n
                            ):
                                contact_window.append(last_sample)
                                debounce = (
                                    tactile_probe.sensor
                                    .contact_debounce_samples
                                )
                                contact_window = contact_window[
                                    -debounce:
                                ]
                            else:
                                if physical_contact:
                                    rejection_reasons = []
                                    if not wrench_contact_candidate:
                                        rejection_reasons.append(
                                            "wrench_contact_candidate_false"
                                        )
                                    if (
                                        signed_compression
                                        < contact_on_compression_n
                                    ):
                                        rejection_reasons.append(
                                            "signed_compression_below_"
                                            "contact_on"
                                        )
                                    ft_sample = last_sample[
                                        "wrist_ft_sample"
                                    ]
                                    rejected_physical_contact_samples.append(
                                        {
                                            "global_step": ft_sample[
                                                "global_step"
                                            ],
                                            "rejection_reasons": (
                                                rejection_reasons
                                            ),
                                            "wrench_contact_candidate": bool(
                                                wrench_contact_candidate
                                            ),
                                            "signed_compression_n": float(
                                                signed_compression
                                            ),
                                            "minimum_required_compression_n": (
                                                contact_on_compression_n
                                            ),
                                            "raw_wrench": ft_sample[
                                                "raw_wrench"
                                            ],
                                            "canonical_wrench_sensor": (
                                                ft_sample[
                                                    "canonical_wrench_sensor"
                                                ]
                                            ),
                                            "compensated_wrench_sensor": (
                                                ft_sample[
                                                    "compensated_wrench_sensor"
                                                ]
                                            ),
                                            "compensated_wrench_task": [
                                                float(value)
                                                for value in last_sample[
                                                    "wrench"
                                                ]
                                            ],
                                            "estimated_gap_m": float(
                                                observation.estimated_gap_m
                                            ),
                                            "loose_fixed_contact_records": (
                                                last_sample[
                                                    "loose_fixed_contact_"
                                                    "records"
                                                ]
                                            ),
                                            (
                                                "intended_lip_contact_pairs"
                                            ): list(
                                                last_sample[
                                                    "intended_lip_contact_"
                                                    "pairs"
                                                ]
                                            ),
                                            (
                                                "unexpected_loose_fixed_"
                                                "contact_pairs"
                                            ): list(
                                                last_sample[
                                                    "unexpected_loose_fixed_"
                                                    "contact_pairs"
                                                ]
                                            ),
                                            (
                                                "loose_fixture_contact_pairs"
                                            ): list(
                                                last_sample[
                                                    "loose_fixture_contact_"
                                                    "pairs"
                                                ]
                                            ),
                                            "loose_table_contact_pairs": list(
                                                last_sample[
                                                    "loose_table_contact_pairs"
                                                ]
                                            ),
                                        }
                                    )
                                    # The bounded ring prevents an extended
                                    # sub-threshold scrape from inflating the
                                    # report while retaining the last causal
                                    # samples for postmortem inspection.
                                    rejected_physical_contact_samples = (
                                        rejected_physical_contact_samples[-12:]
                                    )
                                contact_window.clear()
                            if len(contact_window) == (
                                tactile_probe.sensor.contact_debounce_samples
                            ):
                                return (
                                    last_sample,
                                    tuple(contact_window),
                                    False,
                                    motion_evidence_payload(),
                                )
                            if (
                                drive_frozen
                                and physical_contact_hold_steps
                                >= (
                                    contact_hold_limit
                                )
                            ):
                                break
                        if evaluate_release:
                            if task_z_progress_samples_m[-1] >= (
                                tactile_probe.motion
                                .unload_retract_distance_m
                            ):
                                # Unloading can recover the loaded-drive bias
                                # before the command path reaches its endpoint.
                                # Freeze at the first measured +0.3 mm crossing
                                # so actual displacement, not command FK, caps
                                # the contact retreat.
                                drive_frozen = True
                                if not retract_distance_reached:
                                    retract_distance_reached = True
                            # Release debounce starts only after measured TCP
                            # has reached the full +0.3 mm retreat.  Samples
                            # collected while still unloading cannot satisfy
                            # the terminal gate early.
                            if (
                                (
                                    retract_distance_reached
                                    or not release_after_retract
                                )
                                and not physical_contact
                                and contact_release_candidate(
                                    observation, tactile_probe
                                )
                            ):
                                release_window.append(last_sample)
                                debounce = (
                                    tactile_probe.sensor
                                    .release_debounce_samples
                                )
                                release_window = release_window[
                                    -debounce:
                                ]
                            else:
                                release_window.clear()
                            release_evidence.append(
                                {
                                    "task_z_progress_m": (
                                        task_z_progress_samples_m[-1]
                                    ),
                                    "physical_contact": physical_contact,
                                    "signed_compression_n": float(
                                        tactile_probe.motion
                                        .compressive_axial_force_sign_candidate
                                        * observation.axial_force_n
                                    ),
                                    "bending_torque_nm": float(
                                        np.linalg.norm(
                                            observation.bending_torque_xy_nm
                                        )
                                    ),
                                    "release_candidate": bool(
                                        (
                                            retract_distance_reached
                                            or not release_after_retract
                                        )
                                        and not physical_contact
                                        and contact_release_candidate(
                                            observation, tactile_probe
                                        )
                                    ),
                                    "intended_lip_contact_pairs": list(
                                        last_sample[
                                            "intended_lip_contact_pairs"
                                        ]
                                    ),
                                }
                            )
                            if retract_distance_reached:
                                post_retract_release_ticks += 1
                            if (
                                retract_distance_reached
                                and len(release_window)
                                == (
                                    tactile_probe.sensor
                                    .release_debounce_samples
                                )
                            ):
                                break
                            if (
                                retract_distance_reached
                                and post_retract_release_ticks
                                >= (
                                    tactile_probe.sensor
                                    .release_debounce_samples
                                )
                            ):
                                # Once measured motion reaches +0.3 mm the
                                # command is frozen and only this finite
                                # six-tick release window remains.  A missing
                                # debounce returns false; it never waits or
                                # advances the command indefinitely.
                                break
                        loop_cursor += 1
                    release_ready = bool(
                        evaluate_release
                        and retract_distance_reached
                        and len(release_window)
                        == tactile_probe.sensor.release_debounce_samples
                    )
                    motion_evidence = motion_evidence_payload()
                    return (
                        last_sample,
                        tuple(contact_window),
                        release_ready,
                        motion_evidence,
                    )

                def command_continuous_retract_plan(
                    retract_distance_m,
                    maximum_command_speed_m_s=None,
                    strict_discrete_headroom=False,
                ):
                    """Plan +Z without replacing the loaded drive command.

                    The measured arm is evidence only.  A stiff position drive
                    supports its load with a persistent command/measured bias;
                    resetting its target to measured q would remove that bias
                    in one tick and can drive the TCP down before a +Z retreat.
                    """

                    command_start = current_arm.copy()
                    if command_start.shape != (7,) or not np.all(
                        np.isfinite(command_start)
                    ):
                        raise RuntimeError(
                            "retract command start is not finite shape (7,)"
                        )
                    measured_all = np.asarray(
                        robot.get_joint_positions(), dtype=np.float64
                    )
                    measured_arm = measured_all[arm_indices].copy()
                    if measured_arm.shape != (7,) or not np.all(
                        np.isfinite(measured_arm)
                    ):
                        raise RuntimeError(
                            "retract measured arm is not finite shape (7,)"
                        )
                    actual_tcp, _ = _world_pose(
                        Gf, Usd, UsdGeom, tcp_prim
                    )
                    actual_tcp = np.asarray(actual_tcp, dtype=np.float64)
                    measured_fk = np.asarray(
                        iiwa14_grasp_tcp_transform(
                            seven_float_arm_tuple(
                                measured_arm, "retract_measured_arm"
                            )
                        ),
                        dtype=np.float64,
                    )[:3, 3]
                    command_start_transform = np.asarray(
                        iiwa14_grasp_tcp_transform(
                            seven_float_arm_tuple(
                                command_start, "retract_command_start"
                            )
                        ),
                        dtype=np.float64,
                    )

                    def solve_retract_target(
                        start_arm, target_position, start_rotation
                    ):
                        solved = np.asarray(
                            solve_fixed_q7_tcp_pose(
                                seven_float_arm_tuple(
                                    start_arm, "retract_ik_seed"
                                ),
                                three_float_position_tuple(
                                    target_position,
                                    "retract_ik_target_position",
                                ),
                                target_rotation=np.asarray(
                                    start_rotation, dtype=np.float64
                                ),
                                maximum_iterations=(
                                    preinsert_probe.ik.maximum_iterations
                                ),
                                damping=preinsert_probe.ik.damping,
                            ),
                            dtype=np.float64,
                        )
                        if (
                            solved.shape != (7,)
                            or not np.all(np.isfinite(solved))
                            or not np.isclose(
                                solved[6], command_start[6], atol=1.0e-12
                            )
                            or np.max(np.abs(solved - center_arm))
                            > (
                                preinsert_probe.ik
                                .maximum_abs_joint_delta_from_nominal_rad
                            )
                        ):
                            raise RuntimeError(
                                "retract target left local fixed-q7 bounds"
                            )
                        return solved

                    command_speed_m_s = (
                        tactile_probe.motion.unload_retract_speed_m_s
                        if maximum_command_speed_m_s is None
                        else float(maximum_command_speed_m_s)
                    )
                    path = build_command_continuous_retract_path(
                        command_start,
                        task_z_world,
                        retract_distance_m,
                        rate_hz,
                        command_speed_m_s,
                        iiwa14_grasp_tcp_transform,
                        solve_retract_target,
                        maximum_fk_position_error_m=(
                            preinsert_probe.ik.maximum_fk_position_error_m
                        ),
                        maximum_fk_orientation_error_rad=(
                            preinsert_probe.ik
                            .maximum_fk_orientation_error_rad
                        ),
                        strict_discrete_headroom=(
                            strict_discrete_headroom
                        ),
                    )
                    diagnostics = {
                        "last_applied_command_arm_rad": [
                            float(value) for value in command_start
                        ],
                        "measured_start_arm_rad": [
                            float(value) for value in measured_arm
                        ],
                        "command_continuous_start_arm_rad": [
                            float(value) for value in command_start
                        ],
                        "measured_q7_tracking_error_rad": float(
                            measured_arm[6] - center_arm[6]
                        ),
                        "command_vs_measured_max_abs_rad": float(
                            np.max(np.abs(command_start - measured_arm))
                        ),
                        "measured_start_tcp_prim_world_m": [
                            float(value) for value in actual_tcp
                        ],
                        "measured_arm_fk_tcp_world_m": [
                            float(value) for value in measured_fk
                        ],
                        "measured_fk_to_tcp_prim_error_m": float(
                            np.linalg.norm(measured_fk - actual_tcp)
                        ),
                        "command_start_fk_tcp_world_m": [
                            float(value)
                            for value in command_start_transform[:3, 3]
                        ],
                        "requested_command_target_tcp_world_m": list(
                            path["requested_target_position_m"]
                        ),
                        "target_command_fk_tcp_world_m": list(
                            path["target_fk_position_m"]
                        ),
                        "command_fk_axial_progress_m": list(
                            path["command_fk_axial_progress_m"]
                        ),
                        "command_fk_lateral_error_m": list(
                            path["command_fk_lateral_error_m"]
                        ),
                        "command_fk_orientation_error_rad": list(
                            path["command_fk_orientation_error_rad"]
                        ),
                        "peak_command_fk_axial_speed_m_s": path[
                            "peak_command_fk_axial_speed_m_s"
                        ],
                        "maximum_commanded_tcp_speed_m_s": (
                            command_speed_m_s
                        ),
                        "first_command_exact": path["first_command_exact"],
                        "base_required_command_count": path[
                            "base_required_steps"
                        ],
                        "added_headroom_command_count": path[
                            "added_discrete_headroom_steps"
                        ],
                        "final_moving_command_count": path[
                            "steps_excluding_exact_start_hold"
                        ],
                        "target_q7_rad": float(path["target_arm_rad"][6]),
                        "registered_center_q7_rad": float(center_arm[6]),
                        "measured_state_used_for_command": False,
                    }
                    return path, actual_tcp, diagnostics

                def perform_tactile_abort_retract(
                    *, manifold_capture_success=False
                ):
                    """Execute the state machine's bounded +Z terminal abort.

                    This recovery does not search, center, or resume a trial.
                    It commands exactly the contract's 0.3 mm task-axis
                    retract, verifies finite/contact/2 Nm gates on every step,
                    and records whether force plus physical contact released.
                    """

                    nonlocal current_arm
                    nonlocal phase
                    manifold_mode = bool(
                        arguments.tactile_lip_manifold_capture
                    )
                    command_speed_m_s = (
                        TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S
                        if manifold_mode
                        else None
                    )
                    path, start_tcp, diagnostics = (
                        command_continuous_retract_plan(
                            tactile_probe.motion.unload_retract_distance_m,
                            command_speed_m_s,
                            strict_discrete_headroom=manifold_mode,
                        )
                    )
                    phase = (
                        "mixed_grip_preinsert_tactile_manifold_"
                        "terminal_retract"
                        if manifold_capture_success
                        else "mixed_grip_preinsert_tactile_lip_abort_retract"
                    )
                    abort_peaks = {}
                    (
                        final_sample,
                        _,
                        release_ready,
                        motion_evidence,
                    ) = run_tactile_motion(
                        None,
                        None,
                        search_offset_xy_m=(
                            TACTILE_LIP_OFFSET_M,
                            0.0,
                        )
                        if manifold_mode
                        else (0.0, 0.0),
                        contact_attempts=0,
                        trial_peaks=abort_peaks,
                        allow_fixed_contact=True,
                        evaluate_release=True,
                        command_path=path["commands"],
                        evidence_sink=tactile_abort_runtime_evidence,
                        record_full_progress=manifold_mode,
                        record_raw_guarded_samples=manifold_mode,
                        release_after_retract=manifold_mode,
                        minimum_gap_m=(
                            tactile_probe.motion.entry_gap_m
                            if manifold_mode
                            else None
                        ),
                        maximum_command_speed_m_s=(
                            TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S
                            if manifold_mode
                            else None
                        ),
                        maximum_measured_speed_m_s=(
                            tactile_probe.motion.unload_retract_speed_m_s
                            if manifold_mode
                            else None
                        ),
                        maximum_negative_progress_m=(
                            TACTILE_FIXED_NEGATIVE_RETRACT_BOUND_M
                            if manifold_mode
                            else None
                        ),
                        raw_window_role=(
                            "stage_a_terminal_retract"
                            if manifold_mode
                            else None
                        ),
                    )
                    measured_retract_m = float(
                        np.dot(
                            final_sample["tcp_position"]
                            - start_tcp,
                            task_z_world,
                        )
                    )
                    retract_report = {
                        "attempted": True,
                        "commanded_retract_m": (
                            tactile_probe.motion.unload_retract_distance_m
                        ),
                        "measured_robot_fk_retract_m": measured_retract_m,
                        "measured_robot_fk_retract_m_deprecated": True,
                        "measured_tcp_prim_retract_m": measured_retract_m,
                        "minimum_task_z_progress_m": motion_evidence[
                            "minimum_task_z_progress_m"
                        ],
                        "final_task_z_progress_m": motion_evidence[
                            "final_task_z_progress_m"
                        ],
                        "release_samples": motion_evidence[
                            "release_samples"
                        ],
                        "transition_ring": tactile_abort_runtime_evidence.get(
                            "transition_ring", []
                        ),
                        "start_and_target_diagnostics": diagnostics,
                        "release_debounced": release_ready,
                        "terminal_state": (
                            "TERMINAL_MANIFOLD_CAPTURE"
                            if manifold_capture_success
                            else "TERMINAL_ABORT"
                        ),
                        "resume_attempted": False,
                    }
                    if manifold_mode:
                        retract_report.update(
                            {
                                "motion_evidence": motion_evidence,
                                "peak_experimental_observations": {
                                    name: float(
                                        abort_peaks.get(name, 0.0)
                                    )
                                    for name in (
                                        TACTILE_PREFLIGHT_PEAK_FIELDS
                                    )
                                },
                            }
                        )
                    return retract_report

                tactile_abort_retract = perform_tactile_abort_retract

                trials = tactile_report["trials"]
                if arguments.tactile_retract_preflight:
                    # This independent mode proves only the loaded position
                    # drive's mid-trajectory reversal behavior outside entry.
                    # No lip contact is allowed and no calibration touch runs.
                    tactile_touch_motion_started = True
                    preflight_peaks = {}
                    # Match the exact dynamic state that failed in the first
                    # Stage-A unload: interrupt the *full* 2.5 mm axial plan at
                    # its unique discrete peak-speed command, then hold that
                    # drive target for five more ticks.  The moving interrupt
                    # tick plus those five frozen ticks is the same six-frame
                    # age used by contact capture.  This remains >10 mm from
                    # entry and never enables collision or preload.
                    matched_report = {
                        "passed": False,
                        "status": "RUNNING_MATCHED_MID_SLOPE_REVERSAL",
                        "fixed_negative_progress_ceiling_m": (
                            TACTILE_FIXED_NEGATIVE_RETRACT_BOUND_M
                        ),
                        "effective_negative_progress_ceiling_m": (
                            TACTILE_FIXED_NEGATIVE_RETRACT_BOUND_M
                        ),
                        "observed_negative_reversal_progress_m": None,
                        "terminal_retract": {
                            "attempted": False,
                            "started": False,
                            "completed": False,
                            "hard_stop": False,
                            "zero_step_abort": False,
                            "release_debounced": False,
                            "resume_attempted": False,
                            "terminal_state": "NOT_STARTED",
                            "failure_phase": None,
                            "failure_global_step": None,
                            "failure_motion_index": None,
                            "failure_reason": None,
                            "world_steps_after_original_failure": 0,
                        },
                    }
                    tactile_report["matched_profile_reversal"] = (
                        matched_report
                    )
                    matched_approach_target = (
                        center_tcp_target
                        - approach_travel_m * task_z_world
                    )
                    matched_approach_arm = solve_tactile_arm(
                        current_arm, matched_approach_target
                    )
                    matched_headroom = (
                        minimum_jerk_steps_with_strict_headroom(
                            approach_travel_m,
                            rate_hz,
                            TACTILE_LIP_MANIFOLD_APPROACH_SPEED_M_S,
                        )
                    )
                    matched_interrupt_plan = (
                        build_discrete_mid_slope_interrupt_plan(
                            current_arm,
                            matched_approach_arm,
                            matched_headroom["final_command_count"],
                            task_z_world,
                            rate_hz,
                            iiwa14_grasp_tcp_transform,
                            strict_headroom_plan=matched_headroom,
                        )
                    )
                    if matched_interrupt_plan[
                        "maximum_downward_speed_m_s"
                    ] > TACTILE_LIP_MANIFOLD_APPROACH_SPEED_M_S:
                        raise TactileSafetyStop(
                            "matched reversal planned command exceeded the "
                            "axial profile ceiling",
                            zero_step_abort=True,
                        )
                    matched_report["discrete_interrupt_plan"] = (
                        matched_interrupt_plan
                    )
                    matched_moving_sink = {}
                    matched_report["moving_interrupt_runtime"] = (
                        matched_moving_sink
                    )
                    phase = (
                        "mixed_grip_preinsert_tactile_retract_preflight_"
                        "matched_mid_slope_approach"
                    )
                    (
                        matched_moving_sample,
                        _,
                        _,
                        matched_moving_evidence,
                    ) = run_tactile_motion(
                        matched_approach_arm,
                        float(matched_headroom["final_command_count"])
                        / float(rate_hz),
                        search_offset_xy_m=(0.0, 0.0),
                        contact_attempts=0,
                        trial_peaks=preflight_peaks,
                        allow_fixed_contact=False,
                        evidence_sink=matched_moving_sink,
                        record_full_progress=True,
                        record_raw_guarded_samples=True,
                        minimum_gap_m=tactile_probe.motion.entry_gap_m,
                        maximum_command_speed_m_s=(
                            TACTILE_LIP_MANIFOLD_APPROACH_SPEED_M_S
                        ),
                        maximum_measured_speed_m_s=(
                            tactile_probe.motion.unload_retract_speed_m_s
                        ),
                        command_count_override=(
                            matched_headroom["final_command_count"]
                        ),
                        stop_after_motion_index=matched_interrupt_plan[
                            "unique_argmax_motion_index"
                        ],
                        raw_window_role="matched_moving_toward_interrupt",
                    )
                    matched_report["moving_interrupt_evidence"] = (
                        matched_moving_evidence
                    )
                    matched_freeze_sink = {}
                    matched_report["frozen_hold_runtime"] = (
                        matched_freeze_sink
                    )
                    phase = (
                        "mixed_grip_preinsert_tactile_retract_preflight_"
                        "matched_frozen_hold"
                    )
                    frozen_commands = tuple(
                        current_arm.copy()
                        for _ in range(
                            TACTILE_MATCHED_REVERSAL_WINDOW_SAMPLES - 1
                        )
                    )
                    (
                        matched_frozen_sample,
                        _,
                        _,
                        matched_frozen_evidence,
                    ) = run_tactile_motion(
                        None,
                        None,
                        search_offset_xy_m=(0.0, 0.0),
                        contact_attempts=0,
                        trial_peaks=preflight_peaks,
                        allow_fixed_contact=False,
                        command_path=frozen_commands,
                        evidence_sink=matched_freeze_sink,
                        record_full_progress=True,
                        record_raw_guarded_samples=True,
                        minimum_gap_m=tactile_probe.motion.entry_gap_m,
                        maximum_command_speed_m_s=(
                            TACTILE_LIP_MANIFOLD_APPROACH_SPEED_M_S
                        ),
                        maximum_measured_speed_m_s=(
                            tactile_probe.motion.unload_retract_speed_m_s
                        ),
                        raw_window_role="matched_frozen_drive_target",
                    )
                    matched_report["frozen_hold_evidence"] = (
                        matched_frozen_evidence
                    )
                    moving_raw = matched_moving_evidence[
                        "guarded_tick_samples"
                    ]
                    frozen_raw = matched_frozen_evidence[
                        "guarded_tick_samples"
                    ]
                    matched_window_raw = (
                        moving_raw[-1:],
                        frozen_raw,
                    )
                    matched_window_raw = list(
                        matched_window_raw[0] + matched_window_raw[1]
                    )
                    if (
                        len(matched_window_raw)
                        != TACTILE_MATCHED_REVERSAL_WINDOW_SAMPLES
                        or any(
                            sample["global_step"]
                            != matched_window_raw[0]["global_step"] + index
                            for index, sample in enumerate(matched_window_raw)
                        )
                        or any(
                            sample["command_arm_rad"]
                            != matched_window_raw[0]["command_arm_rad"]
                            for sample in matched_window_raw
                        )
                    ):
                        raise TactileSafetyStop(
                            "matched reversal exact-six frozen window failed",
                            zero_step_abort=True,
                        )
                    matched_state_limits = tactile_reversal_state_features(
                        matched_window_raw,
                        task_rotation,
                        rate_hz,
                    )
                    matched_report.update(
                        {
                            "equivalence_window_semantics": (
                                "moving_argmax_tick_plus_five_frozen_ticks"
                            ),
                            "equivalence_window_raw": matched_window_raw,
                            "state_equivalence_limits": matched_state_limits,
                            "planned_argmax_neighborhood": (
                                matched_interrupt_plan[
                                    "planned_argmax_neighborhood"
                                ]
                            ),
                            "executed_interrupt_neighborhood_raw": list(
                                moving_raw[-2:] + frozen_raw[:1]
                            ),
                            "last_frozen_downward_drift_m": (
                                matched_state_limits[
                                    "negative_task_z_drift_after_frozen_tick_m"
                                ]
                            ),
                        }
                    )
                    # From this point onward every failure is terminal.  In
                    # particular, the fixed negative-progress guard must never
                    # be followed by a generic recovery/abort world step.
                    tactile_terminal_retract_started = True
                    matched_report["terminal_retract"].update(
                        {
                            "attempted": True,
                            "started": True,
                            "terminal_state": "RUNNING_FIXED_BOUND_RETRACT",
                        }
                    )
                    matched_retract_path, _, matched_retract_diagnostics = (
                        command_continuous_retract_plan(
                            tactile_probe.motion.unload_retract_distance_m,
                            TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S,
                            strict_discrete_headroom=True,
                        )
                    )
                    matched_retract_sink = {}
                    matched_report["terminal_retract_runtime"] = (
                        matched_retract_sink
                    )
                    phase = (
                        "mixed_grip_preinsert_tactile_retract_preflight_"
                        "matched_command_continuous_retract"
                    )
                    (
                        matched_retract_sample,
                        _,
                        _,
                        matched_retract_evidence,
                    ) = run_tactile_motion(
                        None,
                        None,
                        search_offset_xy_m=(0.0, 0.0),
                        contact_attempts=0,
                        trial_peaks=preflight_peaks,
                        allow_fixed_contact=False,
                        command_path=matched_retract_path["commands"],
                        evidence_sink=matched_retract_sink,
                        record_full_progress=True,
                        record_raw_guarded_samples=True,
                        minimum_gap_m=tactile_probe.motion.entry_gap_m,
                        maximum_command_speed_m_s=(
                            TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S
                        ),
                        maximum_measured_speed_m_s=(
                            tactile_probe.motion.unload_retract_speed_m_s
                        ),
                        maximum_negative_progress_m=(
                            TACTILE_FIXED_NEGATIVE_RETRACT_BOUND_M
                        ),
                        raw_window_role="matched_terminal_retract",
                    )
                    matched_report["terminal_retract_evidence"] = (
                        matched_retract_evidence
                    )
                    matched_observed_negative_m = max(
                        0.0,
                        -float(
                            matched_retract_evidence[
                                "minimum_task_z_progress_m"
                            ]
                        ),
                    )
                    matched_report.update(
                        {
                            "observed_negative_reversal_progress_m": (
                                matched_observed_negative_m
                            ),
                            "terminal_retract_diagnostics": (
                                matched_retract_diagnostics
                            ),
                        }
                    )
                    matched_report["terminal_retract"].update(
                        {
                            "completed": True,
                            "terminal_state": "RETRACT_COMPLETED",
                            "minimum_task_z_progress_m": (
                                matched_retract_evidence[
                                    "minimum_task_z_progress_m"
                                ]
                            ),
                            "partial_motion_evidence": (
                                matched_retract_evidence
                            ),
                        }
                    )
                    current_command_fk = np.asarray(
                        iiwa14_grasp_tcp_transform(
                            seven_float_arm_tuple(
                                current_arm, "matched_recovery_command"
                            )
                        ),
                        dtype=np.float64,
                    )[:3, 3]
                    matched_recovery_distance_m = float(
                        np.dot(
                            center_tcp_target - current_command_fk,
                            task_z_world,
                        )
                    )
                    if matched_recovery_distance_m <= 0.0:
                        raise TactileSafetyStop(
                            "matched reversal recovery distance is invalid",
                            zero_step_abort=True,
                        )
                    matched_recovery_path, _, matched_recovery_diagnostics = (
                        command_continuous_retract_plan(
                            matched_recovery_distance_m,
                            TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S,
                            strict_discrete_headroom=True,
                        )
                    )
                    matched_recovery_sink = {}
                    matched_report["recovery_runtime"] = (
                        matched_recovery_sink
                    )
                    phase = (
                        "mixed_grip_preinsert_tactile_retract_preflight_"
                        "matched_recover_preinsert"
                    )
                    (
                        matched_final_sample,
                        _,
                        _,
                        matched_recovery_evidence,
                    ) = run_tactile_motion(
                        None,
                        None,
                        search_offset_xy_m=(0.0, 0.0),
                        contact_attempts=0,
                        trial_peaks=preflight_peaks,
                        allow_fixed_contact=False,
                        command_path=matched_recovery_path["commands"],
                        evidence_sink=matched_recovery_sink,
                        record_full_progress=True,
                        record_raw_guarded_samples=True,
                        minimum_gap_m=tactile_probe.motion.entry_gap_m,
                        maximum_command_speed_m_s=(
                            TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S
                        ),
                        maximum_measured_speed_m_s=(
                            tactile_probe.motion.unload_retract_speed_m_s
                        ),
                        raw_window_role="matched_recover_preinsert",
                    )
                    matched_report.update(
                        {
                            "recovery_evidence": matched_recovery_evidence,
                            "recovery_diagnostics": (
                                matched_recovery_diagnostics
                            ),
                            "final_preinsert_gap_error_m": abs(
                                matched_final_sample[
                                    "observation"
                                ].estimated_gap_m
                                - tactile_probe.motion.preinsert_gap_m
                            ),
                        }
                    )
                    matched_phase_evidence = (
                        matched_moving_evidence,
                        matched_frozen_evidence,
                        matched_retract_evidence,
                        matched_recovery_evidence,
                    )
                    matched_command_ceilings = (
                        TACTILE_LIP_MANIFOLD_APPROACH_SPEED_M_S,
                        TACTILE_LIP_MANIFOLD_APPROACH_SPEED_M_S,
                        TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S,
                        TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S,
                    )
                    matched_safety_gate = bool(
                        matched_observed_negative_m
                        <= TACTILE_FIXED_NEGATIVE_RETRACT_BOUND_M
                        and matched_report["final_preinsert_gap_error_m"]
                        <= (
                            preinsert_probe.insertion.acceptance
                            .maximum_preinsert_gap_error_m
                        )
                        and all(
                            evidence["all_samples_finite"] is True
                            and evidence[
                                "minimum_body_contact_finger_count"
                            ]
                            == 3
                            and evidence[
                                "applied_action_all_float32_equivalent"
                            ]
                            is True
                            and evidence[
                                "applied_action_maximum_abs_arm_error_"
                                "float32_rad"
                            ]
                            == 0.0
                            and not any(
                                evidence["contact_record_totals"].values()
                            )
                            and min(evidence["estimated_gap_samples_m"])
                            >= tactile_probe.motion.entry_gap_m
                            and evidence["peak_abs_tcp_speed_m_s"]
                            <= tactile_probe.motion.unload_retract_speed_m_s
                            and evidence[
                                "peak_abs_command_fk_tcp_speed_m_s"
                            ]
                            <= command_ceiling
                            for evidence, command_ceiling in zip(
                                matched_phase_evidence,
                                matched_command_ceilings,
                            )
                        )
                    )
                    matched_report.update(
                        {
                            "phase_names": [
                                "mid_slope_approach",
                                "exact_five_tick_frozen_hold",
                                "command_continuous_retract",
                                "recover_preinsert",
                            ],
                            "phase_sample_counts": [
                                evidence["checked_sample_count"]
                                for evidence in matched_phase_evidence
                            ],
                            "all_samples_outside_entry_gate": all(
                                min(evidence["estimated_gap_samples_m"])
                                >= tactile_probe.motion.entry_gap_m
                                for evidence in matched_phase_evidence
                            ),
                            "actual_speed_hard_ceiling_m_s": (
                                tactile_probe.motion.unload_retract_speed_m_s
                            ),
                            "approach_command_speed_ceiling_m_s": (
                                TACTILE_LIP_MANIFOLD_APPROACH_SPEED_M_S
                            ),
                            "other_command_speed_ceiling_m_s": (
                                TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S
                            ),
                            "safety_gate": matched_safety_gate,
                            "passed": matched_safety_gate,
                            "status": (
                                "PASSED_MATCHED_MID_SLOPE_REVERSAL"
                                if matched_safety_gate
                                else "REJECTED_FAIL_CLOSED"
                            ),
                        }
                    )
                    matched_report["terminal_retract"].update(
                        {
                            "terminal_state": "RECOVERED_PREINSERT",
                            "resume_attempted": False,
                        }
                    )
                    if not matched_safety_gate:
                        raise TactileSafetyStop(
                            "matched reversal evidence failed closed",
                            zero_step_abort=True,
                        )
                    phase = (
                        "mixed_grip_preinsert_tactile_retract_preflight_"
                        "descent"
                    )
                    preflight_descent_target = (
                        center_tcp_target
                        - TACTILE_RETRACT_PREFLIGHT_COMMAND_DESCENT_M
                        * task_z_world
                    )
                    preflight_descent_arm = solve_tactile_arm(
                        center_arm, preflight_descent_target
                    )
                    descent_steps = minimum_jerk_steps_for_peak_speed(
                        TACTILE_RETRACT_PREFLIGHT_COMMAND_DESCENT_M,
                        rate_hz,
                        TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S,
                    )
                    descent_duration_s = (
                        float(descent_steps) / float(rate_hz)
                    )
                    (
                        descent_sample,
                        _,
                        _,
                        descent_evidence,
                    ) = run_tactile_motion(
                        preflight_descent_arm,
                        descent_duration_s,
                        search_offset_xy_m=(0.0, 0.0),
                        contact_attempts=0,
                        trial_peaks=preflight_peaks,
                        allow_fixed_contact=False,
                        evidence_sink=tactile_report,
                        record_full_progress=True,
                    )
                    if (
                        descent_sample["loose_fixed_contact_records"] != 0
                        or descent_sample["observation"].estimated_gap_m
                        < (
                            tactile_probe.motion.preinsert_gap_m
                            - TACTILE_RETRACT_PREFLIGHT_DESCENT_M
                            - preinsert_probe.insertion.acceptance
                            .maximum_preinsert_gap_error_m
                        )
                    ):
                        raise RuntimeError(
                            "retract preflight descent left no-contact scope"
                        )
                    preflight_path, reversal_start_tcp, diagnostics = (
                        command_continuous_retract_plan(
                            tactile_probe.motion.unload_retract_distance_m,
                            TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S,
                        )
                    )
                    phase = (
                        "mixed_grip_preinsert_tactile_retract_preflight_"
                        "command_continuous_reversal"
                    )
                    (
                        reversal_sample,
                        _,
                        _,
                        reversal_evidence,
                    ) = run_tactile_motion(
                        None,
                        None,
                        search_offset_xy_m=(0.0, 0.0),
                        contact_attempts=0,
                        trial_peaks=preflight_peaks,
                        allow_fixed_contact=False,
                        command_path=preflight_path["commands"],
                        evidence_sink=tactile_report,
                        record_full_progress=True,
                    )
                    measured_reversal_m = float(
                        np.dot(
                            reversal_sample["tcp_position"]
                            - reversal_start_tcp,
                            task_z_world,
                        )
                    )
                    observed_negative_progress_bound_m = max(
                        0.0,
                        -float(
                            reversal_evidence[
                                "minimum_task_z_progress_m"
                            ]
                        ),
                    )
                    phase = (
                        "mixed_grip_preinsert_tactile_retract_preflight_"
                        "recover_preinsert"
                    )
                    return_distance_m = max(
                        0.0,
                        tactile_probe.motion.preinsert_gap_m
                        - reversal_sample["observation"].estimated_gap_m,
                    )
                    recovery_steps = minimum_jerk_steps_for_peak_speed(
                        max(return_distance_m, 1.0e-12),
                        rate_hz,
                        TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S,
                    )
                    recovery_duration_s = (
                        float(recovery_steps) / float(rate_hz)
                    )
                    final_preflight_sample, _, _, recovery_evidence = (
                        run_tactile_motion(
                            center_arm,
                            recovery_duration_s,
                            search_offset_xy_m=(0.0, 0.0),
                            contact_attempts=0,
                            trial_peaks=preflight_peaks,
                            allow_fixed_contact=False,
                            evidence_sink=tactile_report,
                            record_full_progress=True,
                        )
                    )
                    final_gap_error_m = abs(
                        final_preflight_sample[
                            "observation"
                        ].estimated_gap_m
                        - tactile_probe.motion.preinsert_gap_m
                    )
                    preflight_phase_evidence = (
                        descent_evidence,
                        reversal_evidence,
                        recovery_evidence,
                    )
                    all_preflight_gap_samples_m = tuple(
                        float(gap)
                        for evidence in preflight_phase_evidence
                        for gap in evidence["estimated_gap_samples_m"]
                    )
                    minimum_measured_gap_m = min(
                        all_preflight_gap_samples_m
                    )
                    measured_descent_tcp_prim_m = max(
                        0.0,
                        -float(
                            descent_evidence[
                                "minimum_task_z_progress_m"
                            ]
                        ),
                    )
                    all_trajectory_samples_outside_entry_gate = bool(
                        minimum_measured_gap_m
                        >= tactile_probe.motion.entry_gap_m
                    )
                    measured_descent_bound_gate = bool(
                        measured_descent_tcp_prim_m
                        <= TACTILE_RETRACT_PREFLIGHT_DESCENT_M
                    )
                    reversal_negative_progress_bound_gate = bool(
                        observed_negative_progress_bound_m
                        <= TACTILE_RETRACT_PREFLIGHT_DESCENT_M
                    )
                    measured_phase_peak_speeds_m_s = {
                        name: float(
                            evidence["peak_abs_task_z_speed_m_s"]
                        )
                        for name, evidence in zip(
                            ("descent", "reversal", "recovery"),
                            preflight_phase_evidence,
                        )
                    }
                    all_phase_measured_speed_gates = bool(
                        all(
                            value
                            <= tactile_probe.motion.unload_retract_speed_m_s
                            for value in (
                                measured_phase_peak_speeds_m_s.values()
                            )
                        )
                    )
                    command_fk_phase_peak_speeds_m_s = {
                        name: float(
                            evidence[
                                "peak_abs_command_fk_task_z_speed_m_s"
                            ]
                        )
                        for name, evidence in zip(
                            ("descent", "reversal", "recovery"),
                            preflight_phase_evidence,
                        )
                    }
                    all_phase_command_speed_gates = bool(
                        all(
                            value
                            <= TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S
                            for value in (
                                command_fk_phase_peak_speeds_m_s.values()
                            )
                        )
                    )
                    # A short transition ring is diagnostic, not proof of the
                    # whole no-contact run.  These cumulative counters and
                    # extrema therefore cover every guarded tick in all three
                    # phases and become part of the preflight PASS decision.
                    total_checked_sample_count = sum(
                        int(evidence["checked_sample_count"])
                        for evidence in preflight_phase_evidence
                    )
                    total_finite_sample_count = sum(
                        int(evidence["finite_sample_count"])
                        for evidence in preflight_phase_evidence
                    )
                    phase_sample_count_gates = bool(
                        total_checked_sample_count > 0
                        and all(
                            int(evidence["checked_sample_count"])
                            == len(evidence["task_z_progress_samples_m"])
                            == len(
                                evidence[
                                    "measured_tcp_prim_world_samples_m"
                                ]
                            )
                            == len(evidence["estimated_gap_samples_m"])
                            == len(
                                evidence[
                                    "command_fk_task_z_progress_samples_m"
                                ]
                            )
                            == len(
                                evidence[
                                    "command_fk_tcp_world_samples_m"
                                ]
                            )
                            for evidence in preflight_phase_evidence
                        )
                    )
                    all_samples_finite_gate = bool(
                        phase_sample_count_gates
                        and total_finite_sample_count
                        == total_checked_sample_count
                        and all(
                            evidence["all_samples_finite"] is True
                            for evidence in preflight_phase_evidence
                        )
                    )
                    minimum_body_contact_finger_count = min(
                        int(
                            evidence[
                                "minimum_body_contact_finger_count"
                            ]
                        )
                        for evidence in preflight_phase_evidence
                    )
                    three_finger_body_contact_gate = bool(
                        minimum_body_contact_finger_count == 3
                    )
                    applied_action_precheck_count = sum(
                        int(evidence["applied_action_precheck_count"])
                        for evidence in preflight_phase_evidence
                    )
                    applied_action_postcheck_count = sum(
                        int(evidence["applied_action_postcheck_count"])
                        for evidence in preflight_phase_evidence
                    )
                    applied_action_max_error_float32_rad = max(
                        float(
                            evidence[
                                "applied_action_maximum_abs_arm_error_"
                                "float32_rad"
                            ]
                        )
                        for evidence in preflight_phase_evidence
                    )
                    applied_action_evidence_gate = bool(
                        applied_action_precheck_count
                        == total_checked_sample_count
                        and applied_action_postcheck_count
                        == total_checked_sample_count
                        and all(
                            evidence[
                                "applied_action_all_float32_equivalent"
                            ]
                            is True
                            for evidence in preflight_phase_evidence
                        )
                        and applied_action_max_error_float32_rad == 0.0
                    )
                    contact_record_totals = {
                        name: sum(
                            int(evidence["contact_record_totals"][name])
                            for evidence in preflight_phase_evidence
                        )
                        for name in TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS
                    }
                    all_contact_record_totals_zero_gate = bool(
                        all(
                            value == 0
                            for value in contact_record_totals.values()
                        )
                    )
                    experimental_abort_ceilings = {
                        "absolute_axial_force_n": (
                            tactile_probe.abort.maximum_absolute_axial_force_n
                        ),
                        "lateral_force_n": (
                            tactile_probe.abort.maximum_lateral_force_n
                        ),
                        "bending_torque_nm": (
                            tactile_probe.abort.maximum_bending_torque_nm
                        ),
                        "absolute_tightening_torque_nm": (
                            tactile_probe.abort.maximum_tightening_torque_nm
                        ),
                        "absolute_finger_base_torque_nm": (
                            tactile_probe.abort.maximum_finger_base_torque_nm
                        ),
                    }
                    peak_experimental_observations = {
                        name: float(preflight_peaks.get(name, float("nan")))
                        for name in TACTILE_PREFLIGHT_PEAK_FIELDS
                    }
                    experimental_abort_envelope_gate = bool(
                        set(preflight_peaks)
                        == set(TACTILE_PREFLIGHT_PEAK_FIELDS)
                        and all(
                            np.isfinite(peak_experimental_observations[name])
                            and peak_experimental_observations[name] >= 0.0
                            and peak_experimental_observations[name]
                            <= experimental_abort_ceilings[name]
                            for name in TACTILE_PREFLIGHT_PEAK_FIELDS
                        )
                    )
                    preflight_passed = bool(
                        descent_sample["loose_fixed_contact_records"] == 0
                        and reversal_sample["loose_fixed_contact_records"] == 0
                        and final_preflight_sample[
                            "loose_fixed_contact_records"
                        ]
                        == 0
                        and measured_reversal_m >= 0.0
                        and all_trajectory_samples_outside_entry_gate
                        and measured_descent_bound_gate
                        and reversal_negative_progress_bound_gate
                        and all_phase_measured_speed_gates
                        and all_phase_command_speed_gates
                        and phase_sample_count_gates
                        and all_samples_finite_gate
                        and three_finger_body_contact_gate
                        and applied_action_evidence_gate
                        and all_contact_record_totals_zero_gate
                        and experimental_abort_envelope_gate
                        and final_gap_error_m
                        <= (
                            preinsert_probe.insertion.acceptance
                            .maximum_preinsert_gap_error_m
                        )
                    )
                    tactile_report.update(
                        {
                            "status": (
                                "PASSED_NO_CONTACT_COMMAND_REVERSAL_"
                                "AT_PREINSERT"
                                if preflight_passed
                                else "REJECTED_FAIL_CLOSED"
                            ),
                            "passed": preflight_passed,
                            "commanded_descent_m": (
                                TACTILE_RETRACT_PREFLIGHT_COMMAND_DESCENT_M
                            ),
                            "maximum_actual_descent_m": (
                                TACTILE_RETRACT_PREFLIGHT_DESCENT_M
                            ),
                            "maximum_commanded_tcp_speed_m_s": (
                                TACTILE_RETRACT_PREFLIGHT_COMMAND_SPEED_M_S
                            ),
                            "maximum_measured_tcp_speed_m_s": (
                                tactile_probe.motion
                                .unload_retract_speed_m_s
                            ),
                            "commanded_reversal_distance_m": (
                                tactile_probe.motion
                                .unload_retract_distance_m
                            ),
                            "measured_reversal_tcp_prim_m": (
                                measured_reversal_m
                            ),
                            "measured_descent_tcp_prim_m": (
                                measured_descent_tcp_prim_m
                            ),
                            "minimum_measured_gap_m": minimum_measured_gap_m,
                            "entry_gap_floor_m": (
                                tactile_probe.motion.entry_gap_m
                            ),
                            (
                                "all_trajectory_samples_outside_"
                                "entry_gate"
                            ): all_trajectory_samples_outside_entry_gate,
                            "measured_descent_bound_gate": (
                                measured_descent_bound_gate
                            ),
                            "reversal_negative_progress_bound_gate": (
                                reversal_negative_progress_bound_gate
                            ),
                            "measured_phase_peak_speeds_m_s": (
                                measured_phase_peak_speeds_m_s
                            ),
                            "all_phase_measured_speed_gates": (
                                all_phase_measured_speed_gates
                            ),
                            "command_fk_phase_peak_speeds_m_s": (
                                command_fk_phase_peak_speeds_m_s
                            ),
                            "all_phase_command_speed_gates": (
                                all_phase_command_speed_gates
                            ),
                            "total_checked_sample_count": (
                                total_checked_sample_count
                            ),
                            "total_finite_sample_count": (
                                total_finite_sample_count
                            ),
                            "phase_sample_count_gates": (
                                phase_sample_count_gates
                            ),
                            "all_samples_finite_gate": (
                                all_samples_finite_gate
                            ),
                            "minimum_body_contact_finger_count": (
                                minimum_body_contact_finger_count
                            ),
                            "three_finger_body_contact_gate": (
                                three_finger_body_contact_gate
                            ),
                            "applied_action_precheck_count": (
                                applied_action_precheck_count
                            ),
                            "applied_action_postcheck_count": (
                                applied_action_postcheck_count
                            ),
                            "applied_action_all_float32_equivalent": bool(
                                applied_action_evidence_gate
                            ),
                            (
                                "applied_action_maximum_abs_arm_error_"
                                "float32_rad"
                            ): (
                                applied_action_max_error_float32_rad
                            ),
                            "applied_action_evidence_gate": (
                                applied_action_evidence_gate
                            ),
                            "contact_record_totals": contact_record_totals,
                            "all_contact_record_totals_zero_gate": (
                                all_contact_record_totals_zero_gate
                            ),
                            "peak_experimental_observations": (
                                peak_experimental_observations
                            ),
                            "experimental_abort_ceilings": (
                                experimental_abort_ceilings
                            ),
                            "experimental_abort_envelope_gate": (
                                experimental_abort_envelope_gate
                            ),
                            "trajectory_sample_counts": {
                                "descent": len(
                                    descent_evidence[
                                        "estimated_gap_samples_m"
                                    ]
                                ),
                                "reversal": len(
                                    reversal_evidence[
                                        "estimated_gap_samples_m"
                                    ]
                                ),
                                "recovery": len(
                                    recovery_evidence[
                                        "estimated_gap_samples_m"
                                    ]
                                ),
                            },
                            (
                                "observed_negative_reversal_"
                                "progress_bound_m"
                            ): observed_negative_progress_bound_m,
                            "descent_evidence": descent_evidence,
                            "reversal_evidence": reversal_evidence,
                            "recovery_evidence": recovery_evidence,
                            "reversal_start_and_target_diagnostics": (
                                diagnostics
                            ),
                            "final_preinsert_gap_error_m": (
                                final_gap_error_m
                            ),
                            "lip_contact_executed": False,
                            "touch_trials_executed": 0,
                            "engage_executed": False,
                            "insertion_executed": False,
                            "twist_executed": False,
                            "home_return_executed": False,
                            "assembly_success_claimed": False,
                        }
                    )
                    passed = bool(passed and preflight_passed)
                    report["passed"] = passed
                    raise TactilePreflightComplete

                if arguments.tactile_lip_manifold_capture:
                    # Stage A is intentionally not a force calibration.  It
                    # observes exactly one +X collision manifold, freezes the
                    # first-contact drive target for a total of six frames,
                    # then performs the sole terminal +0.3 mm unload.  No
                    # contact-dependent preload, centering, or continuation
                    # is reachable from this branch.
                    tactile_report["retract_preflight_evidence_required"] = (
                        True
                    )
                    tactile_touch_motion_started = True
                    direction_name, direction_xy = (
                        TACTILE_LIP_MANIFOLD_DIRECTION
                    )
                    direction_world = (
                        direction_xy[0] * task_x_world
                        + direction_xy[1] * task_y_world
                    )
                    offset_world = TACTILE_LIP_OFFSET_M * direction_world
                    offset_tcp_target = center_tcp_target + offset_world
                    offset_arm = solve_tactile_arm(
                        center_arm, offset_tcp_target
                    )
                    approach_tcp_target = (
                        offset_tcp_target
                        - approach_travel_m * task_z_world
                    )
                    approach_arm = solve_tactile_arm(
                        offset_arm, approach_tcp_target
                    )
                    offset_xy_m = (
                        TACTILE_LIP_OFFSET_M,
                        0.0,
                    )
                    manifold_peaks = {}
                    phase = (
                        "mixed_grip_preinsert_tactile_manifold_plus_x_"
                        "offset"
                    )
                    lateral_steps = minimum_jerk_steps_for_peak_speed(
                        TACTILE_LIP_OFFSET_M,
                        rate_hz,
                        TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S,
                    )
                    (
                        lateral_sample,
                        _,
                        _,
                        lateral_evidence,
                    ) = run_tactile_motion(
                        offset_arm,
                        float(lateral_steps) / float(rate_hz),
                        search_offset_xy_m=offset_xy_m,
                        contact_attempts=0,
                        trial_peaks=manifold_peaks,
                        allow_fixed_contact=False,
                        evidence_sink=tactile_report,
                        record_full_progress=True,
                        record_raw_guarded_samples=True,
                        minimum_gap_m=tactile_probe.motion.entry_gap_m,
                        maximum_command_speed_m_s=(
                            TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S
                        ),
                        maximum_measured_speed_m_s=(
                            tactile_probe.motion.unload_retract_speed_m_s
                        ),
                    )
                    phase = (
                        "mixed_grip_preinsert_tactile_manifold_plus_x_"
                        "guarded_approach"
                    )
                    approach_steps = minimum_jerk_steps_for_peak_speed(
                        approach_travel_m,
                        rate_hz,
                        TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S,
                    )
                    (
                        manifold_sample,
                        manifold_window,
                        _,
                        approach_evidence,
                    ) = run_tactile_motion(
                        approach_arm,
                        float(approach_steps) / float(rate_hz),
                        search_offset_xy_m=offset_xy_m,
                        contact_attempts=0,
                        trial_peaks=manifold_peaks,
                        allow_fixed_contact=True,
                        detect_contact=True,
                        evidence_sink=tactile_report,
                        record_full_progress=True,
                        record_raw_guarded_samples=True,
                        capture_manifold=True,
                        minimum_gap_m=tactile_probe.motion.entry_gap_m,
                        maximum_command_speed_m_s=(
                            TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S
                        ),
                        maximum_measured_speed_m_s=(
                            tactile_probe.motion.unload_retract_speed_m_s
                        ),
                    )

                    manifold_frames = []
                    for frame_index, sample in enumerate(manifold_window):
                        ft_sample = sample["wrist_ft_sample"]
                        post_state = sample["transition_post_state"]
                        manifold_frames.append(
                            {
                                "frame_index": int(frame_index),
                                "global_step": int(
                                    ft_sample["global_step"]
                                ),
                                "command_arm_rad": [
                                    float(value)
                                    for value in sample["command_arm_rad"]
                                ],
                                "command_fk_tcp_world_m": [
                                    float(value)
                                    for value in sample[
                                        "command_fk_tcp_world_m"
                                    ]
                                ],
                                "measured_arm_rad": list(
                                    post_state["measured_arm_rad"]
                                ),
                                "applied_action_arm_readback_float32": list(
                                    post_state[
                                        "applied_action_arm_readback_float32"
                                    ]
                                ),
                                "expected_command_float32": list(
                                    post_state[
                                        "expected_previous_command_float32"
                                    ]
                                ),
                                "applied_action_arm_error_float32_rad": list(
                                    post_state[
                                        "applied_action_arm_error_float32_rad"
                                    ]
                                ),
                                "applied_action_float32_equivalent_gate": (
                                    post_state[
                                        "applied_action_float32_"
                                        "equivalent_gate"
                                    ]
                                ),
                                "measured_tcp_prim_world_m": [
                                    float(value)
                                    for value in sample["tcp_position"]
                                ],
                                "raw_wrench": ft_sample["raw_wrench"],
                                "canonical_wrench_sensor": ft_sample[
                                    "canonical_wrench_sensor"
                                ],
                                "compensated_wrench_sensor": ft_sample[
                                    "compensated_wrench_sensor"
                                ],
                                "compensated_wrench_task": [
                                    float(value)
                                    for value in sample["wrench"]
                                ],
                                "estimated_gap_m": float(
                                    sample["observation"].estimated_gap_m
                                ),
                                "intended_lip_contact_pairs": list(
                                    sample["intended_lip_contact_pairs"]
                                ),
                                "unexpected_loose_fixed_contact_pairs": (
                                    list(
                                        sample[
                                            "unexpected_loose_fixed_"
                                            "contact_pairs"
                                        ]
                                    )
                                ),
                                "loose_fixture_contact_pairs": list(
                                    sample[
                                        "loose_fixture_contact_pairs"
                                    ]
                                ),
                                "loose_table_contact_pairs": list(
                                    sample["loose_table_contact_pairs"]
                                ),
                            }
                        )

                    expected_frame_count = (
                        tactile_probe.sensor.contact_debounce_samples
                    )
                    frame_count_gate = bool(
                        len(manifold_frames) == expected_frame_count
                        and approach_evidence[
                            "frozen_contact_hold_steps"
                        ]
                        == expected_frame_count
                        and not approach_evidence[
                            "manifold_contact_lost_after_first"
                        ]
                    )
                    consecutive_step_gate = bool(
                        frame_count_gate
                        and all(
                            manifold_frames[index]["global_step"]
                            == manifold_frames[0]["global_step"] + index
                            for index in range(expected_frame_count)
                        )
                    )
                    frozen_command_gate = bool(
                        frame_count_gate
                        and all(
                            frame["command_arm_rad"]
                            == manifold_frames[0]["command_arm_rad"]
                            and frame["command_fk_tcp_world_m"]
                            == manifold_frames[0][
                                "command_fk_tcp_world_m"
                            ]
                            for frame in manifold_frames
                        )
                    )
                    exact_contact_manifold_gate = bool(
                        frame_count_gate
                        and all(
                            frame["intended_lip_contact_pairs"]
                            and not frame[
                                "unexpected_loose_fixed_contact_pairs"
                            ]
                            and not frame["loose_fixture_contact_pairs"]
                            and not frame["loose_table_contact_pairs"]
                            and sum(
                                pair["contact_records"]
                                for pair in frame[
                                    "intended_lip_contact_pairs"
                                ]
                            )
                            > 0
                            and all(
                                (
                                    pair["contact_records"] == 0
                                    and "contact_manifold" not in pair
                                )
                                or (
                                    pair.get(
                                        "contact_manifold", {}
                                    ).get("contact_point_count")
                                    == pair["contact_records"]
                                    and pair["contact_manifold"].get(
                                        "normal_convention"
                                    )
                                    == (
                                        TACTILE_LIP_MANIFOLD_NORMAL_CONVENTION
                                    )
                                )
                                for pair in frame[
                                    "intended_lip_contact_pairs"
                                ]
                            )
                            for frame in manifold_frames
                        )
                    )

                    # The terminal unload is commanded even when Stage-A
                    # evidence is incomplete, provided no hard zero-step gate
                    # fired.  It never resumes the contact approach.
                    tactile_terminal_retract_started = True
                    terminal_retract = perform_tactile_abort_retract(
                        manifold_capture_success=True
                    )
                    retract_evidence = terminal_retract["motion_evidence"]
                    phase_evidence = (
                        lateral_evidence,
                        approach_evidence,
                        retract_evidence,
                    )
                    phase_names = (
                        "plus_x_offset",
                        "guarded_approach_and_frozen_hold",
                        "terminal_retract",
                    )
                    actual_phase_peak_speeds_m_s = {
                        name: float(evidence["peak_abs_tcp_speed_m_s"])
                        for name, evidence in zip(
                            phase_names, phase_evidence
                        )
                    }
                    command_phase_peak_speeds_m_s = {
                        name: float(
                            evidence[
                                "peak_abs_command_fk_tcp_speed_m_s"
                            ]
                        )
                        for name, evidence in zip(
                            phase_names, phase_evidence
                        )
                    }
                    actual_speed_gate = bool(
                        all(
                            speed
                            <= tactile_probe.motion.unload_retract_speed_m_s
                            for speed in (
                                actual_phase_peak_speeds_m_s.values()
                            )
                        )
                    )
                    command_speed_gate = bool(
                        all(
                            speed
                            <= TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S
                            + 1.0e-12
                            for speed in (
                                command_phase_peak_speeds_m_s.values()
                            )
                        )
                    )
                    minimum_measured_gap_m = min(
                        float(gap)
                        for evidence in phase_evidence
                        for gap in evidence["estimated_gap_samples_m"]
                    )
                    outside_entry_gate = bool(
                        minimum_measured_gap_m
                        >= tactile_probe.motion.entry_gap_m
                    )
                    total_checked_samples = sum(
                        int(evidence["checked_sample_count"])
                        for evidence in phase_evidence
                    )
                    total_finite_samples = sum(
                        int(evidence["finite_sample_count"])
                        for evidence in phase_evidence
                    )
                    cumulative_finite_gate = bool(
                        total_checked_samples > 0
                        and total_finite_samples == total_checked_samples
                        and all(
                            evidence["all_samples_finite"] is True
                            for evidence in phase_evidence
                        )
                    )
                    minimum_body_fingers = min(
                        int(
                            evidence[
                                "minimum_body_contact_finger_count"
                            ]
                        )
                        for evidence in phase_evidence
                    )
                    three_finger_gate = bool(minimum_body_fingers == 3)
                    applied_prechecks = sum(
                        int(evidence["applied_action_precheck_count"])
                        for evidence in phase_evidence
                    )
                    applied_postchecks = sum(
                        int(evidence["applied_action_postcheck_count"])
                        for evidence in phase_evidence
                    )
                    applied_max_error = max(
                        float(
                            evidence[
                                "applied_action_maximum_abs_arm_error_"
                                "float32_rad"
                            ]
                        )
                        for evidence in phase_evidence
                    )
                    applied_action_gate = bool(
                        applied_prechecks == total_checked_samples
                        and applied_postchecks == total_checked_samples
                        and applied_max_error == 0.0
                        and all(
                            evidence[
                                "applied_action_all_float32_equivalent"
                            ]
                            is True
                            for evidence in phase_evidence
                        )
                    )
                    contact_record_totals = {
                        name: sum(
                            int(evidence["contact_record_totals"][name])
                            for evidence in phase_evidence
                        )
                        for name in TACTILE_PREFLIGHT_CONTACT_TOTAL_FIELDS
                    }
                    contact_scope_gate = bool(
                        contact_record_totals["intended_lip"] > 0
                        and contact_record_totals["unexpected_loose_fixed"]
                        == 0
                        and contact_record_totals["loose_fixture"] == 0
                        and contact_record_totals["loose_table"] == 0
                    )
                    experimental_abort_ceilings = {
                        "absolute_axial_force_n": (
                            tactile_probe.abort.maximum_absolute_axial_force_n
                        ),
                        "lateral_force_n": (
                            tactile_probe.abort.maximum_lateral_force_n
                        ),
                        "bending_torque_nm": (
                            tactile_probe.abort.maximum_bending_torque_nm
                        ),
                        "absolute_tightening_torque_nm": (
                            tactile_probe.abort.maximum_tightening_torque_nm
                        ),
                        "absolute_finger_base_torque_nm": (
                            tactile_probe.abort.maximum_finger_base_torque_nm
                        ),
                    }
                    peak_experimental_observations = {
                        name: max(
                            float(
                                evidence[
                                    "peak_experimental_observations"
                                ][name]
                            )
                            for evidence in phase_evidence
                        )
                        for name in TACTILE_PREFLIGHT_PEAK_FIELDS
                    }
                    experimental_abort_gate = bool(
                        all(
                            np.isfinite(
                                peak_experimental_observations[name]
                            )
                            and peak_experimental_observations[name]
                            <= experimental_abort_ceilings[name]
                            for name in TACTILE_PREFLIGHT_PEAK_FIELDS
                        )
                    )
                    retract_gate = bool(
                        terminal_retract["release_debounced"]
                        and terminal_retract[
                            "measured_tcp_prim_retract_m"
                        ]
                        >= tactile_probe.motion.unload_retract_distance_m
                        and terminal_retract[
                            "minimum_task_z_progress_m"
                        ]
                        >= -tactile_negative_progress_bound_m
                    )
                    manifold_passed = bool(
                        frame_count_gate
                        and consecutive_step_gate
                        and frozen_command_gate
                        and exact_contact_manifold_gate
                        and actual_speed_gate
                        and command_speed_gate
                        and outside_entry_gate
                        and cumulative_finite_gate
                        and three_finger_gate
                        and applied_action_gate
                        and contact_scope_gate
                        and experimental_abort_gate
                        and retract_gate
                        and report["object_pose_writes_after_physics"] == 0
                    )
                    tactile_report.update(
                        {
                            "status": (
                                "VALIDATING_RAW_MANIFOLD_EVIDENCE_"
                                "FAIL_CLOSED"
                            ),
                            # A candidate may satisfy online summary gates but
                            # is never serialized as passed before the pure
                            # raw-evidence validator returns successfully.
                            "passed": False,
                            "manifold_capture_only": True,
                            "ft_sign_calibrated": False,
                            "stage_b_authorized": False,
                            "direction": direction_name,
                            "known_offset_task_xy_m": list(offset_xy_m),
                            "maximum_commanded_tcp_speed_m_s": (
                                TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S
                            ),
                            "maximum_measured_tcp_speed_m_s": (
                                tactile_probe.motion.unload_retract_speed_m_s
                            ),
                            "contact_runtime_identity": {
                                # Avoid importing package metadata (and any
                                # dependency) on default/legacy paths.  The
                                # concrete SimulationApp/query owner types
                                # below are the runtime SDK identity.
                                "contact_query_callable_module": (
                                    type(
                                        get_physx_simulation_interface()
                                        .get_full_contact_report
                                    ).__module__
                                ),
                                "contact_query_callable_name": (
                                    getattr(
                                        get_physx_simulation_interface()
                                        .get_full_contact_report,
                                        "__name__",
                                        "get_full_contact_report",
                                    )
                                ),
                                "contact_query_owner_type_module": (
                                    type(
                                        get_physx_simulation_interface()
                                    ).__module__
                                ),
                                "contact_query_owner_type_name": (
                                    type(
                                        get_physx_simulation_interface()
                                    ).__name__
                                ),
                                "normal_convention": (
                                    TACTILE_LIP_MANIFOLD_NORMAL_CONVENTION
                                ),
                                "simulation_app_type_module": (
                                    type(simulation_app).__module__
                                ),
                                "simulation_app_type_name": (
                                    type(simulation_app).__name__
                                ),
                            },
                            "entry_gap_floor_m": (
                                tactile_probe.motion.entry_gap_m
                            ),
                            "release_compression_ceiling_n": (
                                tactile_probe.contact
                                .contact_off_compressive_axial_force_n
                            ),
                            "release_bending_ceiling_nm": (
                                tactile_probe.contact
                                .contact_off_bending_torque_nm
                            ),
                            "expected_manifold_frame_count": (
                                expected_frame_count
                            ),
                            "manifold_frames": manifold_frames,
                            "frame_count_gate": frame_count_gate,
                            "consecutive_step_gate": consecutive_step_gate,
                            "frozen_command_gate": frozen_command_gate,
                            "exact_contact_manifold_gate": (
                                exact_contact_manifold_gate
                            ),
                            "actual_phase_peak_speeds_m_s": (
                                actual_phase_peak_speeds_m_s
                            ),
                            "command_phase_peak_speeds_m_s": (
                                command_phase_peak_speeds_m_s
                            ),
                            "actual_speed_gate": actual_speed_gate,
                            "command_speed_gate": command_speed_gate,
                            "minimum_measured_gap_m": (
                                minimum_measured_gap_m
                            ),
                            "outside_entry_gate": outside_entry_gate,
                            "total_checked_sample_count": (
                                total_checked_samples
                            ),
                            "total_finite_sample_count": (
                                total_finite_samples
                            ),
                            "cumulative_finite_gate": (
                                cumulative_finite_gate
                            ),
                            "minimum_body_contact_finger_count": (
                                minimum_body_fingers
                            ),
                            "three_finger_body_contact_gate": (
                                three_finger_gate
                            ),
                            "applied_action_precheck_count": (
                                applied_prechecks
                            ),
                            "applied_action_postcheck_count": (
                                applied_postchecks
                            ),
                            (
                                "applied_action_maximum_abs_arm_error_"
                                "float32_rad"
                            ): applied_max_error,
                            "applied_action_gate": applied_action_gate,
                            "contact_record_totals": contact_record_totals,
                            "contact_scope_gate": contact_scope_gate,
                            "peak_experimental_observations": (
                                peak_experimental_observations
                            ),
                            "experimental_abort_ceilings": (
                                experimental_abort_ceilings
                            ),
                            "experimental_abort_envelope_gate": (
                                experimental_abort_gate
                            ),
                            "lateral_motion_evidence": lateral_evidence,
                            "approach_and_hold_motion_evidence": (
                                approach_evidence
                            ),
                            "terminal_retract": terminal_retract,
                            "retract_gate": retract_gate,
                            "truth_pose_used_for_touch_control": False,
                            "engage_executed": False,
                            "insertion_executed": False,
                            "twist_executed": False,
                            "home_return_executed": False,
                            "production_control_authorized": False,
                            "hardware_safety_calibration_claimed": False,
                            "assembly_success_claimed": False,
                        }
                    )
                    manifold_validation = (
                        validate_tactile_manifold_capture_evidence(
                            tactile_report,
                            body_root=body_root,
                            fixed_root=(
                                tabletop.asset.fixed_receptacle_prim_path
                            ),
                            task_rotation_world=task_rotation,
                            task_origin_world=preinsert_reference_tcp,
                            physics_dt_s=1.0 / float(rate_hz),
                            sample_rate_hz=rate_hz,
                            expected_frame_count=expected_frame_count,
                            expected_offset_task_xy_m=offset_xy_m,
                            maximum_command_speed_m_s=(
                                TACTILE_LIP_MANIFOLD_COMMAND_SPEED_M_S
                            ),
                            maximum_measured_speed_m_s=(
                                tactile_probe.motion
                                .unload_retract_speed_m_s
                            ),
                            minimum_gap_m=(
                                tactile_probe.motion.entry_gap_m
                            ),
                            expected_preinsert_gap_m=(
                                tactile_probe.motion.preinsert_gap_m
                            ),
                            required_retract_distance_m=(
                                tactile_probe.motion
                                .unload_retract_distance_m
                            ),
                            maximum_negative_retract_progress_m=(
                                tactile_negative_progress_bound_m
                            ),
                            expected_release_compression_ceiling_n=(
                                tactile_probe.contact
                                .contact_off_compressive_axial_force_n
                            ),
                            expected_release_bending_ceiling_nm=(
                                tactile_probe.contact
                                .contact_off_bending_torque_nm
                            ),
                            compressive_axial_force_sign_candidate=(
                                tactile_probe.motion
                                .compressive_axial_force_sign_candidate
                            ),
                            expected_experimental_abort_ceilings=(
                                experimental_abort_ceilings
                            ),
                        )
                    )
                    tactile_report["raw_evidence_validation"] = (
                        manifold_validation
                    )
                    manifold_passed = bool(
                        manifold_passed
                        and manifold_validation.get("validated") is True
                    )
                    tactile_report["passed"] = manifold_passed
                    tactile_report["status"] = (
                        "PASSED_PLUS_X_LIP_MANIFOLD_CAPTURE_"
                        "AND_TERMINAL_RETRACT"
                        if manifold_passed
                        else "REJECTED_FAIL_CLOSED"
                    )
                    passed = bool(passed and manifold_passed)
                    report["passed"] = passed
                    if not manifold_passed:
                        raise RuntimeError(
                            "plus_x manifold capture failed closed after "
                            "terminal retract"
                        )
                    raise TactileManifoldComplete

                if arguments.tactile_lip_calibration:
                    tactile_report["retract_preflight_evidence_required"] = (
                        True
                    )
                for (
                    attempt_index,
                    (
                        direction_name,
                        direction_xy,
                        moment_name,
                        moment_index,
                    ),
                ) in enumerate(
                    TACTILE_LIP_DIRECTIONS
                    if arguments.tactile_lip_calibration
                    else ()
                ):
                    direction_world = (
                        direction_xy[0] * task_x_world
                        + direction_xy[1] * task_y_world
                    )
                    offset_world = TACTILE_LIP_OFFSET_M * direction_world
                    offset_tcp_target = center_tcp_target + offset_world
                    offset_arm = solve_tactile_arm(
                        center_arm, offset_tcp_target
                    )
                    approach_tcp_target = (
                        offset_tcp_target
                        - approach_travel_m * task_z_world
                    )
                    approach_arm = solve_tactile_arm(
                        offset_arm, approach_tcp_target
                    )
                    offset_xy_m = tuple(
                        TACTILE_LIP_OFFSET_M * value
                        for value in direction_xy
                    )
                    trial_peaks = {}
                    trial = {
                        "direction": direction_name,
                        "known_offset_task_xy_m": list(offset_xy_m),
                        "known_offset_world_m": [
                            float(value) for value in offset_world
                        ],
                        "expected_moment_component": moment_name,
                        "absolute_moment_sign_claimed": False,
                        "pairwise_response_direction_only": True,
                        "contact_debounce_samples": (
                            tactile_probe.sensor.contact_debounce_samples
                        ),
                        "release_debounce_samples": (
                            tactile_probe.sensor.release_debounce_samples
                        ),
                        "passed": False,
                    }
                    trials.append(trial)
                    tactile_touch_motion_started = True

                    # Lateral offset occurs at the collision-free 12 mm gap.
                    phase = (
                        "mixed_grip_preinsert_tactile_lip_"
                        f"{direction_name}_offset"
                    )
                    lateral_duration_s = (
                        1.875
                        * TACTILE_LIP_OFFSET_M
                        / tactile_probe.motion.maximum_xy_speed_m_s
                    )
                    run_tactile_motion(
                        offset_arm,
                        lateral_duration_s,
                        search_offset_xy_m=offset_xy_m,
                        contact_attempts=attempt_index,
                        trial_peaks=trial_peaks,
                        allow_fixed_contact=False,
                    )

                    phase = (
                        "mixed_grip_preinsert_tactile_lip_"
                        f"{direction_name}_guarded_approach"
                    )
                    approach_duration_s = (
                        1.875
                        * approach_travel_m
                        / tactile_probe.motion.guarded_approach_speed_m_s
                    )
                    (
                        contact_sample,
                        contact_window,
                        _,
                        contact_motion_evidence,
                    ) = run_tactile_motion(
                        approach_arm,
                        approach_duration_s,
                        search_offset_xy_m=offset_xy_m,
                        contact_attempts=attempt_index,
                        trial_peaks=trial_peaks,
                        allow_fixed_contact=True,
                        detect_contact=True,
                        evidence_sink=trial,
                    )
                    if len(contact_window) != (
                        tactile_probe.sensor.contact_debounce_samples
                    ):
                        last_ft_sample = contact_sample["wrist_ft_sample"]
                        signed_last_compression_n = float(
                            tactile_probe.motion
                            .compressive_axial_force_sign_candidate
                            * contact_sample["wrench"][2]
                        )
                        # Persist the bounded rejected-contact ring and final
                        # guarded sample before failing.  In particular, do
                        # not command a nominal no-contact return: a physical
                        # lip touch may have occurred below the strict force
                        # gate.  The outer terminal abort performs the sole
                        # command-continuous 0.3 mm unload instead.
                        trial.update(
                            {
                                "status": (
                                    "REJECTED_CONTACT_DID_NOT_MEET_"
                                    "STRICT_COMPRESSION_DEBOUNCE"
                                ),
                                "contact_debounced": False,
                                "touch_passed": False,
                                "guarded_approach_motion_evidence": (
                                    contact_motion_evidence
                                ),
                                "rejected_physical_contact_samples": (
                                    contact_motion_evidence[
                                        "rejected_physical_contact_samples"
                                    ]
                                ),
                                "last_guarded_approach_sample": {
                                    "global_step": last_ft_sample[
                                        "global_step"
                                    ],
                                    "raw_wrench": last_ft_sample[
                                        "raw_wrench"
                                    ],
                                    "canonical_wrench_sensor": (
                                        last_ft_sample[
                                            "canonical_wrench_sensor"
                                        ]
                                    ),
                                    "compensated_wrench_sensor": (
                                        last_ft_sample[
                                            "compensated_wrench_sensor"
                                        ]
                                    ),
                                    "compensated_wrench_task": [
                                        float(value)
                                        for value in contact_sample["wrench"]
                                    ],
                                    "signed_compression_n": (
                                        signed_last_compression_n
                                    ),
                                    "estimated_gap_m": float(
                                        contact_sample[
                                            "observation"
                                        ].estimated_gap_m
                                    ),
                                    "loose_fixed_contact_records": (
                                        contact_sample[
                                            "loose_fixed_contact_records"
                                        ]
                                    ),
                                    "intended_lip_contact_pairs": list(
                                        contact_sample[
                                            "intended_lip_contact_pairs"
                                        ]
                                    ),
                                    (
                                        "unexpected_loose_fixed_contact_"
                                        "pairs"
                                    ): list(
                                        contact_sample[
                                            "unexpected_loose_fixed_"
                                            "contact_pairs"
                                        ]
                                    ),
                                    "loose_fixture_contact_pairs": list(
                                        contact_sample[
                                            "loose_fixture_contact_pairs"
                                        ]
                                    ),
                                    "loose_table_contact_pairs": list(
                                        contact_sample[
                                            "loose_table_contact_pairs"
                                        ]
                                    ),
                                },
                                "peak_experimental_observations": dict(
                                    sorted(trial_peaks.items())
                                ),
                            }
                        )
                        raise RuntimeError(
                            f"{direction_name} lip contact did not debounce"
                        )

                    contact_wrenches = np.stack(
                        [sample["wrench"] for sample in contact_window]
                    )
                    contact_window_evidence = []
                    for sample_index, sample in enumerate(contact_window):
                        ft_sample = sample["wrist_ft_sample"]
                        signed_compression = float(
                            tactile_probe.motion
                            .compressive_axial_force_sign_candidate
                            * sample["wrench"][2]
                        )
                        contact_window_evidence.append(
                            {
                                "sample_index": sample_index,
                                "global_step": ft_sample["global_step"],
                                "raw_wrench": ft_sample["raw_wrench"],
                                "canonical_wrench_sensor": ft_sample[
                                    "canonical_wrench_sensor"
                                ],
                                "compensated_wrench_sensor": ft_sample[
                                    "compensated_wrench_sensor"
                                ],
                                "compensated_wrench_task": ft_sample[
                                    "compensated_wrench_task"
                                ],
                                "signed_compression_n": (
                                    signed_compression
                                ),
                                "intended_lip_contact_pairs": list(
                                    sample["intended_lip_contact_pairs"]
                                ),
                                "unexpected_loose_fixed_contact_pairs": list(
                                    sample[
                                        "unexpected_loose_fixed_contact_pairs"
                                    ]
                                ),
                                "loose_fixture_contact_pairs": list(
                                    sample["loose_fixture_contact_pairs"]
                                ),
                                "loose_table_contact_pairs": list(
                                    sample["loose_table_contact_pairs"]
                                ),
                            }
                        )
                    contact_pair_evidence = []
                    for sample in contact_window:
                        contact_pair_evidence.extend(
                            sample["intended_lip_contact_pairs"]
                        )
                    # Persist all six raw/canonical/compensated samples,
                    # exact collider paths, and peaks before normalization so
                    # any later fail-closed path remains auditable.
                    trial.update(
                        {
                            "status": (
                                "CONTACT_DEBOUNCED_PENDING_NORMALIZATION"
                            ),
                            "contact_debounced": True,
                            "contact_window_evidence": (
                                contact_window_evidence
                            ),
                            "contact_pair_evidence": (
                                contact_pair_evidence
                            ),
                            "peak_experimental_observations": dict(
                                sorted(trial_peaks.items())
                            ),
                        }
                    )
                    representative_wrench = np.mean(
                        contact_wrenches, axis=0
                    )
                    signed_compression_samples = (
                        tactile_probe.motion
                        .compressive_axial_force_sign_candidate
                        * contact_wrenches[:, 2]
                    )
                    if (
                        not np.all(np.isfinite(signed_compression_samples))
                        or np.any(
                            signed_compression_samples
                            < (
                                tactile_probe.contact
                                .contact_on_compressive_axial_force_n
                            )
                        )
                    ):
                        raise RuntimeError(
                            f"{direction_name} debounce contains a "
                            "non-compressive sample"
                        )
                    # Normalize moment by each sample's positive compression
                    # before pairing.  For ideal axial contact, rx=-My/F and
                    # ry=Mx/F; unlike raw moment differences this removes the
                    # unknown-center term when the two forces are unequal.
                    inferred_lever_samples_m = (
                        -contact_wrenches[:, moment_index]
                        / signed_compression_samples
                        if moment_index == 4
                        else contact_wrenches[:, moment_index]
                        / signed_compression_samples
                    )
                    if not np.all(np.isfinite(inferred_lever_samples_m)):
                        raise RuntimeError(
                            f"{direction_name} normalized lever is non-finite"
                        )
                    compression = (
                        tactile_probe.motion
                        .compressive_axial_force_sign_candidate
                        * float(representative_wrench[2])
                    )
                    compression_sign_gate = bool(
                        compression
                        >= tactile_probe.contact
                        .contact_on_compressive_axial_force_n
                    )
                    (
                        retract_path,
                        contact_tcp_position,
                        retract_diagnostics,
                    ) = command_continuous_retract_plan(
                        tactile_probe.motion.unload_retract_distance_m
                    )
                    phase = (
                        "mixed_grip_preinsert_tactile_lip_"
                        f"{direction_name}_retract_unload"
                    )
                    (
                        retract_sample,
                        _,
                        release_gate,
                        retract_motion_evidence,
                    ) = run_tactile_motion(
                        None,
                        None,
                        search_offset_xy_m=offset_xy_m,
                        contact_attempts=attempt_index,
                        trial_peaks=trial_peaks,
                        allow_fixed_contact=True,
                        evaluate_release=True,
                        command_path=retract_path["commands"],
                        evidence_sink=trial,
                    )
                    measured_retract_m = float(
                        np.dot(
                            retract_sample["tcp_position"]
                            - contact_tcp_position,
                            task_z_world,
                        )
                    )
                    retract_diagnostics.update(
                        {
                            "measured_robot_fk_retract_m": (
                                measured_retract_m
                            ),
                            "measured_robot_fk_retract_m_deprecated": True,
                            "measured_tcp_prim_retract_m": (
                                measured_retract_m
                            ),
                        }
                    )

                    # Commit contact and unload evidence before applying the
                    # release gate.  A failed release is terminal, but its raw
                    # wrench history, exact collider paths, measured-state
                    # resynchronization, and task-Z progress must still reach
                    # the unique report written by the outer fail-closed path.
                    trial.update(
                        {
                            "status": (
                                "RETRACT_RELEASE_VALIDATED_PENDING_RETURN"
                                if release_gate
                                else "REJECTED_RELEASE_DID_NOT_DEBOUNCE"
                            ),
                            "touch_passed": False,
                            "passed": False,
                            "contact_debounced": True,
                            "representative_compensated_wrench_task": [
                                float(value)
                                for value in representative_wrench
                            ],
                            "compression_sign_gate": compression_sign_gate,
                            "signed_compression_response_n": compression,
                            "minimum_debounced_compression_n": float(
                                np.min(signed_compression_samples)
                            ),
                            "inferred_lever_response_mean_m": float(
                                np.mean(inferred_lever_samples_m)
                            ),
                            (
                                "inferred_lever_response_"
                                "standard_deviation_m"
                            ): float(
                                np.std(inferred_lever_samples_m, ddof=1)
                            ),
                            "inferred_lever_response_samples_m": [
                                float(value)
                                for value in inferred_lever_samples_m
                            ],
                            "contact_estimated_gap_m": (
                                contact_sample[
                                    "observation"
                                ].estimated_gap_m
                            ),
                            "contact_loose_fixed_records": (
                                contact_sample[
                                    "loose_fixed_contact_records"
                                ]
                            ),
                            "contact_pair_evidence": (
                                contact_pair_evidence
                            ),
                            "accepted_contact_class": (
                                "BodyAssembly/MatingShell/Segment_*_to_"
                                "FixedReceptacle/EntryShell/Segment_*"
                            ),
                            "unexpected_loose_fixed_contact_pairs": [],
                            "proxy_collision_filter_enabled": (
                                proxy_collision_filter["enabled"]
                            ),
                            "proxy_collision_filter_pair_count": (
                                proxy_collision_filter["pair_count"]
                            ),
                            "commanded_retract_m": (
                                tactile_probe.motion
                                .unload_retract_distance_m
                            ),
                            "measured_robot_fk_retract_m": (
                                measured_retract_m
                            ),
                            "measured_robot_fk_retract_m_deprecated": True,
                            "measured_tcp_prim_retract_m": (
                                measured_retract_m
                            ),
                            "retract_start_and_target_diagnostics": (
                                retract_diagnostics
                            ),
                            "guarded_approach_motion_evidence": (
                                contact_motion_evidence
                            ),
                            "retract_motion_evidence": (
                                retract_motion_evidence
                            ),
                            "release_debounced": release_gate,
                            "peak_experimental_observations": dict(
                                sorted(trial_peaks.items())
                            ),
                        }
                    )

                    # Release is proven after the full 0.3 mm command.  Only
                    # then may the robot retreat to offset-preinsert and move
                    # laterally back to the original visual preinsert target.
                    if not release_gate:
                        raise RuntimeError(
                            f"{direction_name} force/contact release failed "
                            "after 0.3 mm retract"
                        )
                    phase = (
                        "mixed_grip_preinsert_tactile_lip_"
                        f"{direction_name}_axial_return"
                    )
                    return_distance_m = max(
                        0.0,
                        tactile_probe.motion.preinsert_gap_m
                        - retract_sample["observation"].estimated_gap_m,
                    )
                    axial_return_duration_s = max(
                        1.0 / float(rate_hz),
                        1.875
                        * return_distance_m
                        / tactile_probe.motion.unload_retract_speed_m_s,
                    )
                    run_tactile_motion(
                        offset_arm,
                        axial_return_duration_s,
                        search_offset_xy_m=offset_xy_m,
                        contact_attempts=attempt_index,
                        trial_peaks=trial_peaks,
                        allow_fixed_contact=False,
                    )
                    phase = (
                        "mixed_grip_preinsert_tactile_lip_"
                        f"{direction_name}_center_return"
                    )
                    final_trial_sample, _, _, _ = run_tactile_motion(
                        center_arm,
                        lateral_duration_s,
                        search_offset_xy_m=(0.0, 0.0),
                        contact_attempts=attempt_index,
                        trial_peaks=trial_peaks,
                        allow_fixed_contact=False,
                    )
                    returned_preinsert_gate = bool(
                        final_trial_sample["loose_fixed_contact_records"] == 0
                        and abs(
                            final_trial_sample[
                                "observation"
                            ].estimated_gap_m
                            - tactile_probe.motion.preinsert_gap_m
                        )
                        <= (
                            preinsert_probe.insertion.acceptance
                            .maximum_preinsert_gap_error_m
                        )
                    )
                    touch_passed = bool(
                        compression_sign_gate
                        and release_gate
                        and returned_preinsert_gate
                    )
                    trial.update(
                        {
                            "status": (
                                "TOUCH_PASSED_PENDING_PAIRWISE_GATE"
                                if touch_passed
                                else "REJECTED_TOUCH_GATE"
                            ),
                            "touch_passed": touch_passed,
                            "returned_to_preinsert": (
                                returned_preinsert_gate
                            ),
                        }
                    )
                    if not touch_passed:
                        raise RuntimeError(
                            f"{direction_name} tactile compression/release "
                            "trial failed closed"
                        )

                # The ±0.6 mm commands are referenced to visual XY, whose
                # residual can exceed 0.6 mm.  Therefore an individual touch's
                # absolute Mx/My sign cannot prove which side of the physical
                # center it reached.  Divide each moment sample by its own
                # positive compression first.  Then ideal axial contact gives
                # inferred rx=-My/F and ry=Mx/F, and plus-minus inferred lever
                # response must increase.  The 3*sample-mean-SE threshold is a
                # diagnostic, unregistered separation heuristic for correlated
                # 240 Hz samples; it is not a confidence or safety statement.
                trial_by_direction = {
                    trial["direction"]: trial for trial in trials
                }
                pairwise_response_gates = {}
                failed_response_axes = []
                for (
                    axis_name,
                    plus_name,
                    minus_name,
                ) in (
                    ("x_to_normalized_lever_x", "plus_x", "minus_x"),
                    ("y_to_normalized_lever_y", "plus_y", "minus_y"),
                ):
                    plus_trial = trial_by_direction[plus_name]
                    minus_trial = trial_by_direction[minus_name]
                    delta_m = float(
                        plus_trial["inferred_lever_response_mean_m"]
                        - minus_trial["inferred_lever_response_mean_m"]
                    )
                    sample_count = (
                        tactile_probe.sensor.contact_debounce_samples
                    )
                    pooled_sample_mean_standard_error_m = float(
                        np.sqrt(
                            (
                                plus_trial[
                                    "inferred_lever_response_"
                                    "standard_deviation_m"
                                ]
                                ** 2
                                + minus_trial[
                                    "inferred_lever_response_"
                                    "standard_deviation_m"
                                ]
                                ** 2
                            )
                            / float(sample_count)
                        )
                    )
                    minimum_increase_m = max(
                        1.0e-6,
                        3.0 * pooled_sample_mean_standard_error_m,
                    )
                    pair_passed = bool(
                        delta_m >= minimum_increase_m
                    )
                    pair_report = {
                        "plus_direction": plus_name,
                        "minus_direction": minus_name,
                        "moment_component": plus_trial[
                            "expected_moment_component"
                        ],
                        "normalization": (
                            "inferred_rx_equals_minus_My_over_compression"
                            if axis_name.startswith("x_")
                            else "inferred_ry_equals_Mx_over_compression"
                        ),
                        "expected_plus_minus_delta_sign": 1,
                        "plus_minus_inferred_lever_delta_m": delta_m,
                        "pooled_sample_mean_standard_error_m": (
                            pooled_sample_mean_standard_error_m
                        ),
                        "minimum_diagnostic_increase_m": (
                            minimum_increase_m
                        ),
                        "separation_threshold_status": (
                            "diagnostic_unregistered_correlated_samples_"
                            "not_statistical_confidence"
                        ),
                        "absolute_single_touch_sign_claimed": False,
                        "interpretation": (
                            "local_moment_response_jacobian_sign_only_"
                            "not_physical_center_localization"
                        ),
                        "truth_pose_used": False,
                        "passed": pair_passed,
                    }
                    pairwise_response_gates[axis_name] = pair_report
                    for trial in (plus_trial, minus_trial):
                        trial["pairwise_response_gate"] = pair_report
                        trial["passed"] = bool(
                            trial["touch_passed"] and pair_passed
                        )
                    if not pair_passed:
                        failed_response_axes.append(axis_name)

                tactile_report["pairwise_moment_response_gates"] = (
                    pairwise_response_gates
                )
                tactile_report["pairwise_response_scope"] = (
                    "local_moment_response_jacobian_only_not_center_"
                    "localization"
                )
                if failed_response_axes:
                    raise RuntimeError(
                        "paired moment response direction failed closed: "
                        f"{failed_response_axes}"
                    )

                final_tactile_tcp, _ = _world_pose(
                    Gf, Usd, UsdGeom, tcp_prim
                )
                final_tactile_gap_m = measured_tactile_gap_m(
                    final_tactile_tcp
                )
                final_tactile_positions = np.asarray(
                    robot.get_joint_positions(), dtype=np.float64
                )
                final_tactile_tracking_error = float(
                    np.max(
                        np.abs(
                            final_tactile_positions[arm_indices]
                            - center_arm
                        )
                    )
                )
                final_tactile_body_in_tcp = body_in_tcp_frame(
                    body.get_world_pose()[0]
                )
                posthoc_body_tcp_slip_m = float(
                    np.linalg.norm(
                        final_tactile_body_in_tcp
                        - preinsert_reference_body_in_tcp
                    )
                )
                tactile_fixed_position, tactile_fixed_orientation = (
                    _world_pose(Gf, Usd, UsdGeom, fixed_prim)
                )
                posthoc_fixed_translation_drift_m = float(
                    np.linalg.norm(
                        np.asarray(tactile_fixed_position, dtype=np.float64)
                        - np.asarray(
                            fixed_initial_position, dtype=np.float64
                        )
                    )
                )
                posthoc_fixed_rotation_drift_rad = (
                    _gf_quaternion_error_radians(
                        fixed_initial_orientation,
                        tactile_fixed_orientation,
                    )
                )
                final_preinsert_gate = bool(
                    latest_loose_fixed_contact_records == 0
                    and abs(
                        final_tactile_gap_m
                        - tactile_probe.motion.preinsert_gap_m
                    )
                    <= (
                        preinsert_probe.insertion.acceptance
                        .maximum_preinsert_gap_error_m
                    )
                    and final_tactile_tracking_error
                    <= (
                        preinsert_probe.insertion.acceptance
                        .maximum_arm_tracking_error_rad
                    )
                )
                posthoc_scene_gate = bool(
                    posthoc_body_tcp_slip_m
                    <= (
                        preinsert_probe.insertion.acceptance
                        .maximum_body_tcp_slip_m
                    )
                    and posthoc_fixed_translation_drift_m
                    <= (
                        preinsert_probe.insertion.acceptance
                        .maximum_fixed_translation_drift_m
                    )
                    and posthoc_fixed_rotation_drift_rad
                    <= (
                        preinsert_probe.insertion.acceptance
                        .maximum_fixed_rotation_drift_rad
                    )
                )
                tactile_passed = bool(
                    len(trials) == len(TACTILE_LIP_DIRECTIONS)
                    and all(trial["passed"] for trial in trials)
                    and final_preinsert_gate
                    and posthoc_scene_gate
                    and finite_throughout
                    and preinsert_minimum_body_contact_fingers == 3
                    and float(np.max(maximum_post_tare_delta))
                    <= tactile_probe.abort.maximum_finger_base_torque_nm
                    and object_pose_write_gate
                )
                tactile_report.update(
                    {
                        "status": (
                            "PASSED_SIGNED_LIP_CONTACT_CALIBRATION_"
                            "AT_PREINSERT"
                            if tactile_passed
                            else "REJECTED_FAIL_CLOSED"
                        ),
                        "passed": tactile_passed,
                        "checked_physics_steps": (
                            global_step - tactile_start_step
                        ),
                        "minimum_body_contact_finger_count": (
                            preinsert_minimum_body_contact_fingers
                        ),
                        "maximum_finger_base_torque_delta_nm": float(
                            np.max(maximum_post_tare_delta)
                        ),
                        "final_estimated_gap_from_robot_fk_m": (
                            final_tactile_gap_m
                        ),
                        "final_preinsert_gate": final_preinsert_gate,
                        "posthoc_scene_gate": posthoc_scene_gate,
                        "posthoc_body_tcp_slip_m": posthoc_body_tcp_slip_m,
                        "posthoc_fixed_translation_drift_m": (
                            posthoc_fixed_translation_drift_m
                        ),
                        "posthoc_fixed_rotation_drift_rad": (
                            posthoc_fixed_rotation_drift_rad
                        ),
                        "virtual_wrist_ft_monitor": (
                            wrist_ft_monitor.report()
                        ),
                        "object_pose_writes_after_physics": (
                            report["object_pose_writes_after_physics"]
                        ),
                        "truth_pose_used_for_touch_control": False,
                        "engage_executed": False,
                        "insertion_executed": False,
                        "twist_executed": False,
                        "home_return_executed": False,
                        "production_control_authorized": False,
                        "hardware_safety_calibration_claimed": False,
                        "assembly_success_claimed": False,
                    }
                )
                passed = bool(passed and tactile_passed)
                report["passed"] = passed
    except TactilePreflightComplete:
        # The preflight is a deliberate terminal milestone.  It never falls
        # through into lip contact and preserves the pass/fail result already
        # written above for the shared finally/report/SimulationApp shutdown.
        pass
    except TactileManifoldComplete:
        # Stage A deliberately terminates after its one +X manifold capture
        # and guarded unload; it cannot fall through to four-way calibration.
        pass
    except BaseException as exception:
        tactile_failure_phase = locals().get("phase")
        tactile_failure_step = locals().get("global_step")
        tactile_abort_report = None
        zero_step_tokens = (
            "experimental_axial_force_ceiling",
            "experimental_lateral_force_ceiling",
            "experimental_bending_torque_ceiling",
            "experimental_tightening_torque_ceiling",
            "finger_base_torque_hard_stop",
            "2 Nm finger torque hard stop exceeded",
            "non-finite",
            "unexpected hand2arm reaction wrench shape",
            "lacks a protected INSERT wrench",
            "all three fingers must retain BodyAssembly contact",
            "applied action readback does not equal",
            "forbidden ",
            "disallowed loose/fixed contact",
            "grasp_contact_lost",
        )
        tactile_zero_step_abort_reason = (
            exception.reason
            if isinstance(exception, TactileSafetyStop)
            and exception.zero_step_abort
            else next(
                (
                    token
                    for token in zero_step_tokens
                    if token in str(exception)
                ),
                None,
            )
        )
        if (
            tactile_runtime_requested
            and tactile_touch_motion_started
            and not tactile_terminal_retract_started
            and callable(tactile_abort_retract)
        ):
            if tactile_zero_step_abort_reason is not None:
                tactile_abort_report = {
                    "attempted": False,
                    "world_steps_after_original_failure": 0,
                    "zero_step_reason": tactile_zero_step_abort_reason,
                    "release_debounced": False,
                    "terminal_state": "TERMINAL_ABORT",
                    "resume_attempted": False,
                    "transition_ring": list(tactile_transition_ring),
                }
            else:
                try:
                    tactile_abort_report = tactile_abort_retract()
                except BaseException as retract_error:
                    tactile_abort_report = {
                        "attempted": True,
                        "release_debounced": False,
                        "terminal_state": "TERMINAL_ABORT",
                        "resume_attempted": False,
                        "transition_ring": (
                            tactile_abort_runtime_evidence.get(
                                "transition_ring", []
                            )
                        ),
                        "error": (
                            f"{type(retract_error).__name__}: "
                            f"{retract_error}"
                        ),
                    }
        if (
            arguments.preinsert_probe
            and isinstance(report.get("preinsert_probe"), dict)
            and report["preinsert_probe"].get("status") == "RUNNING"
        ):
            # Preserve the exact fail-fast location for a long Isaac run.
            # diagnostics do not resume or weaken any rejected continuation.
            report["preinsert_probe"].update(
                {
                    "status": "FAILED_RUNTIME_FAIL_CLOSED",
                    "passed": False,
                    "failure_phase": locals().get("phase"),
                    "failure_global_step": locals().get("global_step"),
                    "checked_physics_steps": locals().get(
                        "preinsert_checked_steps", 0
                    ),
                    "minimum_body_contact_finger_count": locals().get(
                        "preinsert_minimum_body_contact_fingers"
                    ),
                    "loose_fixed_contact_records": locals().get(
                        "preinsert_loose_fixed_contact_records", 0
                    ),
                }
            )
        if (
            wrist_ft_guarded_requested
            and isinstance(
                report.get("wrist_ft_guarded_insertion"), dict
            )
            and not report["wrist_ft_guarded_insertion"].get(
                "passed", False
            )
        ):
            report["wrist_ft_guarded_insertion"].update(
                {
                    "status": "FAILED_RUNTIME_FAIL_CLOSED",
                    "passed": False,
                    "failure_phase": locals().get("phase"),
                    "failure_global_step": locals().get("global_step"),
                    "failure_reason": (
                        f"{type(exception).__name__}: {exception}"
                    ),
                    "fingertip_tactile_sensor_used": False,
                    "physx_contact_truth_used_for_control": False,
                    "simulator_truth_used_for_control": False,
                    "twist_executed": False,
                    "home_return_executed": False,
                }
            )
        if (
            tactile_runtime_requested
            and isinstance(report.get(tactile_report_key), dict)
            and not report[tactile_report_key].get("passed", False)
        ):
            # Preserve partial direction evidence, but never convert a failed
            # calibration into an engage authorization or assembly claim.
            report[tactile_report_key].update(
                {
                    "status": "FAILED_RUNTIME_FAIL_CLOSED",
                    "passed": False,
                    "failure_phase": tactile_failure_phase,
                    "failure_global_step": tactile_failure_step,
                    "failure_reason": (
                        f"{type(exception).__name__}: {exception}"
                    ),
                    "abort_retract": tactile_abort_report,
                    "failure_transition_ring": list(
                        tactile_transition_ring
                    ),
                    "first_forbidden_contact": (
                        first_tactile_forbidden_contact
                    ),
                    "engage_executed": False,
                    "insertion_executed": False,
                    "twist_executed": False,
                    "home_return_executed": False,
                    "production_control_authorized": False,
                    "hardware_safety_calibration_claimed": False,
                    "assembly_success_claimed": False,
                }
            )
        report.update(
            {
                "passed": False,
                "error": f"{type(exception).__name__}: {exception}",
                "traceback": traceback.format_exc(),
            }
        )
        passed = False
    finally:
        if tactile_runtime_requested or wrist_ft_guarded_requested:
            runtime_source_finalize_sha256 = _sha256(RUNTIME_SOURCE_PATH)
            runtime_source_unchanged = bool(
                RUNTIME_SOURCE_IMPORT_SHA256
                == runtime_source_start_sha256
                == runtime_source_finalize_sha256
            )
            report.update(
                {
                    "runtime_source_finalize_sha256": (
                        runtime_source_finalize_sha256
                    ),
                    "runtime_source_unchanged": runtime_source_unchanged,
                }
            )
            if not runtime_source_unchanged:
                report["runtime_source_change_error"] = (
                    "visual wrist-FT runtime source changed between import, "
                    "start, and finalize"
                )
                passed = False
                report["passed"] = False
                runtime_section = (
                    report.get("wrist_ft_guarded_insertion")
                    if wrist_ft_guarded_requested
                    else report.get(tactile_report_key)
                )
                if isinstance(runtime_section, dict):
                    runtime_section.update(
                        {
                            "passed": False,
                            "status": "FAILED_RUNTIME_SOURCE_CHANGED",
                        }
                    )
        if world is not None:
            try:
                world.stop()
            except BaseException as exception:
                report["world_stop_error"] = (
                    f"{type(exception).__name__}: {exception}"
                )
                passed = False
                report["passed"] = False
        try:
            output_value = report.get("output_directory")
            if output_value:
                report_path = Path(output_value) / (
                    probe.report_filename
                    if "probe" in locals()
                    else "report.json"
                )
                report["report_path"] = str(report_path)
                report_path.write_text(
                    json.dumps(
                        report, allow_nan=False, indent=2, sort_keys=True
                    )
                    + "\n",
                    encoding="utf-8",
                )
        except BaseException as exception:
            report["report_write_error"] = (
                f"{type(exception).__name__}: {exception}"
            )
            passed = False
            report["passed"] = False
        print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
        if arguments.wrist_ft_guarded_insertion:
            result_marker = WRIST_FT_GUARDED_RESULT_MARKER
        elif arguments.tactile_retract_preflight:
            result_marker = TACTILE_RETRACT_PREFLIGHT_RESULT_MARKER
        elif arguments.tactile_lip_manifold_capture:
            result_marker = TACTILE_LIP_MANIFOLD_RESULT_MARKER
        elif arguments.tactile_lip_calibration:
            result_marker = TACTILE_LIP_RESULT_MARKER
        elif arguments.preinsert_probe:
            result_marker = PREINSERT_RESULT_MARKER
        else:
            result_marker = RESULT_MARKER
        print(
            f"{result_marker} {'PASSED' if passed else 'FAILED'}",
            flush=True,
        )
        # Isaac fast shutdown may terminate the process inside ``close``.
        # Carry the probe result into that boundary so a failed run cannot be
        # reported to the caller as a successful shell command.
        simulation_app.close(exit_code=0 if passed else 1)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
