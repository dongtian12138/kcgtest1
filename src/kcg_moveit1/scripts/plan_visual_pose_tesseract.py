#!/usr/bin/env python3
"""Plan one visual hand pose with Tesseract KDL and OMPL.

The planner keeps the existing detailed collision meshes, checks OMPL edges
with LVS-discrete subdivision, and does not run TrajOpt or path smoothing.
Isaac remains the executor and performs the final 240 Hz task-specific FCL
check before any motion.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

import tesseract_robotics
from tesseract_robotics.planning import (
    MotionProgram,
    Pose,
    Robot,
    StateTarget,
    box,
    create_obstacle,
    cylinder,
)
from tesseract_robotics.tesseract_collision import (
    CollisionEvaluatorType,
    ContactRequest,
    ContactResultMap,
    ContactTestType_FIRST,
)
from tesseract_robotics.tesseract_common import CollisionMarginData
from tesseract_robotics.tesseract_command_language import (
    CompositeInstruction,
    InstructionPoly_as_MoveInstructionPoly,
    MoveInstruction,
    MoveInstructionPoly_wrap_MoveInstruction,
    MoveInstructionType_FREESPACE,
    ProfileDictionary,
    StateWaypoint,
    StateWaypointPoly_wrap_StateWaypoint,
    WaypointPoly_as_JointWaypointPoly,
    WaypointPoly_as_StateWaypointPoly,
)
from tesseract_robotics.tesseract_motion_planners import (
    PlannerRequest,
    assignCurrentStateAsSeed,
)
from tesseract_robotics.tesseract_motion_planners_ompl import (
    OMPLMotionPlanner,
    OMPLRealVectorPlanProfile,
    OMPLSolverConfig,
    RNG_setSeed,
    RRTConnectConfigurator,
)
from tesseract_robotics.tesseract_motion_planners_simple import (
    generateInterpolatedProgram,
)
from tesseract_robotics.tesseract_time_parameterization import (
    TimeOptimalTrajectoryGeneration,
    TOTGCompositeProfile,
)


ARM_JOINT_NAMES = tuple(f"iiwa_joint_{index}" for index in range(1, 8))
HAND_JOINT_NAMES = ("f1j1", "f1j2", "f2j1", "f3j2")
HAND_MIMIC_JOINTS = (
    ("f1j3", "f1j2"),
    ("f2j2", "f2j1"),
    ("f3j1", "f1j1"),
    ("f3j3", "f3j2"),
)
OMPL_NAMESPACE = "OMPLMotionPlannerTask"
COLLISION_CLEARANCE_M = 0.0005
LVS_JOINT_SEGMENT_RAD = 0.02
OUTPUT_INTERPOLATION_RAD = math.radians(5.0)
GLOBAL_IK_RANDOM_SEED = 20260903
GLOBAL_IK_SEED_COUNT = 64
OMPL_RANDOM_SEED = 20260903
TASK_NOMINAL_ELBOW_DOWN_SEED = np.asarray(
    (0.0, 0.0, 0.0, -math.pi / 2.0, 0.0, math.pi / 2.0, 2.5),
    dtype=np.float64,
)


def _finite_vector(value: object, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {length} finite values")
    return result


def _matrix4(value: object, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.size != 16:
        raise ValueError(f"{label} must contain 16 values")
    result = result.reshape(4, 4)
    if (
        not np.all(np.isfinite(result))
        or not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12)
    ):
        raise ValueError(f"{label} is not a finite homogeneous transform")
    return result


def _robot_joint_state(arm: np.ndarray, hand: np.ndarray) -> dict[str, float]:
    """Resolve the four commanded hand joints into the full physical hand state."""
    state = {
        name: float(value)
        for name, value in zip(ARM_JOINT_NAMES, arm, strict=True)
    }
    active_hand = {
        name: float(value)
        for name, value in zip(HAND_JOINT_NAMES, hand, strict=True)
    }
    state.update(active_hand)
    state.update(
        {
            mimic_name: active_hand[source_name]
            for mimic_name, source_name in HAND_MIMIC_JOINTS
        }
    )
    return state


def _prepare_robot_files(
    repository: Path,
    request: dict[str, object],
    temporary: Path,
) -> tuple[Path, Path]:
    xacro_path = repository / "src/kcg_moveit1/config/handarm.urdf.xacro"
    completed = subprocess.run(
        ["xacro", str(xacro_path), "use_gazebo:=false"],
        cwd=str(repository),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"xacro failed: {completed.stderr.strip()}")
    urdf_text = completed.stdout
    for link_name in ("f1Link3", "f2Link2", "f3Link3"):
        old = f"meshes/hand/collision/{link_name}_convex.stl"
        new = f"meshes/hand/connector_no_nail/{link_name}_nailfree.stl"
        if urdf_text.count(old) != 1:
            raise RuntimeError(f"unexpected collision mesh count for {link_name}")
        urdf_text = urdf_text.replace(old, new)

    urdf_root = ET.fromstring(urdf_text)
    for mimic_name, source_name in HAND_MIMIC_JOINTS:
        joint = urdf_root.find(f"./joint[@name='{mimic_name}']")
        mimic = None if joint is None else joint.find("mimic")
        if (
            mimic is None
            or mimic.get("joint") != source_name
            or float(mimic.get("multiplier", "1")) != 1.0
            or float(mimic.get("offset", "0")) != 0.0
        ):
            raise RuntimeError(
                f"URDF mimic relation differs: {mimic_name} -> {source_name}"
            )
    requested_bounds = dict(request.get("path_joint_bounds_rad", {}))
    for joint_name in ARM_JOINT_NAMES:
        joint = urdf_root.find(f"./joint[@name='{joint_name}']")
        if joint is None:
            raise RuntimeError(f"URDF omits {joint_name}")
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"URDF omits limits for {joint_name}")
        if joint_name in requested_bounds:
            bounds = _finite_vector(
                requested_bounds[joint_name], 2, f"bounds for {joint_name}"
            )
            if bounds[0] >= bounds[1]:
                raise ValueError(f"reversed bounds for {joint_name}")
            limit.set("lower", repr(float(bounds[0])))
            limit.set("upper", repr(float(bounds[1])))
    urdf_text = ET.tostring(urdf_root, encoding="unicode")
    if not urdf_text.startswith("<robot "):
        raise RuntimeError("expanded URDF has an unexpected root")
    urdf_text = urdf_text.replace(
        "<robot ",
        '<robot xmlns:tesseract="http://ros.org/wiki/tesseract" '
        'tesseract:make_convex="false" ',
        1,
    )
    urdf_output = temporary / "handarm_tesseract.urdf"
    urdf_output.write_text(urdf_text, encoding="utf-8")

    srdf_source = repository / "src/kcg_moveit1/config/handarm.srdf"
    srdf_root = ET.parse(srdf_source).getroot()
    for child in list(srdf_root):
        if (
            child.tag == "group"
            and child.get("name") == "handarm"
            or child.tag == "group_state"
            and child.get("group") == "handarm"
            or child.tag in (
                "kinematics_plugin_config",
                "contact_managers_plugin_config",
            )
        ):
            srdf_root.remove(child)
    ET.SubElement(
        srdf_root,
        "kinematics_plugin_config",
        filename=(
            repository / "src/kcg_moveit1/config/tesseract_kinematics.yaml"
        ).resolve().as_uri(),
    )
    contact_plugins = (
        Path(tesseract_robotics.__file__).resolve().parent
        / "data/tesseract/support/urdf/contact_manager_plugins.yaml"
    )
    if not contact_plugins.is_file():
        raise RuntimeError("Tesseract contact-manager configuration is missing")
    ET.SubElement(
        srdf_root,
        "contact_managers_plugin_config",
        filename=contact_plugins.as_uri(),
    )
    srdf_output = temporary / "handarm_tesseract.srdf"
    ET.ElementTree(srdf_root).write(
        srdf_output, encoding="utf-8", xml_declaration=True
    )
    return urdf_output, srdf_output


def _add_obstacles(robot: Robot, request: dict[str, object]) -> list[str]:
    identifiers: list[str] = []
    for row in request.get("collision_objects", []):
        identifier = str(row["id"])
        pose = Pose(_matrix4(row["world_from_primitive_row_major"], identifier))
        dimensions = row["dimensions_m"]
        kind = str(row["type"]).lower()
        if kind == "box":
            values = _finite_vector(dimensions, 3, f"box {identifier}")
            geometry = box(*map(float, values))
        elif kind == "cylinder":
            values = _finite_vector(dimensions, 2, f"cylinder {identifier}")
            geometry = cylinder(float(values[1]), float(values[0]))
        else:
            raise ValueError(f"unsupported collision primitive: {kind}")
        if np.any(values <= 0.0):
            raise ValueError(f"non-positive obstacle dimensions for {identifier}")
        if not create_obstacle(
            robot,
            f"environment_{identifier}",
            geometry,
            pose,
            parent_link="world",
        ):
            raise RuntimeError(f"Tesseract rejected obstacle {identifier}")
        identifiers.append(identifier)
    return identifiers


def _positions(instructions: CompositeInstruction) -> np.ndarray:
    rows: list[np.ndarray] = []
    for instruction in instructions.getInstructions():
        move = InstructionPoly_as_MoveInstructionPoly(instruction)
        waypoint = move.getWaypoint()
        if waypoint.isJointWaypoint():
            value = WaypointPoly_as_JointWaypointPoly(waypoint).getPosition()
        elif waypoint.isStateWaypoint():
            value = WaypointPoly_as_StateWaypointPoly(waypoint).getPosition()
        else:
            raise RuntimeError("Tesseract returned a non-joint waypoint")
        rows.append(np.asarray(value, dtype=np.float64).reshape(-1))
    result = np.stack(rows)
    if result.ndim != 2 or result.shape[1] != 7 or not np.all(np.isfinite(result)):
        raise RuntimeError("Tesseract returned invalid arm positions")
    return result


def _goal_is_collision_free(
    robot: Robot,
    arm: np.ndarray,
    hand: np.ndarray,
) -> bool:
    robot.set_joints(_robot_joint_state(arm, hand))
    manager = robot.env.getDiscreteContactManager()
    manager.setActiveCollisionObjects(robot.env.getActiveLinkNames())
    manager.setCollisionMarginData(CollisionMarginData(COLLISION_CLEARANCE_M))
    manager.setCollisionObjectsTransform(robot.env.getState().link_transforms)
    contacts = ContactResultMap()
    manager.contactTest(contacts, ContactRequest(ContactTestType_FIRST))
    return contacts.empty()


def _linear_joint_positions(
    start: np.ndarray,
    goal: np.ndarray,
    maximum_step_rad: float,
) -> np.ndarray:
    segment_count = max(
        1,
        int(math.ceil(float(np.max(np.abs(goal - start))) / maximum_step_rad)),
    )
    return np.linspace(start, goal, segment_count + 1)


def _first_colliding_state(
    robot: Robot,
    positions: np.ndarray,
    hand: np.ndarray,
) -> int | None:
    manager = robot.env.getDiscreteContactManager()
    manager.setActiveCollisionObjects(robot.env.getActiveLinkNames())
    manager.setCollisionMarginData(CollisionMarginData(COLLISION_CLEARANCE_M))
    request = ContactRequest(ContactTestType_FIRST)
    for index, arm in enumerate(positions):
        robot.set_joints(_robot_joint_state(arm, hand))
        manager.setCollisionObjectsTransform(robot.env.getState().link_transforms)
        contacts = ContactResultMap()
        manager.contactTest(contacts, request)
        if not contacts.empty():
            return index
    return None


def _solve_goal(
    robot: Robot,
    target: np.ndarray,
    start_arm: np.ndarray,
    start_hand: np.ndarray,
    bounds: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    pose = Pose(target)
    nominal = robot.ik(
        "kuka",
        pose,
        seed=TASK_NOMINAL_ELBOW_DOWN_SEED,
        tip_link="handbase_link",
    )
    if nominal is not None:
        nominal = np.asarray(nominal, dtype=np.float64).reshape(7)
        inside = bool(
            np.all(nominal >= bounds[:, 0])
            and np.all(nominal <= bounds[:, 1])
        )
        if inside and _goal_is_collision_free(robot, nominal, start_hand):
            return nominal, {
                "method": "TASK_NOMINAL_ELBOW_DOWN_SEED",
                "seed_count": 1,
                "ik_solution_count": 1,
                "collision_free_solution_count": 1,
            }

    direct = robot.ik(
        "kuka", pose, seed=start_arm, tip_link="handbase_link"
    )
    if direct is not None:
        direct = np.asarray(direct, dtype=np.float64).reshape(7)
        inside = bool(
            np.all(direct >= bounds[:, 0])
            and np.all(direct <= bounds[:, 1])
        )
        if inside and _goal_is_collision_free(robot, direct, start_hand):
            return direct, {
                "method": "CURRENT_STATE_SEED_AFTER_TASK_NOMINAL_FAILED",
                "seed_count": 2,
                "ik_solution_count": 1,
                "collision_free_solution_count": 1,
            }

    generator = np.random.default_rng(GLOBAL_IK_RANDOM_SEED)
    candidates: list[np.ndarray] = []
    ik_solution_count = 0
    for seed in generator.uniform(
        bounds[:, 0], bounds[:, 1], size=(GLOBAL_IK_SEED_COUNT - 2, 7)
    ):
        solution = robot.ik(
            "kuka", pose, seed=seed, tip_link="handbase_link"
        )
        if solution is None:
            continue
        ik_solution_count += 1
        solution = np.asarray(solution, dtype=np.float64).reshape(7)
        if (
            np.all(solution >= bounds[:, 0])
            and np.all(solution <= bounds[:, 1])
            and _goal_is_collision_free(robot, solution, start_hand)
            and all(
                np.linalg.norm(solution - existing) > 1.0e-3
                for existing in candidates
            )
        ):
            candidates.append(solution)
    if not candidates:
        raise RuntimeError(
            "Tesseract found no collision-free IK goal inside the soft limits"
        )
    goal = min(candidates, key=lambda row: float(np.linalg.norm(row - start_arm)))
    return goal, {
        "method": "DETERMINISTIC_MULTI_SEED_KDL",
        "random_seed": GLOBAL_IK_RANDOM_SEED,
        "seed_count": GLOBAL_IK_SEED_COUNT,
        "ik_solution_count": ik_solution_count,
        "collision_free_solution_count": len(candidates),
    }


def _state_program(
    positions: np.ndarray,
    joint_names: list[str],
    manipulator_info: object,
) -> CompositeInstruction:
    result = CompositeInstruction("DEFAULT")
    result.setManipulatorInfo(manipulator_info)
    for row in positions:
        waypoint = StateWaypoint(joint_names, row)
        move = MoveInstruction(
            StateWaypointPoly_wrap_StateWaypoint(waypoint),
            MoveInstructionType_FREESPACE,
            "DEFAULT",
        )
        result.appendMoveInstruction(
            MoveInstructionPoly_wrap_MoveInstruction(move)
        )
    return result


def _trajectory(instructions: CompositeInstruction) -> dict[str, object]:
    points: list[dict[str, object]] = []
    for instruction in instructions.getInstructions():
        waypoint = WaypointPoly_as_StateWaypointPoly(
            InstructionPoly_as_MoveInstructionPoly(instruction).getWaypoint()
        )
        points.append(
            {
                "time_from_start_s": float(waypoint.getTime()),
                "positions_rad": np.asarray(waypoint.getPosition())
                .reshape(-1)
                .tolist(),
                "velocities_rad_s": np.asarray(waypoint.getVelocity())
                .reshape(-1)
                .tolist(),
                "accelerations_rad_s2": np.asarray(
                    waypoint.getAcceleration()
                )
                .reshape(-1)
                .tolist(),
            }
        )
    return {"joint_names": list(ARM_JOINT_NAMES), "points": points}


def plan(repository: Path, request: dict[str, object]) -> dict[str, object]:
    start_names = [str(name) for name in request["start_joint_names"]]
    start_values = _finite_vector(
        request["start_joint_positions_rad"], len(start_names), "start state"
    )
    start_by_name = dict(zip(start_names, start_values))
    if any(name not in start_by_name for name in ARM_JOINT_NAMES + HAND_JOINT_NAMES):
        raise ValueError("planning request omits an active robot joint")
    start_arm = np.asarray([start_by_name[name] for name in ARM_JOINT_NAMES])
    start_hand = np.asarray([start_by_name[name] for name in HAND_JOINT_NAMES])
    target = _matrix4(
        request["world_from_target_link_row_major"], "target pose"
    )
    requested_bounds = dict(request.get("path_joint_bounds_rad", {}))
    bounds = np.asarray(
        [requested_bounds[name] for name in ARM_JOINT_NAMES],
        dtype=np.float64,
    )
    if bounds.shape != (7, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("planning request must define finite soft arm bounds")
    if request.get("group_name", "kuka") != "kuka" or request.get(
        "target_link", "handbase_link"
    ) != "handbase_link":
        raise ValueError("Tesseract adapter only accepts the current kuka hand target")

    with tempfile.TemporaryDirectory(prefix="kcg_tesseract_") as directory:
        urdf, srdf = _prepare_robot_files(
            repository, request, Path(directory)
        )
        robot = Robot.from_files(urdf, srdf)
        obstacle_ids = _add_obstacles(robot, request)
        arm_names = robot.get_joint_names("kuka")
        if tuple(arm_names) != ARM_JOINT_NAMES:
            raise RuntimeError("Tesseract arm joint order differs")
        goal_arm, goal_search = _solve_goal(
            robot, target, start_arm, start_hand, bounds
        )
        robot.set_joints(_robot_joint_state(start_arm, start_hand))
        program = (
            MotionProgram(
                "kuka",
                tcp_frame="handbase_link",
                working_frame="world",
                profile="DEFAULT",
            )
            .set_joint_names(arm_names)
            .start_at(StateTarget(start_arm, names=arm_names, profile="DEFAULT"))
            .move_to(StateTarget(goal_arm, names=arm_names, profile="DEFAULT"))
        )
        instructions = program.to_composite_instruction(
            arm_names, "handbase_link"
        )
        started = time.perf_counter()
        direct_positions = _linear_joint_positions(
            start_arm, goal_arm, LVS_JOINT_SEGMENT_RAD
        )
        direct_first_collision = _first_colliding_state(
            robot, direct_positions, start_hand
        )
        if direct_first_collision is None:
            positions = direct_positions
            selected_path_type = "DIRECT_JOINT_SEGMENT"
            planner_message = "Direct joint-space segment is collision-free"
            ompl_elapsed = None
        else:
            robot.set_joints(_robot_joint_state(start_arm, start_hand))
            assignCurrentStateAsSeed(instructions, robot.env)
            solver = OMPLSolverConfig()
            solver.planning_time = float(
                request.get("allowed_planning_time_s", 8.0)
            )
            solver.optimize = False
            solver.max_solutions = 1
            solver.clearPlanners()
            solver.addPlanner(RRTConnectConfigurator())
            profile = OMPLRealVectorPlanProfile()
            profile.solver_config = solver
            profile.contact_manager_config.default_margin = COLLISION_CLEARANCE_M
            profile.collision_check_config.type = CollisionEvaluatorType.LVS_DISCRETE
            profile.collision_check_config.longest_valid_segment_length = (
                LVS_JOINT_SEGMENT_RAD
            )
            profiles = ProfileDictionary()
            profiles.addProfile(OMPL_NAMESPACE, "DEFAULT", profile)
            planner_request = PlannerRequest()
            planner_request.instructions = instructions
            planner_request.env = robot.env
            planner_request.profiles = profiles
            RNG_setSeed(OMPL_RANDOM_SEED)
            ompl_started = time.perf_counter()
            response = OMPLMotionPlanner(OMPL_NAMESPACE).solve(planner_request)
            ompl_elapsed = time.perf_counter() - ompl_started
            if not response.successful:
                return {
                    "schema_version": "kcg_tesseract_kdl_planner_v2",
                    "success": False,
                    "failure_stage": "tesseract_ompl",
                    "message": str(response.message),
                    "planning_time_s": time.perf_counter() - started,
                    "goal_search": goal_search,
                    "direct_path_check": {
                        "state_count": len(direct_positions),
                        "first_colliding_state_index": direct_first_collision,
                    },
                    "trajectory": None,
                }
            interpolated = generateInterpolatedProgram(
                response.results,
                robot.env,
                OUTPUT_INTERPOLATION_RAD,
                0.15,
                math.radians(5.0),
                10,
            )
            positions = _positions(interpolated)
            selected_path_type = "OMPL_RRTCONNECT"
            planner_message = str(response.message)
        planning_elapsed = time.perf_counter() - started
        timed = _state_program(
            positions, arm_names, instructions.getManipulatorInfo()
        )
        totg_profile = TOTGCompositeProfile()
        totg_profile.max_velocity_scaling_factor = float(
            request.get("maximum_velocity_scaling_factor", 0.1)
        )
        totg_profile.max_acceleration_scaling_factor = float(
            request.get("maximum_acceleration_scaling_factor", 0.1)
        )
        time_profiles = ProfileDictionary()
        time_profiles.addProfile("TOTG", "DEFAULT", totg_profile)
        if not TimeOptimalTrajectoryGeneration().compute(
            timed, robot.env, time_profiles
        ):
            return {
                "schema_version": "kcg_tesseract_kdl_planner_v2",
                "success": False,
                "failure_stage": "tesseract_totg",
                "message": "Tesseract time parameterization failed",
                "planning_time_s": planning_elapsed,
                "trajectory": None,
            }

        final_fk = np.asarray(
            robot.fk("kuka", positions[-1], tip_link="handbase_link").matrix,
            dtype=np.float64,
        )
        rotation_delta = final_fk[:3, :3].T @ target[:3, :3]
        position_error = float(
            np.linalg.norm(final_fk[:3, 3] - target[:3, 3])
        )
        rotation_error = float(
            math.acos(
                np.clip(0.5 * (np.trace(rotation_delta) - 1.0), -1.0, 1.0)
            )
        )
        position_tolerance = float(request.get("position_tolerance_m", 1.0e-4))
        rotation_tolerance = float(
            request.get("orientation_tolerance_rad", 5.0e-4)
        )
        if position_error > position_tolerance or rotation_error > rotation_tolerance:
            return {
                "schema_version": "kcg_tesseract_kdl_planner_v2",
                "success": False,
                "failure_stage": "endpoint_tolerance",
                "message": "Tesseract endpoint is outside the requested tolerance",
                "planning_time_s": planning_elapsed,
                "endpoint_position_error_m": position_error,
                "endpoint_orientation_error_rad": rotation_error,
                "trajectory": None,
            }
        return {
            "schema_version": "kcg_tesseract_kdl_planner_v2",
            "success": True,
            "failure_stage": None,
            "message": planner_message,
            "planner": "TESSERACT_KDL_DIRECT_IF_CLEAR_ELSE_OMPL_RRTCONNECT",
            "selected_path_type": selected_path_type,
            "tesseract_distribution_version": importlib.metadata.version(
                "tesseract-robotics-nanobind"
            ),
            "planning_time_s": planning_elapsed,
            "ompl_planning_time_s": ompl_elapsed,
            "ompl_random_seed": OMPL_RANDOM_SEED,
            "goal_search": goal_search,
            "direct_path_check": {
                "state_count": len(direct_positions),
                "first_colliding_state_index": direct_first_collision,
            },
            "goal_arm_positions_rad": goal_arm.tolist(),
            "minimum_goal_soft_limit_margin_rad": float(
                np.min(
                    np.minimum(
                        goal_arm - bounds[:, 0], bounds[:, 1] - goal_arm
                    )
                )
            ),
            "collision_object_ids": obstacle_ids,
            "collision_model": "DETAILED_ORIGINAL_LINK_MESHES",
            "planning_collision_check": {
                "type": "LVS_DISCRETE",
                "longest_valid_joint_segment_rad": LVS_JOINT_SEGMENT_RAD,
                "required_clearance_m": COLLISION_CLEARANCE_M,
                "continuous_collision_proof_claimed": False,
            },
            "path_smoothing_applied": False,
            "endpoint_position_error_m": position_error,
            "endpoint_orientation_error_rad": rotation_error,
            "maximum_interpolated_joint_step_rad": float(
                np.max(np.abs(np.diff(positions, axis=0)))
            ),
            "trajectory": _trajectory(timed),
            "requires_isaac_exact_fcl_recheck": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    request_text = (
        sys.stdin.read()
        if args.request == "-"
        else Path(args.request).read_text(encoding="utf-8")
    )
    request = json.loads(request_text)
    if not isinstance(request, dict):
        raise ValueError("planning request must be one object")
    repository = Path(__file__).resolve().parents[3]
    result = plan(repository, request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
