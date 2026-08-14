import os
import re
from glob import glob

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    GroupAction,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.actions import SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    package_share = get_package_share_directory("kcg_moveit1")
    description_share = get_package_share_directory("iiwa_description")
    gazebo_share = get_package_share_directory("gazebo_ros")
    start_at_cylinder_pregrasp = LaunchConfiguration(
        "start_at_cylinder_pregrasp"
    )

    gazebo_model_paths = [os.path.dirname(description_share)]
    gazebo_model_paths.extend(sorted(glob("/usr/share/gazebo-*/models")))
    gazebo_model_paths.extend(
        path
        for path in os.environ.get("GAZEBO_MODEL_PATH", "").split(os.pathsep)
        if path
    )
    gazebo_model_path = os.pathsep.join(dict.fromkeys(gazebo_model_paths))

    moveit_config = (
        MoveItConfigsBuilder("handarm", package_name="kcg_moveit1")
        .robot_description(
            file_path="config/handarm.urdf.xacro",
            mappings={
                "use_gazebo": "true",
                "start_at_cylinder_pregrasp": start_at_cylinder_pregrasp,
            },
        )
        .robot_description_semantic(file_path="config/handarm.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .trajectory_execution(
            file_path="config/moveit_controllers.yaml",
            moveit_manage_controllers=False,
        )
        .to_moveit_configs()
    )

    description_key = "robot_description"
    description = moveit_config.robot_description[description_key]
    if isinstance(description, str):
        moveit_config.robot_description[description_key] = re.sub(
            r"<!--.*?-->", "", description, flags=re.DOTALL
        ).replace("\n", "")

    world = LaunchConfiguration("world")
    gui = LaunchConfiguration("gui")
    use_rviz = LaunchConfiguration("use_rviz")
    start_moveit = LaunchConfiguration("start_moveit")
    start_trajectory_controllers = LaunchConfiguration(
        "start_trajectory_controllers"
    )
    activate_hand_controller = LaunchConfiguration("activate_hand_controller")
    run_stability_check = LaunchConfiguration("run_stability_check")
    stability_duration = LaunchConfiguration("stability_duration")

    move_group_params = [
        moveit_config.to_dict(),
        {
            "allow_trajectory_execution": True,
            "publish_robot_description_semantic": True,
            "publish_planning_scene": True,
            "publish_geometry_updates": True,
            "publish_state_updates": True,
            "publish_transforms_updates": True,
            "use_sim_time": True,
        },
    ]

    stability_check = Node(
        package="kcg_moveit1",
        executable="check_physics_stability.py",
        name="physics_stability_check",
        output="screen",
        parameters=[
            {
                "duration": ParameterValue(stability_duration, value_type=float),
                "joint_state_topic": (
                    "/physics_joint_state_broadcaster/joint_states"
                ),
                "use_sim_time": True,
            }
        ],
        condition=IfCondition(run_stability_check),
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
            SetEnvironmentVariable("GAZEBO_MODEL_PATH", gazebo_model_path),
            DeclareLaunchArgument(
                "world",
                default_value=os.path.join(package_share, "worlds", "physics.world"),
            ),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("start_moveit", default_value="true"),
            DeclareLaunchArgument(
                "start_trajectory_controllers", default_value="true"
            ),
            DeclareLaunchArgument(
                "activate_hand_controller",
                default_value="true",
                description=(
                    "Spawn the simulated hand controller active.  Task launches "
                    "may keep it inactive while the ideally-positioned arm makes "
                    "a large initial move, then activate it through controller_manager."
                ),
            ),
            DeclareLaunchArgument(
                "start_at_cylinder_pregrasp",
                default_value="false",
                description="Initialize Gazebo at the cylinder task pregrasp.",
            ),
            DeclareLaunchArgument("run_stability_check", default_value="false"),
            DeclareLaunchArgument("stability_duration", default_value="30.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(gazebo_share, "launch", "gazebo.launch.py")
                ),
                launch_arguments={"world": world, "gui": gui}.items(),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[moveit_config.robot_description, {"use_sim_time": True}],
            ),
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=["-entity", "kcg_handarm", "-topic", "robot_description"],
                output="screen",
            ),
            # Spawners wait for controller_manager themselves.  Starting them
            # immediately prevents a non-home initial pose from falling under
            # gravity for several seconds before its controllers are active.
            TimerAction(
                period=0.0,
                actions=[
                    Node(
                        package="controller_manager",
                        executable="spawner",
                        arguments=[
                            "joint_state_broadcaster",
                            "-c",
                            "/controller_manager",
                        ],
                        output="screen",
                    ),
                    Node(
                        package="controller_manager",
                        executable="spawner",
                        arguments=[
                            "finger_torque_broadcaster",
                            "-c",
                            "/controller_manager",
                        ],
                        output="screen",
                    ),
                    Node(
                        package="controller_manager",
                        executable="spawner",
                        arguments=[
                            "controller_gazebo_kuka",
                            "-c",
                            "/controller_manager",
                        ],
                        output="screen",
                        condition=IfCondition(start_trajectory_controllers),
                    ),
                    GroupAction(
                        condition=IfCondition(start_trajectory_controllers),
                        actions=[
                            Node(
                                package="controller_manager",
                                executable="spawner",
                                arguments=[
                                    "controller_gazebo_hand",
                                    "-c",
                                    "/controller_manager",
                                ],
                                output="screen",
                                condition=IfCondition(activate_hand_controller),
                            ),
                            Node(
                                package="controller_manager",
                                executable="spawner",
                                arguments=[
                                    "controller_gazebo_hand",
                                    "-c",
                                    "/controller_manager",
                                    "--inactive",
                                ],
                                output="screen",
                                condition=UnlessCondition(activate_hand_controller),
                            ),
                        ],
                    ),
                    Node(
                        package="controller_manager",
                        executable="spawner",
                        arguments=[
                            "physics_joint_state_broadcaster",
                            "-c",
                            "/controller_manager",
                        ],
                        output="screen",
                        condition=IfCondition(run_stability_check),
                    ),
                ],
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=move_group_params,
                condition=IfCondition(start_moveit),
            ),
            GroupAction(
                actions=[
                    SetParameter(name="use_sim_time", value=True),
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            os.path.join(
                                package_share,
                                "launch",
                                "moveit_rviz.launch.py",
                            )
                        )
                    ),
                ],
                condition=IfCondition(use_rviz),
            ),
            TimerAction(period=6.0, actions=[stability_check]),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=stability_check,
                    on_exit=[EmitEvent(event=Shutdown(reason="Stability check completed"))],
                ),
                condition=IfCondition(run_stability_check),
            ),
        ]
    )
