import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import IfElseSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    task_share = get_package_share_directory("kcg_grasping")
    moveit_share = get_package_share_directory("kcg_moveit1")

    gui = LaunchConfiguration("gui")
    use_rviz = LaunchConfiguration("use_rviz")
    run_baseline = LaunchConfiguration("run_baseline")
    fast_reset = LaunchConfiguration("fast_reset")
    shutdown_on_completion = LaunchConfiguration("shutdown_on_completion")
    start_at_cylinder_pregrasp = LaunchConfiguration(
        "start_at_cylinder_pregrasp"
    )
    task_config = os.path.join(task_share, "config", "cylinder_task.yaml")
    task_world = IfElseSubstitution(
        fast_reset,
        os.path.join(task_share, "worlds", "cylinder_grasp.world"),
        os.path.join(task_share, "worlds", "cylinder_grasp_demo.world"),
    )

    baseline = Node(
        package="kcg_grasping",
        executable="scripted_cylinder_grasp.py",
        name="scripted_cylinder_grasp",
        output="screen",
        parameters=[task_config],
        condition=IfCondition(run_baseline),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("run_baseline", default_value="false"),
            DeclareLaunchArgument(
                "fast_reset",
                default_value="false",
                description=(
                    "Use the RL reset world and teleport the cylinder at "
                    "episode reset. The default demonstration mode starts "
                    "with the cylinder already on the pedestal."
                ),
            ),
            DeclareLaunchArgument(
                "start_at_cylinder_pregrasp",
                default_value="false",
                description=(
                    "Initialize the simulated robot at the validated cylinder "
                    "pregrasp.  Intended for fast RL episode startup."
                ),
            ),
            DeclareLaunchArgument(
                "shutdown_on_completion", default_value="false"
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(moveit_share, "launch", "gazebo.launch.py")
                ),
                launch_arguments={
                    "world": task_world,
                    "gui": gui,
                    "use_rviz": use_rviz,
                    "start_moveit": "false",
                    "start_trajectory_controllers": "true",
                    "activate_hand_controller": "true",
                    "start_at_cylinder_pregrasp": start_at_cylinder_pregrasp,
                }.items(),
            ),
            Node(
                package="kcg_grasping",
                executable="cylinder_task_manager.py",
                name="cylinder_task_manager",
                output="screen",
                parameters=[
                    task_config,
                    {
                        "teleport_on_reset": ParameterValue(
                            fast_reset, value_type=bool
                        )
                    },
                ],
            ),
            TimerAction(period=8.0, actions=[baseline]),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=baseline,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(reason="Scripted grasp completed")
                        )
                    ],
                ),
                condition=IfCondition(shutdown_on_completion),
            ),
        ]
    )
