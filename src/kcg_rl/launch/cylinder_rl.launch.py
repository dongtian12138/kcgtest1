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
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    grasping_share = get_package_share_directory("kcg_grasping")
    rl_share = get_package_share_directory("kcg_rl")

    gui = LaunchConfiguration("gui")
    use_rviz = LaunchConfiguration("use_rviz")
    run_smoke = LaunchConfiguration("run_smoke")
    smoke_episodes = LaunchConfiguration("smoke_episodes")
    shutdown_on_completion = LaunchConfiguration("shutdown_on_completion")

    smoke = Node(
        package="kcg_rl",
        executable="cylinder_rl_smoke",
        name="cylinder_rl_env",
        output="screen",
        parameters=[os.path.join(rl_share, "config", "cylinder_rl.yaml")],
        arguments=["--episodes", smoke_episodes],
        condition=IfCondition(run_smoke),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("run_smoke", default_value="true"),
            DeclareLaunchArgument("smoke_episodes", default_value="1"),
            DeclareLaunchArgument(
                "shutdown_on_completion", default_value="true"
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        grasping_share, "launch", "cylinder_grasp.launch.py"
                    )
                ),
                launch_arguments={
                    "gui": gui,
                    "use_rviz": use_rviz,
                    "run_baseline": "false",
                    "shutdown_on_completion": "false",
                    "fast_reset": "true",
                    "start_at_cylinder_pregrasp": "true",
                }.items(),
            ),
            TimerAction(period=6.0, actions=[smoke]),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=smoke,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(reason="RL smoke test completed")
                        )
                    ],
                ),
                condition=IfCondition(shutdown_on_completion),
            ),
        ]
    )
