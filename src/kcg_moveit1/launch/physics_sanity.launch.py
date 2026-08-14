from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    duration = LaunchConfiguration("duration")
    return LaunchDescription(
        [
            DeclareLaunchArgument("duration", default_value="30.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("kcg_moveit1"), "launch", "gazebo.launch.py"]
                    )
                ),
                launch_arguments={
                    "gui": "false",
                    "use_rviz": "false",
                    "start_moveit": "false",
                    "start_trajectory_controllers": "false",
                    "run_stability_check": "true",
                    "stability_duration": duration,
                }.items(),
            ),
        ]
    )
