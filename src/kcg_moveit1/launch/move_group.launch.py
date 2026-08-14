from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("handarm", package_name="kcg_moveit1")
        .robot_description(
            file_path="config/handarm.urdf.xacro",
            mappings={"use_gazebo": "false"},
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
    return generate_move_group_launch(moveit_config)
