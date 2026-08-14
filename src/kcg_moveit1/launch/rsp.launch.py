from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_rsp_launch


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("handarm", package_name="kcg_moveit1")
        .robot_description(
            file_path="config/handarm.urdf.xacro",
            mappings={"use_gazebo": "false"},
        )
        .to_moveit_configs()
    )
    return generate_rsp_launch(moveit_config)
