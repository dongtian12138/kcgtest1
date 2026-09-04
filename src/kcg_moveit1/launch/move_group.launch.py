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
    robot_description = moveit_config.robot_description["robot_description"]
    for link_name in ("f1Link3", "f2Link2", "f3Link3"):
        old_mesh = f"meshes/hand/collision/{link_name}_convex.stl"
        new_mesh = f"meshes/hand/connector_no_nail/{link_name}_nailfree.stl"
        if robot_description.count(old_mesh) != 1:
            raise RuntimeError(
                f"expected exactly one MoveIt collision mesh for {link_name}"
            )
        robot_description = robot_description.replace(old_mesh, new_mesh)
    moveit_config.robot_description["robot_description"] = robot_description
    return generate_move_group_launch(moveit_config)
