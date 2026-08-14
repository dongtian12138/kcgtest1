"""Expand the authoritative KUKA Xacro into an Isaac-importable URDF."""

import argparse
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree


_STRIPPED_TAGS = {"gazebo", "ros2_control", "transmission"}


def _local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _strip_simulator_tags(parent):
    for child in list(parent):
        if _local_name(child.tag) in _STRIPPED_TAGS:
            parent.remove(child)
        else:
            _strip_simulator_tags(child)


def sanitize_urdf(xml_text, package_paths):
    """Remove ROS/Gazebo extensions and resolve every package mesh URI."""
    root = ElementTree.fromstring(xml_text)
    if _local_name(root.tag) != "robot":
        raise ValueError("expanded Xacro root must be <robot>")

    _strip_simulator_tags(root)
    resolved_mesh_count = 0
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if not filename.startswith("package://"):
            continue
        package_reference = filename.removeprefix("package://")
        package_name, separator, relative_path = package_reference.partition("/")
        if not separator or package_name not in package_paths:
            raise ValueError(f"unresolved mesh URI: {filename}")
        resolved = (Path(package_paths[package_name]) / relative_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"mesh does not exist: {resolved}")
        mesh.set("filename", str(resolved))
        resolved_mesh_count += 1

    if resolved_mesh_count == 0:
        raise ValueError("no package mesh URIs were resolved")
    ElementTree.indent(root, space="  ")
    body = ElementTree.tostring(root, encoding="unicode")
    return '<?xml version="1.0"?>\n' + body + "\n"


def export_urdf(output_path):
    """Expand the installed KUKA Xacro and write a clean standalone URDF."""
    from ament_index_python.packages import get_package_share_directory

    xacro_executable = shutil.which("xacro")
    if xacro_executable is None:
        raise RuntimeError("xacro executable is unavailable; source ROS 2 Humble")

    moveit_share = Path(get_package_share_directory("kcg_moveit1"))
    description_share = Path(get_package_share_directory("iiwa_description"))
    xacro_path = moveit_share / "config" / "handarm.urdf.xacro"
    completed = subprocess.run(
        [
            xacro_executable,
            str(xacro_path),
            "use_gazebo:=false",
            "start_at_cylinder_pregrasp:=false",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    sanitized = sanitize_urdf(
        completed.stdout,
        {"iiwa_description": description_share},
    )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(sanitized, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export the authoritative KUKA/KCG Xacro for Isaac Sim."
    )
    parser.add_argument("--output", required=True, help="Destination .urdf path")
    arguments = parser.parse_args()
    output = export_urdf(arguments.output)
    print(f"ISAAC URDF EXPORTED: {output}")


if __name__ == "__main__":
    main()
