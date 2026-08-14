import xml.etree.ElementTree as ElementTree

import pytest

from kcg_connector.export_isaac_urdf import sanitize_urdf


def test_sanitize_resolves_meshes_and_removes_simulator_extensions(tmp_path):
    package = tmp_path / "description"
    mesh = package / "meshes" / "link.stl"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"solid link\nendsolid link\n")
    source = """
    <robot name="test">
      <link name="base">
        <visual>
          <geometry>
            <mesh filename="package://description/meshes/link.stl"/>
          </geometry>
        </visual>
      </link>
      <gazebo reference="base"/>
      <ros2_control name="test" type="system"/>
      <transmission name="test"/>
    </robot>
    """

    result = sanitize_urdf(source, {"description": package})
    root = ElementTree.fromstring(result)
    assert root.find("gazebo") is None
    assert root.find("ros2_control") is None
    assert root.find("transmission") is None
    assert root.find("./link/visual/geometry/mesh").get("filename") == str(
        mesh.resolve()
    )


def test_sanitize_rejects_unknown_package_uri():
    source = """
    <robot name="test">
      <link name="base">
        <visual><geometry><mesh filename="package://missing/a.stl"/></geometry></visual>
      </link>
    </robot>
    """
    with pytest.raises(ValueError, match="unresolved mesh URI"):
        sanitize_urdf(source, {})
