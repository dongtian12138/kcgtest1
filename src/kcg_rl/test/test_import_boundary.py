"""Verify that pure RL imports do not cross the ROS Python ABI boundary."""

from pathlib import Path
import os
import subprocess
import sys


def test_package_import_does_not_load_ros_modules():
    package_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)
    script = """
import importlib
import sys

package = importlib.import_module("kcg_rl")
assert package.__all__ == ["CylinderEnvConfig", "KcgCylinderEnv"]
assert "kcg_rl.cylinder_env" not in sys.modules
for name in (
    "rclpy",
    "builtin_interfaces",
    "gazebo_msgs",
    "rosgraph_msgs",
    "sensor_msgs",
    "std_msgs",
    "std_srvs",
    "trajectory_msgs",
):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_connector_residual_contract_stays_ros_and_simulator_free():
    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "kcg_rl"), str(source_root / "kcg_connector"))
    )
    script = """
import importlib
import sys

module = importlib.import_module("kcg_connector.residual_rl")
for name in (
    "load_connector_residual_config",
    "ConnectorResidualState",
    "decode_residual_action",
    "residual_observation",
    "evaluate_residual_state",
    "calculate_residual_reward",
):
    assert hasattr(module, name), name
for name in ("rclpy", "omni", "isaacsim"):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
