"""Process-local environment locations; no control or geometry parameters."""
import os
from pathlib import Path


def isaac_environment_prefix():
    repository = Path(__file__).resolve().parents[3]
    return Path(os.environ.get("ISAAC_ENV_PREFIX", repository.parent / "isaacsim/.conda-env")).expanduser().absolute()


def sam6d_runtime(repository, *, root=None, python=None):
    repository = Path(repository).resolve()
    packaged = repository / ".deps/SAM-6D"
    default_root = packaged / "SAM-6D" if packaged.is_dir() else Path.home() / ".cache/kcgtest1-sam6d/SAM-6D"
    default_python = packaged / ".venv/bin/python" if (packaged / ".venv/bin/python").is_file() else Path.home() / ".cache/kcgtest1-sam6d/.venv/bin/python"
    source = Path(root or os.environ.get("KCG_SAM6D_ROOT", default_root)).expanduser().resolve()
    # Resolving a venv Python symlink can select the base interpreter instead.
    executable = Path(python or os.environ.get("KCG_SAM6D_PYTHON", default_python)).expanduser().absolute()
    return source, executable


def planner_python(repository):
    return Path(os.environ.get("KCG_PLANNER_PYTHON", Path(repository) / ".venv/bin/python")).expanduser().absolute()


def ros_setup_bash():
    return Path(os.environ.get("KCG_ROS_SETUP", "/opt/ros/humble/setup.bash")).expanduser().absolute()
