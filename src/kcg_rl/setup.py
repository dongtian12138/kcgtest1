from glob import glob
import os

from setuptools import find_packages, setup


package_name = "kcg_rl"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [os.path.join("resource", package_name)],
        ),
        (os.path.join("share", package_name), ["package.xml", "README.md"]),
        (
            os.path.join("share", package_name),
            ["requirements-training.txt", "requirements-isaac-rl.txt"],
        ),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="dong",
    maintainer_email="974622934@qq.com",
    description=(
        "RL adapters for KCG Gazebo grasping and Isaac connector tasks."
    ),
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "cylinder_rl_smoke = kcg_rl.rl_smoke:main",
            "d38999_rl_readiness = kcg_rl.full_skill_readiness:main",
        ],
    },
)
