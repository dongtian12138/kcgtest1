from glob import glob
import os

from setuptools import find_packages, setup


package_name = "kcg_connector"


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
            [
                "requirements-isaacsim.txt",
                "requirements-torch-cu128.txt",
            ],
        ),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "isaac"), glob("isaac/*.py")),
        (
            os.path.join(
                "share",
                package_name,
                "assets",
                "public_specs",
                "mil_dtl_38999",
            ),
            glob("assets/public_specs/mil_dtl_38999/*"),
        ),
        (
            os.path.join("share", package_name, "docs"),
            glob("docs/*.md"),
        ),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "PyYAML",
        "mpmath==1.2.1",
        "scipy==1.8.0",
    ],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="dong",
    maintainer_email="974622934@qq.com",
    description=(
        "Connector assembly task logic for the KUKA iiwa and KCG hand."
    ),
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "export_isaac_urdf = kcg_connector.export_isaac_urdf:main",
        ],
    },
)
