import os
from glob import glob
from setuptools import setup

package_name = 'perception'

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        # Install marker file in the package index
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        # Include our package.xml file
        (os.path.join("share", package_name), ["package.xml"]),
        # Include all launch files.
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py")),
        ),
        # Include all config files.
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="watonomous",
    maintainer_email="watonomous@uwaterloo.ca",
    description="Perception pipeline for humanoid robot",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dummy_publisher_node = perception.dummy_publisher_node:main",
            "pose_estimation_node = perception.pose_estimation_node:main",
            "dummy_tracknet_output = perception.dummy_tracknet_output:main",
            "trajectory_EKF_prediction_node = "
            "perception.trajectory_EKF_prediction_node:main",
            "dummy_shuttlecock_publisher = "
            "perception.dummy_shuttlecock_publisher:main",
            "shuttlecock_localizer_node = "
            "perception.shuttlecock_localizer_node:main",
            "latency_logger_node = perception.latency_logger_node:main",
        ],
    },
)
