import os

from setuptools import find_packages, setup


PACKAGE_NAME = "race_auv_camera_pkg"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + PACKAGE_NAME]),
        (os.path.join("share", PACKAGE_NAME), ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="dev@example.com",
    description=(
        "Camera + per-camera AprilTag detector scripts (no TF, no fuser). "
        "All configuration lives in race_auv_bringup; this package only "
        "ships the detector executable and its private modules."
    ),
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "apriltag_detector_node = race_auv_camera_pkg.apriltag_detector_node:main",
            "apriltag_fuser_node     = race_auv_camera_pkg.apriltag_fuser_node:main",
        ],
    },
)