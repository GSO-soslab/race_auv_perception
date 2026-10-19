import os

from setuptools import find_packages, setup

PACKAGE_NAME = "race_auv_camera_tag_factorgraph"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),
        (os.path.join("share", PACKAGE_NAME), ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dark3090",
    maintainer_email="yweifu@uri.edu",
    description="Minimal tag36h11 corner + pose detector for race_auv_factorgraph.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "tag_detector_node = race_auv_camera_tag_factorgraph.detector_node:main",
        ],
    },
)
