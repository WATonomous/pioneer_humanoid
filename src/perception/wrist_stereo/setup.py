from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'wrist_stereo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pioneer Humanoid Team',
    maintainer_email='pioneer@uwaterloo.ca',
    description='Dynamic baseline stereo depth from wrist-mounted cameras',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_depth_node = wrist_stereo.stereo_depth_node:main',
            'calibrate_intrinsics = wrist_stereo.calibrate_intrinsics:main',
            'calibrate_hand_eye = wrist_stereo.calibrate_hand_eye:main',
        ],
    },
)
