import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'vision_odometry_noise'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
        (os.path.join('share', package_name, 'data'), glob(os.path.join(package_name, 'data', '*.csv'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mgardenswartz@ad.ufl.edu',
    maintainer_email='max.gardenswartz@proton.me',
    description='Injects noisy, jittered external-vision odometry sourced from Gazebo ground truth, published to PX4 as vehicle_visual_odometry.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_odometry_noise_node = vision_odometry_noise.vision_odometry_noise_node:main',
        ],
    },
)
