import os
from glob import glob

from setuptools import find_packages, setup

# colcon_detected = "COLCON_CURRENT_PREFIX" in os.environ

package_name = 'grad_sdf'
setup_kwargs = dict()

# Avoid stdout in colcon metadata parsing.
setup_kwargs["data_files"] = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.py') if os.path.exists('launch') else []),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml') if os.path.exists('config') else []),
    (os.path.join('share', package_name, 'configs'), glob('configs/**/*.yaml', recursive=True) if os.path.exists('configs') else []),
]

setup_kwargs["entry_points"] = {
    'console_scripts': [
        'mapping_node = grad_sdf.node.mapping_node:main',
        'mapping_sim_node = grad_sdf.node.mapping_sim_node:main',
        'mapping_quad_node = grad_sdf.node.mapping_quad_node:main',
    ],
}


setup(
    name="grad_sdf",
    version="0.1",
    author="Zhirui Dai, Qihao Qian",
    author_email="zhdai@ucsd.edu, q2qian@ucsd.edu",
    description="A package for gradient-based signed distance functions",
    url="https://github.com/ExistentialRobotics/grad-sdf",
    install_requires=['setuptools'],
    license='MIT',
    packages=find_packages(),
    zip_safe=True,
    **setup_kwargs,
)
