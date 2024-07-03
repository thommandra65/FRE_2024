from setuptools import find_packages, setup

package_name = 'fre_2024'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='megatron',
    maintainer_email='megatron@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	"ransac_plant = fre_2024.Ransac_plant:main",
            	"point0_node = fre_2024.point0:main",
            	"point0_test_node = fre_2024.point0_test:main",
            	"ransac_gazebo = fre_2024.Ransac_gazebo:main",
            	"ransac_wt = fre_2024.Ransac_wt:main",
            	"sequence2_node = fre_2024.sequence2:main",
            	"odometry_transform_node = fre_2024.odometry_transform:main",
            	"ransac_gp = fre_2024.goal_publisher:main",
            	"ransac_plant_test = fre_2024.Ransac_plant_test:main",
        ],
    },
)
