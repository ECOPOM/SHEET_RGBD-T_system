import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    node = Node(
        namespace='',
        package='label_generator',
        name='label_generator_node',
        executable='label_generator_node',
        parameters=[
            {"destination": "/home/rovermsi/Desktop/Ros_image_extractor/SHEET_061022_ALBERO_RMSE_TEMP/"},
            {"fps_count": 5}
        ]
    )
    
    ld.add_action(node)
    return ld
