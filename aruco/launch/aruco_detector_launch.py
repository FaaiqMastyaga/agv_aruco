from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # V4L2 Camera Node
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='camera',
            parameters=[{
                'video_device': '/dev/video2',
                'image_size': [640, 480],
                'pixel_format': 'YUYV',
                'camera_frame_id': 'camera_frame'
            }],
        ),

        # ArUco Detector node
        Node(
            package='aruco',
            executable='aruco_detector_node',
            name='aruco_detector',
        ),
    ])