from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Pose detection node
    pose_detect_node = Node(
        package='your_package_name',
        executable='pose_detect',
        name='pose_detect',
        output='screen'
    )

    # TurtleBot movement node
    turtlebot_movement_node = Node(
        package='your_package_name',
        executable='turtlebot_movement',
        name='turtlebot_movement',
        output='screen'
    )

    # Stop follow node
    stop_follow_node = Node(
        package='your_package_name',
        executable='stop_follow',
        name='stop_follow',
        output='screen'
    )

    return LaunchDescription([
        pose_detect_node,
        turtlebot_movement_node,
        stop_follow_node
    ])
