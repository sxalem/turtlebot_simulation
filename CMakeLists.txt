cmake_minimum_required(VERSION 3.10)
project(turtlebot_simulation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  tf
  cv_bridge
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  std_srvs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    tf
    cv_bridge
    sensor_msgs
    nav_msgs
    geometry_msgs
    std_msgs
    visualization_msgs
    std_srvs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# install **all** your Python scripts (including tasks_class.py)
catkin_install_python(
  PROGRAMS
    scripts/aruco_detection.py
    scripts/task_builder.py
    scripts/task_priority_node.py
    scripts/tasks_class.py
    scripts/kinematics.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

