cmake_minimum_required(VERSION 2.8.3)
project(lili_om)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  message_generation
  pcl_ros
  pcl_conversions
  livox_ros_driver
)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED)

message(STATUS "Using catkin version ${catkin_VERSION}")
message(STATUS "Using Ceres version ${CERES_VERSION}")

catkin_package(
    # LIBRARIES ${PROJECT_NAME}
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp rospy message_runtime std_msgs nav_msgs geometry_msgs
    DEPENDS EIGEN3
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
)

link_directories(
    include
    ${GTSAM_LIBRARY_DIRS}
)

## Declare a C++ executable
add_executable(FormatConvert src/FormatConvert.cpp)
target_link_libraries(FormatConvert ${catkin_LIBRARIES})

add_executable(Preprocessing src/Preprocessing.cpp)
target_link_libraries(Preprocessing ${catkin_LIBRARIES})

add_executable(LidarOdometry src/LidarOdometry.cpp)
target_link_libraries(LidarOdometry ${catkin_LIBRARIES} ${CERES_LIBRARIES})

add_executable(BackendFusion src/BackendFusion.cpp src/MarginalizationFactor.cpp)
target_link_libraries(BackendFusion ${catkin_LIBRARIES} ${CERES_LIBRARIES} gtsam)

#For Debug
set(CMAKE_BUILD_TYPE "Debug")
