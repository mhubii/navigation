cmake_minimum_required(VERSION 2.8.3)
project(beginner_tutorials)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  image_transport
  sensor_msgs
  cv_bridge
)

find_package(YARP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(octomap REQUIRED)
find_package(Pangolin REQUIRED)
find_package(ORB_SLAM2 REQUIRED)
find_package(PCL REQUIRED)
find_package(OctoMap REQUIRED)
find_package(Boost REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES beginner_tutorials
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(talker src/talker.cpp)
target_link_libraries(talker ${catkin_LIBRARIES})
#add_dependencies(talker beginner_tutorials_generate_messages_cpp)

add_executable(listener src/listener.cpp)
target_link_libraries(listener ${catkin_LIBRARIES})
#add_dependencies(listener beginner_tutorials_generate_messages_cpp)

add_executable(yarp_to_ros_pub src/yarp_to_ros_pub.cpp)
target_link_libraries(yarp_to_ros_pub ${catkin_LIBRARIES} ${YARP_LIBRARIES})

include_directories(${PROJECT_SOURCE_DIR}/../../../third_party/ORB_SLAM2 ${PROJECT_SOURCE_DIR}/../../../third_party/ORB_SLAM2/include)
add_executable(stereo_slam src/stereo_slam.cpp)
target_link_libraries(stereo_slam ${catkin_LIBRARIES} ${ORB_SLAM2_LIBRARY} ${Pangolin_LIBRARIES} ${YARP_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES} ${BOOST_LIBRARIES})
    
