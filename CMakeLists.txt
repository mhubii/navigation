cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(path_finding)
set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ORB SLAM2.
#configure_file(cmake/CMakeLists.txt.in_orb_slam2 orb_slam2_download/CMakeLists.txt)

#execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
#    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/orb_slam2_download" )

#execute_process(COMMAND "${CMAKE_COMMAND}" --build .
#    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/orb_slam2_download")

# Build.
#add_custom_target(orb_slam2 ALL 
#    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build/orb_slam2/build.sh
#    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/orb_slam2)

# Utilities.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
add_subdirectory(utils)

# Source.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(src)

# Gazebo plugins.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vehicle/plugins)
add_subdirectory(vehicle)

