cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# Utilities.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
add_subdirectory(utils)

# Gazebo plugins.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugins)
add_subdirectory(irobot)

