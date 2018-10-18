cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(path_finding)
set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Utilities.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
add_subdirectory(utils)

# Source.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(src)

# Gazebo plugins.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/gazebo/plugins)
add_subdirectory(gazebo)

