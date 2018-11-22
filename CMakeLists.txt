cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(path_finding)
set(CMAKE_CXX_STANDARD 11)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(gazebo REQUIRED)

include_directories(${Torch_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

# Utilities.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
add_subdirectory(utils)

# Source.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(src)

# Gazebo models.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/world/vehicle/plugins)
add_subdirectory(vehicle)

