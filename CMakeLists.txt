cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

set(CMAKE_INSTALL_PREFIX $ENV{HOME}/navigation CACHE STRING "Install path prefix default.")
set(CMAKE_CXX_STANDARD 11)

project(navigation)

# Cmake.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Options.
option(BUILD_WITH_TORCH "Whether to build Torch. This enables learning features." OFF)
option(BUILD_WITH_ORB_SLAM2 "Whether to build with ORB SLAM2. This enables SLAM." OFF) # TODO

# Include.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/navigation)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# Sources.
set(NAVIGATION_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NAVIGATION_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/navigation)
list(APPEND NAVIGATION_SRCS ${NAVIGATION_SRC_DIR}/keyboard.cpp)
list(APPEND NAVIGATION_INCLUDES ${NAVIGATION_INCLUDE_DIR}/keyboard.h)

# Optional.
if (${BUILD_WITH_TORCH})

  find_package(Torch REQUIRED)
  find_package(OpenCV REQUIRED)

  include_directories(${Torch_INCLUDE_DIRS})
  include_directories(${OpenCV_INCLUDE_DIRS})

  list(APPEND NAVIGATION_SRCS ${NAVIGATION_SRC_DIR}/q_learning.cpp  
                              ${NAVIGATION_SRC_DIR}/replay_memory.cpp 
                              ${NAVIGATION_SRC_DIR}/ornstein_uhlenbeck_process.cpp
                              ${NAVIGATION_SRC_DIR}/ddpg_continuous_control.cpp)
  
  list(APPEND NAVIGATION_INCLUDES ${NAVIGATION_INCLUDE_DIR}/q_learning.h 
                                  ${NAVIGATION_INCLUDE_DIR}/replay_memory.h 
                                  ${NAVIGATION_INCLUDE_DIR}/models.h 
                                  ${NAVIGATION_INCLUDE_DIR}/ornstein_uhlenbeck_process.h
                                  ${NAVIGATION_INCLUDE_DIR}/ddpg_continuous_control.h)
  
  list(APPEND NAVIGATION_LIBRARIES ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

endif()

if (${BUILD_WITH_ORB_SLAM2})

  find_package(ORB_SLAM2 REQUIRED)

endif()

# Add navigation library.
add_library(navigation SHARED ${NAVIGATION_SRCS})
target_link_libraries(navigation ${NAVIGATION_LIBRARIES})

# Install directives.
install(TARGETS navigation DESTINATION lib)
install(FILES ${NAVIGATION_INCLUDES} DESTINATION include/navigation)

# Uninstall directives.
add_custom_target (uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/navigation_uninstall.cmake)

