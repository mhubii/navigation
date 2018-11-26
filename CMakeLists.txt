cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

set(CMAKE_INSTALL_PREFIX $ENV{HOME}/navigation CACHE STRING "Install path prefix default.")
set(CMAKE_CXX_STANDARD 11)

project(navigation)

# Options.
option(BUILD_WITH_TORCH "Whether to build Torch. This enables learning features." OFF)
option(BUILD_WITH_ORB_SLAM2 "Whether to build with ORB SLAM2. This enables SLAM." OFF) # TODO

# Include.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/navigation)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# Sources.
set(NAVIGATION_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NAVIGATION_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/navigation)
set(NAVIGATION_SRCS ${NAVIGATION_SRC_DIR}/keyboard.cpp)
set(NAVIGATION_INCLUDES ${NAVIGATION_INCLUDE_DIR}/keyboard.h)

# Optional.
if (${BUILD_WITH_TORCH})

  find_package(Torch REQUIRED)
  find_package(OpenCV REQUIRED)

  include_directories(${Torch_INCLUDE_DIRS})
  include_directories(${OpenCV_INCLUDE_DIRS})

  list(APPEND NAVIGATION_SRCS ${NAVIGATION_SRC_DIR}/q_learning.cpp)
  list(APPEND NAVIGATION_INCLUDES ${NAVIGATION_INCLUDE_DIR}/q_learning.h)
  list(APPEND NAVIGATION_LIBS ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

endif()

if (${BUILD_WITH_ORB_SLAM2})

  find_package(ORB_SLAM2 REQUIRED)

endif()

# Add navigation library.
add_library(navigation SHARED ${NAVIGATION_SRCS})
target_link_libraries(navigation ${NAVIGATION_LIBS})

# Install directives.
install(TARGETS navigation DESTINATION lib)
install(FILES ${NAVIGATION_INCLUDES} DESTINATION include/navigation)

# Uninstall directives.
add_custom_target (uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/navigation_uninstall.cmake)







