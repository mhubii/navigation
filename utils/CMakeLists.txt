# Keyboard input user navigation.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(keyboard SHARED keyboard.cpp)

# Read cameras from yarp server.
find_package(YARP REQUIRED)
find_package(OpenCV REQUIRED)

add_library(read_cams SHARED read_cams.cpp)
target_link_libraries(read_cams ${YARP_LIBRARIES} ${OpenCV_LIBRARIES})

# Test ReadCams.
add_executable(test test.cpp)
target_link_libraries(test read_cams ${OpenCV_LIBRARIES})

