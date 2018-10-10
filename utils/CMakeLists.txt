# Keyboard input user navigation.
add_library(keyboard SHARED keyboard.cpp)

# Read cameras from yarp server.
find_package(YARP REQUIRED)
find_package(OpenCV REQUIRED)

add_library(read_cams SHARED read_cams.cpp)
target_link_libraries(read_cams ${YARP_LIBRARIES} ${OpenCV_LIBRARIES})

# Example for ReadCams.
add_executable(example_read_cams example_read_cams.cpp)
target_link_libraries(example_read_cams read_cams ${OpenCV_LIBRARIES})

# Tensor utility functions.
find_library(CAFFE2_LIB caffe2)

if(NOT CAFFE2_LIB)
  message(FATAL_ERROR "Caffe2 lib not found")
endif()

add_library(tensor_util SHARED tensor_util.cpp)
target_link_libraries(tensor_util ${CAFFE2_LIB} ${OpenCV_LIBARIES})

# Example for TensorUtil.
add_executable(example_tensor_util example_tensor_util.cpp)
target_link_libraries(example_tensor_util read_cams tensor_util ${OpenCV_LIBRARIES})
