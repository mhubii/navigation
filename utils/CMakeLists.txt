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
find_library(GLOG_LIB glog)
find_package(Protobuf 3 REQUIRED)

if(NOT CAFFE2_LIB)
  message(FATAL_ERROR "Caffe2 lib not found")
endif()

#set(SOURCE
#    ${CMAKE_CURRENT_SOURCE_DIR}/net_util.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/net_util_operator.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/model_base.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_util.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/blob_util.cpp
#)

#add_library(neural_net SHARED ${SOURCE})
#target_link_libraries(neural_net ${CAFFE2_LIB} ${OpenCV_LIBRARIES} ${PROTOBUF_LIBRARIES})

# Example for TensorUtil.
#add_executable(example_tensor_util example_tensor_util.cpp)
#target_link_libraries(example_tensor_util read_cams neural_net ${OpenCV_LIBRARIES})

# Example TestModel.
#add_executable(example_test_model example_test_model.cpp)
#target_link_libraries(example_test_model neural_net)

add_executable(play_ground play_ground.cpp)
target_link_libraries(play_ground ${CAFFE2_LIB} ${PROTOBUF_LIBRARIES} ${GLOG_LIB})
