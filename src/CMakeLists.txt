# SLAM.
find_package(YARP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(ORB_SLAM2 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/third_party/ORB_SLAM2 
    ${PROJECT_SOURCE_DIR}/third_party/ORB_SLAM2/include 
    ${PCL_INCLUDE_DIRS}
    ${CAFFE2_INCLUDE_DIRS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_executable(orb_slam stereo_slam.cpp)
target_link_libraries(orb_slam ${ORB_SLAM2_LIBRARY} ${Pangolin_LIBRARIES} ${YARP_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

file(COPY ${PROJECT_SOURCE_DIR}/third_party/ORB_SLAM2/Vocabulary/ORBvoc.txt ${CMAKE_CURRENT_SOURCE_DIR}/config/
    DESTINATION ${PROJECT_SOURCE_DIR}/bin/config)


# Deep Q-Learning.
find_library(CAFFE2_LIB caffe2)
find_library(CAFFE2_GPU_LIB caffe2_gpu)
find_package(Protobuf REQUIRED)

if(NOT CAFFE2_LIB)
  message(FATAL_ERROR "Caffe2 lib not found")
endif()

add_library(dqn SHARED dqn.cpp)
target_link_libraries(dqn ${CAFFE2_LIB} ${PROTOBUF_LIBRARY})

# Test DQN.
add_executable(test_dqn test_dqn.cpp)
target_link_libraries(test_dqn dqn)

