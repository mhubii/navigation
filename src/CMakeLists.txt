find_package(YARP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(ORB_SLAM2 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Protobuf REQUIRED)

find_library(CAFFE2_LIB caffe2)

if(NOT CAFFE2_LIB)
  message(FATAL_ERROR "Caffe2 lib not found")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/third_party/ORB_SLAM2 
    ${PROJECT_SOURCE_DIR}/third_party/ORB_SLAM2/include 
    ${PCL_INCLUDE_DIRS}
    ${CAFFE2_INCLUDE_DIRS}
)

add_executable(a3c ${CMAKE_CURRENT_SOURCE_DIR}/a3c.cpp)
target_link_libraries(a3c ${CAFFE2_LIB} ${PROTOBUF_LIBRARY})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_executable(orb_slam stereo_slam.cpp)
target_link_libraries(orb_slam ${ORB_SLAM2_LIBRARY} ${Pangolin_LIBRARIES} ${YARP_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

file(COPY ${PROJECT_SOURCE_DIR}/third_party/ORB_SLAM2/Vocabulary/ORBvoc.txt ${CMAKE_CURRENT_SOURCE_DIR}/config/
    DESTINATION ${PROJECT_SOURCE_DIR}/bin/config)

# Reinforcement learning agent.
#find_package(CUDA REQUIRED)
find_package(PythonLibs REQUIRED)

add_library(dqn_agent SHARED rl_agent.cpp dqn_agent.cpp)
target_link_libraries(dqn_agent)
