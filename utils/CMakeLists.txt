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

# Deep Q-Learning.
find_package(Torch REQUIRED)

#set(SOURCE)

#add_library(dqn SHARED ${SOURCE})
#target_link_libraries(dqn ${TORCH_LIBRARIES})

# Example for Deep Q-Learning.
add_executable(example_dqn example_dqn.cpp)
target_link_libraries(example_dqn ${TORCH_LIBRARIES})
