# Keyboard input user navigation.
add_library(keyboard SHARED keyboard.cpp)

# Deep Q-Learning.
find_package(Torch REQUIRED)

#set(SOURCE)

#add_library(dqn SHARED ${SOURCE})
#target_link_libraries(dqn ${TORCH_LIBRARIES})

# Example for Deep Q-Learning.
add_executable(example_dqn example_dqn.cpp)
target_link_libraries(example_dqn ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})
