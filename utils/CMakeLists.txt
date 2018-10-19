# Keyboard input user navigation.
add_library(keyboard SHARED keyboard.cpp)

# Q-Learning.
add_library(q_learning SHARED q_learning.cpp)
target_link_libraries(q_learning ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

# Example for Deep Q-Learning.
add_executable(example_dqn example_dqn.cpp)
target_link_libraries(example_dqn ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})
