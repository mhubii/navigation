find_package(gazebo REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(gazeboVehiclePlugin SHARED ${CMAKE_CURRENT_SOURCE_DIR}/plugins/vehicle_plugin.cpp)
target_link_libraries(gazeboVehiclePlugin ${GAZEBO_LIBRARIES} keyboard q_learning)

