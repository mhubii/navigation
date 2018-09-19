# Find the ORB SLAM2 library.
find_path(ORB_SLAM2_INCLUDE_DIR System.h ${PROJECT_SOURCE_DIR}/third_party/ORB_SLAM2/include)

find_library(ORB_SLAM2_LIBRARY
    NAMES ORB_SLAM2
    PATHS ${PROJECT_SOURCE_DIR}/third_party/ORB_SLAM2/lib
) 

if (ORB_SLAM2_INCLUDE_DIR AND ORB_SLAM2_LIBRARY)
   set(ORB_SLAM2_FOUND TRUE)
endif (ORB_SLAM2_INCLUDE_DIR AND ORB_SLAM2_LIBRARY)

if (ORB_SLAM2_FOUND)
  if (NOT ORB_SLAM2_FIND_QUIETLY)
    message(STATUS "Found ORB_SLAM2: ${ORB_SLAM2_LIBRARY}")
  endif (NOT ORB_SLAM2_FIND_QUIETLY)
else (ORB_SLAM2_FOUND)
  if (ORB_SLAM2_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find ORB SLAM2.")
  endif (ORB_SLAM2_FIND_REQUIRED)
endif (ORB_SLAM2_FOUND)
