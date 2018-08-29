# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/martin/Documents/path_finding

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/martin/Documents/path_finding/build

# Include any dependencies generated for this target.
include utils/CMakeFiles/keyboard.dir/depend.make

# Include the progress variables for this target.
include utils/CMakeFiles/keyboard.dir/progress.make

# Include the compile flags for this target's objects.
include utils/CMakeFiles/keyboard.dir/flags.make

utils/CMakeFiles/keyboard.dir/keyboard.cpp.o: utils/CMakeFiles/keyboard.dir/flags.make
utils/CMakeFiles/keyboard.dir/keyboard.cpp.o: ../utils/keyboard.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/martin/Documents/path_finding/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object utils/CMakeFiles/keyboard.dir/keyboard.cpp.o"
	cd /home/martin/Documents/path_finding/build/utils && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/keyboard.dir/keyboard.cpp.o -c /home/martin/Documents/path_finding/utils/keyboard.cpp

utils/CMakeFiles/keyboard.dir/keyboard.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/keyboard.dir/keyboard.cpp.i"
	cd /home/martin/Documents/path_finding/build/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/martin/Documents/path_finding/utils/keyboard.cpp > CMakeFiles/keyboard.dir/keyboard.cpp.i

utils/CMakeFiles/keyboard.dir/keyboard.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/keyboard.dir/keyboard.cpp.s"
	cd /home/martin/Documents/path_finding/build/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/martin/Documents/path_finding/utils/keyboard.cpp -o CMakeFiles/keyboard.dir/keyboard.cpp.s

# Object files for target keyboard
keyboard_OBJECTS = \
"CMakeFiles/keyboard.dir/keyboard.cpp.o"

# External object files for target keyboard
keyboard_EXTERNAL_OBJECTS =

utils/libkeyboard.so: utils/CMakeFiles/keyboard.dir/keyboard.cpp.o
utils/libkeyboard.so: utils/CMakeFiles/keyboard.dir/build.make
utils/libkeyboard.so: utils/CMakeFiles/keyboard.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/martin/Documents/path_finding/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libkeyboard.so"
	cd /home/martin/Documents/path_finding/build/utils && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/keyboard.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
utils/CMakeFiles/keyboard.dir/build: utils/libkeyboard.so

.PHONY : utils/CMakeFiles/keyboard.dir/build

utils/CMakeFiles/keyboard.dir/clean:
	cd /home/martin/Documents/path_finding/build/utils && $(CMAKE_COMMAND) -P CMakeFiles/keyboard.dir/cmake_clean.cmake
.PHONY : utils/CMakeFiles/keyboard.dir/clean

utils/CMakeFiles/keyboard.dir/depend:
	cd /home/martin/Documents/path_finding/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/martin/Documents/path_finding /home/martin/Documents/path_finding/utils /home/martin/Documents/path_finding/build /home/martin/Documents/path_finding/build/utils /home/martin/Documents/path_finding/build/utils/CMakeFiles/keyboard.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : utils/CMakeFiles/keyboard.dir/depend

