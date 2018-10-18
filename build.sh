#!/bin/bash

# Go to location of bash script.
DIR=`dirname $(readlink -f \$0)`
cd ${DIR}

# Build ORB SLAM2.
cd third_party/ORB_SLAM2
sh build.sh
cd ${DIR}

# Download libtorch.
if ! [ -d "third_party/libtorch" ]; then

	cd third_party
	wget https://download.pytorch.org/libtorch/nightly/cu90/libtorch-shared-with-deps-latest.zip
	unzip libtorch-shared-with-deps-latest.zip
	rm libtorch-shared-with-deps-latest.zip
	cd ..
fi

export CMAKE_PREFIX_PATH="${DIR}/third_party/libtorch"

# Build the project.
if ! [ -d "build" ]; then

	mkdir build
fi

cd build
cmake ..
make -j8



