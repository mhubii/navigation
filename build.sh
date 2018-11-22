#!/bin/bash

# Go to location of bash script.
DIR=`dirname $(readlink -f \$0)`
cd ${DIR}

# Download libtorch.
if ! [ -d "third_party/libtorch" ]; then

	mkdir third_party
	cd third_party
	wget https://download.pytorch.org/libtorch/nightly/cu92/libtorch-shared-with-deps-latest.zip
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



