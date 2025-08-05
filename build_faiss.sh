#!/bin/bash

set -e  # Exit on any error

cd build

# Run CMake
cmake \
  -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_GPU=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  ..

# Build with make
make -j"$(nproc)"

cd faiss/python
python setup.py build install


