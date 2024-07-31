cmake_minimum_required(VERSION 3.25)
project(gBLAS
        VERSION 0.1.0
        DESCRIPTION "A BLAS implementation in C++ dedicated for AI"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(src_files ${CMAKE_SOURCE_DIR}/src/gTensor/DataBuffer.cpp
              ${CMAKE_SOURCE_DIR}/src/gTensor/gTensor.cpp
              ${CMAKE_SOURCE_DIR}/src/operations/axpy.cpp src/gTensor/gTensorIterator.cpp src/gTensor/gTensorIterator.h)
add_library(gBLAS SHARED ${src_files})
target_include_directories(gBLAS PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(tests)
