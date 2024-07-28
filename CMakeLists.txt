cmake_minimum_required(VERSION 3.25)
project(gBLAS
        VERSION 0.1.0
        DESCRIPTION "A BLAS implementation in C++ dedicated for AI"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)

add_library(gBLAS SHARED src/gTensor/DataBuffer.cpp
                         src/gTensor/DataBuffer.h
                         src/gTensor/gTensor.cpp
                         src/gTensor/gTensor.h)

add_subdirectory(tests)
