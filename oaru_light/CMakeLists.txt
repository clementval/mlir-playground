cmake_minimum_required(VERSION 3.10)

project(oaru LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)
find_package(CUDA 10.0 REQUIRED)

add_library(oaru SHARED oaru.cu)

install(TARGETS oaru DESTINATION bin)
