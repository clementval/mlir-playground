cmake_minimum_required(VERSION 3.10)

project(smalllib C)

enable_language(CUDA)
find_package(CUDA 10.0 REQUIRED)


add_library(smalllib SHARED smalllib.cu)
