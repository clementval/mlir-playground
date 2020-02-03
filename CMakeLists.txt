cmake_minimum_required(VERSION 3.10)

project(mlir-playground LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)
find_package(CUDA 10.0 REQUIRED)

if(NOT DEFINED LLVM_BUILD_DIR)
  message(FATAL_ERROR "LLVM_BUILD_DIR need to be specified")
endif()

if(NOT DEFINED LIT_PATH)
  message(FATAL_ERROR "LIT_PATH need to be defined to lit.py file path")
endif()

message(${LLVM_BUILD_DIR})
message(${LIT_PATH})



find_file(MLIR_LIT_SITE_CONFIG lit.site.cfg.py PATHS "${LLVM_BUILD_DIR}/tools/mlir/test")
if(NOT DEFINED MLIR_LIT_SITE_CONFIG)
  message(FATAL_ERROR "Cannot find lit site config file for MLIR")
endif()

add_subdirectory(oaru_light)
add_subdirectory(openacc)
