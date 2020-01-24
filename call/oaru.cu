#include <cuda.h>
#include <stdio.h>

void print_cuda_error(CUresult cuResult) {
  switch (cuResult) {
  case CUDA_SUCCESS:
    printf("CUDA_SUCCESS\n");
    break;
  case CUDA_ERROR_INVALID_VALUE:
    printf("CUDA_ERROR_INVALID_VALUE\n");
    break;
  case CUDA_ERROR_OUT_OF_MEMORY:
    printf("CUDA_ERROR_OUT_OF_MEMORY\n");
    break;
  case CUDA_ERROR_NOT_INITIALIZED:
    printf("CUDA_ERROR_NOT_INITIALIZED\n");
    break;
  case CUDA_ERROR_DEINITIALIZED:
    printf("CUDA_ERROR_DEINITIALIZED\n");
    break;
  case CUDA_ERROR_NO_DEVICE:
    printf("CUDA_ERROR_NO_DEVICE\n");
    break;
  case CUDA_ERROR_INVALID_DEVICE:
    printf("CUDA_ERROR_INVALID_DEVICE\n");
    break;
  case CUDA_ERROR_INVALID_IMAGE:
    printf("CUDA_ERROR_INVALID_IMAGE\n");
    break;
  case CUDA_ERROR_INVALID_CONTEXT:
    printf("CUDA_ERROR_INVALID_CONTEXT\n");
    break;
  case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
    printf("CUDA_ERROR_CONTEXT_ALREADY_CURRENT\n");
    break;
  case CUDA_ERROR_MAP_FAILED:
    printf("CUDA_ERROR_MAP_FAILED\n");
    break;
  case CUDA_ERROR_UNMAP_FAILED:
    printf("CUDA_ERROR_UNMAP_FAILED\n");
    break;
  case CUDA_ERROR_ARRAY_IS_MAPPED:
    printf("CUDA_ERROR_ARRAY_IS_MAPPED\n");
    break;
  case CUDA_ERROR_ALREADY_MAPPED:
    printf("CUDA_ERROR_ALREADY_MAPPED\n");
    break;
  case CUDA_ERROR_NO_BINARY_FOR_GPU:
    printf("CUDA_ERROR_NO_BINARY_FOR_GPU\n");
    break;
  case CUDA_ERROR_ALREADY_ACQUIRED:
    printf("CUDA_ERROR_ALREADY_ACQUIRED\n");
    break;
  case CUDA_ERROR_NOT_MAPPED:
    printf("CUDA_ERROR_NOT_MAPPED\n");
    break;
  case CUDA_ERROR_INVALID_SOURCE:
    printf("CUDA_ERROR_INVALID_SOURCE\n");
    break;
  case CUDA_ERROR_FILE_NOT_FOUND:
    printf("CUDA_ERROR_FILE_NOT_FOUND\n");
    break;
  case CUDA_ERROR_INVALID_HANDLE:
    printf("CUDA_ERROR_INVALID_HANDLE\n");
    break;
  case CUDA_ERROR_NOT_FOUND:
    printf("CUDA_ERROR_NOT_FOUND\n");
    break;
  case CUDA_ERROR_NOT_READY:
    printf("CUDA_ERROR_NOT_READY\n");
    break;
  case CUDA_ERROR_LAUNCH_FAILED:
    printf("CUDA_ERROR_LAUNCH_FAILED\n");
    break;
  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
    printf("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES\n");
    break;
  case CUDA_ERROR_LAUNCH_TIMEOUT:
    printf("CUDA_ERROR_LAUNCH_TIMEOUT\n");
    break;
  case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
    printf("CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING\n");
    break;
  case CUDA_ERROR_UNKNOWN:
    printf("CUDA_ERROR_UNKNOWN\n");
    break;
  default:
    printf("Unknown error code\n");
    break;
  }
}

extern "C" int oaru_get_num_devices() {
  int deviceCount = 0;
  CUresult cuResult = cuDeviceGetCount(&deviceCount);
  if (cuResult != CUDA_SUCCESS) {
    print_cuda_error(cuResult);
    return 0;
  }
  return deviceCount;
}

extern "C" void oaru_init() {
  CUresult cuResult = cuInit(0);
  if (cuResult != CUDA_SUCCESS) {
    print_cuda_error(cuResult);
  }
}

extern "C" void oaru_print_i32(int val) {
  printf("%d\n", val);
}

