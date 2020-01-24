#include <cuda.h>
#include <stdio.h>

extern "C"int oaru_get_num_devices() {
  int deviceCount = 0;
  CUresult cuResult = cuDeviceGetCount(&deviceCount);
  if (cuResult != CUDA_SUCCESS) {
    printf("[ERROR] Cannot read number of devices\n");
  }
  return deviceCount;
}

