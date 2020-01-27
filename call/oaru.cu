/*
 * Set of basic function call from MLIR just for testing purpose
 */

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
  CUdevice device_;
  cuResult = cuDeviceGet(&device_, 0);
  if (cuResult != CUDA_SUCCESS) {
    print_cuda_error(cuResult);
    exit(1);
  }

  CUcontext context_;
  cuResult = cuCtxCreate(&context_, 0, device_);
  if (cuResult != CUDA_SUCCESS) {
    print_cuda_error(cuResult);
    cuCtxDestroy(context_);
    exit(1);
  }
}

template <typename T, int N> struct MemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" void* oaru_allocate(void* hostPtr, size_t size) {
  void *devPtr;
  CUresult cuResult = CUDA_SUCCESS;
  CUdeviceptr cuPtr;
  cuResult = cuMemAlloc(&cuPtr, size);
  if (cuResult == CUDA_SUCCESS) {
    devPtr = (void *)(uintptr_t)cuPtr;
    return devPtr;
  } else {
    print_cuda_error(cuResult);
    exit(1);
  }
  return NULL;
}

extern "C" void oaru_print_i32(int val) {
  printf("%d\n", val);
}

template<typename T, int N>
MemRefType<T, N> oaru_allocate_memref(const MemRefType<T, N> *arg) {
  T* devicePtr = (T*)oaru_allocate(arg->basePtr, arg->sizes[0] * sizeof(T));
  struct MemRefType<T, N> allocated;  
  allocated.basePtr = devicePtr;
  allocated.data = devicePtr;
  allocated.offset = arg->offset;
  allocated.sizes[0] = arg->sizes[0];
  allocated.strides[0] = arg->strides[0];
  return allocated;
}

extern "C" MemRefType<float, 1>
oaru_allocate_memref_1d_float(const MemRefType<float, 1> *arg) {
  return oaru_allocate_memref(arg);
}

template<typename T, int N> 
void oaru_free(const MemRefType<T, N> *arg) {
  CUresult cuResult = CUDA_SUCCESS;
  CUdeviceptr dptr;
  dptr = (CUdeviceptr) (uintptr_t) arg->basePtr;
  cuResult = cuMemFree(dptr);
  if (cuResult != CUDA_SUCCESS) {
    print_cuda_error(cuResult);
  }
}

extern "C" void
oaru_free_memref_1d_float(const MemRefType<float, 1> *arg) {
  oaru_free(arg);
}

template<typename T, int N1, int N2>
void oaru_update_device(const MemRefType<T, N1> *host, 
    const MemRefType<T, N2> *device) 
{
  CUresult cuResult = CUDA_SUCCESS;
  CUdeviceptr dptr;
  dptr = (CUdeviceptr) (uintptr_t) device->basePtr;
  cuResult = cuMemcpyHtoD(dptr, host->basePtr, count(host) * sizeof(T));
  if (cuResult != CUDA_SUCCESS) {
    print_cuda_error(cuResult);
    exit(1);
  }
}

extern "C" void
oaru_update_device_1d_float(const MemRefType<float, 1> *host, 
    const MemRefType<float, 1> *device) 
{
  oaru_update_device(host, device);
}

template<typename T, int N1, int N2>
void oaru_update_host(const MemRefType<T, N1> *host, 
    const MemRefType<T, N2> *device) 
{
  CUresult cuResult = CUDA_SUCCESS;
  CUdeviceptr dptr;
  dptr = (CUdeviceptr) (uintptr_t) device->basePtr;
  cuResult = cuMemcpyDtoH(host->basePtr, dptr, count(host) * sizeof(T));
  if (cuResult != CUDA_SUCCESS) {
    print_cuda_error(cuResult);
    exit(1);
  }
}
template<typename T, int N> 
int64_t count(const MemRefType<T, N> *arg) {
  int count = arg->sizes[0];
  for(int i = 1; i < N; ++i) {
    count *= arg->sizes[i];
  }
  return count;
}

extern "C" void
oaru_update_host_1d_float(const MemRefType<float, 1> *host, 
    const MemRefType<float, 1> *device) 
{
  oaru_update_host(host, device);
}