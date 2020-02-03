// RUN: mlir-opt --canonicalize --convert-linalg-to-loops --convert-loop-to-std --gpu-kernel-outlining %s | mlir-cuda-runner --shared-libs=%cuda_wrapper_library_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libcuda-runtime-wrappers%shlibext,%oaru_library_dir/liboaru%shlibext --entry-point-result=void | FileCheck %s

// Simple code to call external function of various kind

func @main() {
  %ci1 = constant 1 : index
  %cf1 = constant 1.0 : f32
  %c11 = constant 11.0 : f32
  %nblock = constant 2 : index
  %nthread = constant 5 : index
  %n = constant 10 : index
  call @oaru_init() : () -> ()
  %0 = call @oaru_get_num_devices() : () -> i32
  call @oaru_print_i32(%0) : (i32) -> ()

  // Allocate on host
  %host_A = alloc() : memref<10xf32>
  // Fill memref with initial data
  linalg.fill(%host_A, %cf1) : memref<10xf32>, f32

  call @print_memref_1d_f32(%host_A) : (memref<10xf32>) -> ()
  // CHECK: Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [10] strides = [1] data = 
  // CHECK-NEXT: [1,  1,  1,  1,  1,  1,  1,  1,  1,  1]

  // Allocate memory on the device
  %device_A = call @oaru_allocate_memref_1d_float(%host_A) : (memref<10xf32>) -> memref<10xf32,5>

  // Update device memory with host memory
  call @oaru_update_device_1d_float(%host_A, %device_A) : (memref<10xf32>, memref<10xf32, 5>) -> ()

  gpu.launch
    blocks(%bx, %by, %bz) in (%nbx = %nblock, %nby = %ci1, %nbz = %ci1)
    threads(%tx, %ty, %tz) in (%ntx = %nthread, %nty = %ci1, %ntz = %ci1) {

    // i = blockIdx.x * blockDim.x + threadIdx.x
    %tidx = "gpu.thread_id"() {dimension = "x"} : () -> (index)
    %bidx = "gpu.block_id"() {dimension = "x"} : () -> (index)
    %bdimx = "gpu.block_dim"() {dimension = "x"} : () -> (index)
    %blockPos = muli %bidx, %bdimx : index
    %idx = addi %blockPos, %tidx : index

    // if (i < n)
    %inside = cmpi "slt", %idx, %n : index
    cond_br %inside, ^bb1, ^bb2

    ^bb1:
      %x = load %device_A[%idx] : memref<10xf32, 5>
      %xi = addf %x, %c11 : f32
      store %xi, %device_A[%idx] : memref<10xf32, 5>
      gpu.terminator
    ^bb2:
      gpu.terminator
  }

  // Update host memory 
  call @oaru_update_host_1d_float(%host_A, %device_A) : (memref<10xf32>, memref<10xf32, 5>) -> ()

  // CHECK: Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [10] strides = [1] data = 
  // CHECK-NEXT: [12,  12,  12,  12,  12,  12,  12,  12,  12,  12]
  call @print_memref_1d_f32(%host_A) : (memref<10xf32>) -> ()
  call @oaru_free_memref_1d_float(%device_A) : (memref<10xf32, 5>) -> ()
  return
}

func @oaru_init()
func @oaru_get_num_devices() -> i32
func @oaru_allocate_memref_1d_float(memref<10xf32>) -> memref<10xf32,5>
func @oaru_update_device_1d_float(memref<10xf32>, memref<10xf32,5>)
func @oaru_update_host_1d_float(memref<10xf32>, memref<10xf32,5>)
func @oaru_free_memref_1d_float(memref<10xf32, 5>)
func @oaru_print_i32(%val: i32) -> ()

func @print_memref_1d_f32(memref<10xf32>)
