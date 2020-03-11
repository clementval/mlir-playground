// RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module attributes {gpu.container_module} {
  func @compute(%arg0: memref<5x5xf32>, %arg1: memref<5xf32>, %arg2: index) -> memref<5xf32> {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c8 = constant 8 : index
    %c5 = constant 5 : index
    "gpu.launch_func"(%c5, %c1, %c1, %c5, %c1, %c1, %arg0, %arg1, %arg2) {kernel = "compute_acc_parallel", kernel_module = @compute_acc_parallel} : (index, index, index, index, index, index, memref<5x5xf32>, memref<5xf32>, index) -> ()
    return %arg1 : memref<5xf32>
  }
  gpu.module @compute_acc_parallel {
    gpu.func @compute_acc_parallel(%arg0: memref<5x5xf32>, %arg1: memref<5xf32>, %arg2: index) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.grid_dim"() {dimension = "x"} : () -> index
      %3 = "gpu.block_dim"() {dimension = "x"} : () -> index
      %c0 = constant 0 : index

      %val = load %arg0[%0, %1] : memref<5x5xf32>

      %sum = "gpu.all_reduce"(%val) ({}) { op = "add" } : (f32) -> (f32)
      store %sum, %arg1[%0] : memref<5xf32>
      gpu.return
    }
  }
  func @main() {
    %0 = alloc() : memref<5x5xf32>
    %sum = alloc() : memref<5xf32>
    %cst0 = constant 0.0 : f32
    %cst1 = constant 1.0 : f32
    %cst2 = constant 2.0 : f32
    %cst3 = constant 3.0 : f32
    %cst4 = constant 4.0 : f32

    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c4 = constant 4 : index
    %n = constant 5 : index
    
    loop.for %i = %c0 to %n step %c1 {
      store %cst0, %0[%i, %c0] : memref<5x5xf32>
      store %cst1, %0[%i, %c1] : memref<5x5xf32>
      store %cst2, %0[%i, %c2] : memref<5x5xf32>
      store %cst3, %0[%i, %c3] : memref<5x5xf32>
      store %cst4, %0[%i, %c4] : memref<5x5xf32>
    }

    %2 = call @compute(%0, %sum, %n) : (memref<5x5xf32>, memref<5xf32>, index) -> memref<5xf32>
    %ptr = memref_cast %2 : memref<5xf32> to memref<*xf32>
    call @print_memref_f32(%ptr) : (memref<*xf32>) -> ()
    return
  }
  func @print_memref_f32(memref<*xf32>)
}
