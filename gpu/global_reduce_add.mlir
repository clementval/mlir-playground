// RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @compute_acc_parallel {
    gpu.func @compute_acc_parallel(%arg0: memref<5x5xf32>, %arg1: memref<5xf32>, %arg2: index) kernel {
      %bidx = "gpu.block_id"() {dimension = "x"} : () -> index
      %tidx = "gpu.thread_id"() {dimension = "x"} : () -> index

      %c0 = constant 0 : index

      %val = load %arg0[%bidx, %tidx] : memref<5x5xf32>

      %sum = "gpu.all_reduce"(%val) ({}) { op = "add" } : (f32) -> (f32)

      %istid0 = cmpi "eq", %tidx, %c0 : index
      cond_br %istid0, ^tidxblock, ^continue
    ^tidxblock :
        store %sum, %arg1[%bidx] : memref<5xf32>
        br ^continue
    ^continue :
      gpu.return
    }
  }
  func @main() {
    %data = alloc() : memref<5x5xf32>
    %sum = alloc() : memref<1xf32>
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
    %c8 = constant 8 : index
    %c5 = constant 5 : index
    
    loop.for %i = %c0 to %n step %c1 {
      store %cst0, %data[%i, %c0] : memref<5x5xf32>
      store %cst1, %data[%i, %c1] : memref<5x5xf32>
      store %cst2, %data[%i, %c2] : memref<5x5xf32>
      store %cst3, %data[%i, %c3] : memref<5x5xf32>
      store %cst4, %data[%i, %c4] : memref<5x5xf32>
    }

    // Allocate a global memory buffer to store workgroup 
    %sum_reduce_buffer = alloc() : memref<5xf32>
    // Init reduction value
    store %cst0, %sum[%c0] : memref<1xf32>
    // Call the kernel
    "gpu.launch_func"(%c5, %c1, %c1, %c5, %c1, %c1, %data, %sum_reduce_buffer, %n) {kernel = "compute_acc_parallel", kernel_module = @compute_acc_parallel} : (index, index, index, index, index, index, memref<5x5xf32>, memref<5xf32>, index) -> ()
    // Sum the reduction_buffer
    loop.for %i = %c0 to %c5 step %c1 {
      %tmp = load %sum_reduce_buffer[%i] : memref<5xf32>
      %crt_sum = load %sum[%c0] : memref<1xf32>
      %tmp_sum = addf %tmp, %crt_sum : f32
      store %tmp_sum, %sum[%c0] : memref<1xf32>
    }

    %ptr = memref_cast %sum : memref<1xf32> to memref<*xf32>
    call @print_memref_f32(%ptr) : (memref<*xf32>) -> ()
    return
  }
  func @print_memref_f32(memref<*xf32>)
}
