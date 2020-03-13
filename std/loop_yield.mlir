// RUN: mlir-opt --convert-loop-to-std --convert-std-to-llvm %s | mlir-cpu-runner --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void 

func @reduce(%buffer: memref<5xf32>, %lb: index, %ub: index, %step: index) -> (f32) {
  // Initial sum set to 0.
  %sum_0 = constant 0.0 : f32
  // iter_args binds initial values to the loop's region arguments.
  %sum = loop.for %iv = %lb to %ub step %step iter_args(%sum_iter = %sum_0) -> (f32) {
    %t = load %buffer[%iv] : memref<5xf32>
    %sum_next = addf %sum_iter, %t : f32
    loop.yield %sum_next : f32
  }
  return %sum : f32
}

func @main() {
  %data = alloc() : memref<5xf32>
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
    
  loop.for %i = %c0 to %n step %c1 {
    store %cst0, %data[%c0] : memref<5xf32>
    store %cst1, %data[%c1] : memref<5xf32>
    store %cst2, %data[%c2] : memref<5xf32>
    store %cst3, %data[%c3] : memref<5xf32>
    store %cst4, %data[%c4] : memref<5xf32>
  }

  %sum_rtn = call @reduce(%data, %c0, %n, %c1) : (memref<5xf32>, index, index, index) -> (f32) 

  store %sum_rtn, %sum[%c0] : memref<1xf32>

  %ptr = memref_cast %sum : memref<1xf32> to memref<*xf32>
  call @print_memref_f32(%ptr) : (memref<*xf32>) -> ()
  return
}

func @print_memref_f32(memref<*xf32>)
