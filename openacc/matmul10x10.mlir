// Check sequential CPU
// ../../llvm-project/build/bin/mlir-opt --canonicalize --convert-openacc-to-seq -convert-std-to-llvm matmul10x10.mlir | ../../llvm-project/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=../../llvm-project/build/lib/libmlir_runner_utils.dylib
// Check GPU
// ../../llvm-project/build/bin/mlir-opt --canonicalize --convert-openacc-to-gpu -gpu-kernel-outlining --linalg-lower-to-loops --convert-loop-to-std matmul10x10.mlir | ../../llvm-project/build/bin/mlir-cuda-runner --shared-libs=../../llvm-project/build/lib/libcuda-runtime-wrappers.so,./lib/libmlir_runner_utils.so --entry-point-result=void
func @main() {

  %A = alloc() : memref<10x10xf32>
  %B = alloc() : memref<10x10xf32>
  %C = alloc() : memref<10x10xf32>

  %cf0 = constant 0.00000e+00 : f32
  %cf1 = constant 1.00000e+00 : f32

  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index
  loop.for %arg0 = %c0 to %c10 step %c1 {
    loop.for %arg1 = %c0 to %c10 step %c1 {
      store %cf1, %A[%arg0, %arg1] : memref<10x10xf32>
      store %cf1, %B[%arg0, %arg1] : memref<10x10xf32>
      store %cf0, %C[%arg0, %arg1] : memref<10x10xf32>
    }
  }
  acc.parallel {
    acc.loop {
      loop.for %arg3 = %c0 to %c10 step %c1 {
        loop.for %arg4 = %c0 to %c10 step %c1 {
          loop.for %arg5 = %c0 to %c10 step %c1 {
            %a = load %A[%arg3, %arg5] : memref<10x10xf32>
            %b = load %B[%arg5, %arg4] : memref<10x10xf32>
            %cij = load %C[%arg3, %arg4] : memref<10x10xf32>
            %p = mulf %a, %b : f32
            %co = addf %cij, %p : f32
            store %co, %C[%arg3, %arg4] : memref<10x10xf32>
          }
        }
      }
    } attributes { collapse = 3 }
    acc.loop {}
  } attributes { async = 1 }

  call @print_memref_2d_f32(%C): (memref<10x10xf32>) -> ()
  return
}

func @print_memref_2d_f32(memref<10x10xf32>)
