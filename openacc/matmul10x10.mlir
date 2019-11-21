// RUN: ../../llvm-project/build/bin/mlir-opt --canonicalize --convert-openacc-to-gpu -linalg-lower-to-loops -convert-linalg-to-llvm -convert-std-to-llvm %s | FileCheck %s

func @compute(%A: memref<10x10xf32>, %B: memref<10x10xf32>,
  %C: memref<10x10xf32>) -> memref<10x10xf32> {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index

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

  return %C : memref<10x10xf32>
}

func @main() {
  %A = alloc() : memref<10x10xf32>
  %B = alloc() : memref<10x10xf32>
  %C = alloc() : memref<10x10xf32>

  %cf0 = constant 0.00000e+00 : f32
  %cf1 = constant 1.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<10x10xf32>, f32
  linalg.fill(%B, %cf1) : memref<10x10xf32>, f32
  linalg.fill(%C, %cf0) : memref<10x10xf32>, f32

  call @compute(%A, %B, %C) : (memref<10x10xf32>, memref<10x10xf32>, memref<10x10xf32>) -> memref<10x10xf32>
  call @print_memref_2d_f32(%C): (memref<10x10xf32>) -> ()
  return
}


func @print_memref_2d_f32(memref<10x10xf32>)
