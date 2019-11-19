

func @compute(%A: memref<10x10xf32>, %B: memref<10x10xf32>,
  %C: memref<10x10xf32>) -> memref<10x10xf32> {

  affine.for %arg3 = 0 to 10 {
    affine.for %arg4 = 0 to 10 {
      affine.for %arg5 = 0 to 10 {
        %a = affine.load %A[%arg3, %arg5] : memref<10x10xf32>
        %b = affine.load %B[%arg5, %arg4] : memref<10x10xf32>
        %cij = affine.load %C[%arg3, %arg4] : memref<10x10xf32>
        %p = mulf %a, %b : f32
        %co = addf %cij, %p : f32
        affine.store %co, %C[%arg3, %arg4] : memref<10x10xf32>
      }
    }
  }

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
