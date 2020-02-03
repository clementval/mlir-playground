// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s

func @compute(%A: memref<10xf32>, %B: memref<10xf32>) -> memref<10xf32> {
  %c0 = constant 0 : index

  acc.parallel {
    %tmp = load %A[%c0] : memref<10xf32> 
    store %tmp, %B[%c0] : memref<10xf32>
  }

  return %B : memref<10xf32>
}

// CHECK:       func @compute(%{{.*}}: memref<10xf32>, %{{.*}}: memref<10xf32>) -> memref<10xf32> {
//  CHECK-NEXT:   %{{.*}}= constant 0 : index
//  CHECK-NEXT:   %{{.*}}= constant 1 : index
//  CHECK-NEXT:   gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) {
//  CHECK-NEXT:      %{{.*}} = load %{{.*}}[%{{.*}}] : memref<10xf32>
//  CHECK-NEXT:      store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
//  CHECK-NEXT:      gpu.terminator
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %arg1 : memref<10xf32>
//  CHECK-NEXT: }


func @main() {
  %x = alloc() : memref<10xf32>
  %y = alloc() : memref<10xf32>

  %c1 = constant 1.0 : f32
  %c2 = constant 2.0 : f32

  linalg.fill(%x, %c1) : memref<10xf32>, f32
  linalg.fill(%y, %c2) : memref<10xf32>, f32

  %z = call @compute(%x, %y) : (memref<10xf32>, memref<10xf32>) -> memref<10xf32>
  call @print_memref_1d_f32(%z): (memref<10xf32>) -> ()
  return
}

func @print_memref_1d_f32(memref<10xf32>)