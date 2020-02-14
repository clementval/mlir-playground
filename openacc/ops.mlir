// RUN: mlir-opt %s | mlir-opt | FileCheck %s

func @compute(%x: memref<10x10x10xf32>, %y: memref<10x10x10xf32>,
  %n: index) -> memref<10x10x10xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // y[i] = a*x[i] + y[i];
  acc.parallel num_gangs(8) num_workers(128) {
    acc.loop gang vector {
      loop.for %arg0 = %c0 to %n step %c1 {
        loop.for %arg1 = %c0 to %n step %c1 {
          loop.for %arg2 = %c0 to %n step %c1 {
            %xi = load %x[%arg0, %arg1, %arg2] : memref<10x10x10xf32>
            %yi = load %y[%arg0, %arg1, %arg2] : memref<10x10x10xf32>
            %yy = mulf %xi, %yi : f32
            store %yy, %y[%arg0, %arg1, %arg2] : memref<10x10x10xf32>
          }
        }
      }
    } attributes { collapse = 3 }
  }
  return %y : memref<10x10x10xf32>
}


// CHECK:      acc.parallel num_gangs(8 : i64) num_workers(128 : i64) {
// CHECK-NEXT:   acc.loop gang vector {