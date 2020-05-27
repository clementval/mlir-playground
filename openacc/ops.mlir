// RUN: mlir-opt --canonicalize %s | mlir-opt | FileCheck %s

func @compute(%x: memref<10x10x10xf32>, %y: memref<10x10x10xf32>,
  %n: index) -> memref<10x10x10xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %gangs = constant 8 : index
  %workers = constant 128 : index

  // y[i] = a*x[i] + y[i];
  // CHECK:      acc.parallel num_gangs(%{{.*}}) num_workers(%{{.*}}) {
  // CHECK-NEXT:   acc.loop gang vector {
  acc.parallel num_gangs(%gangs) num_workers(%workers) {
    acc.loop gang vector {
      scf.for %arg0 = %c0 to %n step %c1 {
        scf.for %arg1 = %c0 to %n step %c1 {
          scf.for %arg2 = %c0 to %n step %c1 {
            %xi = load %x[%arg0, %arg1, %arg2] : memref<10x10x10xf32>
            %yi = load %y[%arg0, %arg1, %arg2] : memref<10x10x10xf32>
            %yy = mulf %xi, %yi : f32
            store %yy, %y[%arg0, %arg1, %arg2] : memref<10x10x10xf32>
          }
        }
      }
  // CHECK: } attributes {collapse = 3 : i64}
  // CHECK-NEXT: }
    } attributes { collapse = 3 }
  }

  // CHECK:      acc.parallel {
  // CHECK-NEXT:   acc.loop seq {
  acc.parallel {
    acc.loop {}
    acc.loop seq {
      scf.for %arg2 = %c0 to %n step %c1 {
        %xi = load %x[%c0, %c1, %arg2] : memref<10x10x10xf32>
        %yi = load %y[%c0, %c1, %arg2] : memref<10x10x10xf32>
        %yy = mulf %xi, %yi : f32
        store %yy, %y[%c0, %c1, %arg2] : memref<10x10x10xf32>
      }
    }
  }

  // CHECK:      acc.parallel {
  // CHECK-NEXT:   acc.gang_redundant {
  // CHECK-NEXT:   }
  // CHECK-NEXT:   acc.loop {
  // CHECK-NEXT:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {  
  // CHECK-NEXT:       %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<10x10x10xf32>
  // CHECK-NEXT:       %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<10x10x10xf32>
  // CHECK-NEXT:       %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:       store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<10x10x10xf32>
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT:   acc.gang_redundant {
  // CHECK-NEXT:   } 
  // CHECK-NEXT: }
  acc.parallel {
    acc.gang_redundant {
    }
    acc.loop {
      scf.for %arg2 = %c0 to %n step %c1 {
        %xi = load %x[%c0, %c1, %arg2] : memref<10x10x10xf32>
        %yi = load %y[%c0, %c1, %arg2] : memref<10x10x10xf32>
        %yy = mulf %xi, %yi : f32
        store %yy, %y[%c0, %c1, %arg2] : memref<10x10x10xf32>
      }
    }
    acc.gang_redundant {
    }
  }


  return %y : memref<10x10x10xf32>
}

