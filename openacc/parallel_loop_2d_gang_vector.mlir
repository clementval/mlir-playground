// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s

func @compute(%x: memref<10x10xf32>, %y: memref<10x10xf32>,
  %n: index) -> memref<10x10xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %i32_10 = constant 10 : index

  // y[i] = a*x[i] + y[i];
  acc.parallel num_gangs(%i32_10) num_workers(%i32_10) {
    acc.loop gang {
      scf.for %arg0 = %c0 to %n step %c1 {
        acc.loop vector {
          scf.for %arg1 = %c0 to %n step %c1 {
            %xi = load %x[%arg0, %arg1] : memref<10x10xf32>
            %yi = load %y[%arg0, %arg1] : memref<10x10xf32>
            %yy = mulf %xi, %yi : f32
            store %yy, %y[%arg0, %arg1] : memref<10x10xf32>
          }
        }
      }
    }
  }
  return %y : memref<10x10xf32>
}

// CHECK:      gpu.module @compute_acc_parallel {
// CHECK-NEXT:   gpu.func @compute_acc_parallel(%{{.*}}: memref<10x10xf32>, %{{.*}}: memref<10x10xf32>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) kernel {
// CHECK-NEXT:     [[BLOCKID:%.*]] = "gpu.block_id"() {dimension = "x"} : () -> index
// CHECK-NEXT:     [[THREADID:%.*]] = "gpu.thread_id"() {dimension = "x"} : () -> index
// CHECK-NEXT:     [[GRIDDIM:%.*]] = "gpu.grid_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT:     [[BLOCKDIM:%.*]] = "gpu.block_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT:     scf.for %{{.*}} = [[BLOCKID]] to %{{.*}} step [[GRIDDIM]] {
// CHECK-NEXT:       scf.for %{{.*}} = [[THREADID]] to %{{.*}} step [[BLOCKDIM]] {
// CHECK-NEXT:         %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:         %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:         %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:         store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     gpu.return
// CHECK-NEXT:   }
// CHECK-NEXT: }


func @main() {
  %x = alloc() : memref<10x10xf32>
  %y = alloc() : memref<10x10xf32>

  %c1 = constant 10.0 : f32
  %c2 = constant 20.0 : f32
  %n = constant 10 : index

  linalg.fill(%x, %c1) : memref<10x10xf32>, f32
  linalg.fill(%y, %c2) : memref<10x10xf32>, f32

  call @compute(%x, %y, %n) : (memref<10x10xf32>, memref<10x10xf32>, index) -> memref<10x10xf32>
  call @print_memref_2d_f32(%y): (memref<10x10xf32>) -> ()
  return
}

func @print_memref_2d_f32(memref<10x10xf32>)
