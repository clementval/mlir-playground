// RUN: mlir-opt --convert-openacc-to-target %s | FileCheck %s

func @compute(%x: memref<10x10x10xf32>, %y: memref<10x10x10xf32>,
  %n: index) -> memref<10x10x10xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // y[i] = a*x[i] + y[i];
  acc.parallel num_gangs(8) num_workers(128) {
    acc.loop {
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

// CHECK:      gpu.module @compute_acc_parallel {
// CHECK-NEXT:   gpu.func @compute_acc_parallel(%{{.*}}: memref<10x10x10xf32>, %{{.*}}: memref<10x10x10xf32>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) kernel {
// CHECK-NEXT:     [[BLOCKID:%.*]] = "gpu.block_id"() {dimension = "x"} : () -> index
// CHECK-NEXT:     [[THREADID:%.*]] = "gpu.thread_id"() {dimension = "x"} : () -> index
// CHECK-NEXT:     [[GRIDDIM:%.*]] = "gpu.grid_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT:     [[BLOCKDIM:%.*]] = "gpu.block_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT:     %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = constant 1 : index
// CHECK-NEXT:     %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = constant 0 : index
// CHECK-NEXT:     %{{.*}} = constant 1 : index
// CHECK-NEXT:     %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = constant 1 : index
// CHECK-NEXT:     %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = constant 0 : index
// CHECK-NEXT:     %{{.*}} = constant 1 : index
// CHECK-NEXT:     %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = constant 1 : index
// CHECK-NEXT:     %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = constant 0 : index
// CHECK-NEXT:     %{{.*}} = constant 1 : index
// CHECK-NEXT:     %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     [[UPPERBOUND:%.*]] = muli %16, %15 : index
// CHECK-NEXT:     [[LBTMP:%.*]] = muli [[BLOCKID]], [[BLOCKDIM]] : index
// CHECK-NEXT:     [[LOWERBOUND:%.*]] = addi [[LBTMP]], [[THREADID]] : index
// CHECK-NEXT:     [[STEP:%.*]] = muli [[GRIDDIM]], [[BLOCKDIM]] : index
// CHECK-NEXT:     loop.for [[IND:%.*]] = [[LOWERBOUND]] to [[UPPERBOUND]] step [[STEP]] {
// CHECK-NEXT:       %{{.*}} = remi_signed [[IND]], %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = divi_signed [[IND]], %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = remi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       [[IDX2:%.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       [[IDX1:%.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       [[IDX0:%.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = load %{{.*}}{{\[}}[[IDX0]], [[IDX1]], [[IDX2]]{{\]}} : memref<10x10x10xf32>
// CHECK-NEXT:       %{{.*}} = load %{{.*}}{{\[}}[[IDX0]], [[IDX1]], [[IDX2]]{{\]}} : memref<10x10x10xf32>
// CHECK-NEXT:       %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       store %{{.*}}, %{{.*}}{{\[}}[[IDX0]], [[IDX1]], [[IDX2]]{{\]}} : memref<10x10x10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     gpu.return
// CHECK-NEXT:   }
// CHECK-NEXT: }


func @main() {
  %x = alloc() : memref<10x10x10xf32>
  %y = alloc() : memref<10x10x10xf32>

  %c1 = constant 10.0 : f32
  %c2 = constant 20.0 : f32
  %n = constant 10 : index

  linalg.fill(%x, %c1) : memref<10x10x10xf32>, f32
  linalg.fill(%y, %c2) : memref<10x10x10xf32>, f32

  call @compute(%x, %y, %n) : (memref<10x10x10xf32>, memref<10x10x10xf32>, index) -> memref<10x10x10xf32>
  call @print_memref_3d_f32(%y): (memref<10x10x10xf32>) -> ()
  return
}

func @print_memref_3d_f32(memref<10x10x10xf32>)
