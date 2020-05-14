// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s

func @compute(%x: memref<10x10x10xf32>, %y: memref<10x10x10xf32>,
  %n: index) -> memref<10x10x10xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %i32_10 = constant 10 : i32

  // y[i] = a*x[i] + y[i];
  acc.parallel num_gangs(%i32_10) num_workers(%i32_10) {
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
    } attributes { collapse = 2 }
  }
  return %y : memref<10x10x10xf32>
}

// CHECK:       gpu.module @compute_acc_parallel {
// CHECK-NEXT:    gpu.func @compute_acc_parallel(%arg0: memref<10x10x10xf32>, %arg1: memref<10x10x10xf32>, %arg2: index, %arg3: index, %arg4: index) kernel {
// CHECK-NEXT:      [[BLOCKID:%.*]] = "gpu.block_id"() {dimension = "x"} : () -> index
// CHECK-NEXT:      [[THREADID:%.*]] = "gpu.thread_id"() {dimension = "x"} : () -> index
// CHECK-NEXT:      [[GRIDDIM:%.*]] = "gpu.grid_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT:      [[BLOCKDIM:%.*]] = "gpu.block_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT:      %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = constant 1 : index
// CHECK-NEXT:      %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = constant 0 : index
// CHECK-NEXT:      %{{.*}} = constant 1 : index
// CHECK-NEXT:      %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = constant 1 : index
// CHECK-NEXT:      %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = constant 0 : index
// CHECK-NEXT:      %{{.*}} = constant 1 : index
// CHECK-NEXT:      [[UPPERBOUND:%.*]] = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      [[LBTMP:%.*]] = muli [[BLOCKID]], [[BLOCKDIM]] : index
// CHECK-NEXT:      [[LOWERBOUND:%.*]] = addi [[LBTMP]], [[THREADID]] : index
// CHECK-NEXT:      [[STEP:%.*]] = muli [[GRIDDIM]], [[BLOCKDIM]] : index
// CHECK-NEXT:      scf.for [[IND:%.*]] = [[LOWERBOUND]] to [[UPPERBOUND]] step [[STEP]] {
// CHECK-NEXT:        %{{.*}} = remi_signed [[IND]], %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = divi_signed [[IND]], %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        [[IDX2:%.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        [[IDX1:%.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        scf.for [[IDX3:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          %{{.*}} = load %{{.*}}{{\[}}[[IDX1]], [[IDX2]], [[IDX3]]{{\]}} : memref<10x10x10xf32>
// CHECK-NEXT:          %{{.*}} = load %{{.*}}{{\[}}[[IDX1]], [[IDX2]], [[IDX3]]{{\]}} : memref<10x10x10xf32>
// CHECK-NEXT:          %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:          store %{{.*}}, %{{.*}}{{\[}}[[IDX1]], [[IDX2]], [[IDX3]]{{\]}} : memref<10x10x10xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      gpu.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }


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
