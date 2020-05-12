// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s

func @compute(%x: memref<1024x1024xf32>, %y: memref<1024x1024xf32>,
  %n: index) -> memref<1024x1024xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // y[i] = a*x[i] + y[i];
  acc.parallel {
    acc.loop gang vector {
      loop.for %arg0 = %c0 to %n step %c1 {
        loop.for %arg1 = %c0 to %n step %c1 {
          %xi = load %x[%arg0, %arg1] : memref<1024x1024xf32>
          %yi = load %y[%arg0, %arg1] : memref<1024x1024xf32>
          %yy = mulf %xi, %yi : f32
          store %yy, %y[%arg0, %arg1] : memref<1024x1024xf32>
        }
      }
    } attributes { collapse = 2 }
  }
  return %y : memref<1024x1024xf32>
}

// CHECK:      func @compute(%{{.*}}: memref<1024x1024xf32>, %{{.*}}: memref<1024x1024xf32>, %{{.*}}: index) -> memref<1024x1024xf32> {
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = constant 127 : index
// CHECK-NEXT:   %{{.*}} = constant 128 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   [[NUMGANGS:%.*]] = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   [[NUMWORKERS:%.*]] = constant 128 : index
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   "gpu.launch_func"([[NUMGANGS]], %{{.*}}, %{{.*}}, [[NUMWORKERS]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {kernel = @compute_acc_parallel::@compute_acc_parallel} : (index, index, index, index, index, index, memref<1024x1024xf32>, memref<1024x1024xf32>, index, index, index) -> ()
// CHECK-NEXT:   return %{{.*}} : memref<1024x1024xf32>
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @compute_acc_parallel {
// CHECK-NEXT:   gpu.func @compute_acc_parallel(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: index, %arg3: index, %arg4: index) kernel {
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
// CHECK-NEXT:     [[UB:%.*]] = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     [[TMPLB:%.*]] = muli [[BLOCKID]], [[BLOCKDIM]] : index
// CHECK-NEXT:     [[LB:%.*]] = addi [[TMPLB]], [[THREADID]] : index
// CHECK-NEXT:     [[STEP:%.*]] = muli [[GRIDDIM]], [[BLOCKDIM]] : index
// CHECK-NEXT:     loop.for [[IND:%.*]] = [[LB]] to [[UB]] step [[STEP]] {
// CHECK-NEXT:       %{{.*}} = remi_signed [[IND]], %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = divi_signed [[IND]], %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       [[IDX2:%.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       [[IDX1:%.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = load %{{.*}}{{\[}}[[IDX1]], [[IDX2]]{{\]}} : memref<1024x1024xf32>
// CHECK-NEXT:       %{{.*}} = load %{{.*}}{{\[}}[[IDX1]], [[IDX2]]{{\]}} : memref<1024x1024xf32>
// CHECK-NEXT:       %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       store %{{.*}}, %{{.*}}{{\[}}[[IDX1]], [[IDX2]]{{\]}} : memref<1024x1024xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     gpu.return
// CHECK-NEXT:   }
// CHECK-NEXT: }


func @main() {
  %x = alloc() : memref<1024x1024xf32>
  %y = alloc() : memref<1024x1024xf32>

  %c1 = constant 10.0 : f32
  %c2 = constant 20.0 : f32
  %n = constant 1024 : index

  linalg.fill(%x, %c1) : memref<1024x1024xf32>, f32
  linalg.fill(%y, %c2) : memref<1024x1024xf32>, f32

  call @compute(%x, %y, %n) : (memref<1024x1024xf32>, memref<1024x1024xf32>, index) -> memref<1024x1024xf32>

  return
}

func @print_memref_f32(memref<*xf32>)
