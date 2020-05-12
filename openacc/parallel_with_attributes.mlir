// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s

func @compute(%A: memref<10xf32>, %B: memref<10xf32>) -> memref<10xf32> {
  %c0 = constant 0 : index
  %i32_2 = constant 2 : i32

  acc.parallel num_gangs(%i32_2) num_workers(%i32_2) {
    acc.gang_redundant {
      %tmp = load %A[%c0] : memref<10xf32>
      store %tmp, %B[%c0] : memref<10xf32>
    }
  }

  return %B : memref<10xf32>
}

// CHECK:       gpu.module @compute_acc_parallel {
// CHECK-NEXT:    gpu.func @compute_acc_parallel(%{{.*}}: memref<10xf32>, %{{.*}}: index, %{{.*}}: memref<10xf32>) kernel {
// CHECK-NEXT:     [[THREADID:%.*]] = "gpu.thread_id"() {dimension = "x"} : () -> index
// CHECK-NEXT:     [[CST0:%.*]] = constant 0 : index
// CHECK-NEXT:     [[ISTHREAD0:%.*]] = cmpi "eq", [[THREADID]], [[CST0]] : index
// CHECK-NEXT:     loop.if [[ISTHREAD0]] {
// CHECK-NEXT:       %{{.*}} = load %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:       store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     gpu.barrier
// CHECK-NEXT:     gpu.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }


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
