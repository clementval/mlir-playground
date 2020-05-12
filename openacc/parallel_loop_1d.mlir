// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s

func @compute(%x: memref<1024xf32>, %y: memref<1024xf32>,
  %n: index, %a: f32) -> memref<1024xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %i32_8 = constant 8 : i32
  %i32_128 = constant 128 : i32  

  // y[i] = a*x[i] + y[i];
  acc.parallel num_gangs(%i32_8) num_workers(%i32_128) {
    acc.loop gang vector {
      loop.for %arg0 = %c0 to %n step %c1 {
        %xi = load %x[%arg0] : memref<1024xf32>
        %yi = load %y[%arg0] : memref<1024xf32>
        %ax = mulf %a, %xi : f32
        %yy = addf %ax, %yi : f32
        store %yy, %y[%arg0] : memref<1024xf32>
      }
    }
  }
  return %y : memref<1024xf32>
}

// CHECK:      gpu.module @compute_acc_parallel {
// CHECK-NEXT:   gpu.func @compute_acc_parallel(%{{.*}}: memref<1024xf32>, %{{.*}}: memref<1024xf32>, %{{.*}}: f32, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) kernel {
// CHECK-NEXT:     %{{.*}} = "gpu.block_id"() {dimension = "x"} : () -> index
// CHECK-NEXT:     %{{.*}} = "gpu.thread_id"() {dimension = "x"} : () -> index
// CHECK-NEXT:     %{{.*}} = "gpu.grid_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT:     %{{.*}} = "gpu.block_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT:     %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:       %{{.*}} = load %{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK-NEXT:       %{{.*}} = load %{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK-NEXT:       %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     gpu.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

func @main() {
  %x = alloc() : memref<1024xf32>
  %y = alloc() : memref<1024xf32>

  %a = constant 10.0 : f32
  %c1 = constant 1.0 : f32
  %c2 = constant 2.0 : f32
  %n = constant 1024 : index

  linalg.fill(%x, %c1) : memref<1024xf32>, f32
  linalg.fill(%y, %c2) : memref<1024xf32>, f32

  call @compute(%x, %y, %n, %a) : (memref<1024xf32>, memref<1024xf32>, index, f32) -> memref<1024xf32>
  call @print_memref_1d_f32(%y): (memref<1024xf32>) -> ()
  return
}

func @print_memref_1d_f32(memref<1024xf32>)
