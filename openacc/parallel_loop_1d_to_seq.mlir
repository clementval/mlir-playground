// RUN: mlir-opt --convert-openacc-to-seq %s | FileCheck %s
// RUN: mlir-opt --canonicalize --convert-linalg-to-loops --convert-openacc-to-seq --convert-loop-to-std -convert-linalg-to-llvm --convert-std-to-llvm %s | mlir-cpu-runner --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void 

func @compute(%x: memref<20xf32>, %n: index) -> memref<20xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // x[i] = x[i] + x[i-1];
  acc.parallel num_gangs(8) num_workers(128) {
    acc.loop {
      loop.for %arg0 = %c1 to %n step %c1 {
        %xi = load %x[%arg0] : memref<20xf32>
        %im1 = subi %arg0, %c1 : index
        %xim1 = load %x[%im1] : memref<20xf32>
        %tmp = addf %xi, %xim1 : f32
        store %tmp, %x[%arg0] : memref<20xf32>
      }
    } attributes { seq }
  }
  return %x : memref<20xf32>
}

// CHECK:      %{{.*}} = constant 0 : index
// CHECK-NEXT: %{{.*}} = constant 1 : index
// CHECK-NEXT: loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:   %{{.*}} = load %{{.*}}[%{{.*}}] : memref<20xf32>
// CHECK-NEXT:   %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = load %{{.*}}[%{{.*}}] : memref<20xf32>
// CHECK-NEXT:   %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:   store %{{.*}}, %{{.*}}[%{{.*}}] : memref<20xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %{{.*}} : memref<20xf32>

// EXEC: [1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20]

func @main() {
  %x = alloc() : memref<20xf32>

  %c1 = constant 1.0 : f32
  %n = constant 20 : index

  linalg.fill(%x, %c1) : memref<20xf32>, f32

  call @compute(%x, %n) : (memref<20xf32>, index) -> memref<20xf32>
  %xp = memref_cast %x : memref<20xf32> to memref<*xf32>
  call @print_memref_f32(%xp) : (memref<*xf32>) -> ()
  return
}

func @print_memref_f32(%ptr : memref<*xf32>)