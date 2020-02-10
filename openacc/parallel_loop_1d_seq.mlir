// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s
// RUN: mlir-opt --canonicalize --convert-linalg-to-loops --convert-openacc-to-gpu --convert-loop-to-std --gpu-kernel-outlining %s | mlir-cuda-runner --shared-libs=%cuda_wrapper_library_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libcuda-runtime-wrappers%shlibext,%oaru_library_dir/liboaru%shlibext --entry-point-result=void 

func @compute(%x: memref<20xf32>, %n: index) -> memref<20xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // x[i] = x[i] + x[i-1];
  acc.parallel {
    acc.loop {
      loop.for %arg0 = %c1 to %n step %c1 {
        %xi = load %x[%arg0] : memref<20xf32>
        %im1 = subi %arg0, %c1 : index
        %xim1 = load %x[%im1] : memref<20xf32>
        %tmp = addf %xi, %xim1 : f32
        store %tmp, %x[%arg0] : memref<20xf32>
      }
    } attributes { seq }
  } attributes { num_gangs = 8, num_workers = 128 }
  return %x : memref<20xf32>
}

// CHECK:      gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) {
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}} = cmpi "eq", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = cmpi "eq", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = and %0, %1 : i1
// CHECK-NEXT:   loop.if %{{.*}} {
// CHECK-NEXT:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:       %{{.*}} = load %{{.*}}[%{{.*}}] : memref<20xf32>
// CHECK-NEXT:       %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = load %{{.*}}[%{{.*}}] : memref<20xf32>
// CHECK-NEXT:       %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       store %{{.*}}, %{{.*}}[%{{.*}}] : memref<20xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   gpu.barrier
// CHECK-NEXT:   gpu.terminator
// CHECK-NEXT: }

// EXEC: [1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20]

func @main() {
  %x = alloc() : memref<20xf32>

  %c1 = constant 1.0 : f32
  %n = constant 20 : index

  linalg.fill(%x, %c1) : memref<20xf32>, f32

  call @compute(%x, %n) : (memref<20xf32>, index) -> memref<20xf32>
  call @print_memref_1d_f32(%x): (memref<20xf32>) -> ()
  return
}

func @print_memref_1d_f32(memref<20xf32>)
