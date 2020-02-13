// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s
// mlir-opt --canonicalize --convert-openacc-to-gpu --convert-linalg-to-loops --convert-loop-to-std --gpu-kernel-outlining %s | mlir-cuda-runner --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrapper%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void

func @compute(%x: memref<1024xf32>, %y: memref<1024xf32>,
  %n: index, %a: f32) -> memref<1024xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // y[i] = a*x[i] + y[i];
  acc.parallel num_gangs(8) num_workers(1) {
    acc.loop {
      loop.for %arg0 = %c0 to %n step %c1 {
        %xi = load %x[%arg0] : memref<1024xf32>
        %yi = load %y[%arg0] : memref<1024xf32>
        %ax = mulf %a, %xi : f32
        %yy = addf %ax, %yi : f32
        store %yy, %y[%arg0] : memref<1024xf32>
      }
    } attributes { gang }
  }
  return %y : memref<1024xf32>
}

// CHECK: %c0 = constant 0 : index
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   %{{.*}} = constant 8 : index
//  CHECK-NEXT:   gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) {
//  CHECK-NEXT:      loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:        %{{.*}} = load %{{.*}}[%{{.*}}] : memref<1024xf32>
//  CHECK-NEXT:        %{{.*}} = load %{{.*}}[%{{.*}}] : memref<1024xf32>
//  CHECK-NEXT:        %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:        %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:        store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32>
//  CHECK-NEXT:      }
//  CHECK-NEXT:      gpu.terminator
//  CHECK-NEXT:    }


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
