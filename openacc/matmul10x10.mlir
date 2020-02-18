// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s
// RUN: mlir-opt --canonicalize --convert-linalg-to-loops --convert-openacc-to-gpu --convert-loop-to-std --gpu-kernel-outlining %s | mlir-cuda-runner --shared-libs=%cuda_wrapper_library_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libcuda-runtime-wrappers%shlibext,%oaru_library_dir/liboaru%shlibext --entry-point-result=void 

func @compute(%arg0: memref<100xf32>, %arg1: memref<100xf32>,
  %arg2: memref<100xf32>) -> () {
  %cst = constant 1 : index

  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index
  acc.parallel num_gangs(10) num_workers(10) {
    acc.loop gang vector {
      loop.for %i = %c0 to %c10 step %c1 {
        loop.for %j = %c0 to %c10 step %c1 {
          loop.for %k = %c0 to %c10 step %c1 {
            
            // c[i∗n+j]+=a[i∗n+k] ∗b[k∗n+j];

            // c-index = i*n+j
            %ixn = muli %i, %c10 : index
            %ci = addi %ixn, %j : index

            // a-index = i*n+k
            %ai = addi %ixn, %k : index

            // b-index = k*n+j
            %kxn = muli %k, %c10 : index
            %bi = addi %kxn, %j : index

            %a = load %arg0[%ai] : memref<100xf32>
            %b = load %arg1[%bi] : memref<100xf32>
            %c = load %arg2[%ci] : memref<100xf32>
              
            %tmp1 = mulf %a, %b : f32
            %tmp2 = addf %c, %tmp1 : f32

            store %tmp2, %arg2[%ci] : memref<100xf32>
          }
        }
      } 
    } attributes { collapse = 2 }
  }

  return

// CHECK: func @compute(%arg0: memref<100xf32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>) {
// CHECK-NEXT: %{{.*}} = constant 1 : index
// CHECK-NEXT: %{{.*}} = constant 0 : index
// CHECK-NEXT: %{{.*}} = constant 10 : index
// CHECK-NEXT: %{{.*}} = constant 1 : index
// CHECK-NEXT: %{{.*}} = constant 1 : index
// CHECK-NEXT: %{{.*}} = constant 10 : index
// CHECK-NEXT: %{{.*}} = constant 10 : index
// CHECK-NEXT: gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c10_2, %arg10 = %c1_1, %arg11 = %c1_1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c10_3, %arg13 = %c1_1, %arg14 = %c1_1) {
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}}2 : index
// CHECK-NEXT:   loop.for %{{.*}} = %{{.*}} t{{.*}} %{{.*}} step %{{.*}} {
// CHECK-NEXT:     %{{.*}} = remi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:       %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = load %{{.*}}[%{{.*}}] : memref<100xf32>
// CHECK-NEXT:       %{{.*}} = load %{{.*}}[%{{.*}}] : memref<100xf32>
// CHECK-NEXT:       %{{.*}} = load %{{.*}}[%{{.*}}] : memref<100xf32>
// CHECK-NEXT:       %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:       store %{{.*}}, %{{.*}}[%{{.*}}] : memref<100xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   gpu.terminator
// CHECK-NEXT: }
// CHECK-NEXT: return


}

func @main() {
  %A = alloc() : memref<100xf32>
  %B = alloc() : memref<100xf32>
  %C = alloc() : memref<100xf32>

  %cf0 = constant 0.00000e+00 : f32
  %cf1 = constant 1.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<100xf32>, f32
  linalg.fill(%B, %cf1) : memref<100xf32>, f32
  linalg.fill(%C, %cf0) : memref<100xf32>, f32

  call @compute(%A, %B, %C) : (memref<100xf32>, memref<100xf32>, memref<100xf32>) -> ()

  %C_ptr = memref_cast %C : memref<100xf32> to memref<*xf32>
  call @print_memref_f32(%C_ptr) : (memref<*xf32>) -> ()
  return
}

func @print_memref_f32(memref<*xf32>)
