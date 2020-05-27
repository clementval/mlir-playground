// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s
// RUN: mlir-opt --canonicalize --convert-linalg-to-loops --convert-openacc-to-gpu --convert-scf-to-std --gpu-kernel-outlining %s | mlir-cuda-runner --shared-libs=%cuda_wrapper_library_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libcuda-runtime-wrappers%shlibext,%oaru_library_dir/liboaru%shlibext --entry-point-result=void 

func @compute(%arg0: memref<100xf32>, %arg1: memref<100xf32>,
  %arg2: memref<100xf32>) -> () {
  %cst = constant 1 : index

  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index
  %i32_10 = constant 10 : index
  acc.parallel num_gangs(%i32_10) num_workers(%i32_10) {
    acc.loop gang vector {
      scf.for %i = %c0 to %c10 step %c1 {
        scf.for %j = %c0 to %c10 step %c1 {
          scf.for %k = %c0 to %c10 step %c1 {
            // c[i*n+j]+=a[i*n+k] * b[k*n+j];

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

// CHECK:       gpu.module @compute_acc_parallel {
// CHECK-NEXT:    gpu.func @compute_acc_parallel(%{{.*}}: index, %{{.*}}: memref<100xf32>, %{{.*}}: memref<100xf32>, %{{.*}}: memref<100xf32>, %{{.*}}: index, %{{.*}}: index) kernel {
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
// CHECK-NEXT:      [[UB:%.*]] = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      [[TMPLB:%.*]] = muli [[BLOCKID]], [[BLOCKDIM]] : index
// CHECK-NEXT:      [[LB:%.*]] = addi [[TMPLB]], [[THREADID]] : index
// CHECK-NEXT:      [[STEP:%.*]] = muli [[GRIDDIM]], [[BLOCKDIM]] : index
// CHECK-NEXT:      scf.for [[IND:%.*]] = [[LB]] to [[UB]] step [[STEP]] {
// CHECK-NEXT:        %{{.*}} = remi_signed [[IND]], %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = divi_signed [[IND]], %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        [[IDX2TMP0:%.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:          [[IDX0:%.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:          [[IDX1:%.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:          [[IDX2TMP1:%.*]] = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:          [[IDX2:%.*]] = addi [[IDX2TMP1]], [[IDX2TMP0]] : index
// CHECK-NEXT:          %{{.*}} = load %{{.*}}{{\[}}[[IDX1]]{{\]}} : memref<100xf32>
// CHECK-NEXT:          %{{.*}} = load %{{.*}}{{\[}}[[IDX2]]{{\]}} : memref<100xf32>
// CHECK-NEXT:          %{{.*}} = load %{{.*}}{{\[}}[[IDX0]]{{\]}} : memref<100xf32>
// CHECK-NEXT:          %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:          %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:          store %{{.*}}, %{{.*}}{{\[}}[[IDX0]]{{\]}} : memref<100xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      gpu.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }


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
