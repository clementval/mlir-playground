// RUN: mlir-opt --convert-openacc-to-target %s | FileCheck %s
// RUN: mlir-opt --canonicalize --convert-linalg-to-loops --convert-openacc-to-target --convert-loop-to-std %s | mlir-cuda-runner --shared-libs=%cuda_wrapper_library_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libcuda-runtime-wrappers%shlibext,%oaru_library_dir/liboaru%shlibext --entry-point-result=void | FileCheck --prefix EXEC

func @main() {
  %a = alloc() : memref<10x10xf32>
  %b = alloc() : memref<10x10xf32>
  %c = alloc() : memref<10xf32>
  %d = alloc() : memref<10xf32>

  %zero = constant 0.0 : f32
  %ca = constant 1.0 : f32
  %cb = constant 2.0 : f32

  %lb = constant 0 : index
  %st = constant 1 : index
  %n = constant 10 : index

  linalg.fill(%a, %ca) : memref<10x10xf32>, f32
  linalg.fill(%b, %cb) : memref<10x10xf32>, f32
  linalg.fill(%c, %zero) : memref<10xf32>, f32
  linalg.fill(%d, %zero) : memref<10xf32>, f32

  acc.parallel num_gangs(10) num_workers(10) private(%c : memref<10xf32>) {
    acc.loop gang {
      // for x = 0 to 10 step 1
      //   for y = 0 to 10 step 1
      //     c[y] = a[x,y] + b[x,y]
      loop.for %x = %lb to %n step %st {
        acc.loop worker {
          loop.for %y = %lb to %n step %st {
            %axy = load %a[%x, %y] : memref<10x10xf32>
            %bxy = load %b[%x, %y] : memref<10x10xf32>
            %tmp = addf %axy, %bxy : f32
            store %tmp, %c[%y] : memref<10xf32>
          }
        }
      
        acc.loop seq {
          // for i = 0 to 10 step 1
          //   d[x] += c[i]
          loop.for %i = %lb to %n step %st {
            %ci = load %c[%i] : memref<10xf32>
            %dx = load %d[%x] : memref<10xf32>
            %z = addf %ci, %dx : f32  
            store %z, %d[%x] : memref<10xf32> 
          }
        }
      }
    }
  }

  %d_ptr = memref_cast %d : memref<10xf32> to memref<*xf32>
  call @print_memref_f32(%d_ptr): (memref<*xf32>) -> ()
  // EXEC: [30,  30,  30,  30,  30,  30,  30,  30,  30,  30]
  return
}

func @print_memref_f32(%ptr : memref<*xf32>)
