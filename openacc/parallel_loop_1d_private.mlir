// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s

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
      loop.for %x = %lb to %n step %st {
        acc.loop worker {
          loop.for %y = %lb to %n step %st {
            %axy = load %a[%x, %y] : memref<10x10xf32>
            %bxy = load %a[%x, %y] : memref<10x10xf32>
            %tmp = addf %axy, %bxy : f32
            store %tmp, %c[%y] : memref<10xf32>
          }
        }

        acc.loop seq {
          loop.for %i = %lb to %n step %st {
            %ci = load %c[%i] : memref<10xf32>
            %di = load %d[%i] : memref<10xf32>
            %z = addf %ci, %di : f32  
            store %z, %d[%i] : memref<10xf32> 
          }
        }
      }
    }
  }

  %d_ptr = memref_cast %d : memref<10xf32> to memref<*xf32>
  call @print_memref_f32(%d_ptr): (memref<*xf32>) -> ()
  // CHECK: [9,  8,  7,  6,  5,  4,  3,  2,  1,  0]
  return
}

func @print_memref_f32(%ptr : memref<*xf32>)
