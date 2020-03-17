// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s

func @compute(%A: memref<10xf32>) -> () {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %n = constant 10 : index
  acc.parallel num_gangs(2) num_workers(5) {
    %reduction = acc.loop gang worker () -> (f32) {
      %sum_0 = constant 0.0 : f32
      %sum = loop.for %i = %c0 to %n step %c1 iter_args(%sum_iter = %sum_0) -> f32 {
        %val = load %A[%i] : memref<10xf32>
        %sum_next = addf %sum_iter, %val : f32
        loop.yield %sum_next : f32
      }
      %res = "acc.reduction"(%sum) {op = "add"} : (f32) -> (f32)
      acc.yield %res
    }
  }

  // %sump = memref_cast %sum   : memref<1xf32> to memref<*xf32>
  // call @print_memref_f32(%sump) : (memref<*xf32>) -> ()
  return
}

func @main() {
  %x = alloc() : memref<10xf32>

  %c1 = constant 1.0 : f32

  linalg.fill(%x, %c1) : memref<10xf32>, f32

  call @compute(%x) : (memref<10xf32>) -> ()

  return
}

func @print_memref_f32(%ptr : memref<*xf32>)
