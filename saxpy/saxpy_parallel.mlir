// RUN: ../../llvm-project/build/bin/mlir-opt --canonicalize -linalg-lower-to-loops -convert-linalg-to-llvm -convert-std-to-llvm  saxpy.mlir | ../../llvm-project/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=../../llvm-project/build/lib/libmlir_runner_utils.dylib

// RUN: ../../llvm-project/build/bin/mlir-opt --canonicalize --convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm  saxpy.mlir | ../../llvm-project/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=../../llvm-project/build/lib/libmlir_runner_utils.so 

func @saxpy(%x: memref<1024xf32>, %y: memref<1024xf32>,
  %n: index, %a: f32) -> memref<1024xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // y[i] = a*x[i] + y[i];
  loop.parallel (%arg0) = (%c0) to (%n) step (%c1) {
    %xi = load %x[%arg0] : memref<1024xf32>
    %yi = load %y[%arg0] : memref<1024xf32>
    %ax = mulf %a, %xi : f32
    %yy = addf %ax, %yi : f32
    store %yy, %y[%arg0] : memref<1024xf32>
    "loop.terminator"() : () -> ()
  }
  return %y : memref<1024xf32>
}


func @main() {
  %x = alloc() : memref<1024xf32>
  %y = alloc() : memref<1024xf32>

  %a = constant 10.0 : f32
  %c1 = constant 1.0 : f32
  %c2 = constant 2.0 : f32
  %n = constant 1024 : index

  linalg.fill(%x, %c1) : memref<1024xf32>, f32
  linalg.fill(%y, %c2) : memref<1024xf32>, f32

  call @saxpy(%x, %y, %n, %a) : (memref<1024xf32>, memref<1024xf32>, index, f32) -> memref<1024xf32>
  call @print_memref_1d_f32(%y): (memref<1024xf32>) -> ()
  return
}

func @print_memref_1d_f32(memref<1024xf32>)
