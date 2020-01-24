// RUN: mlir-opt --canonicalize --convert-linalg-to-loops --convert-loop-to-std --gpu-kernel-outlining %s | mlir-cuda-runner --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrapper%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void

func @saxpy(%x: memref<1024xf32>, %y: memref<1024xf32>,
  %n: index, %a: f32) -> memref<1024xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %nblock = constant 8 : index
  %nthread = constant 128 : index

  gpu.launch
      blocks(%bx, %by, %bz) in (%nbx = %nblock, %nby = %c1, %nbz = %c1)
      threads(%tx, %ty, %tz) in (%ntx = %nthread, %nty = %c1, %ntz = %c1)
      args(%arg0 = %x, %arg1 = %y, %arg2 = %n, %arg3 = %a, %arg4 = %nblock, %arg5 = %nthread) : memref<1024xf32>, memref<1024xf32>, index, f32, index, index {

      // i = blockIdx.x * blockDim.x + threadIdx.x
      %tidx = "gpu.thread_id"() {dimension = "x"} : () -> (index)
      %bidx = "gpu.block_id"() {dimension = "x"} : () -> (index)
      %bdimx = "gpu.block_dim"() {dimension = "x"} : () -> (index)
      %blockPos = muli %bidx, %bdimx : index
      %idx = addi %blockPos, %tidx : index

      // if (i < n)
      %inside = cmpi "slt", %idx, %arg2 : index
      cond_br %inside, ^bb1, ^bb2

^bb1:
      // y[i] = a*x[i] + y[i];
      %xi = load %arg0[%idx] : memref<1024xf32>
      %yi = load %arg1[%idx] : memref<1024xf32>
      %ax = mulf %arg3, %xi : f32
      %yy = addf %ax, %yi : f32
      store %yy, %arg1[%idx] : memref<1024xf32>
      gpu.return
^bb2:
      gpu.return
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

  %y1 = call @saxpy(%x, %y, %n, %a) : (memref<1024xf32>, memref<1024xf32>, index, f32) -> memref<1024xf32>
  call @print_memref_1d_f32(%y1): (memref<1024xf32>) -> ()
  return
}

func @print_memref_1d_f32(memref<1024xf32>)
