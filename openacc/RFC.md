# [RFC] OpenACC dialect in MLIR

We would like to propose an OpenACC[1] dialect to be added to MLIR in a simlar
way than the OpenMP dialect[2].
The overall goal is to have a dialect that is front-end agnostic and that 
represent the capabilities of the 3.0 and newer standard versions. 

The dialect would ultimatly support variety of underlying loops representation
such as `loop.for`, `loop.parallel`, `affine.for` in the dialect region.

The basic operations would be `acc.parallel` and `acc.loop` and those are the
one detailed in this RFC.

The `acc.parallel` opeartion imply that its region must be offloaded to an
accelerator. The mapping of the region to the accelerator follows the standard.
One worker per gang execute the region.

The `acc.loop` operation specifies the mapping of the loops within its region to
the available processors. Several mapping can be specified to the operation and
thoses are attached as attributes. Common mapping can be: `gang`, `gang_vector`,
`vector` ....

Other attributes like loop collapsing can be attached to the operation.

```
%c0 = constant 0 : index
%c1 = constant 1 : index

acc.parallel {
  acc.loop {
    loop.for %arg0 = %c0 to %n step %c1 {
      loop.for %arg1 = %c0 to %n step %c1 {
        %xi = load %x[%arg0, %arg1] : memref<10x10xf32>
        %yi = load %y[%arg0, %arg1] : memref<10x10xf32>
        %yy = mulf %xi, %yi : f32
        store %yy, %y[%arg0, %arg1] : memref<10x10xf32>
      }
    }
  } attributes { collapse = 2, mapping = gang_vector }
} attributes { num_gangs = 8, num_workers = 10 }
```

This could be lowered to the GPU dialect for example: 

```
func @compute(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>, %arg2: index) -> memref<10x10xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c1_0 = constant 1 : index
  %c10 = constant 10 : index
  %c10_1 = constant 10 : index
  gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c10, %arg10 = %c1_0, %arg11 = %c1_0) threads(%arg6, %arg7, %arg8) in (%arg12 = %c10_1, %arg13 = %c1_0, %arg14 = %c1_0) {
    %0 = muli %arg2, %arg2 : index
    %1 = muli %arg3, %arg12 : index
    %2 = addi %1, %arg6 : index
    %3 = muli %arg9, %arg12 : index
    loop.for %arg15 = %2 to %0 step %3 {
      %4 = remi_signed %arg15, %arg2 : index
      %5 = divi_signed %arg15, %arg2 : index
      %6 = load %arg0[%5, %4] : memref<10x10xf32>
      %7 = load %arg1[%5, %4] : memref<10x10xf32>
      %8 = mulf %6, %7 : f32
      store %8, %arg1[%5, %4] : memref<10x10xf32>
    }
    gpu.terminator
  }
  return %arg1 : memref<10x10xf32>
}
```

Obviously, the dialect is meant to represent the full capabilities of the 
OpenACC standard and more operations will come in the future. 
`acc.parallel` and `acc.loop` are good starting point since they are 
representing the most used construct in OpenACC.


### References
[1] https://www.openacc.org
[2] https://llvm.discourse.group/t/rfc-openmp-dialect-in-mlir/397/9
