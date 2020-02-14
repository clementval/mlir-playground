# [RFC] OpenACC dialect in MLIR
We would like to propose an OpenACC[1] dialect to be added to MLIR in a simlar
way than the OpenMP dialect[2] has been added.
The overall goal is to have a dialect that is front-end agnostic and that 
represent the capabilities of the OpenACC 3.0 standard and follow with potential 
new versions.

OpenACC is very important for scientific applications running on big 
supercomputers. About 25% of applications running on the Summit supercomputer 
(#1 in top500) are using OpenACC.

The dialect would ultimatly support variety of underlying dialects and 
loops representation such as `loop.for`, `loop.parallel`, `affine.for` in 
the dialect region.

The operations of the dialect would represents the different constructs found in 
the standard. A parallel construct would be represented by an `acc.parallel` 
operation and a loop construct would be a `acc.loop` operation. Some 
construct could be direct call to runtime library (data movement, allocation 
...)

The `acc.parallel` operation imply that its region must be offloaded to an
accelerator. The mapping of the region to the accelerator follows the standard.
One worker per gang execute the region. The number of gang and workers 
can be specified or determined be the lowering process. The `num_gangs(X)` 
and `num_workers(X)` can be added to the operation to specify those numbers.
This will for example drive the number of workgroup/worker if the operation
is lowered to the GPU dialect.

The `acc.loop` operation specifies the mapping of the loop(s) within its region 
to the available workers. Several mapping can be specified to the operation. 
Common mapping can be: `gang`, `gang vector`, `vector` .... Nested loops can be
collapse with a `collapse(X)` added to the operation. 

Unlike the OpenMP dialect, we are not targeting the OpenMP builder. We are 
targeting a lowering to the GPU dialect as an initial step. 
So a simple lowering from OpenACC to GPU dialects could be done as shown below.

```
func @compute(%x: memref<1024xf32>, %y: memref<1024xf32>,
  %n: index, %a: f32) -> memref<1024xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %c0 = constant 0 : index
  %c1 = constant 1 : index

  acc.parallel num_gangs(8) num_workers(128) {
    acc.loop gang vector {
      loop.for %arg0 = %c0 to %n step %c1 {
        %xi = load %x[%arg0] : memref<1024xf32>
        %yi = load %y[%arg0] : memref<1024xf32>
        %ax = mulf %a, %xi : f32
        %yy = addf %ax, %yi : f32
        store %yy, %y[%arg0] : memref<1024xf32>
      }
    }
  }
 return %y : memref<1024xf32>
} 
```

Once lowered it could look like this.

```
func @compute(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: index, 
    %arg3: f32) -> memref<1024xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c1_0 = constant 1 : index
  %c8 = constant 8 : index
  %c128 = constant 128 : index
  gpu.launch 
      blocks(%arg4, %arg5, %arg6) in (%arg10 = %c8, %arg11 = %c1_0, %arg12 = %c1_0) 
      threads(%arg7, %arg8, %arg9) in (%arg13 = %c128, %arg14 = %c1_0, %arg15 = %c1_0) {
    %0 = muli %arg4, %arg13 : index
    %1 = addi %0, %arg7 : index
    %2 = muli %arg10, %arg13 : index
    loop.for %arg16 = %1 to %arg2 step %2 {
      %3 = load %arg0[%arg16] : memref<1024xf32>
      %4 = load %arg1[%arg16] : memref<1024xf32>
      %5 = mulf %arg3, %3 : f32
      %6 = addf %5, %4 : f32
      store %6, %arg1[%arg16] : memref<1024xf32>
    }
    gpu.terminator
  }
  return %arg1 : memref<1024xf32>
}
```

The `gang` and `vector` information defined how the loop inside the region is
mapped to the workgroups/workers available.

For example, if the loop operation is only mapped with gang `acc.loop gang {`, 
it will distribute the iterations of the associated loops among the workgroups 
only.

The first intent is to use this dialect from f18/flang. It might as well be 
used by any frontend targeting MLIR. We have started work on the parsing/sema
and are looking into the lowering to MLIR when f18 will include the first bit 
of fir/mlir in the master.

Obviously, the dialect is meant to represent the full capabilities of the 
OpenACC standard and more operations will come as we designed them. 
`acc.parallel` and `acc.loop` are good starting point since they are 
representing the most used construct in OpenACC.

As we go, there will probably be some overlap between an OpenACC and the offload
part of the OpenMP dialect. If it make sense, there can share common lowering.

### References
[1] https://www.openacc.org

[2] https://llvm.discourse.group/t/rfc-openmp-dialect-in-mlir/397/9