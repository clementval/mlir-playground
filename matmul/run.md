### Run the MLIR code on CPU

Path to `mlir-opt` and `mlir-cpu-runner` must be in your `PATH`. `libmlir_runner_utils.dylib` is the correct lib for macos. For linux change accordingly.

```
mlir-opt -linalg-lower-to-loops -convert-linalg-to-llvm -lower-to-llvm matmul2048x2048.mlir | mlir-cpu-runner -e main -entry-point-result=void -shared-libs=<PATH_TO_LLVM_REPO>/build/lib/libmlir_runner_utils.dylib
```
