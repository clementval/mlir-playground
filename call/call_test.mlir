// RUN: mlir-opt --convert-std-to-llvm call_test.mlir | mlir-cuda-runner --shared-libs=%cuda_wrapper_library_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libcuda-runtime-wrappers%shlibext,./build/liboaru.so --entry-point-result=void

// Simple code to call external function of various kind

func @main() {
  call @oaru_init() : () -> ()
  %0 = call @oaru_get_num_devices() : () -> i32
  call @oaru_print_i32(%0) : (i32) -> ()
  return
}

func @oaru_get_num_devices() -> i32
func @oaru_init()
func @oaru_print_i32(%val: i32) -> ()
