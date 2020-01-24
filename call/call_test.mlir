


func @main() {
  %0 = call @oaru_get_num_devices() : () -> i32
  return
}

func @oaru_get_num_devices() -> i32
