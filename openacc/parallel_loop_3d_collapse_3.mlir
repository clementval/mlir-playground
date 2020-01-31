
func @compute(%x: memref<10x10x10xf32>, %y: memref<10x10x10xf32>,
  %n: index) -> memref<10x10x10xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // y[i] = a*x[i] + y[i];
  acc.parallel {
    acc.loop {
      loop.for %arg0 = %c0 to %n step %c1 {
        loop.for %arg1 = %c0 to %n step %c1 {
          loop.for %arg2 = %c0 to %n step %c1 {
            %xi = load %x[%arg0, %arg1, %arg2] : memref<10x10x10xf32>
            %yi = load %y[%arg0, %arg1, %arg2] : memref<10x10x10xf32>
            %yy = mulf %xi, %yi : f32
            store %yy, %y[%arg0, %arg1, %arg2] : memref<10x10x10xf32>
          }
        }
      }
    } attributes { collapse = 3 }
  } attributes { num_gangs = 8, num_workers = 128 }
  return %y : memref<10x10x10xf32>
}

// CHECK:      loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:   %{{.*}} = remi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = remi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<10x10x10xf32>
// CHECK-NEXT:   %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<10x10x10xf32>
// CHECK-NEXT:   %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:   store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<10x10x10xf32>
// CHECK-NEXT: }


func @main() {
  %x = alloc() : memref<10x10x10xf32>
  %y = alloc() : memref<10x10x10xf32>

  %c1 = constant 10.0 : f32
  %c2 = constant 20.0 : f32
  %n = constant 10 : index

  linalg.fill(%x, %c1) : memref<10x10x10xf32>, f32
  linalg.fill(%y, %c2) : memref<10x10x10xf32>, f32

  call @compute(%x, %y, %n) : (memref<10x10x10xf32>, memref<10x10x10xf32>, index) -> memref<10x10x10xf32>
  call @print_memref_3d_f32(%y): (memref<10x10x10xf32>) -> ()
  return
}

func @print_memref_3d_f32(memref<10x10x10xf32>)
