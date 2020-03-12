#include <stdio.h>


#define N 5

int main() {


  float data[N][N];
  float sum;

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      data[i][j] = (float) j;
    }
  }


  #pragma acc data copyin(data[N][N]) 
  {
    #pragma acc parallel loop collapse(2) gang vector reduction(+:sum)
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        sum += data[i][j];
      }
    }
  }

  if(sum == 50.0) {
    printf("SUM IS CORRECT - %f\n", sum);
  } else {
    printf("SUM IS WRONG - %f\n", sum);
    return 1;
  }

  return 0;
}
