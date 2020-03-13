#include <stdio.h>


int main() {


  float data[5][5];
  float sum;

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      data[i][j] = (float) j;
    }
  }


  #pragma acc data copyin(data[5][5]) 
  {
    #pragma acc parallel loop collapse(2) gang vector reduction(+:sum)
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) {
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
