#include <stdio.h>

#define N 1024

__global__
void saxpy(int n, float a, float *x, float *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x)
  {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  float x[N];
  float y[N];
  float a = 10.0;
  int n = N;

  for(int i = 0; i < n; ++i) {
    x[i] = 1.0;
    y[i] = 2.0;
  }

  saxpy<<<8, 128>>>(n, a, x, y);

  float sum = 0.0;
  for(int i = 0; i < n; ++i) {
    sum += y[i];
  }

  printf("%f\n", sum);
}
