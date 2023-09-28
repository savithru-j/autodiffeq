#include <stdio.h>
#include <iostream>
#include <autodiffeq/linearalgebra/GPUArray1D.hpp>

using namespace autodiffeq;

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  printf("%d\n", i);
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
#if 0
  int N = 1024;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  cudaError_t err = cudaSuccess;
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
            cudaGetErrorString(err));
    //exit(EXIT_FAILURE);
  }

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);


  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
#endif

  GPUArray1D<double> v0(10, 5.2);
  std::cout << v0 << std::endl;

  v0.SetValue(11);
  std::cout << v0 << std::endl;

  GPUArray1D<double> v1 = {-3.0, 0.2, 10.0, 7.4};
  std::cout << v1 << std::endl;
}
