// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <cuda_runtime.h>
#include <iostream>

#define cudaCheckError(val) check_cuda_error((val), #val, __FILE__, __LINE__)
void check_cuda_error(cudaError_t err, const char* const func, const char* const file,
                      const int line)
{
  if (err != cudaSuccess) {
    printf("CUDA error: %s - %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    printf("%s in %s(%d)\n", func, file, line);
    std::exit(EXIT_FAILURE);
  }

#if 0 //Additional checks
  cudaError_t err_last = cudaGetLastError();
  if (err_last != cudaSuccess) {
    printf("CUDA last error: %s - %s\n", cudaGetErrorName(err_last), cudaGetErrorString(err_last));
    printf("%s in %s(%d)\n", func, file, line);
    std::exit(EXIT_FAILURE);
  }

  cudaError_t err_sync = cudaDeviceSynchronize();
  if (err_sync != cudaSuccess) {
    printf("CUDA sync error: %s - %s\n", cudaGetErrorName(err_sync), cudaGetErrorString(err_sync));
    printf("%s in %s(%d)\n", func, file, line);
    std::exit(EXIT_FAILURE);
  }
#endif
}