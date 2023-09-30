// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <gtest/gtest.h>
#include <autodiffeq/linearalgebra/Array1D.hpp>
#include <autodiffeq/linearalgebra/GPUArray1D.cuh>
#include <autodiffeq/linearalgebra/Array2D.hpp>
#include <autodiffeq/linearalgebra/GPUArray2D.cuh>

#include <complex>

using namespace autodiffeq;

template<typename T = double>
__global__
void matvec(const DeviceArray2D<T>& A, const DeviceArray1D<T>& x, DeviceArray1D<T>& y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int nrows = A.GetNumRows();
  if (i < nrows)
  {
    T sum = 0;
    int ncols = A.GetNumCols();
    for (int j = 0; j < ncols; ++j)
      sum += A(i,j)*x(j);
    y[i] = sum;
  }
}

//----------------------------------------------------------------------------//
TEST( GPUArray2D, Constructor )
{
  {
    GPUArray2D<double> mat(3,5);
    EXPECT_EQ(mat.GetNumRows(), 3);
    EXPECT_EQ(mat.GetNumCols(), 5);
  }

  {
    GPUArray2D<std::complex<double>> mat(5, 4, {0.25, -1.0});
    EXPECT_EQ(mat.GetNumRows(), 5);
    EXPECT_EQ(mat.GetNumCols(), 4);
    auto mat_h = mat.CopyToHost();
    for (int i = 0; i < 20; ++i)
    {
      EXPECT_EQ(mat_h[i].real(), 0.25);
      EXPECT_EQ(mat_h[i].imag(), -1.0);
    }
  }

  {
    Array2D<int> mat_cpu = {{-2, 3, 6}, {42, -4, 8}};
    GPUArray2D<int> mat(mat_cpu);
    EXPECT_EQ(mat.GetNumRows(), 2);
    EXPECT_EQ(mat.GetNumCols(), 3);
    auto mat_h = mat.CopyToHost();
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        EXPECT_EQ(mat_h(i,j), mat_cpu(i,j));
  }

  {
    Array2D<int> mat_cpu = {{-2, 3, 6}, {42, -4, 8}};
    GPUArray2D<int> mat = {{-2, 3, 6}, {42, -4, 8}};
    EXPECT_EQ(mat.GetNumRows(), 2);
    EXPECT_EQ(mat.GetNumCols(), 3);
    auto mat_h = mat.CopyToHost();
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        EXPECT_EQ(mat_h(i,j), mat_cpu(i,j));
  }
}

//----------------------------------------------------------------------------//
TEST( GPUArray2D, MatVec )
{
  int N = 1024;

  Array1D<double> x_h(N);
  Array2D<double> A_h(N,N);
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
      A_h(i,j) = j;
    x_h(i) = i;
  }

  GPUArray1D<double> x(x_h);
  GPUArray2D<double> A(A_h);
  GPUArray1D<double> y(N);

  matvec<<<(N+255)/256, 256>>>(A.GetDeviceArray(), x.GetDeviceArray(), 
                               y.GetDeviceArray());

  double ans = (N-1)*N*(2*N-1)/6.0;
  auto y_h = y.CopyToHost();
  for (int i = 0; i < N; ++i)
    EXPECT_EQ(y_h[i], ans);
}
