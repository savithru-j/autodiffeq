// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <gtest/gtest.h>
#include <autodiffeq/linearalgebra/Array1D.hpp>
#include <autodiffeq/linearalgebra/GPUArray1D.cuh>

#include <complex>

using namespace autodiffeq;

template<typename T = double>
__global__
void add(const DeviceArray1D<T>& x, const DeviceArray1D<T>& y, DeviceArray1D<T>& z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  auto size = x.size();
  if (i < size) 
    z[i] = x[i] + y[i];
}

//----------------------------------------------------------------------------//
TEST( GPUArray1D, Constructor )
{
  {
    GPUArray1D<double> vec(3);
    EXPECT_EQ(vec.size(), 3u);
  }

  {
    GPUArray1D<std::complex<double>> vec(5, {0.25, -1.0});
    EXPECT_EQ(vec.size(), 5u);
    auto vec_h = vec.CopyToHost();
    for (int i = 0; i < 5; ++i)
    {
      EXPECT_EQ(vec_h[i].real(), 0.25);
      EXPECT_EQ(vec_h[i].imag(), -1.0);
    }
  }

  {
    GPUArray1D<int> vec = {-2, 3, 6, 42, -4, 8};
    auto vec_h = vec.CopyToHost();
    EXPECT_EQ(vec.size(), 6u);
    EXPECT_EQ(vec_h.size(), 6u);
    EXPECT_EQ(vec_h[0], -2);
    EXPECT_EQ(vec_h[1], 3);
    EXPECT_EQ(vec_h[2], 6);
    EXPECT_EQ(vec_h[3], 42);
    EXPECT_EQ(vec_h[4], -4);
    EXPECT_EQ(vec_h[5], 8);

    GPUArray1D<int> vec2 = {};
    EXPECT_EQ(vec2.size(), 0u);
  }

  {
    Array1D<int> vec_cpu = {-2, 3, 6, 42, -4, 8};
    GPUArray1D<int> vec(vec_cpu);
    auto vec_h = vec.CopyToHost();
    EXPECT_EQ(vec.size(), 6u);
    EXPECT_EQ(vec_h.size(), 6u);
    EXPECT_EQ(vec_h[0], -2);
    EXPECT_EQ(vec_h[1], 3);
    EXPECT_EQ(vec_h[2], 6);
    EXPECT_EQ(vec_h[3], 42);
    EXPECT_EQ(vec_h[4], -4);
    EXPECT_EQ(vec_h[5], 8);
  }
}

//----------------------------------------------------------------------------//
TEST( GPUArray1D, Add )
{
  int N = 1024;
  GPUArray1D<double> x(N, 5.0);
  GPUArray1D<double> y(N);
  y.SetValue(-2.0);
  GPUArray1D<double> z(N);

  add<<<(N+255)/256, 256>>>(x.GetDeviceArray(), y.GetDeviceArray(), 
                            z.GetDeviceArray());
  
  auto z_h = z.CopyToHost();
  for (int i = 0; i < N; ++i)
    EXPECT_EQ(z_h[i], 3);
}
