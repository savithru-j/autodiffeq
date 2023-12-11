// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <gtest/gtest.h>
#define _USE_MATH_DEFINES //For MSVC
#include <autodiffeq/numerics/ADVarS.hpp>
#include <autodiffeq/linearalgebra/GPUArray1D.cuh>

using namespace autodiffeq;

template<typename T = double>
__global__
void CalcRHS(const DeviceArray1D<T>& sol, DeviceArray1D<T>& rhs)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  auto soldim = sol.size();
  if (i < soldim) 
    rhs[i] = cos(sol[i]) + sol[i]*sol[i];
}

//----------------------------------------------------------------------------//
TEST( ADVarS, GPUDerivatives )
{
  Array1D<ADVarS<2,double>> x(5);
  x(0) = ADVarS<2,double>(0.1, {1.0, 0.0});
  x(1) = ADVarS<2,double>(-2.5, {0.0, 1.0});
  x(2) = ADVarS<2,double>(5.0, {0.2, 0.3});
  x(3) = ADVarS<2,double>(0.7, {-1.0, 2.0});
  x(4) = ADVarS<2,double>(8.0, {0.3, 0.7});

  GPUArray1D<ADVarS<2,double>> sol(x), rhs(5);

  dim3 thread_dim = 64;
  dim3 block_dim = 1;
  CalcRHS<<<block_dim, thread_dim>>>(sol.GetDeviceArray(), rhs.GetDeviceArray());

  auto y = rhs.CopyToHost();
  for (int i = 0; i < 5; ++i)
  {
    EXPECT_EQ(y(i).value(), cos(x[i].value()) + x[i].value()*x[i].value());
    EXPECT_EQ(y(i).deriv(0), (-sin(x[i].value()) + 2.0*x[i].value())*x[i].deriv(0));
    EXPECT_EQ(y(i).deriv(1), (-sin(x[i].value()) + 2.0*x[i].value())*x[i].deriv(1));
  }
}
