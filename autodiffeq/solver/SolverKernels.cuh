// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <cassert>
#include <autodiffeq/linearalgebra/GPUArray1D.cuh>
#include <autodiffeq/linearalgebra/GPUArray2D.cuh>

namespace autodiffeq
{
namespace gpu
{

template<typename T>
__global__
void StepSolution(const DeviceArray1D<T>& rhs, const double dt, DeviceArray1D<T>& sol)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  auto sol_dim = sol.size();
  if (i < sol_dim) 
    sol[i] += rhs[i]*dt;
}

template<typename T>
__global__
void StepSolution(const DeviceArray1D<T>& sol, const DeviceArray1D<T>& rhs, const double dt, DeviceArray1D<T>& new_sol)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  auto sol_dim = sol.size();
  if (i < sol_dim) 
    new_sol[i] = sol[i] + rhs[i]*dt;
}

template<typename T>
__global__
void StepSolutionRK4(const DeviceArray1D<T>& k1, const DeviceArray1D<T>& k2, 
                     const DeviceArray1D<T>& k3, const DeviceArray1D<T>& k4,
                     const double inv6_dt, DeviceArray1D<T>& sol)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  auto sol_dim = sol.size();
  if (i < sol_dim) 
    sol[i] += inv6_dt*(k1[i] + 2.0*(k2[i] + k3[i]) + k4[i]);
}

}
}