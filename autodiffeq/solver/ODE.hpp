// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <autodiffeq/linearalgebra/Array1D.hpp>

#ifdef ENABLE_CUDA
#include <autodiffeq/linearalgebra/GPUArray1D.cuh>
#endif

namespace autodiffeq
{

/* Base class for implementing the RHS function (e.g. time derivative of the state)
 * of an ODE of the form du/dt = f(u, t) */

template<typename T>
class ODE
{
public:

  ODE() = default;

  virtual ~ODE() = default;

  virtual int GetSolutionSize() const = 0;

  virtual void EvalRHS(const Array1D<T>& sol, int step, double time, Array1D<T>& rhs) = 0;

#ifdef ENABLE_CUDA
  virtual void EvalRHSGPU(const DeviceArray1D<T>& sol, int step, double time, DeviceArray1D<T>& rhs)
  {
    std::cout << "EvalRHSGPU function not implemented!" << std::endl;
    exit(1);
  }
#endif

protected:

};

}