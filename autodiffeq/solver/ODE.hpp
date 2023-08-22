// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <autodiffeq/linearalgebra/Array1D.hpp>

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

protected:

};

}