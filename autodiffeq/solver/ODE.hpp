// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <autodiffeq/linearalgebra/Array1D.hpp>

namespace autodiffeq
{

/* Base class for implementing the RHS function (e.g. time derivative of the state)
 * of an ODE of the form du/dt = f(u, t) */

class ODE
{
public:

  ODE() = default;

  virtual ~ODE() = default;

  virtual void EvalRHS(Array1D<double>& sol, int step, double time, Array1D<double>& rhs)
  {
    std::cout << "EvalRHS of ODE not implemented for double datatype!" << std::endl;
    exit(1);
  };

  virtual void EvalRHS(Array1D<ADVar<double>>& sol, int step, double time, Array1D<ADVar<double>>& rhs)
  {
    std::cout << "EvalRHS of ODE not implemented for ADVar<double> datatype!" << std::endl;
    exit(1);
  };


protected:

};

}