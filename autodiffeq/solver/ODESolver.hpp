// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <iostream>
#include <vector>
#include <ostream>
#include "ODE.hpp"
#include "SolutionHistory.hpp"

namespace autodiffeq
{

template<typename T>
class ODESolver
{
public:
  ODESolver(ODE<T>& ode) : ode_(ode) {}
  virtual ~ODESolver() = default;

  virtual SolutionHistory<T> 
  Solve(const Array1D<T>& sol0, const Array1D<double>& time_vec, const int storage_stride = 1) = 0;

  SolutionHistory<T> 
  Solve(const Array1D<T>& sol0, const double t_start, const double t_end, 
        const int num_steps, const int storage_stride = 1)
  {
    const double dt = (t_end - t_start) / (double) num_steps;
    Array1D<double> time_vec(num_steps+1);
    time_vec(0) = t_start;
    for (int step = 1; step <= num_steps; ++step)
      time_vec(step) = time_vec(step-1) + dt;

    return Solve(sol0, time_vec, storage_stride);
  }

protected:
  ODE<T>& ode_;
};

}