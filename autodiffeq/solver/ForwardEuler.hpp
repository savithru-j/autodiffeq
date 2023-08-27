// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <iostream>
#include <vector>
#include <ostream>
#include "ODESolver.hpp"

namespace autodiffeq
{

template<typename T>
class ForwardEuler : public ODESolver<T>
{
public:
  ForwardEuler(ODE<T>& ode) : ODESolver<T>(ode) {}
  ~ForwardEuler() = default;

  using ODESolver<T>::ode_;
  using ODESolver<T>::Solve;

  SolutionHistory<T> 
  Solve(const Array1D<T>& sol0, const Array1D<double>& time_vec,
        const int storage_stride = 1) override
  {
    const int num_steps = time_vec.size() - 1;
    assert(num_steps >= 1);
    const int sol_dim = sol0.size();
    SolutionHistory<T> sol_hist(sol_dim, time_vec, storage_stride);
    sol_hist.SetSolution(0, sol0);

    if (ode_.GetSolutionSize() != sol_dim)
    {
      std::cout << "Solution size (= " << sol_dim << ") does not match ODE size (= " 
                << ode_.GetSolutionSize() << ")!" << std::endl;
      exit(1);
    }

    double time = time_vec(0);
    Array1D<T> sol(sol0);
    Array1D<T> rhs(sol_dim, T(0));

    for (int step = 0; step < num_steps; ++step)
    {
      ode_.EvalRHS(sol, step, time, rhs);

      double dt = time_vec(step+1) - time_vec(step);
      for (int i = 0; i < sol_dim; ++i)
        sol(i) += rhs(i)*dt;

      if ((step+1) % storage_stride == 0)
        sol_hist.SetSolution(step+1, sol);
      time += dt;
    }

    return sol_hist;
  }
};

}