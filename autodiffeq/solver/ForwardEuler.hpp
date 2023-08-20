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

namespace ForwardEuler
{

template<typename T>
SolutionHistory<T> Solve(ODE& ode, const Array1D<T>& sol0, const Array1D<double>& time_vec)
{
  const int num_steps = time_vec.size() - 1;
  assert(num_steps >= 1);
  const int sol_dim = sol0.size();
  SolutionHistory<T> sol_hist(sol_dim, time_vec);
  sol_hist.SetSolution(0, sol0);

  if (ode.GetSolutionSize() != sol_dim)
  {
    std::cout << "Solution size (= " << sol_dim << ") does not match ODE size (= " 
              << ode.GetSolutionSize() << ")!" << std::endl;
    exit(1);
  }

  double time = time_vec(0);
  Array1D<T> sol(sol0);
  Array1D<T> rhs(sol_dim, T(0));

  for (int step = 0; step < num_steps; ++step)
  {
    ode.EvalRHS(sol, step, time, rhs);

    double dt = time_vec(step+1) - time_vec(step);
    for (int i = 0; i < sol_dim; ++i)
      sol(i) += rhs(i)*dt;

    sol_hist.SetSolution(step+1, sol);
    time += dt;
  }

  return sol_hist;
}

template<typename T>
SolutionHistory<T> Solve(ODE& ode, const Array1D<T>& sol0, 
                         const double t_start, const double t_end, const int num_steps)
{
  const double dt = (t_end - t_start) / (double) num_steps;
  Array1D<double> time_vec(num_steps+1);
  time_vec(0) = t_start;
  for (int step = 1; step <= num_steps; ++step)
    time_vec(step) = time_vec(step-1) + dt;

  return Solve(ode, sol0, time_vec);
}


}


}