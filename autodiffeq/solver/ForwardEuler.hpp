// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <iostream>
#include <vector>
#include <ostream>
#include "SolutionHistory.hpp"

namespace autodiffeq
{

namespace ForwardEuler
{

template<typename T>
SolutionHistory<T> Solve(const Array1D<T>& sol0, const Array1D<double>& time_vec)
{
  const int num_steps = time_vec.size() - 1;
  assert(num_steps >= 1);
  const int sol_dim = sol0.size();
  std::cout << num_steps << std::endl;
  SolutionHistory<T> sol_hist(sol_dim, num_steps+1);
  sol_hist.SetSolution(0, sol0);

  double time = time_vec(0);
  Array1D<T> sol(sol0);
  Array1D<T> rhs(sol_dim, T(0));

  for (int step = 1; step <= num_steps; ++step)
  {
    //TODO: Evaluate rhs

    double dt = time_vec(step) - time_vec(step-1);
    for (int i = 0; i < sol_dim; ++i)
      sol(i) += rhs(i)*dt;

    time += dt;
  }

  return sol_hist;
}

template<typename T>
SolutionHistory<T> Solve(const Array1D<T>& sol0, const double t_start, const double t_end, const int num_steps)
{
  const double dt = (t_end - t_start) / (double) num_steps;
  Array1D<double> time_vec(num_steps+1);
  time_vec(0) = t_start;
  for (int step = 1; step <= num_steps; ++step)
    time_vec(step) = time_vec(step-1) + dt;

  std::cout << time_vec << std::endl;
  return Solve(sol0, time_vec);
}


}


}