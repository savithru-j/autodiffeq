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

namespace RK4
{

template<typename T>
SolutionHistory<T> Solve(ODE<T>& ode, const Array1D<T>& sol0, const Array1D<double>& time_vec,
                         const int storage_stride = 1)
{
  const int num_steps = time_vec.size() - 1;
  assert(num_steps >= 1);
  const int sol_dim = sol0.size();
  SolutionHistory<T> sol_hist(sol_dim, time_vec, storage_stride);
  sol_hist.SetSolution(0, sol0);

  if (ode.GetSolutionSize() != sol_dim)
  {
    std::cout << "Solution size (= " << sol_dim << ") does not match ODE size (= " 
              << ode.GetSolutionSize() << ")!" << std::endl;
    exit(1);
  }

  double time = time_vec(0);
  Array1D<T> sol(sol0), sol_tmp(sol_dim);
  Array1D<T> k1(sol_dim), k2(sol_dim), k3(sol_dim), k4(sol_dim);
  constexpr double inv6 = 1.0/6.0;

  for (int step = 0; step < num_steps; ++step)
  {
    double dt = time_vec(step+1) - time_vec(step);
    double half_dt = 0.5*dt;
    ode.EvalRHS(sol, step, time, k1);

    for (int i = 0; i < sol_dim; ++i)
      sol_tmp(i) = sol(i) + half_dt*k1(i);
    ode.EvalRHS(sol_tmp, step, time + half_dt, k2);

    for (int i = 0; i < sol_dim; ++i)
      sol_tmp(i) = sol(i) + half_dt*k2(i);
    ode.EvalRHS(sol_tmp, step, time + half_dt, k3);

    for (int i = 0; i < sol_dim; ++i)
      sol_tmp(i) = sol(i) + dt*k3(i);
    ode.EvalRHS(sol_tmp, step, time + dt, k4);
    
    for (int i = 0; i < sol_dim; ++i)
      sol(i) += inv6*dt*(k1(i) + 2.0*(k2(i) + k3(i)) + k4(i));

    if ((step+1) % storage_stride == 0)
      sol_hist.SetSolution(step+1, sol);
    time += dt;
  }

  return sol_hist;
}

template<typename T>
SolutionHistory<T> Solve(ODE<T>& ode, const Array1D<T>& sol0, 
                         const double t_start, const double t_end, const int num_steps,
                         const int storage_stride = 1)
{
  const double dt = (t_end - t_start) / (double) num_steps;
  Array1D<double> time_vec(num_steps+1);
  time_vec(0) = t_start;
  for (int step = 1; step <= num_steps; ++step)
    time_vec(step) = time_vec(step-1) + dt;

  return Solve(ode, sol0, time_vec, storage_stride);
}

}

}