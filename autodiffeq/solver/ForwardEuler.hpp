// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <iostream>
#include <vector>
#include <ostream>
#include "ODESolver.hpp"

#ifdef ENABLE_CUDA
#include <autodiffeq/solver/GPUSolutionHistory.cuh>
#include <autodiffeq/solver/SolverKernels.cuh>
#endif

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
  using ODESolver<T>::solve_on_gpu_;

  SolutionHistory<T> 
  Solve(const Array1D<T>& sol0, const Array1D<double>& time_vec,
        const int storage_stride = 1) override
  {
    if (ode_.GetSolutionSize() != sol0.size())
    {
      std::cout << "Solution size (= " << sol0.size() << ") does not match ODE size (= " 
                << ode_.GetSolutionSize() << ")!" << std::endl;
      exit(1);
    }

    if (!solve_on_gpu_)
      return SolveCPU(sol0, time_vec, storage_stride);
    else
    {
#ifdef ENABLE_CUDA
      return SolveGPU(sol0, time_vec, storage_stride);
#else
      std::cout << "Need to compile with CUDA enabled to solve on GPUs" << std::endl;
      exit(1);
#endif
    }
  }

protected:

  SolutionHistory<T> 
  SolveCPU(const Array1D<T>& sol0, const Array1D<double>& time_vec,
           const int storage_stride = 1)
  {
    const int num_steps = time_vec.size() - 1;
    assert(num_steps >= 1);
    const int sol_dim = sol0.size();
    SolutionHistory<T> sol_hist(sol_dim, time_vec, storage_stride);
    sol_hist.SetSolution(0, sol0);

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

#ifdef ENABLE_CUDA
  SolutionHistory<T> 
  SolveGPU(const Array1D<T>& sol0, const Array1D<double>& time_vec,
           const int storage_stride = 1)
  {
    const int num_steps = time_vec.size() - 1;
    assert(num_steps >= 1);

    GPUArray1D<T> sol(sol0);
    const int sol_dim = sol0.size();
    GPUSolutionHistory<T> gpu_hist(sol_dim, time_vec, storage_stride);
    gpu_hist.SetSolution(0, sol);

    double time = time_vec(0);
    GPUArray1D<T> rhs(sol_dim, T(0));

    dim3 thread_dim = 256;
    dim3 block_dim = (sol_dim + thread_dim.x-1) / thread_dim.x;

    for (int step = 0; step < num_steps; ++step)
    {
      ode_.EvalRHS(sol, step, time, rhs);

      double dt = time_vec(step+1) - time_vec(step);
      gpu::StepSolution<<<block_dim, thread_dim>>>(rhs.GetDeviceArray(), dt, 
                                                   sol.GetDeviceArray());
      if ((step+1) % storage_stride == 0)
        gpu_hist.SetSolution(step+1, sol);
      time += dt;
    }

    SolutionHistory<T> sol_hist(sol_dim, time_vec, storage_stride);
    gpu_hist.GetData().CopyToHost(sol_hist.GetData());
    return sol_hist;
  }
#endif
};

}