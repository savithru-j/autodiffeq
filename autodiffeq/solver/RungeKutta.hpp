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
class RungeKutta : public ODESolver<T>
{
public:
  RungeKutta(ODE<T>& ode, const int order) : ODESolver<T>(ode), order_(order) 
  {
    if (order_ != 4 && order_ != 5)
    {
      std::cout << "Only 4th and 5th order Runge-Kutta schemes are currently implemented." << std::endl;
      exit(1);
    }
  }
  ~RungeKutta() = default;

  using ODESolver<T>::ode_;
  using ODESolver<T>::Solve;

  SolutionHistory<T> 
  Solve(const Array1D<T>& sol0, const Array1D<double>& time_vec,
        const int storage_stride = 1) override
  {
    if (ode_.GetSolutionSize() != (int) sol0.size())
    {
      std::cout << "Solution size (= " << sol0.size() << ") does not match ODE size (= " 
                << ode_.GetSolutionSize() << ")!" << std::endl;
      exit(1);
    }

    if (order_ == 4)
      return RK4(sol0, time_vec, storage_stride);
    else if (order_ == 5)
      return DOPRI5(sol0, time_vec, storage_stride);
  }

protected:
  int order_;

  /* Fourth-order "classical" Runge-Kutta integrator */
  SolutionHistory<T> 
  RK4(const Array1D<T>& sol0, const Array1D<double>& time_vec,
      const int storage_stride = 1)
  {
    const int num_steps = time_vec.size() - 1;
    assert(num_steps >= 1);
    const int sol_dim = sol0.size();
    SolutionHistory<T> sol_hist(sol_dim, time_vec, storage_stride);
    sol_hist.SetSolution(0, sol0);

    double time = time_vec(0);
    Array1D<T> sol(sol0), sol_tmp(sol_dim);
    Array1D<T> k1(sol_dim), k2(sol_dim), k3(sol_dim), k4(sol_dim);
    constexpr double inv6 = 1.0/6.0;

    for (int step = 0; step < num_steps; ++step)
    {
      double dt = time_vec(step+1) - time_vec(step);
      double half_dt = 0.5*dt;
      ode_.EvalRHS(sol, step, time, k1);

      for (int i = 0; i < sol_dim; ++i)
        sol_tmp(i) = sol(i) + half_dt*k1(i);
      ode_.EvalRHS(sol_tmp, step, time + half_dt, k2);

      for (int i = 0; i < sol_dim; ++i)
        sol_tmp(i) = sol(i) + half_dt*k2(i);
      ode_.EvalRHS(sol_tmp, step, time + half_dt, k3);

      for (int i = 0; i < sol_dim; ++i)
        sol_tmp(i) = sol(i) + dt*k3(i);
      ode_.EvalRHS(sol_tmp, step, time + dt, k4);
      
      for (int i = 0; i < sol_dim; ++i)
        sol(i) += inv6*dt*(k1(i) + 2.0*(k2(i) + k3(i)) + k4(i));

      if ((step+1) % storage_stride == 0)
        sol_hist.SetSolution(step+1, sol);
      time += dt;
    }

    return sol_hist;
  }

  /* Fifth-order scheme from the Dormand-Prince method */
  SolutionHistory<T> 
  DOPRI5(const Array1D<T>& sol0, const Array1D<double>& time_vec,
      const int storage_stride = 1)
  {
    const int num_steps = time_vec.size() - 1;
    assert(num_steps >= 1);
    const int sol_dim = sol0.size();
    SolutionHistory<T> sol_hist(sol_dim, time_vec, storage_stride);
    sol_hist.SetSolution(0, sol0);

    double time = time_vec(0);
    Array1D<T> sol(sol0), sol_tmp(sol_dim);
    Array1D<T> k1(sol_dim), k2(sol_dim), k3(sol_dim), k4(sol_dim), k5(sol_dim), k6(sol_dim);

    constexpr double a21 = 0.2;
    constexpr double a31 = 3.0/40.0, a32 = 9.0/40.0;
    constexpr double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
    constexpr double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
    constexpr double a61 = 9017.0/3168.0, a62 = -355.0/33.0, a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;

    constexpr double b1 = 35.0/384.0;
    //constexpr double b2 = 0.0;
    constexpr double b3 = 500.0/1113.0;
    constexpr double b4 = 125.0/192.0;
    constexpr double b5 = -2187.0/6784.0;
    constexpr double b6 = 11.0/84.0;

    constexpr double c2 = 0.2, c3 = 0.3, c4 = 0.8, c5 = 8.0/9.0; //c6 = 1.0, c7 = 1.0;

    ode_.EvalRHS(sol, 0, time, k1);

    for (int step = 0; step < num_steps; ++step)
    {
      double dt = time_vec(step+1) - time_vec(step);

      for (int i = 0; i < sol_dim; ++i)
        sol_tmp(i) = sol(i) + dt*a21*k1(i);
      ode_.EvalRHS(sol_tmp, step, time + c2*dt, k2);

      for (int i = 0; i < sol_dim; ++i)
        sol_tmp(i) = sol(i) + dt*(a31*k1(i) + a32*k2(i));
      ode_.EvalRHS(sol_tmp, step, time + c3*dt, k3);

      for (int i = 0; i < sol_dim; ++i)
        sol_tmp(i) = sol(i) + dt*(a41*k1(i) + a42*k2(i) + a43*k3(i));
      ode_.EvalRHS(sol_tmp, step, time + c4*dt, k4);

      for (int i = 0; i < sol_dim; ++i)
        sol_tmp(i) = sol(i) + dt*(a51*k1(i) + a52*k2(i) + a53*k3(i) + a54*k4(i));
      ode_.EvalRHS(sol_tmp, step, time + c5*dt, k5);

      for (int i = 0; i < sol_dim; ++i)
        sol_tmp(i) = sol(i) + dt*(a61*k1(i) + a62*k2(i) + a63*k3(i) + a64*k4(i) + a65*k5(i));
      ode_.EvalRHS(sol_tmp, step, time + dt, k6);
      
      for (int i = 0; i < sol_dim; ++i)
        sol(i) += dt*(b1*k1(i) + b3*k3(i) + b4*k4(i) + b5*k5(i) + b6*k6(i));

      if ((step+1) % storage_stride == 0)
        sol_hist.SetSolution(step+1, sol);

      time += dt;
      ode_.EvalRHS(sol, 0, time, k1); //Seventh stage of current time-step is the same as the first stage of next time-step
    }

    return sol_hist;
  }
};

}