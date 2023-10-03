// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <gtest/gtest.h>
#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/solver/ForwardEuler.hpp>
#include <chrono>

using namespace autodiffeq;

template<typename T = double>
__global__
void evalRHS(const DeviceArray1D<T>& sol, DeviceArray1D<T>& rhs)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  auto soldim = sol.size();
  if (i < soldim) 
    rhs[i] = sol[i];
}

template<typename T>
class TestODE : public ODE<T>
{
public:
  using ODE<T>::EvalRHS;

  TestODE() = default;

  int GetSolutionSize() const { return 2000; }

  inline void EvalRHS(const Array1D<T>& sol, int step, double time, Array1D<T>& rhs) override
  {
    auto soldim = sol.size();
    for (int i = 0; i < soldim; ++i)
      rhs[i] = sol[i];
  }

  inline void EvalRHS(const GPUArray1D<T>& sol, int step, double time, GPUArray1D<T>& rhs) override
  {
    auto soldim = sol.size();
    evalRHS<<<(soldim+63)/64, 64>>>(sol.GetDeviceArray(), rhs.GetDeviceArray());
  }

protected:
};

//----------------------------------------------------------------------------//
TEST( ForwardEuler, CPU_GPU_Consistency )
{
  using clock = std::chrono::high_resolution_clock;

  TestODE<double> ode;
  const int soldim = ode.GetSolutionSize();

  Array1D<double> sol0(soldim);
  for (int i = 0; i < soldim; ++i)
    sol0(i) = std::sin(i);
 
  int nt = 10000;
  double eps = 1e-12;
  double ts = 0.0, tf = 1.0;

  ForwardEuler<double> solver(ode);
  solver.SetSolveOnGPU(false);

  auto t0 = clock::now();
  auto sol_hist_cpu = solver.Solve(sol0, ts, tf, nt);
  auto t1 = clock::now();
  auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() / 1000.0;
  std::cout << "CPU time: " << cpu_time << " ms" << std::endl;

  solver.SetSolveOnGPU(true);
  t0 = clock::now();
  auto sol_hist_gpu = solver.Solve(sol0, ts, tf, nt);
  t1 = clock::now();
  auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() / 1000.0;
  std::cout << "GPU time: " << gpu_time << " ms" << std::endl;

  for (int t = 0; t < nt; ++t)
    for (int i = 0; i < soldim; ++i)
      EXPECT_NEAR(sol_hist_cpu(t,i), sol_hist_gpu(t,i), eps);

}