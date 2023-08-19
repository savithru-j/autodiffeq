#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/solver/ForwardEuler.hpp>
#include <iostream>

#include <complex>

using namespace autodiffeq;

int main()
{

  ADVar<double> x(1.0, {0.0, 1.0});
  std::cout << x.value() << std::endl;

  Array1D<double> sol0(5, 0.0);
  
  auto sol_hist = ForwardEuler::Solve(sol0, 0, 1, 10);

  std::cout << sol_hist.GetNumSteps() << std::endl;

}