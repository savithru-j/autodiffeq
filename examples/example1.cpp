#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/solver/ForwardEuler.hpp>
#include <iostream>
#include <iomanip>

#include <complex>

using namespace autodiffeq;

class TestODE : public ODE
{
public:

  TestODE() = default;

  void EvalRHS(Array1D<double>& sol, int step, double time, Array1D<double>& rhs)
  {
    rhs(0) = -1.0*sol(0) + 1.8*sol(1);
    rhs(1) =  0.2*sol(0) - 2.0*sol(1);
  }

};

int main()
{

  ADVar<double> x(1.0, {0.0, 1.0});
  std::cout << x.value() << std::endl;

  Array1D<double> sol0(2);
  sol0(0) = 10.0;
  sol0(1) = 7.0;
  
  TestODE ode;
  int nt = 100;
  auto sol_hist = ForwardEuler::Solve(ode, sol0, 0, 1, nt);

  std::cout << std::setprecision(5) << std::scientific;
  std::cout << sol_hist << std::endl;
}