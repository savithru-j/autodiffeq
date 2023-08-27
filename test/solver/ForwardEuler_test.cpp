// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <gtest/gtest.h>
#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/solver/ForwardEuler.hpp>

#include <complex>

using namespace autodiffeq;

template<typename T>
class TestODE : public ODE<T>
{
public:

  TestODE(const Array1D<T>& coeff) : coeff_(coeff) {}

  int GetSolutionSize() const { return 3; }

  void EvalRHS(const Array1D<T>& sol, int step, double time, Array1D<T>& rhs)
  {
    rhs(0) = coeff_(0)*sol(0) + coeff_(1)*sol(1) + coeff_(2)*sol(2);
    rhs(1) = coeff_(3)*sol(0) + coeff_(4)*sol(1) + coeff_(5)*sol(2);
    rhs(2) = coeff_(6)*sol(0) + coeff_(7)*sol(1) + coeff_(8)*sol(2);
  }

protected:
  Array1D<T> coeff_;
};


//----------------------------------------------------------------------------//
TEST( ForwardEuler, CoefficientSensitivityNumDeriv1 )
{
  Array1D<double> sol0(3);
  sol0(0) = 10.0;
  sol0(1) = -3.0;
  sol0(2) = 7.0;

  Array1D<double> coeff(9);
  coeff(0) = -1.0; coeff(1) =  0.2; coeff(2) = -0.1;
  coeff(3) = -0.4; coeff(4) = -3.0; coeff(5) =  0.3;
  coeff(6) =  0.1; coeff(7) = -0.7; coeff(8) = -1.7;
  
  int nt = 10;
  double eps = 1e-9;

  double ts = 0.0, tf = 1.0;

  Array1D<double> dsolfinal0_dcoeff(9);
  Array1D<double> dsolfinal1_dcoeff(9);
  Array1D<double> dsolfinal2_dcoeff(9);

  for (int ic = 0; ic < 9; ++ic)
  {
    coeff(ic) += eps;
    TestODE<double> ode_p(coeff);
    ForwardEuler<double> solver_p(ode_p);
    auto sol_hist_p = solver_p.Solve(sol0, ts, tf, nt);
    
    coeff(ic) -= 2*eps;
    TestODE<double> ode_m(coeff);
    ForwardEuler<double> solver_m(ode_m);
    auto sol_hist_m = solver_m.Solve(sol0, ts, tf, nt);

    coeff(ic) += eps;

    dsolfinal0_dcoeff(ic) = (sol_hist_p(nt, 0) - sol_hist_m(nt, 0)) / (2.0*eps);
    dsolfinal1_dcoeff(ic) = (sol_hist_p(nt, 1) - sol_hist_m(nt, 1)) / (2.0*eps);
    dsolfinal2_dcoeff(ic) = (sol_hist_p(nt, 2) - sol_hist_m(nt, 2)) / (2.0*eps);
  }

  // std::cout << dsolfinal0_dcoeff << std::endl;
  // std::cout << dsolfinal1_dcoeff << std::endl;
  // std::cout << dsolfinal2_dcoeff << std::endl;

  const int num_deriv = 1;
  const double tol = 1e-6;

  Array1D<ADVar<double>> coeff_ad(9);
  for (int i = 0; i < 9; ++i)
    coeff_ad(i) = ADVar<double>(coeff(i), num_deriv);

  for (int ic = 0; ic < 9; ++ic)
  {
    coeff_ad(ic).deriv() = 1.0;

    Array1D<ADVar<double>> sol_ad0(3);
    for (int i = 0; i < 3; ++i)
      sol_ad0(i) = sol0(i);

    TestODE<ADVar<double>> ode_ad(coeff_ad);
    ForwardEuler<ADVar<double>> solver_ad(ode_ad);
    auto sol_hist_ad = solver_ad.Solve(sol_ad0, ts, tf, nt);

    // std::cout << sol_hist_ad(nt, 0).deriv() << ", " << sol_hist_ad(nt, 1).deriv() << ", " << sol_hist_ad(nt, 2).deriv() << std::endl;
    EXPECT_NEAR(sol_hist_ad(nt, 0).deriv(), dsolfinal0_dcoeff(ic), tol);
    EXPECT_NEAR(sol_hist_ad(nt, 1).deriv(), dsolfinal1_dcoeff(ic), tol);
    EXPECT_NEAR(sol_hist_ad(nt, 2).deriv(), dsolfinal2_dcoeff(ic), tol);

    coeff_ad(ic).deriv() = 0.0;
  }
}

//----------------------------------------------------------------------------//
TEST( ForwardEuler, CoefficientSensitivityNumDeriv3 )
{
  Array1D<double> sol0(3);
  sol0(0) = 10.0;
  sol0(1) = -3.0;
  sol0(2) = 7.0;

  Array1D<double> coeff(9);
  coeff(0) = -1.0; coeff(1) =  0.2; coeff(2) = -0.1;
  coeff(3) = -0.4; coeff(4) = -3.0; coeff(5) =  0.3;
  coeff(6) =  0.1; coeff(7) = -0.7; coeff(8) = -1.7;
  
  int nt = 10;
  double eps = 1e-9;

  double ts = 0.0, tf = 1.0;

  Array1D<double> dsolfinal0_dcoeff(9);
  Array1D<double> dsolfinal1_dcoeff(9);
  Array1D<double> dsolfinal2_dcoeff(9);

  for (int ic = 0; ic < 9; ++ic)
  {
    coeff(ic) += eps;
    TestODE<double> ode_p(coeff);
    ForwardEuler<double> solver_p(ode_p);
    auto sol_hist_p = solver_p.Solve(sol0, ts, tf, nt);
    
    coeff(ic) -= 2*eps;
    TestODE<double> ode_m(coeff);
    ForwardEuler<double> solver_m(ode_m);
    auto sol_hist_m = solver_m.Solve(sol0, ts, tf, nt);

    coeff(ic) += eps;

    dsolfinal0_dcoeff(ic) = (sol_hist_p(nt, 0) - sol_hist_m(nt, 0)) / (2.0*eps);
    dsolfinal1_dcoeff(ic) = (sol_hist_p(nt, 1) - sol_hist_m(nt, 1)) / (2.0*eps);
    dsolfinal2_dcoeff(ic) = (sol_hist_p(nt, 2) - sol_hist_m(nt, 2)) / (2.0*eps);
  }

  // std::cout << dsolfinal0_dcoeff << std::endl;
  // std::cout << dsolfinal1_dcoeff << std::endl;
  // std::cout << dsolfinal2_dcoeff << std::endl;

  const int num_deriv = 3;
  const double tol = 1e-6;

  Array1D<ADVar<double>> coeff_ad(9);
  for (int i = 0; i < 9; ++i)
    coeff_ad(i) = ADVar<double>(coeff(i), num_deriv);

  for (int grp = 0; grp < 3; ++grp)
  {
    for (int id = 0; id < 3; ++id)
      coeff_ad(3*grp + id).deriv(id) = 1.0;

    Array1D<ADVar<double>> sol_ad0(3);
    for (int i = 0; i < 3; ++i)
      sol_ad0(i) = sol0(i);

    TestODE<ADVar<double>> ode_ad(coeff_ad);
    ForwardEuler<ADVar<double>> solver_ad(ode_ad);
    auto sol_hist_ad = solver_ad.Solve(sol_ad0, ts, tf, nt);

    for (int id = 0; id < 3; ++id)
    {
      // std::cout << sol_hist_ad(nt, 0).deriv(id) << ", " << sol_hist_ad(nt, 1).deriv(id) << ", " << sol_hist_ad(nt, 2).deriv(id) << std::endl;
      EXPECT_NEAR(sol_hist_ad(nt, 0).deriv(id), dsolfinal0_dcoeff(3*grp + id), tol);
      EXPECT_NEAR(sol_hist_ad(nt, 1).deriv(id), dsolfinal1_dcoeff(3*grp + id), tol);
      EXPECT_NEAR(sol_hist_ad(nt, 2).deriv(id), dsolfinal2_dcoeff(3*grp + id), tol);
    }

    for (int id = 0; id < 3; ++id)
      coeff_ad(3*grp + id).deriv(id) = 0.0;
  }
}

//----------------------------------------------------------------------------//
TEST( ForwardEuler, CoefficientSensitivityNumDeriv9 )
{
  Array1D<double> sol0(3);
  sol0(0) = 10.0;
  sol0(1) = -3.0;
  sol0(2) = 7.0;

  Array1D<double> coeff(9);
  coeff(0) = -1.0; coeff(1) =  0.2; coeff(2) = -0.1;
  coeff(3) = -0.4; coeff(4) = -3.0; coeff(5) =  0.3;
  coeff(6) =  0.1; coeff(7) = -0.7; coeff(8) = -1.7;
  
  int nt = 10;
  double eps = 1e-9;

  double ts = 0.0, tf = 1.0;

  Array1D<double> dsolfinal0_dcoeff(9);
  Array1D<double> dsolfinal1_dcoeff(9);
  Array1D<double> dsolfinal2_dcoeff(9);

  for (int ic = 0; ic < 9; ++ic)
  {
    coeff(ic) += eps;
    TestODE<double> ode_p(coeff);
    ForwardEuler<double> solver_p(ode_p);
    auto sol_hist_p = solver_p.Solve(sol0, ts, tf, nt);
    
    coeff(ic) -= 2*eps;
    TestODE<double> ode_m(coeff);
    ForwardEuler<double> solver_m(ode_m);
    auto sol_hist_m = solver_m.Solve(sol0, ts, tf, nt);

    coeff(ic) += eps;

    dsolfinal0_dcoeff(ic) = (sol_hist_p(nt, 0) - sol_hist_m(nt, 0)) / (2.0*eps);
    dsolfinal1_dcoeff(ic) = (sol_hist_p(nt, 1) - sol_hist_m(nt, 1)) / (2.0*eps);
    dsolfinal2_dcoeff(ic) = (sol_hist_p(nt, 2) - sol_hist_m(nt, 2)) / (2.0*eps);
  }

  // std::cout << dsolfinal0_dcoeff << std::endl;
  // std::cout << dsolfinal1_dcoeff << std::endl;
  // std::cout << dsolfinal2_dcoeff << std::endl;

  const int num_deriv = 9;
  const double tol = 1e-6;

  Array1D<ADVar<double>> coeff_ad(9);
  for (int i = 0; i < 9; ++i)
    coeff_ad(i) = ADVar<double>(coeff(i), num_deriv);

  for (int id = 0; id < 9; ++id)
    coeff_ad(id).deriv(id) = 1.0;

  Array1D<ADVar<double>> sol_ad0(3);
  for (int i = 0; i < 3; ++i)
    sol_ad0(i) = sol0(i);

  TestODE<ADVar<double>> ode_ad(coeff_ad);
  ForwardEuler<ADVar<double>> solver_ad(ode_ad);
  auto sol_hist_ad = solver_ad.Solve(sol_ad0, ts, tf, nt);

  for (int id = 0; id < 9; ++id)
  {
    // std::cout << sol_hist_ad(nt, 0).deriv(id) << ", " << sol_hist_ad(nt, 1).deriv(id) << ", " << sol_hist_ad(nt, 2).deriv(id) << std::endl;
    EXPECT_NEAR(sol_hist_ad(nt, 0).deriv(id), dsolfinal0_dcoeff(id), tol);
    EXPECT_NEAR(sol_hist_ad(nt, 1).deriv(id), dsolfinal1_dcoeff(id), tol);
    EXPECT_NEAR(sol_hist_ad(nt, 2).deriv(id), dsolfinal2_dcoeff(id), tol);
  }
}