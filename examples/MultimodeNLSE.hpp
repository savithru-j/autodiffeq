#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/solver/ODE.hpp>
#include <autodiffeq/linearalgebra/Array1D.hpp>
#include <autodiffeq/linearalgebra/Array2D.hpp>
#include <iostream>
#include <iomanip>

#include <complex>

namespace autodiffeq
{

template<typename T>
class MultimodeNLSE : public ODE<T>
{
public:

  static_assert(std::is_same<T, std::complex<double>>::value ||
                std::is_same<T, ADVar<std::complex<double>>>::value, 
                "Template datatype needs to be std::complex<double> or ADVar<std::complex>!");

  MultimodeNLSE(const int num_modes, const int num_time_points, 
                const double tmin, const double tmax, const Array2D<double>& beta_mat) :
    num_modes_(num_modes), num_time_points_(num_time_points), tmin_(tmin), tmax_(tmax),
    beta_mat_(beta_mat)
  {
    const int nt = num_time_points_-1;
    const double dt = (tmax_ - tmin_) / (double) nt;
    tvec_.resize(num_time_points_);
    tvec_(0) = tmin_;
    for (int step = 1; step <= nt; ++step)
      tvec_(step) = tvec_(step-1) + dt;

    assert(beta_mat_.GetNumCols() == num_modes);
  }

  int GetSolutionSize() const { return num_modes_*num_time_points_; }

  void EvalRHS(Array1D<T>& sol, int step, double z, Array1D<T>& rhs)
  {
    // rhs(0) = -1.0*sol(0) + 1.8*sol(1);
    // rhs(1) =  0.2*sol(0) - 2.0*sol(1);
  }

  Array1D<T> GetInitialSolutionGaussian(T Et, T t_FWHM, T t_center) 
  {
    const T tmp = std::sqrt(1665.0*Et / ((double)num_modes_ * t_FWHM * std::sqrt(M_PI)));

    Array1D<T> sol(GetSolutionSize());

    return sol;
  }

protected:
  const int num_modes_;
  const int num_time_points_;
  const double tmin_, tmax_;
  Array1D<double> tvec_;
  const Array2D<double>& beta_mat_;
};

}