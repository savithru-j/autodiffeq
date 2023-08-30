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
    dt_((tmax_ - tmin_) / (double) (num_time_points_-1)), beta_mat_(beta_mat)
  {
    tvec_.resize(num_time_points_);
    tvec_(0) = tmin_;
    for (int step = 1; step < num_time_points_; ++step)
      tvec_(step) = tvec_(step-1) + dt_;

    assert(beta_mat_.GetNumCols() == num_modes);
  }

  int GetSolutionSize() const { return num_modes_*num_time_points_; }

  void EvalRHS(const Array1D<T>& sol, int step, double z, Array1D<T>& rhs)
  {
    constexpr std::complex<double> imag(0.0, 1.0);
    const auto& beta00 = beta_mat_(0,0);
    const auto& beta10 = beta_mat_(1,0);

    const int max_Ap_tderiv = beta_mat_.GetNumRows()-1;
    Array2D<T> sol_tderiv(max_Ap_tderiv, num_time_points_); //Stores the time-derivatives (e.g. d/dt, d^2/dt^2 ...) of a particular solution mode
    constexpr double inv6 = 1.0/6.0;
    constexpr double inv24 = 1.0/24.0;

    for (int p = 0; p < num_modes_; ++p)
    {
      const int offset = p*num_time_points_;
      const auto& beta0p = beta_mat_(0,p);
      const auto& beta1p = beta_mat_(1,p);
      const auto& beta2p = beta_mat_(2,p);
      const auto& beta3p = beta_mat_(3,p);
      const auto& beta4p = beta_mat_(4,p);
      ComputeTimeDerivativesOrder2(p, sol, sol_tderiv);

      for (int i = 0; i < num_time_points_; ++i) 
      {
        rhs(offset+i) = imag*(beta0p - beta00)*sol(offset+i)
                            -(beta1p - beta10)*sol_tderiv(0,i) //d/dt
                      - imag* beta2p*0.5      *sol_tderiv(1,i) //d^2/dt^2
                            + beta3p*inv6     *sol_tderiv(2,i) //d^3/dt^3
                      + imag* beta4p*inv24    *sol_tderiv(3,i); //d^4/dt^4
      }
    }
  }

  void ComputeTimeDerivativesOrder2(const int mode, const Array1D<T>& sol, Array2D<T>& tderiv)
  {
    const int offset = mode*num_time_points_;
    const int max_deriv = tderiv.GetNumRows();
    assert(max_deriv >= 2 && max_deriv <= 4);

    const double inv_dt2 = 1.0 / (dt_*dt_);

    //First derivative d/dt
    tderiv(0,0) = 0.0;
    tderiv(0,num_time_points_-1) = 0.0;

    //Second derivative d^2/dt^2
    tderiv(1,0) = 2.0*(sol(offset+1) - sol(offset))*inv_dt2;
    tderiv(1,num_time_points_-1) = 2.0*(sol(offset+num_time_points_-2) 
                                      - sol(offset+num_time_points_-1))*inv_dt2;

    for (int i = 1; i < num_time_points_-1; ++i)
    {
      tderiv(0,i) = (sol(offset+i+1) - sol(offset+i-1))/(2.0*dt_); //d/dt
      tderiv(1,i) = (sol(offset+i+1) - 2.0*sol(offset+i) + sol(offset+i-1))*inv_dt2; //d^2/dt^2
    }

    if (max_deriv >= 3)
    {
      const double inv_dt3 = 1.0 / (dt_*dt_*dt_);

      //Third derivative d^3/dt^3
      tderiv(2,0) = 0.0;
      tderiv(2,1) = (0.5*sol(offset+3) - sol(offset+2) 
                       + sol(offset) - 0.5*sol(offset+1))*inv_dt3;
      tderiv(2,num_time_points_-2) = (0.5*sol(offset+num_time_points_-2) - sol(offset+num_time_points_-1) 
                                        + sol(offset+num_time_points_-3) - 0.5*sol(offset+num_time_points_-4))*inv_dt3;
      tderiv(2,num_time_points_-1) = 0.0;

      for (int i = 2; i < num_time_points_-2; ++i)
      {
        tderiv(2,i) = (0.5*sol(offset+i+2) - sol(offset+i+1) 
                         + sol(offset+i-1) - 0.5*sol(offset+i-2))*inv_dt3; //d^3/dt^3
      }
    }

    if (max_deriv >= 4)
    {
      const double inv_dt4 = 1.0 / (dt_*dt_*dt_*dt_);

      //Fourth derivative d^4/dt^4
      tderiv(3,0) = (2.0*sol(offset+2) - 8.0*sol(offset+1) + 6.0*sol(offset))*inv_dt4;
      tderiv(3,1) = (sol(offset+3) - 4.0*sol(offset+2) + 7.0*sol(offset+1) - 4.0*sol(offset))*inv_dt4;
      tderiv(3,num_time_points_-2) = (- 4.0*sol(offset+num_time_points_-1) + 7.0*sol(offset+num_time_points_-2)  
                                      - 4.0*sol(offset+num_time_points_-3) +     sol(offset+num_time_points_-4))*inv_dt4;
      tderiv(3,num_time_points_-1) = (2.0*sol(offset+num_time_points_-3) - 8.0*sol(offset+num_time_points_-2) 
                                    + 6.0*sol(offset+num_time_points_-1))*inv_dt4;

      for (int i = 2; i < num_time_points_-2; ++i)
      {
        tderiv(3,i) = (sol(offset+i+2) - 4.0*sol(offset+i+1) + 6.0*sol(offset+i)  
                 - 4.0*sol(offset+i-1) +     sol(offset+i-2))*inv_dt4; //d^4/dt^4
      }
    }
  }

  void ComputeTimeDerivativesOrder4(const int mode, const Array1D<T>& sol, Array2D<T>& tderiv)
  {
    const int offset = mode*num_time_points_;
    const int max_deriv = tderiv.GetNumRows();
    assert(max_deriv >= 2 && max_deriv <= 4);

    constexpr double inv_12dt = 1.0/(12.0*dt_);
    constexpr double inv_12dt2 = inv_12dt / dt_;

    //First derivative d/dt
    tderiv(0,0) = 0.0;
    tderiv(0,1) = (sol(offset+1) - 8.0*(sol(offset) - sol(offset+2)) - sol(offset+3))*inv_12dt;
    tderiv(0,num_time_points_-2) = (sol(offset+num_time_points_-4) - 8.0*(sol(offset+num_time_points_-3) - sol(offset+num_time_points_-1)) 
                                   -sol(offset+num_time_points_-2))*inv_12dt;
    tderiv(0,num_time_points_-1) = 0.0;

    //Second derivative d^2/dt^2
    tderiv(1,0) = (-2.0*sol(offset+2) + 32.0*sol(offset+1) - 30.0*sol(offset))*inv_12dt2;
    tderiv(1,1) = (-31.0*sol(offset+1) + 16.0*(sol(offset) + sol(offset+2)) - sol(offset+3))*inv_12dt2;
    tderiv(1,num_time_points_-2) = (-sol(offset+num_time_points_-4) + 16.0*(sol(offset+num_time_points_-3) + sol(offset+num_time_points_-1))
                                    - 31.0*sol(offset+num_time_points_-2))*inv_12dt2;
    tderiv(1,num_time_points_-1) = (-2.0*sol(offset+num_time_points_-3) + 32.0*sol(offset+num_time_points_-2)
                                  - 30.0*sol(offset+num_time_points_-1))*inv_12dt2;

    for (int i = 2; i < num_time_points_-2; ++i)
    {
      tderiv(0,i) = (sol(offset+i-2) - 8.0*(sol(offset+i-1) - sol(offset+i+1)) - sol(offset+i+2))*inv_12dt; //d/dt
      tderiv(1,i) = (-sol(offset+i-2) + 16.0*(sol(offset+i-1) + sol(offset+i+1))
                     - 30.0*sol(offset+i) - sol(offset+i+2))*inv_12dt2; //d^2/dt^2
    }
#if 0 //TODO
    if (max_deriv >= 3)
    {
      const double inv_dt3 = 1.0 / (dt_*dt_*dt_);

      //Third derivative d^3/dt^3
      tderiv(2,0) = 0.0;
      tderiv(2,1) = (0.5*sol(offset+3) - sol(offset+2) 
                       + sol(offset) - 0.5*sol(offset+1))*inv_dt3;
      tderiv(2,num_time_points_-2) = (0.5*sol(offset+num_time_points_-2) - sol(offset+num_time_points_-1) 
                                        + sol(offset+num_time_points_-3) - 0.5*sol(offset+num_time_points_-4))*inv_dt3;
      tderiv(2,num_time_points_-1) = 0.0;

      for (int i = 2; i < num_time_points_-2; ++i)
      {
        tderiv(2,i) = (0.5*sol(offset+i+2) - sol(offset+i+1) 
                         + sol(offset+i-1) - 0.5*sol(offset+i-2))*inv_dt3; //d^3/dt^3
      }
    }

    if (max_deriv >= 4)
    {
      const double inv_dt4 = 1.0 / (dt_*dt_*dt_*dt_);

      //Fourth derivative d^4/dt^4
      tderiv(3,0) = (2.0*sol(offset+2) - 8.0*sol(offset+1) + 6.0*sol(offset))*inv_dt4;
      tderiv(3,1) = (sol(offset+3) - 4.0*sol(offset+2) + 7.0*sol(offset+1) - 4.0*sol(offset))*inv_dt4;
      tderiv(3,num_time_points_-2) = (- 4.0*sol(offset+num_time_points_-1) + 7.0*sol(offset+num_time_points_-2)  
                                      - 4.0*sol(offset+num_time_points_-3) +     sol(offset+num_time_points_-4))*inv_dt4;
      tderiv(3,num_time_points_-1) = (2.0*sol(offset+num_time_points_-3) - 8.0*sol(offset+num_time_points_-2) 
                                    + 6.0*sol(offset+num_time_points_-1))*inv_dt4;

      for (int i = 2; i < num_time_points_-2; ++i)
      {
        tderiv(3,i) = (sol(offset+i+2) - 4.0*sol(offset+i+1) + 6.0*sol(offset+i)  
                 - 4.0*sol(offset+i-1) +     sol(offset+i-2))*inv_dt4; //d^4/dt^4
      }
    }
#endif
  }

  Array1D<T> GetInitialSolutionGaussian(const Array1D<double>& Et, const Array1D<double>& t_FWHM, const Array1D<double>& t_center) 
  {
    assert(num_modes_ == (int) Et.size());
    assert(num_modes_ == (int) t_FWHM.size());
    assert(num_modes_ == (int) t_center.size());
    Array1D<T> sol(GetSolutionSize());

    for (int mode = 0; mode < num_modes_; ++mode)
    {
      const int offset = mode*num_time_points_;
      const double A = std::sqrt(1665.0*Et(mode) / ((double)num_modes_ * t_FWHM(mode) * std::sqrt(M_PI)));
      const double k = -1.665*1.665/(2.0*t_FWHM(mode)*t_FWHM(mode));
      const double& tc = t_center(mode);
      for (int j = 0; j < num_time_points_; ++j)
        sol(offset + j) = A * std::exp(k*(tvec_(j)-tc)*(tvec_(j)-tc));
    }
    return sol;
  }

protected:
  const int num_modes_;
  const int num_time_points_;
  const double tmin_, tmax_, dt_;
  Array1D<double> tvec_;
  const Array2D<double>& beta_mat_;
};

}