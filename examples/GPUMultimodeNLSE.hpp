#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/numerics/Complex.hpp>
#include <autodiffeq/solver/ODE.hpp>
#include <autodiffeq/linearalgebra/GPUArray1D.cuh>
#include <autodiffeq/linearalgebra/GPUArray2D.cuh>
#include <iostream>
#include <iomanip>

namespace autodiffeq
{

namespace gpu
{


template<typename T>
__global__
void mmnlseRHS(const int num_modes, const int num_time_pts, const DeviceArray2D<double>& beta_mat, 
               const double dt, const DeviceArray1D<T>& sol, DeviceArray1D<T>& rhs)
{
  const int t = blockIdx.x*blockDim.x + threadIdx.x; //time-point index
  const int p = threadIdx.y; //mode index (blockIdx.y is always 0 since only 1 block is launched in this dimension)
  const int offset = num_modes*t + p;
  int soldim = sol.size();

  const int num_deriv_rows = beta_mat.GetNumRows();
  const int beta_size = num_deriv_rows * num_modes;
  constexpr int stencil_size = 5;

  extern __shared__ double array[];
  double* beta = array; // num_deriv_rows x num_modes doubles
  T* sol_stencil_shared = (T*) &beta[beta_size]; // (num_threads * stencil_size * num_modes) variables of type T

  if (t < num_deriv_rows)
    beta[offset] = beta_mat(t,p);

  int shared_offset = threadIdx.x * stencil_size * num_modes;
  T* sol_stencil = sol_stencil_shared + shared_offset;
  if (t < num_time_pts)
  {
    for (int i = 0; i < stencil_size; ++i)
    {
      int t_offset = t + i - 2;
      if (t_offset < 0)
        t_offset = -t_offset; //Mirror data at left boundary
      if (t_offset >= num_time_pts)
        t_offset = 2*num_time_pts - 2 - t_offset; //Mirror data at right boundary

      sol_stencil[num_modes*i + p] = sol[num_modes*t_offset + p];
    }
  }

  __syncthreads();

  if (offset >= soldim)
    return;
  
  // printf("%f, %f, %f, %f\n", beta_mat[0], beta_mat[1], beta_mat[2], beta_mat[3]);
  // printf("%f, %f, %f, %f\n", beta[0], beta[1], beta[2], beta[3]);
  constexpr double inv6 = 1.0/6.0;
  constexpr double inv24 = 1.0/24.0;
  constexpr T imag(0.0, 1.0);
  const double inv_dt = 1.0/dt;
  const double inv_dt2 = inv_dt*inv_dt;
  const double inv_dt3 = inv_dt2*inv_dt;
  const double inv_dt4 = inv_dt3*inv_dt;

  T sol_im2 = sol_stencil[              p];
  T sol_im1 = sol_stencil[num_modes   + p];
  T sol_i   = sol_stencil[num_modes*2 + p];
  T sol_ip1 = sol_stencil[num_modes*3 + p];
  T sol_ip2 = sol_stencil[num_modes*4 + p];

  //Calculate solution time-derivatives using stencil data
  T sol_tderiv1 = 0.5*(sol_ip1 - sol_im1) * inv_dt;
  T sol_tderiv2 = (sol_ip1 - 2.0*sol_i + sol_im1) * inv_dt2;
  T sol_tderiv3 = (0.5*sol_ip2 - sol_ip1 + sol_im1 - 0.5*sol_im2) * inv_dt3;
  T sol_tderiv4 = (sol_ip2 - 4.0*sol_ip1 + 6.0*sol_i - 4.0*sol_im1 + sol_im2) * inv_dt4;

  rhs[offset] = imag*(beta[            p] - beta[        0])*sol_i //(beta0p - beta00)
                    -(beta[  num_modes+p] - beta[num_modes])*sol_tderiv1
              - imag* beta[2*num_modes+p]*0.5               *sol_tderiv2
                    + beta[3*num_modes+p]*inv6              *sol_tderiv3
              + imag* beta[4*num_modes+p]*inv24             *sol_tderiv4;
}


}

template<typename T>
class GPUMultimodeNLSE : public ODE<T>
{
public:

  static_assert(std::is_same<T, complex<double>>::value ||
                std::is_same<T, ADVar<complex<double>>>::value, 
                "Template datatype needs to be complex<double> or ADVar<complex<double>>!");

  using ODE<T>::EvalRHS;

  GPUMultimodeNLSE(const int num_modes, const int num_time_points, 
                   const double tmin, const double tmax, const Array2D<double>& beta_mat) :
    GPUMultimodeNLSE(num_modes, num_time_points, tmin, tmax, beta_mat, 0.0, 0.0, {}, false, false) {}

  GPUMultimodeNLSE(const int num_modes, const int num_time_points, 
                   const double tmin, const double tmax, const Array2D<double>& beta_mat,
                   const double n2, const double omega0, const Array4D<double>& Sk, 
                   const bool is_self_steepening, const bool is_nonlinear = true) : 
    num_modes_(num_modes), num_time_points_(num_time_points), tmin_(tmin), tmax_(tmax),
    dt_((tmax_ - tmin_) / (double) (num_time_points_-1)), beta_mat_(beta_mat), n2_(n2),
    omega0_(omega0), Sk_(Sk), is_self_steepening_(is_self_steepening), is_nonlinear_(is_nonlinear)
  {
    tvec_.resize(num_time_points_);
    tvec_(0) = tmin_;
    for (int step = 1; step < num_time_points_; ++step)
      tvec_(step) = tvec_(step-1) + dt_;

    assert(beta_mat_.GetNumCols() == num_modes);
    if (is_nonlinear_)
      for (int d = 0; d < 4; ++d)
        assert(Sk.GetDim(d) == num_modes);

    // const int max_Ap_tderiv = beta_mat_.GetNumRows()-1;
    // sol_tderiv_.resize(max_Ap_tderiv, num_time_points_); //Stores the time-derivatives (e.g. d/dt, d^2/dt^2 ...) of a particular solution mode

    // if (is_nonlinear_)
    // {
    //   kerr_.resize(num_time_points_);
    //   if (is_self_steepening_)
    //     kerr_tderiv_.resize(num_time_points_);
    // }
  }

  int GetSolutionSize() const { return num_modes_*num_time_points_; }

  void EvalRHS(const GPUArray1D<T>& sol, int step, double z, GPUArray1D<T>& rhs)
  {
    constexpr int threads_per_block = 256;
    dim3 thread_dim(threads_per_block, num_modes_, 1);
    dim3 block_dim((num_time_points_ + thread_dim.x-1) / thread_dim.x, 1, 1);

    constexpr int stencil_size = 5;
    int shared_mem_bytes = beta_mat_.size()*sizeof(double)
                         + (threads_per_block * stencil_size * num_modes_)*sizeof(T);
    //std::cout << "shared mem bytes: " << shared_mem_bytes << std::endl;

    gpu::mmnlseRHS<<<block_dim, thread_dim, shared_mem_bytes>>>(
      num_modes_, num_time_points_, beta_mat_.GetDeviceArray(), dt_,
      sol.GetDeviceArray(), rhs.GetDeviceArray());
    cudaCheckLastError();

/*
    static int iter = 0;
    if (iter == 0)
    {
      auto rhs_cpu = rhs.CopyToHost();
      for (int i = 0; i < num_time_points_; ++i)
      {
        for (int p = 0; p < num_modes_; ++p)
        {
          const auto& v = rhs_cpu(num_modes_*i + p);
          if (v.real() != 0.0 || v.imag() != 0.0)
            std::cout << i << ", " << p << ": " << v.real() << ", " << v.imag() << std::endl;
        }
      }
      exit(0);
    }
*/
    iter++;
  }

  void ComputeKerrNonlinearity(const int p, const Array1D<T>& sol)
  {
    #pragma omp for
    for (int i = 0; i < num_time_points_; ++i) 
    {
      T sum = 0.0;
      for (int q = 0; q < num_modes_; ++q)
      {
        const auto& Aq = sol(q*num_time_points_ + i);
        for (int r = 0; r < num_modes_; ++r)
        {
          const auto& Ar = sol(r*num_time_points_ + i);
          for (int s = 0; s < num_modes_; ++s)
          {
            const auto& As = sol(s*num_time_points_ + i);
            sum += Sk_(p,q,r,s)*Aq*Ar*conj(As);
          }
        }
      }
      kerr_(i) = sum;
    }
  }

  void ComputeTimeDerivativesOrder2(const int mode, const Array1D<T>& sol, Array2D<T>& tderiv)
  {
    const int offset = mode*num_time_points_;
    const int max_deriv = tderiv.GetNumRows();
    assert(max_deriv >= 2 && max_deriv <= 4);

    const double inv_dt2 = 1.0 / (dt_*dt_);

    #pragma omp single
    {
      //First derivative d/dt
      tderiv(0,0) = 0.0;
      tderiv(0,num_time_points_-1) = 0.0;

      //Second derivative d^2/dt^2
      tderiv(1,0) = 2.0*(sol(offset+1) - sol(offset))*inv_dt2;
      tderiv(1,num_time_points_-1) = 2.0*(sol(offset+num_time_points_-2) 
                                        - sol(offset+num_time_points_-1))*inv_dt2;
    }

    #pragma omp for
    for (int i = 1; i < num_time_points_-1; ++i)
    {
      tderiv(0,i) = (sol(offset+i+1) - sol(offset+i-1))/(2.0*dt_); //d/dt
      tderiv(1,i) = (sol(offset+i+1) - 2.0*sol(offset+i) + sol(offset+i-1))*inv_dt2; //d^2/dt^2
    }

    if (max_deriv >= 3)
    {
      const double inv_dt3 = 1.0 / (dt_*dt_*dt_);

      #pragma omp single
      {
        //Third derivative d^3/dt^3
        tderiv(2,0) = 0.0;
        tderiv(2,1) = (0.5*sol(offset+3) - sol(offset+2) 
                        + sol(offset) - 0.5*sol(offset+1))*inv_dt3;
        tderiv(2,num_time_points_-2) = (0.5*sol(offset+num_time_points_-2) - sol(offset+num_time_points_-1) 
                                          + sol(offset+num_time_points_-3) - 0.5*sol(offset+num_time_points_-4))*inv_dt3;
        tderiv(2,num_time_points_-1) = 0.0;
      }

      #pragma omp for
      for (int i = 2; i < num_time_points_-2; ++i)
      {
        tderiv(2,i) = (0.5*sol(offset+i+2) - sol(offset+i+1) 
                         + sol(offset+i-1) - 0.5*sol(offset+i-2))*inv_dt3; //d^3/dt^3
      }
    }

    if (max_deriv >= 4)
    {
      const double inv_dt4 = 1.0 / (dt_*dt_*dt_*dt_);

      #pragma omp single
      {
        //Fourth derivative d^4/dt^4
        tderiv(3,0) = (2.0*sol(offset+2) - 8.0*sol(offset+1) + 6.0*sol(offset))*inv_dt4;
        tderiv(3,1) = (sol(offset+3) - 4.0*sol(offset+2) + 7.0*sol(offset+1) - 4.0*sol(offset))*inv_dt4;
        tderiv(3,num_time_points_-2) = (- 4.0*sol(offset+num_time_points_-1) + 7.0*sol(offset+num_time_points_-2)  
                                        - 4.0*sol(offset+num_time_points_-3) +     sol(offset+num_time_points_-4))*inv_dt4;
        tderiv(3,num_time_points_-1) = (2.0*sol(offset+num_time_points_-3) - 8.0*sol(offset+num_time_points_-2) 
                                      + 6.0*sol(offset+num_time_points_-1))*inv_dt4;
      }

      #pragma omp for
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

    const double inv_12dt = 1.0/(12.0*dt_);
    const double inv_12dt2 = inv_12dt / dt_;

    #pragma omp single
    {
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
    }

    #pragma omp for
    for (int i = 2; i < num_time_points_-2; ++i)
    {
      tderiv(0,i) = (sol(offset+i-2) - 8.0*(sol(offset+i-1) - sol(offset+i+1)) - sol(offset+i+2))*inv_12dt; //d/dt
      tderiv(1,i) = (-sol(offset+i-2) + 16.0*(sol(offset+i-1) + sol(offset+i+1))
                     - 30.0*sol(offset+i) - sol(offset+i+2))*inv_12dt2; //d^2/dt^2
    }

    if (max_deriv >= 3)
    {
      const double inv_8dt3 = 1.0 / (8.0*dt_*dt_*dt_);

      //Third derivative d^3/dt^3
      #pragma omp single
      {
        tderiv(2,0) = 0.0;
        tderiv(2,1) = (sol(offset+2) - 8.0*(sol(offset+1) - sol(offset+3)) 
                      + 13.0*(sol(offset) - sol(offset+2)) - sol(offset+4))*inv_8dt3;
        tderiv(2,2) = (sol(offset+1) - 8.0*(sol(offset) - sol(offset+4)) 
                      + 13.0*(sol(offset+1) - sol(offset+3)) - sol(offset+5))*inv_8dt3;
      }

      #pragma omp for
      for (int i = 3; i < num_time_points_-3; ++i)
      {
        tderiv(2,i) = (sol(offset+i-3) - 8.0*(sol(offset+i-2) - sol(offset+i+2)) 
                       + 13.0*(sol(offset+i-1) - sol(offset+i+1)) - sol(offset+i+3))*inv_8dt3; //d^3/dt^3
      }

      #pragma omp single
      {
        tderiv(2,num_time_points_-3) = (sol(offset+num_time_points_-6) - 8.0*(sol(offset+num_time_points_-5) - sol(offset+num_time_points_-1)) 
                                        + 13.0*(sol(offset+num_time_points_-4) - sol(offset+num_time_points_-2)) - sol(offset+num_time_points_-2))*inv_8dt3;
        tderiv(2,num_time_points_-2) = (sol(offset+num_time_points_-5) - 8.0*(sol(offset+num_time_points_-4) - sol(offset+num_time_points_-2)) 
                                        + 13.0*(sol(offset+num_time_points_-3) - sol(offset+num_time_points_-1)) - sol(offset+num_time_points_-3))*inv_8dt3;
        tderiv(2,num_time_points_-1) = 0.0;
      }
    }

    if (max_deriv >= 4)
    {
      const double inv_6dt4 = 1.0 / (6.0*dt_*dt_*dt_*dt_);

      //Fourth derivative d^4/dt^4
      #pragma omp single
      {
        tderiv(3,0) = (- 2.0*sol(offset+3) + 24.0*sol(offset+2) - 78.0*sol(offset+1) + 56.0*sol(offset))*inv_6dt4;
        tderiv(3,1) = (-      (sol(offset+2) + sol(offset+4)) + 12.0*(sol(offset+1) + sol(offset+3))
                        - 39.0*(sol(offset) + sol(offset+2)) + 56.0* sol(offset+1))*inv_6dt4;
        tderiv(3,2) = (-      (sol(offset+1) + sol(offset+5)) + 12.0*(sol(offset) + sol(offset+4))
                       - 39.0*(sol(offset+1) + sol(offset+3)) + 56.0* sol(offset+2))*inv_6dt4;
      }

      #pragma omp for
      for (int i = 3; i < num_time_points_-3; ++i)
      {
        tderiv(3,i) = (-      (sol(offset+i-3) + sol(offset+i+3)) + 12.0*(sol(offset+i-2) + sol(offset+i+2))
                       - 39.0*(sol(offset+i-1) + sol(offset+i+1)) + 56.0* sol(offset+i))*inv_6dt4; //d^4/dt^4
      }

      #pragma omp single
      {                     
        tderiv(3,num_time_points_-3) = (-      (sol(offset+num_time_points_-6) + sol(offset+num_time_points_-2)) 
                                        + 12.0*(sol(offset+num_time_points_-5) + sol(offset+num_time_points_-1))
                                        - 39.0*(sol(offset+num_time_points_-4) + sol(offset+num_time_points_-2)) 
                                        + 56.0* sol(offset+num_time_points_-3))*inv_6dt4;
        tderiv(3,num_time_points_-2) = (-      (sol(offset+num_time_points_-5) + sol(offset+num_time_points_-3)) 
                                        + 12.0*(sol(offset+num_time_points_-4) + sol(offset+num_time_points_-2))
                                        - 39.0*(sol(offset+num_time_points_-3) + sol(offset+num_time_points_-1)) 
                                        + 56.0* sol(offset+num_time_points_-2))*inv_6dt4;
        tderiv(3,num_time_points_-1) = (-  2.0*sol(offset+num_time_points_-4) + 24.0*sol(offset+num_time_points_-3)
                                        - 78.0*sol(offset+num_time_points_-2) + 56.0*sol(offset+num_time_points_-1))*inv_6dt4;
      }
    }
  }

  Array1D<T> GetInitialSolutionGaussian(const Array1D<double>& Et, const Array1D<double>& t_FWHM, const Array1D<double>& t_center) 
  {
    assert(num_modes_ == (int) Et.size());
    assert(num_modes_ == (int) t_FWHM.size());
    assert(num_modes_ == (int) t_center.size());
    Array1D<T> sol(GetSolutionSize());

    for (int mode = 0; mode < num_modes_; ++mode)
    {
      const double A = std::sqrt(1665.0*Et(mode) / ((double)num_modes_ * t_FWHM(mode) * std::sqrt(M_PI)));
      const double k = -1.665*1.665/(2.0*t_FWHM(mode)*t_FWHM(mode));
      const double& tc = t_center(mode);

      for (int j = 0; j < num_time_points_; ++j)
        sol(j*num_modes_ + mode) = A * std::exp(k*(tvec_(j)-tc)*(tvec_(j)-tc));
    }
    return sol;
  }

protected:
  const int num_modes_;
  const int num_time_points_;
  const double tmin_, tmax_, dt_;
  Array1D<double> tvec_;
  GPUArray2D<double> beta_mat_;
  const double n2_; //[m^2 / W]
  const double omega0_; //[rad/ps]
  const Array4D<double>& Sk_; //Kerr nonlinearity tensor
  static constexpr double c_ = 2.99792458e-4; //[m/ps]
  const bool is_self_steepening_ = false;
  const bool is_nonlinear_ = false;
  // Array2D<T> sol_tderiv_;
  Array1D<T> kerr_;
  // Array1D<T> kerr_tderiv_;
};

}