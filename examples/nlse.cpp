#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/solver/ForwardEuler.hpp>
#include <autodiffeq/linearalgebra/Array2D.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "MultimodeNLSE.hpp"

using namespace autodiffeq;

int main()
{
  using Complex = std::complex<double>;
  using ComplexAD = ADVar<Complex>;

  int num_modes = 2;
  int num_time_points = 8192; //8192;
  int sol_dim = num_modes*num_time_points;

  Array2D<double> beta_mat_5x8 = 
    {{ 0.00000000e+00, -5.31830434e+03, -5.31830434e+03, -1.06910098e+04, -1.06923559e+04, -1.07426928e+04, -2.16527479e+04, -3.26533894e+04},
     { 0.00000000e+00,  1.19405403e-01,  1.19405403e-01,  2.44294517e-01,  2.43231165e-01,  2.44450336e-01,  5.03940297e-01,  6.85399771e-01},
     {-2.80794698e-02, -2.82196091e-02, -2.82196091e-02, -2.83665602e-02, -2.83665268e-02, -2.83659537e-02, -2.86356131e-02, -2.77985757e-02},
     { 1.51681819e-04,  1.52043264e-04,  1.52043264e-04,  1.52419435e-04,  1.52419402e-04,  1.52414636e-04,  1.52667612e-04,  1.39629075e-04},
     {-4.95686317e-07, -4.97023237e-07, -4.97023237e-07, -4.98371203e-07, -4.98371098e-07, -4.98311743e-07, -4.94029250e-07, -3.32523455e-07}};

  const int max_Ap_tderiv = 4;
  Array2D<double> beta_mat(max_Ap_tderiv+1, num_modes);
  for (int i = 0; i <= max_Ap_tderiv; ++i)
    for (int j = 0; j < num_modes; ++j)
      beta_mat(i,j) = beta_mat_5x8(i,j);

  double tmin = -40, tmax = 40;
  MultimodeNLSE<Complex> ode(num_modes, num_time_points, tmin, tmax, beta_mat);

  Array1D<double> Et = {9.0, 8.0}; //nJ (in range [6,30] nJ)
  Array1D<double> t_FWHM = {0.1, 0.2}; //ps (in range [0.05, 0.5] ps)
  Array1D<double> t_center = {0.0, 0.0}; //ps
  Array1D<Complex> sol0 = ode.GetInitialSolutionGaussian(Et, t_FWHM, t_center);

  // for (int i = 0; i < num_time_points; ++i)
  //     std::cout << std::abs(sol0(i)) << ", " << std::abs(sol0(num_time_points + i)) << std::endl;

  double z_start = 0, z_end = 0.05; //[m]
  int nz = 1000;
  auto sol_hist = ForwardEuler::Solve(ode, sol0, z_start, z_end, nz);

  for (int mode = 0; mode < num_modes; ++mode)
  {
    std::string filename = "intensity_mode" + std::to_string(mode) + ".txt";
    std::cout << "Writing solution file: " << filename << std::endl;
    std::ofstream f(filename, std::ios_base::out | std::ios::binary);
    f << std::setprecision(6) << std::scientific;
    const int offset = mode*num_time_points;
    for (int i = 0; i < sol_hist.GetNumSteps(); ++i)
    {
      for (int j = 0; j < num_time_points-1; ++j)
        f << std::abs(sol_hist(i, offset + j)) << ", ";
      f << std::abs(sol_hist(i, offset + num_time_points-1)) << std::endl;
    }
    f.close();
  }


  // std::cout << sol_hist << std::endl;
}