// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <gtest/gtest.h>
#include <autodiffeq/numerics/Complex.hpp>
#include <autodiffeq/linearalgebra/Array1D.hpp>
#include <autodiffeq/linearalgebra/GPUArray1D.cuh>
#include <autodiffeq/linearalgebra/Array3D.hpp>
#include <autodiffeq/linearalgebra/GPUArray3D.cuh>

using namespace autodiffeq;

//----------------------------------------------------------------------------//
TEST( GPUArray3D, Constructor )
{
  {
    GPUArray3D<double> mat(3,5,7);
    EXPECT_EQ(mat.GetDim(0), 3);
    EXPECT_EQ(mat.GetDim(1), 5);
    EXPECT_EQ(mat.GetDim(2), 7);
  }

  {
    GPUArray3D<complex<double>> mat(5, 4, 3, {0.25, -1.0});
    EXPECT_EQ(mat.GetDimensions(), (std::array<int,3>{5,4,3}));
    auto mat_h = mat.CopyToHost();
    for (int i = 0; i < 60; ++i)
    {
      EXPECT_EQ(mat_h[i].real(), 0.25);
      EXPECT_EQ(mat_h[i].imag(), -1.0);
    }
  }

  {
    Array1D<double> vec(60);
    for (int i = 0; i < 60; ++i)
      vec[i] = i*i;
    Array3D<double> mat_cpu(4,5,3, vec.GetDataVector());
    GPUArray3D<double> mat(mat_cpu);
    EXPECT_EQ(mat.GetDimensions(), (std::array<int,3>{4,5,3}));
    auto mat_h = mat.CopyToHost();
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 5; j++)
        for (int k = 0; k < 3; k++)
          EXPECT_EQ(mat_h(i,j,k), mat_cpu(i,j,k));
  }
}
