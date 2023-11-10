// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <gtest/gtest.h>
#include <autodiffeq/numerics/Complex.hpp>
#include <autodiffeq/linearalgebra/Array1D.hpp>
#include <autodiffeq/linearalgebra/GPUArray1D.cuh>
#include <autodiffeq/linearalgebra/Array4D.hpp>
#include <autodiffeq/linearalgebra/GPUArray4D.cuh>

using namespace autodiffeq;

//----------------------------------------------------------------------------//
TEST( GPUArray4D, Constructor )
{
  {
    GPUArray4D<double> mat(3,5,2,4);
    EXPECT_EQ(mat.GetDim(0), 3);
    EXPECT_EQ(mat.GetDim(1), 5);
    EXPECT_EQ(mat.GetDim(2), 2);
    EXPECT_EQ(mat.GetDim(3), 4);
  }

  {
    GPUArray4D<complex<double>> mat(5, 4, 3, 2, {0.25, -1.0});
    EXPECT_EQ(mat.GetDimensions(), (std::array<int,4>{5,4,3,2}));
    auto mat_h = mat.CopyToHost();
    for (int i = 0; i < 120; ++i)
    {
      EXPECT_EQ(mat_h[i].real(), 0.25);
      EXPECT_EQ(mat_h[i].imag(), -1.0);
    }
  }

  {
    Array1D<double> vec(120);
    for (int i = 0; i < 120; ++i)
      vec[i] = i*i;
    Array4D<double> mat_cpu(4,5,2,3, vec.GetDataVector());
    GPUArray4D<double> mat(mat_cpu);
    EXPECT_EQ(mat.GetDimensions(), (std::array<int,4>{4,5,2,3}));
    auto mat_h = mat.CopyToHost();
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 5; j++)
        for (int k = 0; k < 2; k++)
          for (int l = 0; l < 3; l++)
            EXPECT_EQ(mat_h(i,j,k,l), mat_cpu(i,j,k,l));
  }
}
