#include <gtest/gtest.h>
#include <autodiffeq/numerics/ADVar.hpp>

#include <complex>

using namespace autodiffeq;

//############################################################################//
//----------------------------------------------------------------------------//
TEST( ADVar, Constructor )
{
  {
    ADVar<double> var(2, {3, 0, 1});
    EXPECT_EQ(var.value(), 2.0);
    EXPECT_EQ(var.size(), 3u);
    EXPECT_EQ(var.deriv(0), 3.0);
    EXPECT_EQ(var.deriv(1), 0.0);
    EXPECT_EQ(var.deriv(2), 1.0);
  }
  {
    ADVar<float> var(42.0f);
    EXPECT_EQ(var.value(), 42.0f);
    EXPECT_EQ(var.size(), 0u);
  }
  {
    ADVar<std::complex<double>> var({{42.0, -5.0}}, 2);
    EXPECT_EQ(var.value(), std::complex<double>(42.0, -5.0));
    EXPECT_EQ(var.size(), 2);
  }
}