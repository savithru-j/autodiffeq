// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <gtest/gtest.h>
#include <autodiffeq/numerics/ADVar.hpp>

#include <complex>

using namespace autodiffeq;

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
    EXPECT_EQ(var.size(), 2u);
  }
  {
    ADVar<int> var(4, {3, -1, 7, 2});
    auto var2(var);
    EXPECT_EQ(var2.value(), 4);
    EXPECT_EQ(var2.size(), 4u);
    EXPECT_EQ(var2.deriv(0), 3);
    EXPECT_EQ(var2.deriv(1), -1);
    EXPECT_EQ(var2.deriv(2), 7);
    EXPECT_EQ(var2.deriv(3), 2);
  }
}

//----------------------------------------------------------------------------//
TEST( ADVar, Assignment )
{
  {
    ADVar<double> var(0.0, 3);
    var = ADVar<double>(2, {3, 0, 1});
    EXPECT_EQ(var.value(), 2.0);
    EXPECT_EQ(var.size(), 3u);
    EXPECT_EQ(var.deriv(0), 3.0);
    EXPECT_EQ(var.deriv(1), 0.0);
    EXPECT_EQ(var.deriv(2), 1.0);
  }
  {
    ADVar<double> var(2, {3, 0, 1});
    ADVar<double> var2 = 35;
    var = var2;
    EXPECT_EQ(var.value(), 35.0);
    EXPECT_EQ(var.size(), 0u);
  }
  {
    ADVar<double> var(0.0, 3);
    var = 42;
    EXPECT_EQ(var.value(), 42.0);
    EXPECT_EQ(var.size(), 0u);
  }
}

//----------------------------------------------------------------------------//
TEST( ADVar, Unary )
{
  ADVar<double> var0(2, {3, -0.5, 1});

  auto var = +var0;
  EXPECT_EQ(var.value(), 2.0);
  EXPECT_EQ(var.size(), 3u);
  EXPECT_EQ(var.deriv(0), 3.0);
  EXPECT_EQ(var.deriv(1), -0.5);
  EXPECT_EQ(var.deriv(2), 1.0);

  var = -var0;
  EXPECT_EQ(var.value(), -2.0);
  EXPECT_EQ(var.size(), 3u);
  EXPECT_EQ(var.deriv(0), -3.0);
  EXPECT_EQ(var.deriv(1), 0.5);
  EXPECT_EQ(var.deriv(2), -1.0);
}

//----------------------------------------------------------------------------//
TEST( ADVar, BinaryAccumulation )
{
  ADVar<double> var0(2, {3, -4});
  ADVar<double> var1(7, {-2, 5});

  var0 += var1;
  EXPECT_EQ(var0.value(), 9.0);
  EXPECT_EQ(var0.size(), 2u);
  EXPECT_EQ(var0.deriv(0), 1.0);
  EXPECT_EQ(var0.deriv(1), 1.0);

  var0 -= var1;
  EXPECT_EQ(var0.value(), 2.0);
  EXPECT_EQ(var0.size(), 2u);
  EXPECT_EQ(var0.deriv(0), 3.0);
  EXPECT_EQ(var0.deriv(1), -4.0);

  var0 *= var1;
  EXPECT_EQ(var0.value(), 14.0);
  EXPECT_EQ(var0.size(), 2u);
  EXPECT_EQ(var0.deriv(0), 17.0);
  EXPECT_EQ(var0.deriv(1), -18.0);

  var0 /= var1;
  EXPECT_DOUBLE_EQ(var0.value(), 2.0);
  EXPECT_EQ(var0.size(), 2u);
  EXPECT_DOUBLE_EQ(var0.deriv(0), 3.0);
  EXPECT_DOUBLE_EQ(var0.deriv(1), -4.0);

  var0 += 9.0;
  EXPECT_DOUBLE_EQ(var0.value(), 11.0);
  EXPECT_EQ(var0.size(), 2u);
  EXPECT_DOUBLE_EQ(var0.deriv(0), 3.0);
  EXPECT_DOUBLE_EQ(var0.deriv(1), -4.0);

  var0 -= 9.0;
  EXPECT_DOUBLE_EQ(var0.value(), 2.0);
  EXPECT_EQ(var0.size(), 2u);
  EXPECT_DOUBLE_EQ(var0.deriv(0), 3.0);
  EXPECT_DOUBLE_EQ(var0.deriv(1), -4.0);

  var0 *= 2.0;
  EXPECT_DOUBLE_EQ(var0.value(), 4.0);
  EXPECT_EQ(var0.size(), 2u);
  EXPECT_DOUBLE_EQ(var0.deriv(0), 6.0);
  EXPECT_DOUBLE_EQ(var0.deriv(1), -8.0);

  var0 /= 2;
  EXPECT_DOUBLE_EQ(var0.value(), 2.0);
  EXPECT_EQ(var0.size(), 2u);
  EXPECT_DOUBLE_EQ(var0.deriv(0), 3.0);
  EXPECT_DOUBLE_EQ(var0.deriv(1), -4.0);
}

//----------------------------------------------------------------------------//
TEST( ADVar, Binary )
{
  ADVar<double> var0(2, {3, -4});
  ADVar<double> var1(7, {-2, 5});

  {
    auto var = var0 + var1;
    EXPECT_EQ(var.value(), 9.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_EQ(var.deriv(0), 1.0);
    EXPECT_EQ(var.deriv(1), 1.0);

    var = var0 + 3.0;
    EXPECT_EQ(var.value(), 5.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_EQ(var.deriv(0), 3.0);
    EXPECT_EQ(var.deriv(1), -4.0);

    var = -1.0 + var0;
    EXPECT_EQ(var.value(), 1.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_EQ(var.deriv(0), 3.0);
    EXPECT_EQ(var.deriv(1), -4.0);
  }
  {
    auto var = var0 - var1;
    EXPECT_EQ(var.value(), -5.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_EQ(var.deriv(0), 5.0);
    EXPECT_EQ(var.deriv(1), -9.0);

    var = var0 - 3.0;
    EXPECT_EQ(var.value(), -1.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_EQ(var.deriv(0), 3.0);
    EXPECT_EQ(var.deriv(1), -4.0);

    var = 10.0 - var0;
    EXPECT_EQ(var.value(), 8.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_EQ(var.deriv(0), -3.0);
    EXPECT_EQ(var.deriv(1), 4.0);

    var = 1.0 - var0 + var1;
    EXPECT_EQ(var.value(), 6.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_EQ(var.deriv(0), -5.0);
    EXPECT_EQ(var.deriv(1), 9.0);
  }
  {
    auto var = var0 * var1;
    EXPECT_EQ(var.value(), 14.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_EQ(var.deriv(0), 17.0);
    EXPECT_EQ(var.deriv(1), -18.0);

    var = var0 * 3.0;
    EXPECT_EQ(var.value(), 6.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_EQ(var.deriv(0), 9.0);
    EXPECT_EQ(var.deriv(1), -12.0);

    var = -3.0 * var0;
    EXPECT_EQ(var.value(), -6.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_EQ(var.deriv(0), -9.0);
    EXPECT_EQ(var.deriv(1), 12.0);
  }
  {
    auto var = var0 / var1;
    EXPECT_DOUBLE_EQ(var.value(), 2.0/7.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_DOUBLE_EQ(var.deriv(0), 25.0/49.0);
    EXPECT_DOUBLE_EQ(var.deriv(1), -38.0/49.0);

    var = var0 / 5.0;
    EXPECT_DOUBLE_EQ(var.value(), 2.0/5.0);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_DOUBLE_EQ(var.deriv(0), 3.0/5.0);
    EXPECT_DOUBLE_EQ(var.deriv(1), -4.0/5.0);

    var = 5.0 / var0;
    EXPECT_DOUBLE_EQ(var.value(), 2.5);
    EXPECT_EQ(var.size(), 2u);
    EXPECT_DOUBLE_EQ(var.deriv(0), -3.75);
    EXPECT_DOUBLE_EQ(var.deriv(1), 5.0);
  }
}

//----------------------------------------------------------------------------//
TEST( ADVar, Relational )
{
  ADVar<double> var0(2, {3, -4});
  ADVar<double> var1(2, {-5, 1});
  ADVar<double> var2(7, {-2, 5});

  EXPECT_TRUE(var0 == var1);
  EXPECT_TRUE(var0 != var2);
  EXPECT_TRUE(var0 <= var1);
  EXPECT_TRUE(var0 >= var1);
  EXPECT_TRUE(var0 < var2);
  EXPECT_TRUE(var2 > var0);

  EXPECT_TRUE(var0 == 2.0);
  EXPECT_TRUE(2.0 == var0);
  EXPECT_TRUE(var0 != 3.0);
  EXPECT_TRUE(3.0 != var0);
  EXPECT_TRUE(var0 <= 2.0);
  EXPECT_TRUE(2.0 <= var0);
  EXPECT_TRUE(var0 >= 2.0);
  EXPECT_TRUE(2.0 >= var0);
  EXPECT_TRUE(var0 < 10.0);
  EXPECT_TRUE(-3.0 < var0);
  EXPECT_TRUE(var2 > 1.0);
  EXPECT_TRUE(42.0 > var2);
}

//----------------------------------------------------------------------------//
TEST( ADVar, Trigonometric )
{
  ADVar<double> x(2, {3, -4});
  ADVar<double> y(0.6, {-2, 5});
  ADVar<double> v(3.0);

  {
    auto z = cos(x);
    EXPECT_DOUBLE_EQ(z.value(), cos(2.0));
    EXPECT_DOUBLE_EQ(z.deriv(0), -sin(2.0)*3.0);
    EXPECT_DOUBLE_EQ(z.deriv(1), -sin(2.0)*-4.0);

    z = cos(v);
    EXPECT_DOUBLE_EQ(z.value(), cos(3.0));
    EXPECT_DOUBLE_EQ(z.size(), 0);
  }
  {
    auto z = sin(x);
    EXPECT_DOUBLE_EQ(z.value(), sin(2.0));
    EXPECT_DOUBLE_EQ(z.deriv(0), cos(2.0)*3.0);
    EXPECT_DOUBLE_EQ(z.deriv(1), cos(2.0)*-4.0);
  }
  {
    auto z = tan(x);
    EXPECT_DOUBLE_EQ(z.value(), tan(2.0));
    EXPECT_DOUBLE_EQ(z.deriv(0), 1.0/(cos(2.0)*cos(2.0))*3.0);
    EXPECT_DOUBLE_EQ(z.deriv(1), 1.0/(cos(2.0)*cos(2.0))*-4.0);
  }
  {
    auto z = acos(y);
    EXPECT_DOUBLE_EQ(z.value(), acos(0.6));
    EXPECT_DOUBLE_EQ(z.deriv(0), -1.0/sqrt(1.0 - 0.6*0.6) * -2.0);
    EXPECT_DOUBLE_EQ(z.deriv(1), -1.0/sqrt(1.0 - 0.6*0.6) * 5.0);
  }
  {
    auto z = asin(y);
    EXPECT_DOUBLE_EQ(z.value(), asin(0.6));
    EXPECT_DOUBLE_EQ(z.deriv(0), 1.0/sqrt(1.0 - 0.6*0.6) * -2.0);
    EXPECT_DOUBLE_EQ(z.deriv(1), 1.0/sqrt(1.0 - 0.6*0.6) * 5.0);
  }
  {
    auto z = atan(y);
    EXPECT_DOUBLE_EQ(z.value(), atan(0.6));
    EXPECT_DOUBLE_EQ(z.deriv(0), 1.0/(1.0 + 0.6*0.6) * -2.0);
    EXPECT_DOUBLE_EQ(z.deriv(1), 1.0/(1.0 + 0.6*0.6) * 5.0);
  }
  {
    auto z = atan2(y,x);
    EXPECT_DOUBLE_EQ(z.value(), atan2(0.6, 2.0));
    EXPECT_DOUBLE_EQ(z.deriv(0), 1.0/(2.0*2.0 + 0.6*0.6) * (2.0*-2.0 - 0.6*3.0));
    EXPECT_DOUBLE_EQ(z.deriv(1), 1.0/(2.0*2.0 + 0.6*0.6) * (2.0* 5.0 - 0.6*-4.0));
  }

}