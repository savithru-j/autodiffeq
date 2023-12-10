// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <cassert>
#include <initializer_list>
#define _USE_MATH_DEFINES //For MSVC
#include <cmath>
#include <limits>
#include <ostream>

#include "Complex.hpp"

#ifdef ENABLE_CUDA
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

namespace autodiffeq
{

// ADVarS : Auto-diff variable for storing a value and a statically-sized vector of derivatives
template<unsigned int N_, typename T>
class ADVarS;

using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;

using std::cosh;
using std::sinh;
using std::tanh;

using std::exp;
using std::expm1;
using std::log;
using std::log10;
using std::log1p;

using std::erf;
using std::erfc;

using std::pow;
using std::sqrt;

using std::ceil;
using std::floor;
using std::abs;
using std::fabs;

template<unsigned int N_, typename T>
class ADVarS
{
public:
  static const unsigned int N = N_;
  static_assert(N_ > 0);

  ADVarS() = default;

  HOST DEVICE ADVarS(const T& v, const T d[], const int num_deriv)
  {
    assert(N_ == num_deriv);
    v_ = v;
    for (unsigned int i = 0; i < N_; ++i)
      d_[i] = d[i];
  }

  HOST DEVICE ADVarS(const T& v, const std::initializer_list<T>& d)
  {
    v_ = v;
    assert(N_ == d.size());
    auto row = d.begin();
    for (unsigned int i = 0; i < N_; ++i)
      d_[i] = *(row++);
  }

  HOST DEVICE ADVarS(const T& v) : v_(v) 
  {
    for (unsigned int i = 0; i < N_; ++i)
      d_[i] = T(0);
  }

  HOST DEVICE ADVarS(const ADVarS& var) : v_(var.v_)
  {
    for (unsigned int i = 0; i < N_; ++i)
      d_[i] = var.d_[i];
  }

  ~ADVarS() = default;

  inline HOST DEVICE unsigned int size() const { return N_; }

  // value accessors
  inline HOST DEVICE T& value() { return v_; }
  inline HOST DEVICE const T& value() const { return v_; }

  // derivative accessors
  inline HOST DEVICE T& deriv(int i = 0) 
  {
    assert(i >= 0 && i < (int) N_); 
    return d_[i]; 
  }
  inline HOST DEVICE T deriv(int i = 0) const 
  { 
    assert(i >= 0 && i < (int) N_); 
    return d_[i]; 
  }

  // assignment operators
  HOST DEVICE ADVarS& operator=(const ADVarS& var);
  HOST DEVICE ADVarS& operator=(const T& v);

  // unary operators
  HOST DEVICE const ADVarS& operator+() const;
  HOST DEVICE const ADVarS  operator-() const;

  // binary accumulation operators
  HOST DEVICE ADVarS& operator+=(const ADVarS& var);
  HOST DEVICE ADVarS& operator+=(const T& v);
  HOST DEVICE ADVarS& operator-=(const ADVarS& var);
  HOST DEVICE ADVarS& operator-=(const T& v);
  HOST DEVICE ADVarS& operator*=(const ADVarS& var);
  HOST DEVICE ADVarS& operator*=(const T& v);
  HOST DEVICE ADVarS& operator/=(const ADVarS& var);
  HOST DEVICE ADVarS& operator/=(const T& v);

  // binary operators
  template<unsigned int N, typename U> friend ADVarS<N,U> operator+( const ADVarS<N,U>& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U, typename V> friend ADVarS<N,U> operator+( const ADVarS<N,U>& a, const V& b);
  template<unsigned int N, typename U, typename V> friend ADVarS<N,U> operator+( const V& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend ADVarS<N,U> operator-( const ADVarS<N,U>& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U, typename V> friend ADVarS<N,U> operator-( const ADVarS<N,U>& a, const V& b);
  template<unsigned int N, typename U, typename V> friend ADVarS<N,U> operator-( const V& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend ADVarS<N,U> operator*( const ADVarS<N,U>& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U, typename V> friend ADVarS<N,U> operator*( const ADVarS<N,U>& a, const V& b);
  template<unsigned int N, typename U, typename V> friend ADVarS<N,U> operator*( const V& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend ADVarS<N,U> operator/( const ADVarS<N,U>& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U, typename V> friend ADVarS<N,U> operator/( const ADVarS<N,U>& a, const V& b);
  template<unsigned int N, typename U, typename V> friend ADVarS<N,U> operator/( const V& a, const ADVarS<N,U>& b);

    // relational operators
  template<unsigned int N, typename U> friend bool operator==(const ADVarS<N,U>& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend bool operator==(const ADVarS<N,U>& a, const U& b);
  template<unsigned int N, typename U> friend bool operator==(const U& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend bool operator!=(const ADVarS<N,U>& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend bool operator!=(const ADVarS<N,U>& a, const U& b);
  template<unsigned int N, typename U> friend bool operator!=(const U& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend bool operator>(const ADVarS<N,U>& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend bool operator>(const ADVarS<N,U>& a, const U& b);
  template<unsigned int N, typename U> friend bool operator>(const U& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend bool operator<(const ADVarS<N,U>& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend bool operator<(const ADVarS<N,U>& a, const U& b);
  template<unsigned int N, typename U> friend bool operator<(const U& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend bool operator>=(const ADVarS<N,U>& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend bool operator>=(const ADVarS<N,U>& a, const U& b);
  template<unsigned int N, typename U> friend bool operator>=(const U& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend bool operator<=(const ADVarS<N,U>& a, const ADVarS<N,U>& b);
  template<unsigned int N, typename U> friend bool operator<=(const ADVarS<N,U>& a, const U& b);
  template<unsigned int N, typename U> friend bool operator<=(const U& a, const ADVarS<N,U>& b);

  // trigonometric functions
  template<unsigned int N, typename U> friend ADVarS<N,U> cos(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> sin(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> tan(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> acos( const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> asin( const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> atan( const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> atan2( const ADVarS<N,U>& y, const ADVarS<N,U>& x);

  // hyperbolic functions
  template<unsigned int N, typename U> friend ADVarS<N,U> cosh(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> sinh(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> tanh(const ADVarS<N,U>& var);

  // exponential and logarithm functions
  template<unsigned int N, typename U> friend ADVarS<N,U> exp(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> expm1(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> log(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> log10(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> log1p(const ADVarS<N,U>& var);

  // error functions
  template<unsigned int N, typename U> friend ADVarS<N,U> erf(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> erfc(const ADVarS<N,U>& var);

  // power functions
  template<unsigned int N, typename U> friend ADVarS<N,U> pow( const ADVarS<N,U>& x, const ADVarS<N,U>& y);
  template<unsigned int N, typename U> friend ADVarS<N,U> pow( const ADVarS<N,U>& x, const U& y);
  template<unsigned int N, typename U> friend ADVarS<N,U> pow( const U& x, const ADVarS<N,U>& y);
  template<unsigned int N, typename U> friend ADVarS<N,U> sqrt(const ADVarS<N,U>& x);

  // rounding and absolute functions
  template<unsigned int N, typename U> friend ADVarS<N,U> ceil(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> floor(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> abs(const ADVarS<N,U>& var);
  template<unsigned int N, typename U> friend ADVarS<N,U> fabs(const ADVarS<N,U>& var);

  // complex functions
  template<unsigned int N, typename U> friend ADVarS<N,complex<U>> abs(const ADVarS<N,complex<U>>& var);
  template<unsigned int N, typename U> friend ADVarS<N,complex<U>> conj(const ADVarS<N,complex<U>>& var);

protected:

  T v_; //value
  T d_[N_]; //derivatives
};

// assignment
template<unsigned int N, typename T>
ADVarS<N,T>& ADVarS<N,T>::operator=(const ADVarS& var)
{
  //Do nothing if assigning self to self
  if ( &var == this ) return *this;

  v_ = var.v_;
  for (unsigned int i = 0; i < N; i++)
    d_[i] = var.d_[i];
  return *this;
}

template<unsigned int N, typename T>
ADVarS<N,T>& ADVarS<N,T>::operator=(const T& v)
{
  v_ = v;
  for (unsigned int i = 0; i < N; i++)
    d_[i] = 0;
  return *this;
}

// unary operators
template<unsigned int N, typename T>
const ADVarS<N,T>& ADVarS<N,T>::operator+() const
{
  return *this;
}

template<unsigned int N, typename T>
const ADVarS<N,T> ADVarS<N,T>::operator-() const
{
  ADVarS<N,T> c;
  c.v_ = -v_;
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = -d_[i];
  return c;
}

// binary accumulation operators
template<unsigned int N, typename T>
ADVarS<N,T>& ADVarS<N,T>::operator+=(const ADVarS& var)
{
  v_ += var.v_;
  for (unsigned int i = 0; i < N; i++)
    d_[i] += var.d_[i];
  return *this;
}

template<unsigned int N, typename T>
ADVarS<N,T>& ADVarS<N,T>::operator+=(const T& v)
{
  v_ += v;
  return *this;
}

template<unsigned int N, typename T>
ADVarS<N,T>& ADVarS<N,T>::operator-=(const ADVarS& var)
{
  v_ -= var.v_;
  for (unsigned int i = 0; i < N; i++)
    d_[i] -= var.d_[i];
  return *this;
}

template<unsigned int N, typename T>
ADVarS<N,T>& ADVarS<N,T>::operator-=(const T& v)
{
  v_ -= v;
  return *this;
}

template<unsigned int N, typename T>
ADVarS<N,T>& ADVarS<N,T>::operator*=(const ADVarS& var)
{
  for (unsigned int i = 0; i < N; i++)
    d_[i] = v_*var.d_[i] + d_[i]*var.v_;
  v_ *= var.v_;
  return *this;
}

template<unsigned int N, typename T>
ADVarS<N,T>& ADVarS<N,T>::operator*=(const T& v)
{
  for (unsigned int i = 0; i < N; i++)
    d_[i] *= v;
  v_ *= v;
  return *this;
}

template<unsigned int N, typename T>
ADVarS<N,T>& ADVarS<N,T>::operator/=(const ADVarS& var)
{
  T tmp = 1.0/(var.v_*var.v_);
  for (unsigned int i = 0; i < N; i++)
    d_[i] = (d_[i]*var.v_ - v_*var.d_[i]) * tmp;
  v_ /= var.v_;
  return *this;
}

template<unsigned int N, typename T>
ADVarS<N,T>& ADVarS<N,T>::operator/=(const T& v)
{
  T tmp = 1.0/v;
  for (unsigned int i = 0; i < N; i++)
    d_[i] *= tmp;
  v_ *= tmp;
  return *this;
}

//binary operators
template<unsigned int N, typename T>
HOST DEVICE ADVarS<N,T> operator+(const ADVarS<N,T>& a, const ADVarS<N,T>& b)
{
  ADVarS<N,T> c;
  c.v_ = a.v_ + b.v_;
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = a.d_[i] + b.d_[i];
  return c;
}

template<unsigned int N, typename T, typename U>
HOST DEVICE ADVarS<N,T> operator+(const ADVarS<N,T>& a, const U& b)
{
  ADVarS<N,T> c;
  c.v_ = a.v_ + b;
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = a.d_[i];
  return c;
}

template<unsigned int N, typename T, typename U>
HOST DEVICE ADVarS<N,T> operator+(const U& a, const ADVarS<N,T>& b)
{
  ADVarS<N,T> c;
  c.v_ = a + b.v_;
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = b.d_[i];
  return c;
}

template<unsigned int N, typename T>
HOST DEVICE ADVarS<N,T> operator-(const ADVarS<N,T>& a, const ADVarS<N,T>& b)
{
  ADVarS<N,T> c;
  c.v_ = a.v_ - b.v_;
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = a.d_[i] - b.d_[i];
  return c;
}

template<unsigned int N, typename T, typename U>
HOST DEVICE ADVarS<N,T> operator-(const ADVarS<N,T>& a, const U& b)
{
  ADVarS<N,T> c;
  c.v_ = a.v_ - b;
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = a.d_[i];
  return c;
}

template<unsigned int N, typename T, typename U>
HOST DEVICE ADVarS<N,T> operator-(const U& a, const ADVarS<N,T>& b)
{
  ADVarS<N,T> c;
  c.v_ = a - b.v_;
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = -b.d_[i];
  return c;
}

template<unsigned int N, typename T>
HOST DEVICE ADVarS<N,T> operator*(const ADVarS<N,T>& a, const ADVarS<N,T>& b)
{
  ADVarS<N,T> c;
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = a.d_[i]*b.v_ + a.v_*b.d_[i];
  c.v_ = a.v_ * b.v_;
  return c;
}

template<unsigned int N, typename T, typename U>
HOST DEVICE ADVarS<N,T> operator*(const ADVarS<N,T>& a, const U& b)
{
  ADVarS<N,T> c;
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = a.d_[i]*b;
  c.v_ = a.v_ * b;
  return c;
}

template<unsigned int N, typename T, typename U>
HOST DEVICE ADVarS<N,T> operator*(const U& a, const ADVarS<N,T>& b)
{
  ADVarS<N,T> c;
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = a*b.d_[i];
  c.v_ = a*b.v_;
  return c;
}

template<unsigned int N, typename T>
HOST DEVICE ADVarS<N,T> operator/(const ADVarS<N,T>& a, const ADVarS<N,T>& b)
{
  ADVarS<N,T> c;
  T tmp = 1.0 / (b.v_ * b.v_);
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = (a.d_[i]*b.v_ - a.v_*b.d_[i]) * tmp;
  c.v_ = a.v_ / b.v_;
  return c;
}

template<unsigned int N, typename T, typename U>
HOST DEVICE ADVarS<N,T> operator/(const ADVarS<N,T>& a, const U& b)
{
  ADVarS<N,T> c;
  T tmp = 1.0 / b;
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = a.d_[i] * tmp;
  c.v_ = a.v_ * tmp;
  return c;
}

template<unsigned int N, typename T, typename U>
HOST DEVICE ADVarS<N,T> operator/(const U& a, const ADVarS<N,T>& b)
{
  ADVarS<N,T> c;
  T tmp = 1.0 / (b.v_ * b.v_);
  for (unsigned int i = 0; i < N; i++)
    c.d_[i] = -a * b.d_[i] * tmp;
  c.v_ = a / b.v_;
  return c;
}

// relational operators

template<unsigned int N, typename T>
HOST DEVICE bool operator==(const ADVarS<N,T>& a, const ADVarS<N,T>& b) { return a.v_ == b.v_; }

template<unsigned int N, typename T>
HOST DEVICE bool operator==(const ADVarS<N,T>& a, const T& b) { return a.v_ == b; }

template<unsigned int N, typename T>
HOST DEVICE bool operator==(const T& a, const ADVarS<N,T>& b) { return a == b.v_; }

template<unsigned int N, typename T>
HOST DEVICE bool operator!=(const ADVarS<N,T>& a, const ADVarS<N,T>& b) { return a.v_ != b.v_; }

template<unsigned int N, typename T>
HOST DEVICE bool operator!=(const ADVarS<N,T>& a, const T& b) { return a.v_ != b; }

template<unsigned int N, typename T>
HOST DEVICE bool operator!=(const T& a, const ADVarS<N,T>& b) { return a != b.v_; }

template<unsigned int N, typename T>
HOST DEVICE bool operator>(const ADVarS<N,T>& a, const ADVarS<N,T>& b) { return a.v_ > b.v_; }

template<unsigned int N, typename T>
HOST DEVICE bool operator>(const ADVarS<N,T>& a, const T& b) { return a.v_ > b; }

template<unsigned int N, typename T>
HOST DEVICE bool operator>(const T& a, const ADVarS<N,T>& b) { return a > b.v_; }

template<unsigned int N, typename T>
HOST DEVICE bool operator<(const ADVarS<N,T>& a, const ADVarS<N,T>& b) { return a.v_ < b.v_; }

template<unsigned int N, typename T>
HOST DEVICE bool operator<(const ADVarS<N,T>& a, const T& b) { return a.v_ < b; }

template<unsigned int N, typename T>
HOST DEVICE bool operator<(const T& a, const ADVarS<N,T>& b) { return a < b.v_; }

template<unsigned int N, typename T>
HOST DEVICE bool operator>=(const ADVarS<N,T>& a, const ADVarS<N,T>& b) { return a.v_ >= b.v_; }

template<unsigned int N, typename T>
HOST DEVICE bool operator>=(const ADVarS<N,T>& a, const T& b) { return a.v_ >= b; }

template<unsigned int N, typename T>
HOST DEVICE bool operator>=(const T& a, const ADVarS<N,T>& b) { return a >= b.v_; }

template<unsigned int N, typename T>
HOST DEVICE bool operator<=(const ADVarS<N,T>& a, const ADVarS<N,T>& b) { return a.v_ <= b.v_; }

template<unsigned int N, typename T>
HOST DEVICE bool operator<=(const ADVarS<N,T>& a, const T& b) { return a.v_ <= b; }

template<unsigned int N, typename T>
HOST DEVICE bool operator<=(const T& a, const ADVarS<N,T>& b) { return a <= b.v_; }

#define ADVAR_FUNC(NAME, VALUE, DERIV) \
template<unsigned int N, typename T> \
HOST DEVICE ADVarS<N,T> \
NAME(const ADVarS<N,T>& var) \
{ \
  T tmp = DERIV; \
  ADVarS<N,T> z; \
  z.v_ = VALUE; \
  for (unsigned int i = 0; i < N; i++) \
    z.d_[i] = tmp*var.d_[i]; \
  return z; \
}

// trigonometric functions
ADVAR_FUNC( cos, cos(var.v_), -sin(var.v_) )
ADVAR_FUNC( sin, sin(var.v_),  cos(var.v_) )
ADVAR_FUNC( tan, tan(var.v_),  T(1)/(cos(var.v_)*cos(var.v_)) )
ADVAR_FUNC( acos, acos(var.v_), -T(1)/sqrt(1 - var.v_*var.v_) )
ADVAR_FUNC( asin, asin(var.v_),  T(1)/sqrt(1 - var.v_*var.v_) )
ADVAR_FUNC( atan, atan(var.v_),  T(1)/(1 + var.v_*var.v_) )

template<unsigned int N, typename T>
HOST DEVICE ADVarS<N,T>
atan2(const ADVarS<N,T>& y, const ADVarS<N,T>& x)
{
  T tmp = T(1) / (y.v_*y.v_ + x.v_*x.v_);
  ADVarS<N,T> z;
  z.v_ = atan2(y.v_, x.v_);
  for (unsigned int i = 0; i < N; i++)
    z.d_[i] = tmp * (x.v_*y.d_[i] - y.v_*x.d_[i]);
  return z;
}

// hyperbolic functions
ADVAR_FUNC( cosh, cosh(var.v_), sinh(var.v_) )
ADVAR_FUNC( sinh, sinh(var.v_), cosh(var.v_) )
ADVAR_FUNC( tanh, tanh(var.v_), T(1)/(cosh(var.v_)*cosh(var.v_)) )

// exponential and logarithm functions
ADVAR_FUNC( exp, exp(var.v_), exp(var.v_) )
ADVAR_FUNC( expm1, expm1(var.v_), exp(var.v_) )
ADVAR_FUNC( log, log(var.v_), T(1)/var.v_ )
ADVAR_FUNC( log10, log10(var.v_), T(1)/(var.v_*log(10.)) )
ADVAR_FUNC( log1p, log1p(var.v_), T(1)/( 1 + var.v_ ) )

// error functions
ADVAR_FUNC( erf , erf(var.v_) ,  T(2./sqrt(M_PI))*exp(-(var.v_*var.v_)) )
ADVAR_FUNC( erfc, erfc(var.v_), -T(2./sqrt(M_PI))*exp(-(var.v_*var.v_)) )

// power functions

// abs functions
template<unsigned int N, typename T>
HOST DEVICE ADVarS<N,T> abs(const ADVarS<N,T>& var)
{
  return (var.v_ < 0) ? -var : var;
}

template<unsigned int N, typename T>
HOST DEVICE ADVarS<N,complex<T>> abs(const ADVarS<N,complex<T>>& var)
{
  T abs_val = abs(var.v_);
  if (abs_val == 0 || (std::is_floating_point<T>::value && 
                       abs_val < 10*std::numeric_limits<T>::min()))
    return ADVarS<N,complex<T>>(complex<T>(0,0));

  T inv_abs  = T(1) / abs_val;
  ADVarS<N,complex<T>> z;
  z.v_ = abs_val;
  for (unsigned int i = 0; i < N; i++)
    z.d_[i] = complex<T>(inv_abs*(var.v_.real()*var.d_[i].real() + var.v_.imag()*var.d_[i].imag()), 0);
  return z;
}

template<unsigned int N, typename T>
HOST DEVICE ADVarS<N,complex<T>> conj(const ADVarS<N,complex<T>>& var)
{
  ADVarS<N,complex<T>> z;
  z.v_ = conj(var.v_);
  for (unsigned int i = 0; i < N; i++)
    z.d_[i] = complex<T>(var.d_[i].real(), -var.d_[i].imag());
  return z;
}


// ostream
template<int N>
inline std::ostream& operator<<(std::ostream& os, const ADVarS<N,double>& var)
{
  os << "(" << var.value() << "; ";
  for (int i = 0; i < N-1; ++i)
    os << var.deriv(i) << ", ";
  os << var.deriv(N-1) << ")";
  return os;
}

}
