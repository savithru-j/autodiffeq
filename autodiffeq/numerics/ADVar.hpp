// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <cassert>
#include <initializer_list>
#include <cmath>

namespace autodiffeq
{

// ADVar : Auto-diff variable for storing a value and a vector of derivatives
template<typename T>
class ADVar;

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

template<typename T>
class ADVar
{
public:
  ADVar() = default;

  ADVar(const T& v, const T d[], const int num_deriv)
  {
    assert(num_deriv > 0);
    v_ = v;
    d_ = new T[num_deriv];
    N_ = num_deriv;
    for (unsigned int i = 0; i < N_; ++i)
      d_[i] = d[i];
  }

  ADVar(const T& v, const std::initializer_list<T>& d)
  {
    v_ = v;
    N_ = d.size();
    if (N_ > 0) 
    {
      d_ = new T[N_];
      auto row = d.begin();
      for (unsigned int i = 0; i < N_; ++i)
        d_[i] = *(row++);
    }
  }

  explicit ADVar(const T& v, const int num_deriv) : v_(v) 
  {
    N_ = num_deriv;
    if (N_ > 0)
      d_ = new T[N_];
  }
  
  ADVar(const T& v) : v_(v), d_(nullptr), N_(0) {}

  ADVar(const ADVar& var) : v_(var.v_), N_(var.N_)
  {
    if (N_ > 0) 
    {
      d_ = new T[N_];
      for (unsigned int i = 0; i < N_; ++i)
        d_[i] = var.d_[i];
    }
  }

  ~ADVar() { delete [] d_; }

  inline unsigned int size() const { return N_; }

  // value accessors
  inline T& value() { return v_; }
  inline const T& value() const { return v_; }

  // derivative accessors
  inline T& deriv(int i = 0) 
  {
    assert(N_ > 0); 
    return d_[i]; 
  }
  inline T deriv(int i = 0) const { return N_ > 0 ? d_[i] : T(0.0); }

  // assignment operators
  ADVar& operator=(const ADVar& var);
  ADVar& operator=(const T& v);

  // unary operators
  const ADVar& operator+() const;
  const ADVar  operator-() const;

  // binary accumulation operators
  ADVar& operator+=(const ADVar& var);
  ADVar& operator+=(const T& v);
  ADVar& operator-=(const ADVar& var);
  ADVar& operator-=(const T& v);
  ADVar& operator*=(const ADVar& var);
  ADVar& operator*=(const T& v);
  ADVar& operator/=(const ADVar& var);
  ADVar& operator/=(const T& v);

  // binary operators
  template<typename U> friend ADVar<U> operator+( const ADVar<U>& a, const ADVar<U>& b);
  template<typename U> friend ADVar<U> operator+( const ADVar<U>& a, const U& b);
  template<typename U> friend ADVar<U> operator+( const U& a, const ADVar<U>& b);
  template<typename U> friend ADVar<U> operator-( const ADVar<U>& a, const ADVar<U>& b);
  template<typename U> friend ADVar<U> operator-( const ADVar<U>& a, const U& b);
  template<typename U> friend ADVar<U> operator-( const U& a, const ADVar<U>& b);
  template<typename U> friend ADVar<U> operator*( const ADVar<U>& a, const ADVar<U>& b);
  template<typename U> friend ADVar<U> operator*( const ADVar<U>& a, const U& b);
  template<typename U> friend ADVar<U> operator*( const U& a, const ADVar<U>& b);
  template<typename U> friend ADVar<U> operator/( const ADVar<U>& a, const ADVar<U>& b);
  template<typename U> friend ADVar<U> operator/( const ADVar<U>& a, const U& b);
  template<typename U> friend ADVar<U> operator/( const U& a, const ADVar<U>& b);

    // relational operators
  template<typename U> friend bool operator==(const ADVar<U>& a, const ADVar<U>& b);
  template<typename U> friend bool operator==(const ADVar<U>& a, const U& b);
  template<typename U> friend bool operator==(const U& a, const ADVar<U>& b);
  template<typename U> friend bool operator!=(const ADVar<U>& a, const ADVar<U>& b);
  template<typename U> friend bool operator!=(const ADVar<U>& a, const U& b);
  template<typename U> friend bool operator!=(const U& a, const ADVar<U>& b);
  template<typename U> friend bool operator>(const ADVar<U>& a, const ADVar<U>& b);
  template<typename U> friend bool operator>(const ADVar<U>& a, const U& b);
  template<typename U> friend bool operator>(const U& a, const ADVar<U>& b);
  template<typename U> friend bool operator<(const ADVar<U>& a, const ADVar<U>& b);
  template<typename U> friend bool operator<(const ADVar<U>& a, const U& b);
  template<typename U> friend bool operator<(const U& a, const ADVar<U>& b);
  template<typename U> friend bool operator>=(const ADVar<U>& a, const ADVar<U>& b);
  template<typename U> friend bool operator>=(const ADVar<U>& a, const U& b);
  template<typename U> friend bool operator>=(const U& a, const ADVar<U>& b);
  template<typename U> friend bool operator<=(const ADVar<U>& a, const ADVar<U>& b);
  template<typename U> friend bool operator<=(const ADVar<U>& a, const U& b);
  template<typename U> friend bool operator<=(const U& a, const ADVar<U>& b);

  // trigonometric functions
  template<typename U> friend ADVar<U> cos(const ADVar<U>& var);
  template<typename U> friend ADVar<U> sin(const ADVar<U>& var);
  template<typename U> friend ADVar<U> tan(const ADVar<U>& var);
  template<typename U> friend ADVar<U> acos( const ADVar<U>& var);
  template<typename U> friend ADVar<U> asin( const ADVar<U>& var);
  template<typename U> friend ADVar<U> atan( const ADVar<U>& var);
  template<typename U> friend ADVar<U> atan2( const ADVar<U>& y, const ADVar<U>& x);

  // hyperbolic functions
  template<typename U> friend ADVar<U> cosh(const ADVar<U>& var);
  template<typename U> friend ADVar<U> sinh(const ADVar<U>& var);
  template<typename U> friend ADVar<U> tanh(const ADVar<U>& var);

  // exponential and logarithm functions
  template<typename U> friend ADVar<U> exp(const ADVar<U>& var);
  template<typename U> friend ADVar<U> expm1(const ADVar<U>& var);
  template<typename U> friend ADVar<U> log(const ADVar<U>& var);
  template<typename U> friend ADVar<U> log10(const ADVar<U>& var);
  template<typename U> friend ADVar<U> log1p(const ADVar<U>& var);

  // error functions
  template<typename U> friend ADVar<U> erf(const ADVar<U>& var);
  template<typename U> friend ADVar<U> erfc(const ADVar<U>& var);

  // power functions
  template<typename U> friend ADVar<U> pow( const ADVar<U>& x, const ADVar<U>& y);
  template<typename U> friend ADVar<U> pow( const ADVar<U>& x, const U& y);
  template<typename U> friend ADVar<U> pow( const U& x, const ADVar<U>& y);
  template<typename U> friend ADVar<U> sqrt(const ADVar<U>& x);

  // rounding and absolute functions
  template<typename U> friend ADVar<U> ceil(const ADVar<U>& var);
  template<typename U> friend ADVar<U> floor(const ADVar<U>& var);
  template<typename U> friend ADVar<U> abs(const ADVar<U>& var);
  template<typename U> friend ADVar<U> fabs(const ADVar<U>& var);

protected:

  T v_; //value
  T* d_; //derivatives
  unsigned int N_; //number of derivatives
};

// assignment
template<typename T>
ADVar<T>& ADVar<T>::operator=(const ADVar& var)
{
  //Do nothing if assigning self to self
  if ( &var == this ) return *this;

  if ((N_ == 0) && (var.N_ > 0))
  {
    N_ = var.N_;
    d_ = new T[N_];
  }
  else if ( var.N_ == 0 )
  {
    (*this) = var.v_; //var is a scalar
    return *this;
  }
  else
    assert(N_ == var.N_);

  v_ = var.v_;
  for (unsigned int i = 0; i < N_; i++)
    d_[i] = var.d_[i];
  return *this;
}

template<typename T>
ADVar<T>& ADVar<T>::operator=(const T& v)
{
  v_ = v;
  delete [] d_;
  d_ = nullptr;
  N_ = 0;
  return *this;
}

// unary operators
template<typename T>
const ADVar<T>& ADVar<T>::operator+() const
{
  return *this;
}

template<typename T>
const ADVar<T> ADVar<T>::operator-() const
{
  ADVar c(-v_, N_);
  for (unsigned int i = 0; i < N_; i++)
    c.d_[i] = -d_[i];
  return c;
}

// binary accumulation operators
template<typename T>
ADVar<T>& ADVar<T>::operator+=(const ADVar& var)
{
  if ((N_ == 0) && (var.N_ > 0))
  {
    N_ = var.N_;
    d_ = new T[N_](); //d_ is value-initialized
  }
  else if (var.N_ == 0)
  {
    v_ += var.v_; //var is a scalar
    return *this;
  }
  else
    assert(N_ == var.N_);

  v_ += var.v_;
  for (unsigned int i = 0; i < N_; i++)
    d_[i] += var.d_[i];
  return *this;
}

template<typename T>
ADVar<T>& ADVar<T>::operator+=(const T& v)
{
  v_ += v;
  return *this;
}

template<typename T>
ADVar<T>& ADVar<T>::operator-=(const ADVar& var)
{
  if ((N_ == 0) && (var.N_ > 0))
  {
    N_ = var.N_;
    d_ = new T[N_](); //d_ is value-initialized
  }
  else if (var.N_ == 0)
  {
    v_ -= var.v_; //var is a scalar
    return *this;
  }
  else
    assert(N_ == var.N_);

  v_ -= var.v_;
  for (unsigned int i = 0; i < N_; i++)
    d_[i] -= var.d_[i];
  return *this;
}

template<typename T>
ADVar<T>& ADVar<T>::operator-=(const T& v)
{
  v_ -= v;
  return *this;
}

template<typename T>
ADVar<T>& ADVar<T>::operator*=(const ADVar& var)
{
  if ((N_ == 0) && (var.N_ > 0))
  {
    N_ = var.N_;
    d_ = new T[N_](); //d_ is value-initialized
  }
  else if (var.N_ == 0)
  {
    v_ *= var.v_; //var is a scalar
    return *this;
  }
  else
    assert(N_ == var.N_);

  for (unsigned int i = 0; i < N_; i++)
    d_[i] = v_*var.d_[i] + d_[i]*var.v_;
  v_ *= var.v_;
  return *this;
}

template<typename T>
ADVar<T>& ADVar<T>::operator*=(const T& v)
{
  for (unsigned int i = 0; i < N_; i++)
    d_[i] *= v;
  v_ *= v;
  return *this;
}

template<typename T>
ADVar<T>& ADVar<T>::operator/=(const ADVar& var)
{
  if ((N_ == 0) && (var.N_ > 0))
  {
    N_ = var.N_;
    d_ = new T[N_](); //d_ is value-initialized
  }
  else if (var.N_ == 0)
  {
    v_ /= var.v_; //var is a scalar
    return *this;
  }
  else
    assert(N_ == var.N_);

  T tmp = 1.0/(var.v_*var.v_);
  for (unsigned int i = 0; i < N_; i++)
    d_[i] = (d_[i]*var.v_ - v_*var.d_[i]) * tmp;
  v_ /= var.v_;
  return *this;
}

template<typename T>
ADVar<T>& ADVar<T>::operator/=(const T& v)
{
  T tmp = 1.0/v;
  for (unsigned int i = 0; i < N_; i++)
    d_[i] *= tmp;
  v_ *= tmp;
  return *this;
}

//binary operators
template<typename T>
ADVar<T> operator+(const ADVar<T>& a, const ADVar<T>& b)
{
  if (a.N_ == 0 && b.N_ == 0)
    return ADVar<T>(a.v_ + b.v_);
  else if (a.N_ > 0 && b.N_ == 0)
    return ADVar<T>(a.v_ + b.v_, a.d_, a.N_);
  else if (a.N_ == 0 && b.N_ > 0)
    return ADVar<T>(a.v_ + b.v_, b.d_, b.N_);
  
  assert(a.N_ == b.N_);
  ADVar<T> c(a.v_ + b.v_, a.N_);
  for (unsigned int i = 0; i < c.N_; i++)
    c.d_[i] = a.d_[i] + b.d_[i];
  return c;
}

template<typename T>
ADVar<T> operator+(const ADVar<T>& a, const T& b)
{
  if (a.N_ == 0 )
    return ADVar<T>(a.v_ + b);
  else
    return ADVar<T>(a.v_ + b, a.d_, a.N_);
}

template<typename T>
ADVar<T> operator+(const T& a, const ADVar<T>& b)
{
  if (b.N_ == 0 )
    return ADVar<T>(a + b.v_);
  else
    return ADVar<T>(a + b.v_, b.d_, b.N_);
}

template<typename T>
ADVar<T> operator-(const ADVar<T>& a, const ADVar<T>& b)
{
  if (a.N_ == 0 && b.N_ == 0)
    return ADVar<T>(a.v_ - b.v_);
  else if (a.N_ > 0 && b.N_ == 0)
    return ADVar<T>(a.v_ - b.v_, a.d_, a.N_);
  else if (a.N_ == 0 && b.N_ > 0)
  {
    ADVar<T> c(a.v_ - b.v_, b.N_);
    for (unsigned int i = 0; i < b.N_; i++)
      c.d_[i] = -b.d_[i];
    return c;
  }
  
  assert(a.N_ == b.N_);
  ADVar<T> c(a.v_ - b.v_, a.N_);
  for (unsigned int i = 0; i < c.N_; i++)
    c.d_[i] = a.d_[i] - b.d_[i];
  return c;
}

template<typename T>
ADVar<T> operator-(const ADVar<T>& a, const T& b)
{
  if (a.N_ == 0 )
    return ADVar<T>(a.v_ - b);
  else
    return ADVar<T>(a.v_ - b, a.d_, a.N_);
}

template<typename T>
ADVar<T> operator-(const T& a, const ADVar<T>& b)
{
  if (b.N_ == 0 )
    return ADVar<T>(a - b.v_);
  else
  {
    ADVar<T> c(a - b.v_, b.N_);
    for (unsigned int i = 0; i < b.N_; i++)
      c.d_[i] = -b.d_[i];
    return c;
  }
}

template<typename T>
ADVar<T> operator*(const ADVar<T>& a, const ADVar<T>& b)
{
  if (a.N_ == 0 && b.N_ == 0)
    return ADVar<T>(a.v_ * b.v_);
  else if (a.N_ > 0 && b.N_ == 0)
  {
    ADVar<T> c(a.v_ * b.v_, a.N_);
    for (unsigned int i = 0; i < a.N_; i++)
      c.d_[i] = a.d_[i] * b.v_;
    return c;
  }
  else if (a.N_ == 0 && b.N_ > 0)
  {
    ADVar<T> c(a.v_ * b.v_, b.N_);
    for (unsigned int i = 0; i < b.N_; i++)
      c.d_[i] = a.v_ * b.d_[i];
    return c;
  }
  
  assert(a.N_ == b.N_);
  ADVar<T> c(a.v_ * b.v_, a.N_);
  for (unsigned int i = 0; i < c.N_; i++)
    c.d_[i] = a.d_[i]*b.v_ + a.v_*b.d_[i];
  return c;
}

template<typename T>
ADVar<T> operator*(const ADVar<T>& a, const T& b)
{
  if (a.N_ == 0 )
    return ADVar<T>(a.v_ * b);
  else
  {
    ADVar<T> c(a.v_ * b, a.N_);
    for (unsigned int i = 0; i < a.N_; i++)
      c.d_[i] = a.d_[i] * b;
    return c;
  }
}

template<typename T>
ADVar<T> operator*(const T& a, const ADVar<T>& b)
{
  if (b.N_ == 0 )
    return ADVar<T>(a * b.v_);
  else
  {
    ADVar<T> c(a * b.v_, b.N_);
    for (unsigned int i = 0; i < b.N_; i++)
      c.d_[i] = a * b.d_[i];
    return c;
  }
}

template<typename T>
ADVar<T> operator/(const ADVar<T>& a, const ADVar<T>& b)
{
  if (a.N_ == 0 && b.N_ == 0)
    return ADVar<T>(a.v_ / b.v_);
  else if (a.N_ > 0 && b.N_ == 0)
  {
    ADVar<T> c(a.v_ / b.v_, a.N_);
    T tmp = 1.0 / b.v_;
    for (unsigned int i = 0; i < a.N_; i++)
      c.d_[i] = a.d_[i] * tmp;
    return c;
  }
  else if (a.N_ == 0 && b.N_ > 0)
  {
    ADVar<T> c(a.v_ / b.v_, b.N_);
    T tmp = 1.0 / (b.v_ * b.v_);
    for (unsigned int i = 0; i < b.N_; i++)
      c.d_[i] = -a.v_ * b.d_[i] * tmp;
    return c;
  }
  
  assert(a.N_ == b.N_);
  ADVar<T> c(a.v_ / b.v_, a.N_);
  T tmp = 1.0 / (b.v_ * b.v_);
  for (unsigned int i = 0; i < c.N_; i++)
    c.d_[i] = (a.d_[i]*b.v_ - a.v_*b.d_[i]) * tmp;
  return c;
}

template<typename T>
ADVar<T> operator/(const ADVar<T>& a, const T& b)
{
  if (a.N_ == 0 )
    return ADVar<T>(a.v_ / b);
  else
  {
    ADVar<T> c(a.v_ / b, a.N_);
    T tmp = 1.0 / b;
    for (unsigned int i = 0; i < a.N_; i++)
      c.d_[i] = a.d_[i] * tmp;
    return c;
  }
}

template<typename T>
ADVar<T> operator/(const T& a, const ADVar<T>& b)
{
  if (b.N_ == 0 )
    return ADVar<T>(a / b.v_);
  else
  {
    ADVar<T> c(a / b.v_, b.N_);
    T tmp = 1.0 / (b.v_ * b.v_);
    for (unsigned int i = 0; i < b.N_; i++)
      c.d_[i] = -a * b.d_[i] * tmp;
    return c;
  }
}

// relational operators

template<typename T>
bool operator==(const ADVar<T>& a, const ADVar<T>& b) { return a.v_ == b.v_; }

template<typename T>
bool operator==(const ADVar<T>& a, const T& b) { return a.v_ == b; }

template<typename T>
bool operator==(const T& a, const ADVar<T>& b) { return a == b.v_; }

template<typename T>
bool operator!=(const ADVar<T>& a, const ADVar<T>& b) { return a.v_ != b.v_; }

template<typename T>
bool operator!=(const ADVar<T>& a, const T& b) { return a.v_ != b; }

template<typename T>
bool operator!=(const T& a, const ADVar<T>& b) { return a != b.v_; }

template<typename T>
bool operator>(const ADVar<T>& a, const ADVar<T>& b) { return a.v_ > b.v_; }

template<typename T>
bool operator>(const ADVar<T>& a, const T& b) { return a.v_ > b; }

template<typename T>
bool operator>(const T& a, const ADVar<T>& b) { return a > b.v_; }

template<typename T>
bool operator<(const ADVar<T>& a, const ADVar<T>& b) { return a.v_ < b.v_; }

template<typename T>
bool operator<(const ADVar<T>& a, const T& b) { return a.v_ < b; }

template<typename T>
bool operator<(const T& a, const ADVar<T>& b) { return a < b.v_; }

template<typename T>
bool operator>=(const ADVar<T>& a, const ADVar<T>& b) { return a.v_ >= b.v_; }

template<typename T>
bool operator>=(const ADVar<T>& a, const T& b) { return a.v_ >= b; }

template<typename T>
bool operator>=(const T& a, const ADVar<T>& b) { return a >= b.v_; }

template<typename T>
bool operator<=(const ADVar<T>& a, const ADVar<T>& b) { return a.v_ <= b.v_; }

template<typename T>
bool operator<=(const ADVar<T>& a, const T& b) { return a.v_ <= b; }

template<typename T>
bool operator<=(const T& a, const ADVar<T>& b) { return a <= b.v_; }

#define ADVAR_FUNC(NAME, VALUE, DERIV) \
template<typename T> \
ADVar<T> \
NAME(const ADVar<T>& var) \
{ \
  if ( var.N_ == 0 ) \
    return ADVar<T>(VALUE); \
  \
  T tmp = DERIV; \
  ADVar<T> z(VALUE, var.N_); \
  for (unsigned int i = 0; i < var.N_; i++) \
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

template<typename T>
ADVar<T>
atan2(const ADVar<T>& y, const ADVar<T>& x)
{
  T val = atan2(y.v_, x.v_);
  if (y.N_ == 0 && x.N_ == 0)
    return ADVar<T>(val);
  else if (y.N_ > 0 && x.N_ == 0)
  {
    T tmp = T(1) / (y.v_*y.v_ + x.v_*x.v_);
    ADVar<T> z(val, y.N_);
    for (unsigned int i = 0; i < y.N_; i++)
      z.d_[i] = tmp * x.v_ * y.d_[i];
    return z;
  }
  else if (y.N_ == 0 && x.N_ > 0)
  {
    T tmp = T(1) / (y.v_*y.v_ + x.v_*x.v_);
    ADVar<T> z(val, x.N_);
    for (unsigned int i = 0; i < x.N_; i++)
      z.d_[i] = -tmp * y.v_ * x.d_[i];
    return z;
  }
  assert(y.N_ == x.N_);
  T tmp = T(1) / (y.v_*y.v_ + x.v_*x.v_);
  ADVar<T> z(val, y.N_);
  for (unsigned int i = 0; i < y.N_; i++)
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

}
