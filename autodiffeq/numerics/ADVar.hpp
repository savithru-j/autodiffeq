// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <cassert>
#include <initializer_list>

namespace autodiffeq
{

// ADVar : Auto-diff variable for storing a value and a vector of derivatives

template<typename T>
class ADVar
{
public:
  ADVar() = default;

  ADVar(const T& v, const T d[], const int n)
  {
    assert(n > 0);
    v_ = v;
    d_ = new T[n];
    N_ = n;
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
    d_ = new T[N_];
  }
  
  ADVar(const T& v) : v_(v), d_(nullptr), N_(0) {}

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

protected:

  T v_; //value
  T* d_; //derivatives
  unsigned int N_; //number of derivatives
};


}
