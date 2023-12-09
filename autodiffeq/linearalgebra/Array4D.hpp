// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <vector>
#include <array>
#include <ostream>

namespace autodiffeq
{

template<class T = double>
class Array4D
{
public:

  Array4D() = default;
  Array4D(int m, int n, int p, int q) : dims_({m,n,p,q}), data_(m*n*p*q) {}
  Array4D(int m, int n, int p, int q, const T& val) : dims_({m,n,p,q}), data_(m*n*p*q, val) {}
  Array4D(const std::array<int,4>& dims) : 
    dims_(dims), data_(dims[0]*dims[1]*dims[2]*dims[3]) {}
  Array4D(const std::array<int,4>& dims, const T& val) : 
    dims_(dims), data_(dims[0]*dims[1]*dims[2]*dims[3], val) {}

  Array4D(int m, int n, int p, int q, const std::vector<T>& data) : dims_({m,n,p,q})
  {
    assert(m*n*p*q == (int) data.size());
    data_ = data;
  }

  inline int GetDim(int axis) const { return dims_[axis]; }
  inline const std::array<int,4>& GetDimensions() const { return dims_; }
  inline std::size_t size() const { return data_.size(); }

  inline const T& operator()(int i, int j, int k, int l) const { return data_[dims_[3]*(dims_[2]*(dims_[1]*i + j) + k) + l]; }
  inline T& operator()(int i, int j, int k, int l) { return data_[dims_[3]*(dims_[2]*(dims_[1]*i + j) + k) + l]; }

  inline const T& operator[](int i) const { return data_[i]; }
  inline T& operator[](int i) { return data_[i]; }

  inline typename std::vector<T>::iterator begin() { return data_.begin(); }
  inline typename std::vector<T>::iterator end() { return data_.end(); }

  inline typename std::vector<T>::const_iterator cbegin() const { return data_.cbegin(); }
  inline typename std::vector<T>::const_iterator cend() const { return data_.cend(); }

  inline void resize(int m, int n, int p, int q, const T& val = T(0)) { 
    dims_ = {m, n, p, q};
    data_.resize(m*n*p*q, val); 
  }

  inline void clear() { 
    dims_ = {0,0,0,0};
    data_.clear(); 
  }

  inline void SetValue(const T& val) { std::fill(data_.begin(), data_.end(), val); }

  inline const std::vector<T>& GetDataVector() const { return data_; }
  inline std::vector<T>& GetDataVector() { return data_; }

protected:
  std::array<int,4> dims_;
  std::vector<T> data_;
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const Array4D<T>& A)
{
  const auto& dims = A.GetDimensions();
  os << "[";
  for (int i = 0; i < dims[0]; ++i)
  {
    if (i == 0)
      os << "[";
    else
      os << " [";
    for (int j = 0; j < dims[1]; ++j)
    {
      if (j == 0)
        os << "[";
      else
        os << "  [";
      for (int k = 0; k < dims[2]; ++k)
      {
        if (k == 0)
          os << "[";
        else
          os << "   [";
        for (int l = 0; l < dims[3]-1; ++l)
          os << A(i,j,k,l) << ", ";
        os << A(i,j,k,dims[3]-1) << "]";
        if (k < dims[2]-1)
          os << ",\n";
      }
      os << "]";
      if (j < dims[1]-1)
        os << ",\n";
    }
    os << "]";
    if (i < dims[0]-1)
      os << ",\n";
  }
  os << "]" << std::endl;
  return os;
}

}