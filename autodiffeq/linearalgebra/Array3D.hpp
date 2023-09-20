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
class Array3D
{
public:

  Array3D() = default;
  Array3D(int m, int n, int p) : dims_({m,n,p}), data_(m*n*p) {}
  Array3D(int m, int n, int p, const T& val) : dims_({m,n,p}), data_(m*n*p, val) {}

  Array3D(int m, int n, int p, const std::vector<T>& data) : dims_({m,n,p})
  {
    assert(m*n*p == (int) data.size());
    data_ = data;
  }

  inline int GetDim(int axis) const { return dims_[axis]; }
  inline const std::array<int,3>& GetDimensions() const { return dims_; }
  inline std::size_t size() const { return data_.size(); }

  inline const T& operator()(int i, int j, int k) const { return data_[dims_[2]*(dims_[1]*i + j) + k]; }
  inline T& operator()(int i, int j, int k) { return data_[dims_[2]*(dims_[1]*i + j) + k]; }

  inline const T& operator[](int i) const { return data_[i]; }
  inline T& operator[](int i) { return data_[i]; }

  inline typename std::vector<T>::iterator begin() { return data_.begin(); }
  inline typename std::vector<T>::iterator end() { return data_.end(); }

  inline typename std::vector<T>::const_iterator cbegin() const { return data_.cbegin(); }
  inline typename std::vector<T>::const_iterator cend() const { return data_.cend(); }

  inline void resize(int m, int n, int p, const T& val = 0.0) { 
    dims_ = {m, n, p};
    data_.resize(m*n*p, val); 
  }

  inline void clear() { 
    dims_ = {0,0,0};
    data_.clear(); 
  }

  inline void SetValue(const T& val) { std::fill(data_.begin(), data_.end(), val); }

  inline const std::vector<T>& GetDataVector() const { return data_; }
  inline std::vector<T>& GetDataVector() { return data_; }

protected:
  std::array<int,3> dims_;
  std::vector<T> data_;
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const Array3D<T>& A)
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
      for (int k = 0; k < dims[2]-1; ++k)
        os << A(i,j,k) << ", ";
      os << A(i,j,dims[2]-1) << "]";
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