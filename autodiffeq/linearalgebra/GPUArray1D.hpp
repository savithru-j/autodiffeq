// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <vector>
#include <ostream>
#include <autodiffeq/cuda/cudaCheckError.h>

namespace autodiffeq
{

template<class T = double>
class GPUArray1D
{
public:

  GPUArray1D() = default;
  GPUArray1D(std::size_t m) : size_(m)
  {
    cudaCheckError(cudaMalloc(&d_data_, size_*sizeof(T)));
  }

  GPUArray1D(std::size_t m, const T& val)
  { 
    ResizeAndCopy(std::vector<T>(m, val)); 
  }

  GPUArray1D(const std::vector<T>& v) { ResizeAndCopy(v); }

  GPUArray1D(const std::initializer_list<T>& v)
  {
    if (v.size() > 0) 
    {
      std::vector<T> h_data(v.size());
      auto row = v.begin();
      for (std::size_t i = 0; i < v.size(); ++i)
        h_data[i] = *(row++);
      ResizeAndCopy(v);
    }
  }

  ~GPUArray1D() 
  {
    if (d_data_)
      cudaCheckError(cudaFree(d_data_));
  }

  inline std::size_t m() const { return size_; }
  inline std::size_t size() const { return size_; }

  inline const T& operator()(int i) const { return d_data_[i]; }
  inline T& operator()(int i) { return d_data_[i]; }

  inline const T& operator[](int i) const { return d_data_[i]; }
  inline T& operator[](int i) { return d_data_[i]; }

  inline void ResizeAndCopy(const std::vector<T>& h_data)
  {
    if (d_data_)
      cudaCheckError(cudaFree(d_data_));
    size_ = h_data.size();
    cudaCheckError(cudaMalloc(&d_data_, size_*sizeof(T)));
    cudaCheckError(cudaMemcpy(d_data_, h_data.data(), size_*sizeof(T), cudaMemcpyHostToDevice));
  }

  inline void clear() { 
    if (d_data_)
      cudaCheckError(cudaFree(d_data_));
    size_ = 0;
    d_data_ = nullptr;
  }

  const T* data() const { return d_data_; }
  T* data() { return d_data_; }

  inline void SetValue(const T& val) 
  { 
    if (size_ > 0)
    {
      std::vector<T> data(size_, val);
      cudaCheckError(cudaMemcpy(d_data_, data.data(), size_*sizeof(T), cudaMemcpyHostToDevice));
    }
  }

protected:
  std::size_t size_ = 0; //data size
  T* d_data_ = nullptr; //device pointer to data
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const GPUArray1D<T>& v)
{
  std::size_t m = v.size();
  std::vector<T> vh(m);
  cudaCheckError(cudaMemcpy(vh.data(), v.data(), m*sizeof(T), cudaMemcpyDeviceToHost));
  for (int i = 0; i < m-1; ++i)
    os << vh[i] << ", ";
  os << vh.back();
  return os;
}

}