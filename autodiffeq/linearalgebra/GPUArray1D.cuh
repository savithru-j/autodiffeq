// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <ostream>
#include <autodiffeq/linearalgebra/Array1D.hpp>
#include <autodiffeq/cuda/cudaCheckError.cuh>

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
    cudaCheckError(cudaMalloc(&d_size_, sizeof(std::size_t)));
    cudaCheckError(cudaMemcpy(d_size_, &size_, sizeof(std::size_t), cudaMemcpyHostToDevice));
  }

  GPUArray1D(std::size_t m, const T& val) { 
    Array1D<T> v(m, val);
    ResizeAndCopy(v.data(), v.size()); 
  }
  GPUArray1D(const Array1D<T>& v) { ResizeAndCopy(v.data(), v.size()); }
  GPUArray1D(const std::vector<T>& v) { ResizeAndCopy(v.data(), v.size()); }

  GPUArray1D(const std::initializer_list<T>& v) { ResizeAndCopy(v.begin(), v.size()); }

  ~GPUArray1D() 
  {
    if (d_data_)
      cudaCheckError(cudaFree(d_data_));
    if (d_size_)
      cudaCheckError(cudaFree(d_size_));
  }

  inline __device__ std::size_t m() const { return *d_size_; }
  inline __device__ std::size_t size() const { return *d_size_; }
  std::size_t GetSizeOnHost() const { return size_; }

  inline __device__ const T& operator()(int i) const { return d_data_[i]; }
  inline __device__ T& operator()(int i) { return d_data_[i]; }

  inline __device__ const T& operator[](int i) const { return d_data_[i]; }
  inline __device__ T& operator[](int i) { return d_data_[i]; }

  inline void ResizeAndCopy(const T* h_data, const std::size_t N)
  {
    if (d_data_)
      cudaCheckError(cudaFree(d_data_));
    if (d_size_)
      cudaCheckError(cudaFree(d_size_));
    size_ = N;
    cudaCheckError(cudaMalloc(&d_data_, size_*sizeof(T)));
    cudaCheckError(cudaMemcpy(d_data_, h_data, size_*sizeof(T), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMalloc(&d_size_, sizeof(std::size_t)));
    cudaCheckError(cudaMemcpy(d_size_, &size_, sizeof(std::size_t), cudaMemcpyHostToDevice));
  }

  inline void clear() { 
    if (d_data_)
      cudaCheckError(cudaFree(d_data_));
    if (d_size_)
      cudaCheckError(cudaFree(d_size_));
    size_ = 0;
    d_data_ = nullptr;
    d_size_ = nullptr;
  }

  inline __host__ __device__ const T* data() const { return d_data_; }
  inline __host__ __device__ T* data() { return d_data_; }

  inline void SetValue(const T& val) 
  { 
    if (size_ > 0)
    {
      std::vector<T> data(size_, val);
      cudaCheckError(cudaMemcpy(d_data_, data.data(), size_*sizeof(T), cudaMemcpyHostToDevice));
    }
  }

  inline Array1D<T> CopyToHost() const {
    Array1D<T> vh(size_);
    cudaCheckError(cudaMemcpy(vh.data(), d_data_, size_*sizeof(T), cudaMemcpyDeviceToHost));
    return vh;
  }

protected:
  std::size_t size_ = 0; //data size
  std::size_t* d_size_ = nullptr; //device pointer to array size
  T* d_data_ = nullptr; //device pointer to array data
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const GPUArray1D<T>& v)
{
  Array1D<T> vh = v.CopyToHost();
  int m = (int) vh.size();
  for (int i = 0; i < m-1; ++i)
    os << vh[i] << ", ";
  os << vh.back();
  return os;
}

}