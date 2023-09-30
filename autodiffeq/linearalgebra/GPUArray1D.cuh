// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <ostream>
#include <autodiffeq/linearalgebra/Array1D.hpp>
#include <autodiffeq/cuda/cudaCheckError.cuh>

namespace autodiffeq
{

template<typename T>
class GPUArray1D;

template<typename T = double>
class DeviceArray1D
{
public:

  friend class GPUArray1D<T>;

  DeviceArray1D() = default;
  DeviceArray1D(std::size_t m, T* data) : size_(m), data_(data) {}

  ~DeviceArray1D() = default;

  inline __host__ __device__ std::size_t size() const { return size_; }
  inline __host__ __device__ const T* data() const { return data_; }
  inline __host__ __device__ T* data() { return data_; }

  inline __host__ __device__ const T& operator()(int i) const { return data_[i]; }
  inline __host__ __device__ T& operator()(int i) { return data_[i]; }

  inline __host__ __device__ const T& operator[](int i) const { return data_[i]; }
  inline __host__ __device__ T& operator[](int i) { return data_[i]; }

protected:
  std::size_t size_ = 0; //data size
  T* data_ = nullptr; //pointer to array data on device
};

template<typename T = double>
class GPUArray1D
{
public:

  GPUArray1D() { CreateDeviceArray(); }
  GPUArray1D(std::size_t m)
  {
    CreateDeviceArray();
    Resize(m);
  }

  GPUArray1D(std::size_t m, const T& val) { 
    CreateDeviceArray();
    Resize(m);
    Array1D<T> v(m, val);
    CopyToDevice(v.data());
  }
  
  GPUArray1D(const Array1D<T>& v) { 
    CreateDeviceArray();
    Resize(v.size());
    CopyToDevice(v.data());
  }

  GPUArray1D(const std::vector<T>& v) {
    CreateDeviceArray();
    Resize(v.size());
    CopyToDevice(v.data());
  }

  GPUArray1D(const std::initializer_list<T>& v) {
    CreateDeviceArray();
    Resize(v.size());
    CopyToDevice(v.begin());
  }

  ~GPUArray1D() 
  {
    if (arr_h_.data_)
      cudaCheckError(cudaFree(arr_h_.data_));
    if (arr_d_)
      cudaCheckError(cudaFree(arr_d_));
  }

  inline std::size_t size() const { return arr_h_.size_; }

  const DeviceArray1D<T>& GetDeviceArray() const { return *arr_d_; }
  DeviceArray1D<T>& GetDeviceArray() { return *arr_d_; }

  const T* GetDeviceData() const { return arr_d_->data_; }
  T* GetDeviceData() { return arr_d_->data_; }

  inline void Resize(const std::size_t size)
  {
    if (size != arr_h_.size_)
    {
      //Size has changed, so first free existing data
      if (arr_h_.data_)
        cudaCheckError(cudaFree(arr_h_.data_));

      arr_h_.size_ = size;
      if (size > 0)
        cudaCheckError(cudaMalloc(&arr_h_.data_, size*sizeof(T)));
      else
        arr_h_.data_ = nullptr;

      //Copy host struct to device struct
      cudaCheckError(cudaMemcpy(arr_d_, &arr_h_, sizeof(DeviceArray1D<T>), cudaMemcpyHostToDevice));
    }
  }

  inline void clear() { Resize(0); }

  inline void CopyToDevice(const T* data)
  {
    cudaCheckError(cudaMemcpy(arr_h_.data_, data, arr_h_.size_*sizeof(T), cudaMemcpyHostToDevice));
  }

  inline Array1D<T> CopyToHost() const {
    Array1D<T> vh(arr_h_.size_);
    cudaCheckError(cudaMemcpy(vh.data(), arr_h_.data_, arr_h_.size_*sizeof(T), cudaMemcpyDeviceToHost));
    return vh;
  }

  inline void SetValue(const T& val) 
  { 
    if (arr_h_.size_ > 0)
    {
      std::vector<T> vec(arr_h_.size_, val);
      cudaCheckError(cudaMemcpy(arr_h_.data_, vec.data(), 
                                arr_h_.size_*sizeof(T), cudaMemcpyHostToDevice));
    }
  }

protected:

  inline void CreateDeviceArray() {
    if (!arr_d_)
      cudaCheckError(cudaMalloc(&arr_d_, sizeof(DeviceArray1D<T>)));
  }

  DeviceArray1D<T> arr_h_; //host side struct
  DeviceArray1D<T>* arr_d_ = nullptr; //pointer to device side struct
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const GPUArray1D<T>& v)
{
  os << v.CopyToHost();
  return os;
}

}