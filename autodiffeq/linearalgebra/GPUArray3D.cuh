// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <ostream>
#include <autodiffeq/linearalgebra/Array3D.hpp>
#include <autodiffeq/cuda/cudaCheckError.cuh>

namespace autodiffeq
{

template<typename T>
class GPUArray3D;

template<typename T = double>
class DeviceArray3D
{
public:

  friend class GPUArray3D<T>;

  DeviceArray3D() = default;
  DeviceArray3D(int m, int n, int p, T* data) : data_(data) 
  {
    dims_[0] = m;
    dims_[1] = n;
    dims_[2] = p;
  }

  ~DeviceArray3D() = default;

  inline __host__ __device__ int GetDim(int axis) const { return dims_[axis]; }
  inline __host__ __device__ std::array<int,3> GetDimensions() const { 
    return {dims_[0], dims_[1], dims_[2]};
  }
  inline __host__ __device__ std::size_t size() const { return dims_[0]*dims_[1]*dims_[2]; }
  inline __host__ __device__ const T* data() const { return data_; }
  inline __host__ __device__ T* data() { return data_; }

  inline __host__ __device__ const T& operator()(int i, int j, int k) const { 
    return data_[dims_[2]*(dims_[1]*i + j) + k]; 
  }
  inline __host__ __device__ T& operator()(int i, int j, int k) { 
    return data_[dims_[2]*(dims_[1]*i + j) + k]; 
  }

  inline __host__ __device__ const T& operator[](int i) const { return data_[i]; }
  inline __host__ __device__ T& operator[](int i) { return data_[i]; }

protected:
  int dims_[3];
  T* data_ = nullptr; //pointer to array data on device
};

template<typename T = double>
class GPUArray3D
{
public:

  GPUArray3D() { CreateDeviceArray(); }
  GPUArray3D(int m, int n, int p)
  {
    CreateDeviceArray();
    Resize(m, n, p);
  }

  GPUArray3D(int m, int n, int p, const T& val) { 
    CreateDeviceArray();
    Resize(m, n, p);
    std::vector<T> v(m*n*p, val);
    CopyToDevice(v.data());
  }
  
  GPUArray3D(const Array3D<T>& mat) { 
    CreateDeviceArray();
    const auto& dims = mat.GetDimensions();
    Resize(dims[0], dims[1], dims[2]);
    CopyToDevice(mat.GetDataVector().data());
  }

  ~GPUArray3D() 
  {
    if (arr_h_.data_)
      cudaCheckError(cudaFree(arr_h_.data_));
    if (arr_d_)
      cudaCheckError(cudaFree(arr_d_));
  }

  inline int GetDim(int axis) const { return arr_h_.dims_[axis]; }
  inline std::array<int,3> GetDimensions() const { return arr_h_.GetDimensions(); }
  inline std::size_t size() const { return arr_h_.size(); }

  const DeviceArray3D<T>& GetDeviceArray() const { return *arr_d_; }
  DeviceArray3D<T>& GetDeviceArray() { return *arr_d_; }

  const T* GetDeviceData() const { return arr_d_->data_; }
  T* GetDeviceData() { return arr_d_->data_; }

  inline void Resize(int m, int n, int p)
  {
    const auto& dims = arr_h_.GetDimensions();
    if (m != dims[0] || n != dims[1] || p != dims[2])
    {
      //Size has changed, so first free existing data
      if (arr_h_.data_)
        cudaCheckError(cudaFree(arr_h_.data_));

      arr_h_.dims_[0] = m;
      arr_h_.dims_[1] = n;
      arr_h_.dims_[2] = p;
      if (m*n*p > 0)
        cudaCheckError(cudaMalloc(&arr_h_.data_, m*n*p*sizeof(T)));
      else
        arr_h_.data_ = nullptr;

      //Copy host struct to device struct
      cudaCheckError(cudaMemcpy(arr_d_, &arr_h_, sizeof(DeviceArray3D<T>), cudaMemcpyHostToDevice));
    }
  }

  inline void clear() { Resize(0, 0, 0); }

  inline void CopyToDevice(const T* data)
  {
    cudaCheckError(cudaMemcpy(arr_h_.data_, data, arr_h_.size()*sizeof(T), cudaMemcpyHostToDevice));
  }

  inline Array3D<T> CopyToHost() const {
    Array3D<T> mat_h(arr_h_.GetDimensions());
    cudaCheckError(cudaMemcpy(mat_h.GetDataVector().data(), arr_h_.data_, arr_h_.size()*sizeof(T), cudaMemcpyDeviceToHost));
    return mat_h;
  }

  inline void CopyToHost(Array3D<T>& mat_h) const {
    for (int d = 0; d < 3; ++d)
      assert(mat_h.GetDim(d) == arr_h_.dims_[d]);
    cudaCheckError(cudaMemcpy(mat_h.GetDataVector().data(), arr_h_.data_, arr_h_.size()*sizeof(T), cudaMemcpyDeviceToHost));
  }

  inline void SetValue(const T& val) 
  { 
    if (arr_h_.size() > 0)
    {
      std::vector<T> vec(arr_h_.size(), val);
      cudaCheckError(cudaMemcpy(arr_h_.data_, vec.data(), 
                                arr_h_.size()*sizeof(T), cudaMemcpyHostToDevice));
    }
  }

protected:

  inline void CreateDeviceArray() {
    if (!arr_d_)
      cudaCheckError(cudaMalloc(&arr_d_, sizeof(DeviceArray3D<T>)));
  }

  DeviceArray3D<T> arr_h_; //host side struct
  DeviceArray3D<T>* arr_d_ = nullptr; //pointer to device side struct
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const GPUArray3D<T>& mat)
{
  os << mat.CopyToHost();
  return os;
}

}