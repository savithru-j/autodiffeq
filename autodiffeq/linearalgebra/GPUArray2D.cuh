// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <ostream>
#include <autodiffeq/linearalgebra/Array2D.hpp>
#include <autodiffeq/cuda/cudaCheckError.cuh>

namespace autodiffeq
{

template<typename T>
class GPUArray2D;

template<typename T = double>
class DeviceArray2D
{
public:

  friend class GPUArray2D<T>;

  DeviceArray2D() = default;
  DeviceArray2D(int m, int n, T* data) : nrows_(m), ncols_(n), data_(data) {}

  ~DeviceArray2D() = default;

  inline __host__ __device__ int GetNumRows() const { return nrows_; }
  inline __host__ __device__ int GetNumCols() const { return ncols_; }
  inline __host__ __device__ std::size_t size() const { return nrows_*ncols_; }
  inline __host__ __device__ const T* data() const { return data_; }
  inline __host__ __device__ T* data() { return data_; }

  inline __host__ __device__ const T& operator()(int i, int j) const { return data_[i*ncols_ + j]; }
  inline __host__ __device__ T& operator()(int i, int j) { return data_[i*ncols_ + j]; }

  inline __host__ __device__ const T& operator[](int i) const { return data_[i]; }
  inline __host__ __device__ T& operator[](int i) { return data_[i]; }

protected:
  int nrows_ = 0; //no. of rows
  int ncols_ = 0; //no. of cols
  T* data_ = nullptr; //pointer to array data on device
};

template<typename T = double>
class GPUArray2D
{
public:

  GPUArray2D() { CreateDeviceArray(); }
  GPUArray2D(int m, int n)
  {
    CreateDeviceArray();
    Resize(m, n);
  }

  GPUArray2D(int m, int n, const T& val) { 
    CreateDeviceArray();
    Resize(m, n);
    std::vector<T> v(m*n, val);
    CopyToDevice(v.data());
  }
  
  GPUArray2D(const Array2D<T>& mat) { 
    CreateDeviceArray();
    Resize(mat.GetNumRows(), mat.GetNumCols());
    CopyToDevice(mat.GetDataVector().data());
  }

  GPUArray2D(const std::initializer_list<std::initializer_list<T>>& mat) 
  {
    int m = (int) mat.size();
    int n = 0;
    std::vector<T> data;
    if (m > 0)
    {
      auto row = mat.begin();
      n = row->size();
      data.resize(m*n);
      int ind = 0;
      for (int i = 0; i < m; ++i)
      {
        assert((int) row->size() == n);      
        auto col = row->begin();
        for (int j = 0; j < n; ++j)
          data[ind++] = *(col++);
        row++;
      }
    }
    CreateDeviceArray();
    Resize(m,n);
    CopyToDevice(data.data());
  }

  ~GPUArray2D() 
  {
    if (arr_h_.data_)
      cudaCheckError(cudaFree(arr_h_.data_));
    if (arr_d_)
      cudaCheckError(cudaFree(arr_d_));
  }

  inline int GetNumRows() const { return arr_h_.nrows_; }
  inline int GetNumCols() const { return arr_h_.ncols_; }
  inline std::size_t size() const { return arr_h_.size(); }

  const DeviceArray2D<T>& GetDeviceArray() const { return *arr_d_; }
  DeviceArray2D<T>& GetDeviceArray() { return *arr_d_; }

  const T* GetDeviceData() const { return arr_d_->data_; }
  T* GetDeviceData() { return arr_d_->data_; }

  inline void Resize(int m, int n)
  {
    if (m != arr_h_.nrows_ || n != arr_h_.ncols_)
    {
      //Size has changed, so first free existing data
      if (arr_h_.data_)
        cudaCheckError(cudaFree(arr_h_.data_));

      arr_h_.nrows_ = m;
      arr_h_.ncols_ = n;
      if (m*n > 0)
        cudaCheckError(cudaMalloc(&arr_h_.data_, m*n*sizeof(T)));
      else
        arr_h_.data_ = nullptr;

      //Copy host struct to device struct
      cudaCheckError(cudaMemcpy(arr_d_, &arr_h_, sizeof(DeviceArray2D<T>), cudaMemcpyHostToDevice));
    }
  }

  inline void clear() { Resize(0, 0); }

  inline void CopyToDevice(const T* data)
  {
    cudaCheckError(cudaMemcpy(arr_h_.data_, data, arr_h_.size()*sizeof(T), cudaMemcpyHostToDevice));
  }

  inline Array2D<T> CopyToHost() const {
    Array2D<T> mat_h(arr_h_.nrows_, arr_h_.ncols_);
    cudaCheckError(cudaMemcpy(mat_h.GetDataVector().data(), arr_h_.data_, arr_h_.size()*sizeof(T), cudaMemcpyDeviceToHost));
    return mat_h;
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
      cudaCheckError(cudaMalloc(&arr_d_, sizeof(DeviceArray1D<T>)));
  }

  DeviceArray2D<T> arr_h_; //host side struct
  DeviceArray2D<T>* arr_d_ = nullptr; //pointer to device side struct
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const GPUArray2D<T>& mat)
{
  os << mat.CopyToHost();
  return os;
}

}