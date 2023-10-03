// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <cassert>
#include <autodiffeq/linearalgebra/GPUArray1D.cuh>
#include <autodiffeq/linearalgebra/GPUArray2D.cuh>

namespace autodiffeq
{

#ifdef ENABLE_CUDA
namespace gpu
{
  template<typename T>
  __global__
  void GetSolution(const int timestep, const int storage_stride, const DeviceArray2D<T>& sol_hist,
                   DeviceArray1D<T>& sol)
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int storage_step = timestep / storage_stride;
    auto sol_dim = sol.size();
    if (i < sol_dim) 
      sol(i) = sol_hist(storage_step, i);
  }

  template<typename T>
  __global__
  void SetSolution(const int timestep, const int storage_stride, const DeviceArray1D<T>& sol,
                   DeviceArray2D<T>& sol_hist)
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int storage_step = timestep / storage_stride;
    auto sol_dim = sol.size();
    if (i < sol_dim) 
      sol_hist(storage_step, i) = sol(i);
  }
}
#endif

template<class T = double>
class GPUSolutionHistory
{
public:

  explicit GPUSolutionHistory(int sol_dim, const Array1D<double>& time_vec,
                           int storage_stride = 1) : 
      sol_dim_(sol_dim), time_vec_(time_vec), storage_stride_(storage_stride)
  {
    num_steps_stored_ = time_vec_.size() / storage_stride_;
    if (time_vec.size() % storage_stride_ != 0)
      num_steps_stored_++;
    data_.Resize(num_steps_stored_,sol_dim_);
    data_.SetValue(0);
  }

  inline int GetSolutionSize() const { return sol_dim_; }
  inline int GetNumSteps() const { return time_vec_.size(); }
  inline int GetNumStepsStored() const { return num_steps_stored_; }

  //! Returns the solution at a given step
  inline void GetSolution(const int step, GPUArray1D<T>& sol) const {
    assert(step >= 0 && step < (int) time_vec_.size());
    assert(step % storage_stride_ == 0);
    gpu::GetSolution<<<(sol_dim_+255)/256,256>>>(step, storage_stride_, data_.GetDeviceArray(),
                                                 sol.GetDeviceArray());
  }

  //! Sets the solution at a given step
  inline void SetSolution(const int step, const GPUArray1D<T>& sol) {
    assert(step >= 0 && step < (int) time_vec_.size());
    assert(step % storage_stride_ == 0);
    gpu::SetSolution<<<(sol_dim_+255)/256,256>>>(step, storage_stride_, sol.GetDeviceArray(),
                                                 data_.GetDeviceArray());
  }

  inline double GetTime(int step) const { return time_vec_(step); }

  inline const GPUArray2D<T>& GetData() const { return data_; }
  inline GPUArray2D<T>& GetData() { return data_; }

protected:
  int sol_dim_ = 0;
  Array1D<double> time_vec_;
  int storage_stride_;
  std::size_t num_steps_stored_;
  GPUArray2D<T> data_; //solution history data [num_steps x num_states]
};

}