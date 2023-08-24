// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <cassert>
#include <autodiffeq/linearalgebra/Array1D.hpp>

namespace autodiffeq
{

template<class T = double>
class SolutionHistory
{
public:

  explicit SolutionHistory(int sol_dim, const Array1D<double>& time_vec,
                           int storage_stride = 1) : 
      sol_dim_(sol_dim), time_vec_(time_vec), storage_stride_(storage_stride)
  {
    num_steps_stored_ = time_vec_.size() / storage_stride_ + 1;
    data_.resize(num_steps_stored_*sol_dim_, T(0));
  }

  inline int GetSolutionSize() const { return sol_dim_; }
  inline int GetNumSteps() const { return time_vec_.size(); }
  inline int GetNumStepsStored() const { return num_steps_stored_; }

  inline const T& operator()(int step, int state) const { return data_[(step/storage_stride_)*sol_dim_ + state]; }
  inline T& operator()(int step, int state) { return data_[(step/storage_stride_)*sol_dim_ + state]; }

  //! Returns the solution at a given step
  inline void GetSolution(const int step, Array1D<T>& sol) const {
    assert(step >= 0 && step < (int) time_vec_.size());
    assert(step % storage_stride_ == 0);
    const int offset = (step/storage_stride_)*sol_dim_;
    for (int m = 0; m < sol_dim_; ++m)
      sol[m] = data_[offset + m];
  }

  //! Sets the solution at a given step
  inline void SetSolution(const int step, const Array1D<T>& sol) {
    assert(step >= 0 && step < (int) time_vec_.size());
    assert(step % storage_stride_ == 0);
    const int offset = (step/storage_stride_)*sol_dim_;
    for (int m = 0; m < sol_dim_; ++m)
      data_[offset + m] = sol[m];
  }

  inline double GetTime(int step) const { return time_vec_(step); }

  inline const Array1D<T>& GetData() const { return data_; }
  inline Array1D<T>& GetData() { return data_; }

protected:
  int sol_dim_ = 0;
  Array1D<double> time_vec_;
  int storage_stride_;
  std::size_t num_steps_stored_;
  Array1D<T> data_;
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const SolutionHistory<T>& sol_hist)
{
  int sol_dim = sol_hist.GetSolutionSize();
  int nt = sol_hist.GetNumSteps();
  for (int step = 0; step < nt; ++step)
  {
    for (int i = 0; i < sol_dim-1; ++i)
      os << sol_hist(step, i) << ", ";
    os << sol_hist(step, sol_dim-1) << std::endl;
  }
  return os;
}

}