// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <vector>
#include <ostream>

namespace autodiffeq
{

template<class T = double>
class Array2D
{
public:

  Array2D() = default;
  Array2D(int m, int n) : nrows_(m), ncols_(n), data_(nrows_*ncols_) {}
  Array2D(int m, int n, const T& val) : nrows_(m), ncols_(n), data_(nrows_*ncols_, val) {}

  Array2D(const std::initializer_list<std::initializer_list<T>>& mat) 
  {
    nrows_ = (int) mat.size();
    if (nrows_ > 0)
    {
      auto row = mat.begin();
      ncols_ = row->size();
      data_.resize(nrows_*ncols_);
      int ind = 0;
      for (int i = 0; i < nrows_; ++i)
      {
        assert((int) row->size() == ncols_);      
        auto col = row->begin();
        for (int j = 0; j < ncols_; ++j)
          data_[ind++] = *(col++);
        row++;
      }
    }
  }

  inline int GetNumRows() const { return nrows_; }
  inline int GetNumCols() const { return ncols_; }
  inline std::size_t size() const { return data_.size(); }

  inline const T& operator()(int i, int j) const { return data_[i*ncols_ + j]; }
  inline T& operator()(int i, int j) { return data_[i*ncols_ + j]; }

  inline const T& operator[](int i) const { return data_[i]; }
  inline T& operator[](int i) { return data_[i]; }

  inline typename std::vector<T>::iterator begin() { return data_.begin(); }
  inline typename std::vector<T>::iterator end() { return data_.end(); }

  inline typename std::vector<T>::const_iterator cbegin() const { return data_.cbegin(); }
  inline typename std::vector<T>::const_iterator cend() const { return data_.cend(); }

  inline void resize(int m, int n, const T& val = 0.0) { 
    nrows_ = m;
    ncols_ = n;
    data_.resize(nrows_*ncols_, val); 
  }

  inline void clear() { 
    nrows_ = 0;
    ncols_ = 0;
    data_.clear(); 
  }

  inline void SetValue(const T& val) { std::fill(data_.begin(), data_.end(), val); }

  inline const std::vector<T>& GetDataVector() const { return data_; }
  inline std::vector<T>& GetDataVector() { return data_; }

protected:
  int nrows_, ncols_;
  std::vector<T> data_;
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const Array2D<T>& v)
{
  const int M = v.GetNumRows();
  const int N = v.GetNumCols();
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N-1; ++j)
      os << v(i,j) << ", ";
    os << v(i,N-1) << std::endl;
  }
  return os;
}

}