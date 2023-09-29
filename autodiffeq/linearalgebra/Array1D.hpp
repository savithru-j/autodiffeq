// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#include <vector>
#include <ostream>

namespace autodiffeq
{

template<class T = double>
class Array1D
{
public:

  Array1D() = default;
  Array1D(int m) : data_(m) {}
  Array1D(int m, const T& val) : data_(m, val) {}

  Array1D(const std::vector<T>& v) : data_(v) {}

  Array1D(const std::initializer_list<T>& v)
  {
    if (v.size() > 0) 
    {
      data_.resize(v.size());
      auto row = v.begin();
      for (std::size_t i = 0; i < v.size(); ++i)
        data_[i] = *(row++);
    }
  }

  inline int m() const { return data_.size(); }
  inline std::size_t size() const { return data_.size(); }

  inline const T& operator()(int i) const { return data_[i]; }
  inline T& operator()(int i) { return data_[i]; }

  inline const T& operator[](int i) const { return data_[i]; }
  inline T& operator[](int i) { return data_[i]; }

  inline const T& back() const { return data_.back(); }

  inline typename std::vector<T>::iterator begin() { return data_.begin(); }
  inline typename std::vector<T>::iterator end() { return data_.end(); }

  inline typename std::vector<T>::const_iterator cbegin() const { return data_.cbegin(); }
  inline typename std::vector<T>::const_iterator cend() const { return data_.cend(); }

  inline void resize(int m, const T& val = 0.0) { data_.resize(m, val); }

  inline void clear() { data_.clear(); }

  inline void push_back(const T& val) { data_.push_back(val); }

  inline void insert(typename std::vector<T>::const_iterator pos, const T& val)
  {
    data_.insert(pos, val);
  }

  inline void SetValue(const T& val) { std::fill(data_.begin(), data_.end(), val); }

  inline const T* data() const { return data_.data(); }
  inline T* data() { return data_.data(); }

  inline const std::vector<T>& GetDataVector() const { return data_; }
  inline std::vector<T>& GetDataVector() { return data_; }

protected:
  std::vector<T> data_;
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const Array1D<T>& v)
{
  for (int i = 0; i < v.m()-1; ++i)
    os << v[i] << ", ";
  os << v.back();
  return os;
}

}