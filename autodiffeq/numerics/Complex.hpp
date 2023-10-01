// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#pragma once

#ifdef ENABLE_CUDA

#include <cuda/std/complex>
template<typename T>
using complex = cuda::std::complex<T>;

#else

#include <complex>
template<typename T>
using complex = std::complex<T>;

#endif