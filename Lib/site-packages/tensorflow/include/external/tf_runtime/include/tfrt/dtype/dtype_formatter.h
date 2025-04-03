// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements data formatter classes for DType.

#ifndef TFRT_DTYPE_DTYPE_FORMATTER_H_
#define TFRT_DTYPE_DTYPE_FORMATTER_H_
#include <type_traits>

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/dtype/dtype.h"

namespace tfrt {

namespace detail {

template <typename T>
struct DTypeFormatter {
  const T &value;
  bool full_precision = false;
};

// Format integral data.
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
raw_ostream &operator<<(raw_ostream &os, const DTypeFormatter<T> &f) {
  return os << +f.value;
}

// Format floating point data.
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
raw_ostream &operator<<(raw_ostream &os, const DTypeFormatter<T> &f) {
  if (f.full_precision) {
    os << llvm::format("%.*g", std::numeric_limits<T>::max_digits10, f.value);
  } else {
    os << f.value;
  }
  return os;
}

// Format complex data.
template <typename T>
raw_ostream &operator<<(raw_ostream &os,
                        const DTypeFormatter<std::complex<T>> &f) {
  return os << '(' << DTypeFormatter<T>{f.value.real(), f.full_precision} << ','
            << DTypeFormatter<T>{f.value.imag(), f.full_precision} << ')';
}

// Format other types.
template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value, bool> = true>
raw_ostream &operator<<(raw_ostream &os, const DTypeFormatter<T> &f) {
  return os << f.value;
}

// Format void* as a value of a given DType.
struct AnyDTypeFormatter {
  DType dtype;
  const void *value;
  bool full_precision;
};

inline raw_ostream &operator<<(raw_ostream &os, const AnyDTypeFormatter &f) {
  DispatchByDType(f.dtype, [&](auto data_type) {
    using T = typename decltype(data_type)::Type;
    os << DTypeFormatter<T>{*static_cast<const T *>(f.value), f.full_precision};
  });
  return os;
}

}  // namespace detail

// Format a value as data of a DType.
//
// If full_precision is true, std::numeric_limits<T>::max_digits10 is used as
// the precision for the floating point values.  These many digits are enough to
// make sure number->text->number is guaranteed to get the same number back.
//
// Example usage:
//
// int i = 2;
// float f = 2.0;
// double d = 2.0;
// llvm::outs() << "int32 value is " << FormatDType(i);
// llvm::outs() << "f32 value is " << FormatDType(f);
// // Format with full precision.
// llvm::outs() << "f64 value is " << FormatDType(d, true);
//
// // Format with explicitly DType value.
// llvm::outs() << "f64 value is " << FormatDType(DType::F64, &d, true);

template <typename T>
detail::DTypeFormatter<T> FormatDType(const T &value,
                                      bool full_precision = false) {
  return detail::DTypeFormatter<T>{value, full_precision};
}

inline detail::AnyDTypeFormatter FormatDType(DType dtype, const void *value,
                                             bool full_precision = false) {
  return detail::AnyDTypeFormatter{dtype, value, full_precision};
}

}  // namespace tfrt
#endif  // TFRT_DTYPE_DTYPE_FORMATTER_H_
