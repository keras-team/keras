/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file defines the DType enum and helpers for working with it.

#ifndef TFRT_DTYPE_DTYPE_H_
#define TFRT_DTYPE_DTYPE_H_

#include <complex>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <type_traits>

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/dtype/quantized_types.h"
#include "tfrt/support/bf16.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/fp16.h"

namespace tfrt {

enum class DType : uint8_t {
  Invalid = 0,
  FirstDType = 1,
#define DTYPE(ENUM, VALUE) ENUM = VALUE,
#include "tfrt/dtype/dtype.def"
#undef DTYPE
  LastDType,
  // Valid types that are not natively supported by TFRT.
  Unsupported = LastDType,
};

// Return the size of one value of this dtype when represented on the host.
size_t GetHostSize(DType dtype);

// Return the alignment of this dtype when represented on the host.
size_t GetHostAlignment(DType dtype);

inline bool IsValid(DType dtype) { return dtype != DType::Invalid; }
inline bool IsInvalid(DType dtype) { return dtype == DType::Invalid; }
inline bool IsUnsupported(DType dtype) {
  return dtype == DType::Unsupported || dtype == DType::Resource ||
         dtype == DType::Variant;
}

bool IsTriviallyCopyable(DType dtype);

// Support printing of dtype enums, e.g. i32, f32.
raw_ostream &operator<<(raw_ostream &os, DType dtype);
// Add support for std::ostream to make DType friendly to std::ostream based
// tools, e.g. Google test.
std::ostream &operator<<(std::ostream &os, DType dtype);

// Provides interconversions between C++ type and DTypes at compile time.
//
// GetDType<T>() is the DType for the C++ type T.
//
// TypeForDTypeKind<DT> is the C++ type for DType kind DT. For non-trivial type,
// it can only return a storage-only type.

// Provide a way to get the DType for a specified C++ type at compile time.
//
// Explicitly delete the primary template, so an invalid type T will result in a
// compiler error instead of link error.
template <typename T>
constexpr DType GetDType() = delete;

// Provide a way to get the C++ type for a specified DType Kind at compile
// time.
template <DType K>
struct DTypeData;
template <DType K>
using TypeForDTypeKind = typename DTypeData<K>::Type;

namespace detail {

template <DType dtype>
struct UnsupportedDataType {
  friend raw_ostream &operator<<(raw_ostream &os, UnsupportedDataType data) {
    return os << "UnsupportedDataType<" << DType(dtype) << '>';
  }
};

template <typename T>
struct IsDTypeTriviallyCopyable : std::is_trivially_copyable<T> {};

template <DType dtype>
struct IsDTypeTriviallyCopyable<UnsupportedDataType<dtype>> : std::false_type {
};

}  // namespace detail

// TFRT_REGISTER_DTYPE is a macro to register a non-trivial C++ type to a DType.
#define TFRT_REGISTER_DTYPE(CPP_TYPE, ENUM)     \
  template <>                                   \
  inline constexpr DType GetDType<CPP_TYPE>() { \
    return DType{DType::ENUM};                  \
  }

#define TFRT_DEFINE_DTYPE(ENUM, CPP_TYPE, NAME)        \
  TFRT_REGISTER_DTYPE(CPP_TYPE, ENUM)                  \
  template <>                                          \
  struct DTypeData<DType::ENUM> {                      \
    static constexpr DType kDType{DType::ENUM};        \
    using Type = CPP_TYPE;                             \
    static constexpr const char *kName = NAME;         \
    static constexpr bool kIsTriviallyCopyable =       \
        detail::IsDTypeTriviallyCopyable<CPP_TYPE>();  \
    static constexpr size_t kByteSize =                \
        kIsTriviallyCopyable ? sizeof(CPP_TYPE) : -1;  \
    static constexpr size_t kAlignment =               \
        kIsTriviallyCopyable ? alignof(CPP_TYPE) : -1; \
  };

// LINT.IfChange
TFRT_DEFINE_DTYPE(UI8, uint8_t, "u8")
TFRT_DEFINE_DTYPE(UI16, uint16_t, "u16")
TFRT_DEFINE_DTYPE(UI32, uint32_t, "u32")
TFRT_DEFINE_DTYPE(UI64, uint64_t, "u64")
TFRT_DEFINE_DTYPE(I1, bool, "i1")
TFRT_DEFINE_DTYPE(I8, int8_t, "i8")
TFRT_DEFINE_DTYPE(I16, int16_t, "i16")
TFRT_DEFINE_DTYPE(I32, int32_t, "i32")
TFRT_DEFINE_DTYPE(I64, int64_t, "i64")
TFRT_DEFINE_DTYPE(BF16, bf16, "bf16")
TFRT_DEFINE_DTYPE(F16, fp16, "f16")
TFRT_DEFINE_DTYPE(F32, float, "f32")
TFRT_DEFINE_DTYPE(F64, double, "f64")
// TODO(tfrt-devs): Consider creating a special CPP string type for TFRT.
TFRT_DEFINE_DTYPE(String, std::string, "str")
TFRT_DEFINE_DTYPE(Complex64, std::complex<float>,
                  "complex64")  // Single precision complex.
TFRT_DEFINE_DTYPE(Complex128, std::complex<double>,
                  "complex128")  // Double precision complex.
TFRT_DEFINE_DTYPE(QUI8, quint8, "qu8")
TFRT_DEFINE_DTYPE(QUI16, quint16, "qu16")
TFRT_DEFINE_DTYPE(QI8, qint8, "qi8")
TFRT_DEFINE_DTYPE(QI16, qint16, "qi16")
TFRT_DEFINE_DTYPE(QI32, qint32, "qi32")

TFRT_DEFINE_DTYPE(Resource, detail::UnsupportedDataType<DType::Resource>,
                  "resource")
TFRT_DEFINE_DTYPE(Variant, detail::UnsupportedDataType<DType::Variant>,
                  "variant")

// LINT.ThenChange(//depot/tf_runtime/include/tfrt/dtype/dtype.def)

#undef TFRT_DEFINE_DTYPE

// Dispatch to an overload of function f based on the given dtype.
//
// Example usage:
//
// // Prints the given data assuming it contains a value of the given dtype.
// void Print(DType dtype, void* data) {
//   dispatchByDType(dtype, [data](auto type_tag) {
//     using T = typename decltype(type_tag)::type;
//     llvm::outs() << *static_cast<const T*>(data);
//   });
// }
template <typename F>
decltype(auto) DispatchByDType(DType dtype, F &&f) {
  switch (dtype) {
#define DTYPE(ENUM, VALUE) \
  case DType::ENUM:        \
    return f(DTypeData<DType::ENUM>());
#include "tfrt/dtype/dtype.def"
#undef DTYPE
    default:
      llvm_unreachable("Invalid dtype encountered in DispatchByDType");
  }
}

LLVM_ATTRIBUTE_ALWAYS_INLINE size_t GetHostSize(DType dtype) {
  return DispatchByDType(dtype, [](auto d) { return d.kByteSize; });
}

LLVM_ATTRIBUTE_ALWAYS_INLINE size_t GetHostAlignment(DType dtype) {
  return DispatchByDType(dtype, [](auto d) { return d.kAlignment; });
}

LLVM_ATTRIBUTE_ALWAYS_INLINE bool IsTriviallyCopyable(DType dtype) {
  return DispatchByDType(dtype, [](auto d) { return d.kIsTriviallyCopyable; });
}

}  // namespace tfrt

#endif  // TFRT_DTYPE_DTYPE_H_
