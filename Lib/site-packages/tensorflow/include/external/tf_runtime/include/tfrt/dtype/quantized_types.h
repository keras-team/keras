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

// This file defines quantized integer types: quint8, quint16, qin8, qint16,
// qint32.

#ifndef TFRT_DTYPE_QUANTIZED_TYPES_H_
#define TFRT_DTYPE_QUANTIZED_TYPES_H_

#include "llvm/Support/raw_ostream.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// Templated type mapping quantized types to corresponding integer type. The
// client should get the real C++ type via
// tfrt::TypeForDTypeKind<DType::QUI8>::Type, etc.
template <typename UnderlyingT>
struct QuantizedInteger {
  QuantizedInteger() : value(0) {}
  explicit QuantizedInteger(UnderlyingT v) : value(v) {}
  UnderlyingT value;
};

// Print QuantizedInteger in format as type(value), e.g. qu8(2).
template <typename UnderlyingT>
raw_ostream& operator<<(raw_ostream& os, QuantizedInteger<UnderlyingT> in) {
  return os << 'q' << (std::is_signed<UnderlyingT>() ? 'i' : 'u')
            << (sizeof(UnderlyingT) * 8) << '(' << +in.value << ')';
}

using quint8 = QuantizedInteger<uint8_t>;
using quint16 = QuantizedInteger<uint16_t>;
using qint8 = QuantizedInteger<int8_t>;
using qint16 = QuantizedInteger<int16_t>;
using qint32 = QuantizedInteger<int32_t>;

}  // namespace tfrt

#endif  // TFRT_DTYPE_QUANTIZED_TYPES_H_
