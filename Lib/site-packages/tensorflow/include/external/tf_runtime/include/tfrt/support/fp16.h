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

// This file defines the half precision floating type: fp16.

#ifndef TFRT_SUPPORT_FP16_H_
#define TFRT_SUPPORT_FP16_H_

#include "llvm/Support/raw_ostream.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// This is just a placeholder type telling core TFRT that fp16 has the same
// size as uint16_t. The client should get its real C++ type via
// tfrt::TypeForDTypeKind<DType::F16>::Type.
struct fp16 {
  fp16() : value(0) {}
  explicit fp16(uint16_t v) : value(v) {}
  uint16_t value;
};

inline raw_ostream &operator<<(raw_ostream &os, fp16 v) {
  return os << "fp16(" << v.value << ')';
}
}  // namespace tfrt

#endif  // TFRT_SUPPORT_FP16_H_
