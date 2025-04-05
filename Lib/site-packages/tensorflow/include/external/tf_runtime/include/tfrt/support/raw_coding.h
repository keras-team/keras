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

// Utilities to decode unsigned integer
//
// This file provides utilities to decode unsigned integer.

#ifndef TFRT_SUPPORT_RAW_CODING_H_
#define TFRT_SUPPORT_RAW_CODING_H_

#include <cstring>

#include "tfrt/support/forward_decls.h"

namespace tfrt {

inline bool isLittleEndian() {
  const int32_t i = 1;
  return reinterpret_cast<const char*>(&i)[0] == 1;
}

// Lower-level versions of Get... that read directly from a character buffer
// without any bounds checking.

inline uint16_t DecodeFixed16(const char* ptr) {
  if (isLittleEndian()) {
    // Load the raw bytes
    uint16_t result;
    // gcc optimizes this to a plain load
    std::memcpy(&result, ptr, sizeof(result));
    return result;
  } else {
    return ((static_cast<uint16_t>(static_cast<unsigned char>(ptr[0]))) |
            (static_cast<uint16_t>(static_cast<unsigned char>(ptr[1])) << 8));
  }
}

inline uint32_t DecodeFixed32(const char* ptr) {
  if (isLittleEndian()) {
    // Load the raw bytes
    uint32_t result;
    // gcc optimizes this to a plain load
    std::memcpy(&result, ptr, sizeof(result));
    return result;
  } else {
    return ((static_cast<uint32_t>(static_cast<unsigned char>(ptr[0]))) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[1])) << 8) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[2])) << 16) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[3])) << 24));
  }
}

inline uint64_t DecodeFixed64(const char* ptr) {
  if (isLittleEndian()) {
    // Load the raw bytes
    uint64_t result;
    // gcc optimizes this to a plain load
    std::memcpy(&result, ptr, sizeof(result));
    return result;
  } else {
    uint64_t lo = DecodeFixed32(ptr);
    uint64_t hi = DecodeFixed32(ptr + 4);
    return (hi << 32) | lo;
  }
}

}  // namespace tfrt

#endif  // TFRT_SUPPORT_RAW_CODING_H_
