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

// Hashing Utilities
//
// This file declares hashing utilities.
#ifndef TFRT_SUPPORT_HASH_UTIL_H_
#define TFRT_SUPPORT_HASH_UTIL_H_

#include "llvm/ADT/StringRef.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

uint32_t Hash32(const char* data, size_t n, uint32_t seed);
uint64_t Hash64(const char* data, size_t n, uint64_t seed);

inline uint64_t Hash64(const char* data, size_t n) {
  return Hash64(data, n, 0xDECAFCAFFE);
}

inline uint64_t Hash64(const std::string& str) {
  return Hash64(str.data(), str.size());
}

inline uint64_t Hash64(const string_view& str) {
  return Hash64(str.data(), str.size());
}

inline uint64_t Hash64Combine(uint64_t a, uint64_t b) {
  return a ^ (b + 0x9e3779b97f4a7800ULL + (a << 10) + (a >> 4));
}

}  // namespace tfrt

#endif  // TFRT_SUPPORT_HASH_UTIL_H_
