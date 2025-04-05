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

// crc32c Utilities
//
// This file declares crc32c utils.

#ifndef TFRT_SUPPORT_CRC32C_H_
#define TFRT_SUPPORT_CRC32C_H_

#include <stddef.h>

#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace crc32c {

// Return the crc32c of concat(A, data[0, size-1]) where crc is the crc32c of
// some string A. Extend() is often used to maintain the crc32c of a stream of
// data.
uint32_t Extend(uint32_t crc, const char* buf, size_t size);

// Return the crc32c of data[0,n-1]
inline uint32_t Value(const char* data, size_t n) { return Extend(0, data, n); }

static const uint32_t kMaskDelta = 0xa282ead8ul;

// Return a masked representation of crc.
//
// Motivation: it is problematic to compute the CRC of a string that contains
// embedded CRCs. Therefore we recommend that CRCs stored somewhere (e.g., in
// files) should be masked before being stored.
inline uint32_t Mask(uint32_t crc) {
  // Rotate right by 15 bits and add a constant.
  return ((crc >> 15) | (crc << 17)) + kMaskDelta;
}

// Return the crc whose masked representation is masked_crc.
inline uint32_t Unmask(uint32_t masked_crc) {
  uint32_t rot = masked_crc - kMaskDelta;
  return ((rot >> 17) | (rot << 15));
}

}  // namespace crc32c
}  // namespace tfrt

#endif  // TFRT_SUPPORT_CRC32C_H_
