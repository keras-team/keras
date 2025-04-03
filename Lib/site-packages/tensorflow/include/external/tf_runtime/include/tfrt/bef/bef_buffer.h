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

// This file declares a buffer type for storing BEF binary.

#ifndef TFRT_SUPPORT_BEF_BUFFER_H_
#define TFRT_SUPPORT_BEF_BUFFER_H_

#include "tfrt/bef/bef_encoding.h"
#include "tfrt/support/aligned_buffer.h"

namespace tfrt {

// A BEF file should be loaded to a buffer aligned by kAttributeMaxAlignment.
constexpr size_t GetRequiredBefAlignment() { return kAttributeMaxAlignment; }

// Buffer for storing BEF binary.
using BefBuffer = AlignedBuffer<GetRequiredBefAlignment()>;

}  // namespace tfrt

#endif  // TFRT_SUPPORT_BEF_BUFFER_H_
