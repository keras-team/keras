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

// This file declares AlignedAlloc() for allocating dynamic buffer with
// explicit alignment.
#ifndef TFRT_SUPPORT_ALLOC_H_
#define TFRT_SUPPORT_ALLOC_H_

#include <cstddef>

namespace tfrt {

// Note: The returned pointer *must* be deallocated with AlignedFree().
// Deallocating with e.g. free() instead causes runtime issues on Windows that
// are hard to debug.
void* AlignedAlloc(size_t alignment, size_t size);

void AlignedFree(void* ptr);

}  // namespace tfrt

#endif  // TFRT_SUPPORT_ALLOC_H_
