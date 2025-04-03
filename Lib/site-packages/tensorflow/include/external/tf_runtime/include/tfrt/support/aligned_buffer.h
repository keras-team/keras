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

// This file declares AlignedAllocator and AlignedBuffer.

#ifndef TFRT_SUPPORT_ALIGNED_BUFFER_H_
#define TFRT_SUPPORT_ALIGNED_BUFFER_H_

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "tfrt/support/alloc.h"

namespace tfrt {

namespace internal {

template <typename T, size_t Align = alignof(void*)>
struct AlignedAllocator {
  static_assert(alignof(T) <= Align,
                "The alignment of T must not be larger than Align.");

  using value_type = T;

  constexpr AlignedAllocator() noexcept = default;

  constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;

  template <class Other>
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr AlignedAllocator(const AlignedAllocator<Other, Align>&) noexcept {}

  T* allocate(size_t n) {
    auto* ptr = AlignedAlloc(Align, n * sizeof(T));
    return static_cast<T*>(ptr);
  }
  void deallocate(T* p, size_t) { AlignedFree(p); }

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Align>;
  };
};

template <typename T, size_t Align>
inline bool operator==(const AlignedAllocator<T, Align>& x,
                       const AlignedAllocator<T, Align>& y) {
  return true;
}

template <typename T, size_t Align>
inline bool operator!=(const AlignedAllocator<T, Align>& x,
                       const AlignedAllocator<T, Align>& y) {
  return !(x == y);
}

}  // namespace internal

// AlignedBuffer is a dynamic buffer with explicit alignment. It is used when
// explicit alignment is needed for a sequence of bytes, eg. BEF and binary
// attributes.
template <size_t Align>
using AlignedBuffer =
    std::vector<uint8_t, internal::AlignedAllocator<uint8_t, Align>>;

}  // namespace tfrt

#endif  // TFRT_SUPPORT_ALIGNED_BUFFER_H_
