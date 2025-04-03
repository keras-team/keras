/*
    Copyright (c) 2005-2021 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/
#ifndef __TBB_aligned_space_H
#define __TBB_aligned_space_H

#include <cstddef>

#include "_template_helpers.h"

namespace tbb {
namespace detail {
inline namespace d0 {

//! Block of space aligned sufficiently to construct an array T with N elements.
/** The elements are not constructed or destroyed by this class.
    @ingroup memory_allocation */
template<typename T, std::size_t N = 1>
class aligned_space {
    alignas(alignof(T)) std::uint8_t aligned_array[N * sizeof(T)];

public:
    //! Pointer to beginning of array
    T* begin() const { return punned_cast<T*>(&aligned_array); }

    //! Pointer to one past last element in array.
    T* end() const { return begin() + N; }
};

} // namespace d0
} // namespace detail
} // namespace tbb

#endif /* __TBB_aligned_space_H */
