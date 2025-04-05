/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_BIT_CAST_HPP
#define COMMON_BIT_CAST_HPP

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace dnnl {
namespace impl {
namespace utils {

// Returns a value of type T by reinterpretting the representation of the input
// value (part of C++20).
//
// Provides a safe implementation of type punning.
//
// Constraints:
// - U and T must have the same size
// - U and T must be trivially copyable
template <typename T, typename U>
inline T bit_cast(const U &u) {
    static_assert(sizeof(T) == sizeof(U), "Bit-casting must preserve size.");
    static_assert(std::is_trivial<T>::value, "T must be trivially copyable.");
    static_assert(std::is_trivial<U>::value, "U must be trivially copyable.");

    T t;
    // Since bit_cast is used in SYCL kernels it cannot use std::memcpy as it
    // can be implemented as @llvm.objectsize.* + __memcpy_chk for Release
    // builds which cannot be translated to SPIR-V.
    uint8_t *t_ptr = reinterpret_cast<uint8_t *>(&t);
    const uint8_t *u_ptr = reinterpret_cast<const uint8_t *>(&u);
    for (size_t i = 0; i < sizeof(U); i++)
        t_ptr[i] = u_ptr[i];
    return t;
}

} // namespace utils
} // namespace impl
} // namespace dnnl

#endif
