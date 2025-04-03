/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef COMMON_SERIALIZATION_STREAM_HPP
#define COMMON_SERIALIZATION_STREAM_HPP

#include <cstdint>
#include <vector>
#include <type_traits>

namespace dnnl {
namespace impl {

struct serialization_stream_t {
    serialization_stream_t() = default;

    template <typename T>
    void write(const T ptr, size_t nelems = 1) {
        using non_pointer_type = typename std::remove_pointer<T>::type;

        static_assert(std::is_pointer<T>::value,
                "T is expected to be a pointer type.");
        static_assert(!std::is_pointer<non_pointer_type>::value,
                "T cannot be a pointer to pointer.");
        static_assert(!std::is_class<non_pointer_type>::value,
                "non-pointer type is expected to be a trivial type to avoid "
                "padding issues.");
        static_assert(!std::is_array<non_pointer_type>::value,
                "non-pointer type cannot be an array.");

        write_impl((const void *)ptr, sizeof(non_pointer_type) * nelems);
    }

    bool empty() const { return data_.empty(); }

    const std::vector<uint8_t> &get_data() const { return data_; }

private:
    void write_impl(const void *ptr, size_t size) {
        const auto *p = reinterpret_cast<const uint8_t *>(ptr);
        data_.insert(data_.end(), p, p + size);
    }

    std::vector<uint8_t> data_;
};

} // namespace impl
} // namespace dnnl

#endif
