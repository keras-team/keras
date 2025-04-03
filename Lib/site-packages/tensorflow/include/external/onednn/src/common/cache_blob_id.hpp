/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef COMMON_CACHE_BLOB_ID_HPP
#define COMMON_CACHE_BLOB_ID_HPP

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>
#include <type_traits>

#include "common/serialization_stream.hpp"

namespace dnnl {
namespace impl {

struct primitive_desc_t;
struct cache_blob_id_t {
    cache_blob_id_t() : is_initialized_ {false} {}
    cache_blob_id_t(const cache_blob_id_t &other)
        : sstream_(other.is_initialized_ ? other.sstream_
                                         : serialization_stream_t {})
        , is_initialized_(!sstream_.empty()) {}

    cache_blob_id_t(cache_blob_id_t &&other) = delete;
    cache_blob_id_t &operator=(const cache_blob_id_t &other) = delete;
    cache_blob_id_t &operator=(cache_blob_id_t &&other) = delete;

    const std::vector<uint8_t> &get(
            const engine_t *engine, const primitive_desc_t *pd);

private:
    serialization_stream_t sstream_;
    std::once_flag flag_;

    // The `std::once_flag` is neither copyable nor movable therefore we
    // define a copy constructor that skips copying the `flag_`. To be able
    // to carry over the `flag_`'s state from the `other` object we introduce
    // an atomic `is_initialized_` flag.
    std::atomic<bool> is_initialized_;
};

} // namespace impl
} // namespace dnnl

#endif
