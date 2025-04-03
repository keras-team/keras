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

#ifndef COMMON_CACHE_BLOB_HPP
#define COMMON_CACHE_BLOB_HPP

#include <cstdint>
#include <cstring>
#include <memory>

#include "c_types_map.hpp"

namespace dnnl {
namespace impl {

struct cache_blob_impl_t {
    cache_blob_impl_t() = delete;
    cache_blob_impl_t(uint8_t *data, size_t size)
        : pos_(0), data_(data), size_(size) {}

    status_t add_binary(const uint8_t *binary, size_t binary_size) {
        if (!binary || binary_size == 0) { return status::invalid_arguments; }
        if (pos_ + sizeof(binary_size) + binary_size > size_) {
            return status::invalid_arguments;
        }

        std::memcpy(data_ + pos_, &binary_size, sizeof(binary_size));
        pos_ += sizeof(binary_size);
        std::memcpy(data_ + pos_, binary, binary_size);
        pos_ += binary_size;
        return status::success;
    }

    status_t get_binary(const uint8_t **binary, size_t *binary_size) {
        if (!binary || !binary_size) { return status::invalid_arguments; }
        if (pos_ >= size_) { return status::invalid_arguments; }
        std::memcpy(binary_size, data_ + pos_, sizeof(*binary_size));
        pos_ += sizeof(*binary_size);
        (*binary) = data_ + pos_;
        pos_ += *binary_size;
        return status::success;
    }

    status_t add_value(const uint8_t *value_ptr, size_t size) {
        if (!value_ptr) { return status::invalid_arguments; }
        if (pos_ + size > size_) { return status::invalid_arguments; }

        std::memcpy(data_ + pos_, value_ptr, size);
        pos_ += size;
        return status::success;
    }

    status_t get_value(uint8_t *value_ptr, size_t size) {
        if (!value_ptr) { return status::invalid_arguments; }
        if (pos_ >= size_) { return status::invalid_arguments; }
        std::memcpy(value_ptr, data_ + pos_, size);
        pos_ += size;
        return status::success;
    }

private:
    size_t pos_;
    uint8_t *data_;
    size_t size_;
};

struct cache_blob_t {
    cache_blob_t() = default;
    cache_blob_t(uint8_t *data, size_t size)
        : impl_(std::make_shared<cache_blob_impl_t>(data, size)) {}

    status_t add_binary(const uint8_t *binary, size_t binary_size) {
        if (!impl_) return status::runtime_error;
        return impl_->add_binary(binary, binary_size);
    }

    status_t get_binary(const uint8_t **binary, size_t *binary_size) const {
        if (!impl_) return status::runtime_error;
        return impl_->get_binary(binary, binary_size);
    }

    status_t add_value(const uint8_t *value_ptr, size_t size) {
        if (!impl_) return status::runtime_error;
        return impl_->add_value(value_ptr, size);
    }

    status_t get_value(uint8_t *value_ptr, size_t size) {
        if (!impl_) return status::runtime_error;
        return impl_->get_value(value_ptr, size);
    }

    explicit operator bool() const { return bool(impl_); }

private:
    std::shared_ptr<cache_blob_impl_t> impl_;
};

} // namespace impl
} // namespace dnnl

#endif
