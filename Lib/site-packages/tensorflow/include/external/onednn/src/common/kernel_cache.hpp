/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef COMMON_KERNEL_CACHE_HPP
#define COMMON_KERNEL_CACHE_HPP

#include <cstddef>
#include <memory>
#include <thread>

#include "c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace kernel_cache {

struct key_impl_t {
    key_impl_t() = default;
    virtual ~key_impl_t() = default;

    key_impl_t(const key_impl_t &) = delete;
    key_impl_t &operator=(const key_impl_t &) = delete;

    virtual bool compare(const key_impl_t *key_impl) const = 0;
    virtual size_t hash() const = 0;
};

// The kernel-cache implementation relies on the copy-constructor. This class is
// marked final to prevent object slicing.
struct key_t final {
    key_t(const std::shared_ptr<key_impl_t> &impl,
            bool has_runtime_dependencies = false)
        : impl_(impl)
        , thread_id_(std::this_thread::get_id())
        , has_runtime_dependencies_(has_runtime_dependencies) {}
    key_t(std::shared_ptr<key_impl_t> &&impl,
            bool has_runtime_dependencies = false)
        : impl_(std::move(impl))
        , thread_id_(std::this_thread::get_id())
        , has_runtime_dependencies_(has_runtime_dependencies) {}

    bool operator==(const key_t &other) const {
        return impl_->compare(other.impl_.get());
    };
    size_t hash() const { return impl_->hash(); };

    key_impl_t *impl() const { return impl_.get(); }

    const std::thread::id &thread_id() const { return thread_id_; }
    bool has_runtime_dependencies() const { return has_runtime_dependencies_; }

protected:
    std::shared_ptr<key_impl_t> impl_;

private:
    // Thread ID is not used as part of the key, it's only used to get
    // information about what thread inserted the key and the corresponding
    // primitive to handle some multithreaded scenarios.
    std::thread::id thread_id_;

    // Used to correctly handle destruction on process termination. If there are
    // runtime dependencies, attempts to destroy the cached object may fail.
    bool has_runtime_dependencies_;
};

struct value_impl_t {
    value_impl_t() = default;
    virtual ~value_impl_t() = default;

    value_impl_t(const value_impl_t &) = delete;
    value_impl_t &operator=(const value_impl_t &) = delete;
};

// The kernel-cache implementation relies on the copy-constructor. This class is
// marked final to prevent object slicing.
struct value_t final {
    value_t() = default;
    value_t(std::nullptr_t) : value_t() {};
    value_t(const std::shared_ptr<value_impl_t> &impl) : impl_(impl) {}
    value_t(std::shared_ptr<value_impl_t> &&impl) : impl_(std::move(impl)) {}
    const std::shared_ptr<value_impl_t> &impl() const { return impl_; }
    std::shared_ptr<value_impl_t> &impl() { return impl_; }
    std::shared_ptr<value_impl_t> release() {
        std::shared_ptr<value_impl_t> ret = nullptr;
        std::swap(ret, impl_);
        return ret;
    }
    bool is_empty() const { return impl_ == nullptr; }

private:
    std::shared_ptr<value_impl_t> impl_;
};

struct iface_t {
    struct cache_t;
    struct result_t {
        result_t() : status(status::success) {};
        result_t(value_t p, status_t s) : value(std::move(p)), status(s) {}
        bool is_empty() const { return value.is_empty(); }
        value_t &get_value() { return value; }
        value_t value;
        status_t status;
    };

    using create_func_t = result_t (&)(void *);
    using create_func_ptr_t = result_t (*)(void *);

    iface_t(cache_t &cache) : cache_(cache) {};

    ~iface_t() = default;

    status_t set_capacity(int capacity);
    int get_capacity() const;
    int get_size() const;

    result_t get_or_create(
            const key_t &key, create_func_t create, void *create_context);

private:
    cache_t &cache_;
};

iface_t get();

} // namespace kernel_cache
} // namespace impl
} // namespace dnnl

#endif
