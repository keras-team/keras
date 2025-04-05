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
#ifndef GRAPH_INTERFACE_CONSTANT_TENSOR_CACHE_HPP
#define GRAPH_INTERFACE_CONSTANT_TENSOR_CACHE_HPP

#include <atomic>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <type_traits>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/rw_mutex.hpp"

#include "graph/interface/allocator.hpp"
#include "graph/interface/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace graph {

class constant_buffer_t {
public:
    using malloc_func_t = void *(*)(size_t, impl::engine_t *, allocator_t *);
    using free_func_t = void (*)(void *, impl::engine_t *, allocator_t *);

    // Backends should provide these malloc and free function handles
    constant_buffer_t(size_t size, impl::engine_t *eng, allocator_t *alc,
            malloc_func_t malloc_func, free_func_t free_func)
        : size_(size)
        , eng_(eng)
        , alc_(alc)
        , malloc_func_(malloc_func)
        , free_func_(free_func) {
        data_ = malloc_func_(size, eng, alc);
        eng_->retain();
    }

    virtual ~constant_buffer_t() {
        free_func_(data_, eng_, alc_);
        eng_->release();
    };

    // Disable assignment and copy
    constant_buffer_t(const constant_buffer_t &) = delete;
    constant_buffer_t(constant_buffer_t &&) = delete;
    constant_buffer_t &operator=(const constant_buffer_t &) = delete;
    constant_buffer_t &operator=(constant_buffer_t &&) = delete;

    template <typename T>
    T *data() const {
        return static_cast<T *>(data_);
    }

    size_t size() const { return size_; }

    // used to notify backend the buffer has been evict. backend can use this
    // api to avoid query constant cache frequently to reduce overhead.
    virtual void notify_evict() {}

protected:
    void *data_;
    size_t size_;
    impl::engine_t *eng_;
    allocator_t *alc_;

private:
    malloc_func_t malloc_func_;
    free_func_t free_func_;
};

struct constant_tensor_cache_t {
    using key_t = size_t;
    using cached_t = std::shared_ptr<constant_buffer_t>;
    using value_t = std::shared_future<cached_t>;

    explicit constant_tensor_cache_t(
            size_t capacity_in_bytes, const std::string &name = "");

    ~constant_tensor_cache_t();

    // This function increments the reference count
    void retain() { counter_.fetch_add(1, std::memory_order_relaxed); }

    void release() {
        if (counter_.fetch_sub(1, std::memory_order_relaxed) == 1) {
            delete this;
        }
    }

    // The capacity set or got through these two method is in MBytes
    status_t set_capacity(size_t capacity);
    size_t get_capacity();

    value_t get_or_add(key_t backend_id, key_t backend_specific_key,
            size_t size, const value_t &value);
    void remove_if_exist(key_t backend_id, key_t backend_specific_key);

    size_t get_size() const;

    // The key_t is composed of two parts: backend id and backend specific key.
    // The backend id occupies 4 bits, and the backend specific key occupies the
    // remained 60 bits. So backends should ensure not encode any information in
    // the first 4 bits of backend specific key, otherwise the information will
    // be ignored during the combination. This requirement is same as the layout
    // id encoding.
    static key_t combine_key(key_t backend_id, key_t backend_specific_key);

private:
    void evict(size_t n);
    value_t get(const key_t &key);
    void add(const key_t &key, size_t size, const value_t &constant);

    void lock_read() { rw_mutex_.lock_read(); }
    void lock_write() { rw_mutex_.lock_write(); }
    void unlock_read() { rw_mutex_.unlock_read(); }
    void unlock_write() { rw_mutex_.unlock_write(); }

    // Disable assignment and copy
    constant_tensor_cache_t(const constant_tensor_cache_t &) = delete;
    constant_tensor_cache_t(constant_tensor_cache_t &&) = delete;
    constant_tensor_cache_t &operator=(const constant_tensor_cache_t &)
            = delete;
    constant_tensor_cache_t &operator=(constant_tensor_cache_t &&) = delete;

    struct timed_entry_t {
        value_t value_;
        std::atomic<size_t> timestamp_;
        timed_entry_t(const value_t &value, size_t timestamp)
            : value_(value), timestamp_(timestamp) {}
    };

    std::unordered_map<key_t, timed_entry_t> &constant_map() {
        return *constant_map_;
    }

    const std::unordered_map<key_t, timed_entry_t> &constant_map() const {
        return *constant_map_;
    }

    // Each entry in the cache has a corresponding key and timestamp.
    // NOTE: pairs that contain atomics cannot be stored in an unordered_map *as
    // an element*, since it invokes the copy constructor of std::atomic, which
    // is deleted.
    std::unique_ptr<std::unordered_map<key_t, timed_entry_t>> constant_map_;
    impl::utils::rw_mutex_t rw_mutex_;
    std::string name_;
    std::atomic<size_t> capacity_in_bytes_;
    std::atomic<int32_t> counter_;
};

constant_tensor_cache_t *get_constant_tensor_cache(
        impl::engine_kind_t eng_kind, size_t index);

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
