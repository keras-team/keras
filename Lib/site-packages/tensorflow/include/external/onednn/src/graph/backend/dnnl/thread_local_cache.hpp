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
#ifndef GRAPH_BACKEND_DNNL_THREAD_LOCAL_CACHE_HPP
#define GRAPH_BACKEND_DNNL_THREAD_LOCAL_CACHE_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <unordered_map>

#include "graph/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// In multithread scenarios, there are two requests to the kernel's execute()
// method:
// 1. To support multithread execution, the kernel instance must be immutable
//    after its creating, so its execute() method must be const and stateless.
// 2. To reduce execution overhead, we want to create some mutable resource once
//    in execute(), cache them and only change few field of them in every
//    iteration. Those resources may be the dnnl::memory objects or others.
// Base on above two requests, those mutable resources can't be a part of the
// kernel instance. So we design this thread_local_cache_t class to cache those
// mutable resources. Kernel should search the resource it needs in the cache,
// if found, it should use the found one, otherwise it should create a new one
// and store it to the cache. At this moment, we observed that those resources
// will not be shared between threads, so we made the cache be thread local to
// reduce the search and sync overhead.

// Note:
// The shared_ptr of resources in ALL threads are cached in the @global_cache_,
// which takes the ownership of cached resources. Besides, each thread will use
// a thread local table to cache the weak_ptr of current thread's resources. The
// thread local table can be get by using the @get_thread_local_cache() method.
// - When looking up the cached value, we will only search the thread local
//   table in thread safe way without lock. If cache hit, we will return the
//   found value, and the performance should be good.
// - If cache miss, we need to add a new value to the global table, and add its
//   weak_ptr to the thread local table correspondingly. Ann we need to use a
//   lock to protect the global table. The performance should be bad, but cache
//   miss should be rare.
// - We can read/write the found resource in each thread without lock, because
//   each thread has its own replica.
// - If a thread existed, the thread local table will be destroyed, during
//   which, we will find the share_ptr in global table for each weak_ptr in
//   thread local table, and release the shared_ptr.
// - If users want to destroy the cached value for a certain key in ALL thread,
//   they can call the @remove_if_exist() method. After that the corresponding
//   shared ptr in global table will be released. Even though the weak ptr in
//   thread local table won't be erased, we think this is acceptable because the
//   underlying instance has been destroyed indeed and the expired weak ptr can
//   be reused by other keys.
template <typename T>
class thread_local_cache_t {
public:
    thread_local_cache_t() = default;

    // Check if we have a cached value for the given key in current thread
    bool has_resource(const size_t &key) {
        cache_type_t &cache = get_thread_local_cache();
        return cache.data().count(key) && !cache.data()[key].expired();
    }

    // return the number of cached values in current thread
    size_t size() {
        cache_type_t &cache = get_thread_local_cache();
        return cache.data().size();
    }

    // Clear the cached values in current thread
    void clear() {
        cache_type_t &cache = get_thread_local_cache();
        for (auto &it : cache.data()) {
            std::shared_ptr<T> value = it.second.lock();
            if (value) {
                std::lock_guard<std::mutex> lock(
                        global_cache_type_t::get_global_cache()->mutex());
                auto &data = global_cache_type_t::get_global_cache()->data();

                auto ret = data.find(it.first);
                if (ret != data.end()) {
                    std::vector<std::shared_ptr<T>> &thread_instances
                            = ret->second;
                    auto pos = std::find_if(thread_instances.begin(),
                            thread_instances.end(),
                            [&](std::shared_ptr<T> &ins) -> bool {
                                return ins.get() == value.get();
                            });
                    assertm(pos != thread_instances.end(),
                            "expected value to exist in cache");
                    thread_instances.erase(pos);
                }
            }
        }
        cache.data().clear();
    }

    // Remove the cached values for the given key in ALL threads
    void remove_if_exist(const size_t &key) {
        std::lock_guard<std::mutex> lock(
                global_cache_type_t::get_global_cache()->mutex());
        auto pos = global_cache_type_t::get_global_cache()->data().find(key);
        if (pos != global_cache_type_t::get_global_cache()->data().end()) {
            pos->second.clear();
        }
    }

    // Get the cached value in current thread. If the value is not cached, we
    // will call the creator to create one and cache it
    T *get_or_add(const size_t &key,
            const std::function<std::shared_ptr<T>()> &creator) {
        cache_type_t &cache = get_thread_local_cache();
        if (has_resource(key)) { // cache hit
            return cache.data()[key].lock().get();
        } else { // cache miss
            // Cache miss shouldn't happen frequently, because the lock is
            // heavy. No double-check is needed here since cached values won't
            // be shared between threads
            std::shared_ptr<T> ins = creator();
            {
                std::lock_guard<std::mutex> lock(
                        global_cache_type_t::get_global_cache()->mutex());
                if (global_cache_type_t::get_global_cache()->data().count(
                            key)) {
                    global_cache_type_t::get_global_cache()
                            ->data()
                            .at(key)
                            .emplace_back(ins);
                } else {
                    global_cache_type_t::get_global_cache()->data().emplace(
                            key, std::vector<std::shared_ptr<T>> {ins});
                }
            }
            cache.data()[key] = ins;
            return ins.get();
        }
    }

    // This function increments the reference count
    void retain() { global_cache_type_t::get_global_cache()->retain(); }

    void release() { global_cache_type_t::get_global_cache()->release(); }

private:
    class global_cache_type_t {
    public:
        global_cache_type_t() : counter_(1) {}
        ~global_cache_type_t() = default;
        std::mutex &mutex() { return mutex_; }
        std::unordered_map<size_t, std::vector<std::shared_ptr<T>>> &data() {
            return data_;
        }

        static global_cache_type_t *get_global_cache() {
            // A global table to store cached values in ALL threads. This global
            // table takes the ownership of cached values
            static auto global_cache = std::shared_ptr<global_cache_type_t>(
                    new global_cache_type_t {},
                    [](global_cache_type_t *ptr) { return ptr->release(); });
            return global_cache.get();
        }

        // This function increments the reference count
        void retain() { counter_.fetch_add(1, std::memory_order_relaxed); }

        void release() {
            if (counter_.fetch_sub(1, std::memory_order_relaxed) == 1) {
                delete this;
            }
        }

    private:
        std::mutex mutex_;
        std::unordered_map<size_t, std::vector<std::shared_ptr<T>>> data_;
        std::atomic<int32_t> counter_;
    };

    class cache_type_t {
    public:
        cache_type_t(global_cache_type_t &global_cache)
            : global_cache_ref_(global_cache) {
            global_cache_ref_.retain();
        }

        ~cache_type_t() {
            // Remove the values of this cache that haven't already expired.
            for (auto &it : data_) {
                std::shared_ptr<T> value = it.second.lock();
                if (value) {
                    std::lock_guard<std::mutex> lock(global_cache_ref_.mutex());

                    // Find the corresponding shared ptr in global table
                    auto ret = global_cache_ref_.data().find(it.first);
                    if (ret != global_cache_ref_.data().end()) {
                        std::vector<std::shared_ptr<T>> &thread_instances
                                = ret->second;
                        auto pos = std::find_if(thread_instances.begin(),
                                thread_instances.end(),
                                [&](std::shared_ptr<T> &ins) -> bool {
                                    return ins.get() == value.get();
                                });
                        assertm(pos != thread_instances.end(),
                                "expected value to exist in cache");
                        // Detroy it
                        thread_instances.erase(pos);
                    }
                }
            }
            global_cache_ref_.release();
        }

        std::unordered_map<size_t, std::weak_ptr<T>> &data() { return data_; }

        global_cache_type_t &global_cache_ref_;
        std::unordered_map<size_t, std::weak_ptr<T>> data_;
    };

    thread_local_cache_t(const thread_local_cache_t &other) = delete;
    thread_local_cache_t &operator=(const thread_local_cache_t &other) = delete;

    static cache_type_t &get_thread_local_cache() {
        static thread_local cache_type_t cache(
                *global_cache_type_t::get_global_cache());
        return cache;
    }
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
