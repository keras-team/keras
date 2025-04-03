/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef COMMON_THREAD_LOCAL_STORAGE_HPP
#define COMMON_THREAD_LOCAL_STORAGE_HPP

#include <assert.h>
#include <thread>
#include <utility>
#include <unordered_map>

#include "rw_mutex.hpp"
#include "z_magic.hpp"

namespace dnnl {
namespace impl {
namespace utils {

template <typename T>
struct thread_local_storage_t {
    thread_local_storage_t() = default;

    DNNL_DISALLOW_COPY_AND_ASSIGN(thread_local_storage_t);

    template <typename U>
    T &set(U &&value) {
        utils::lock_write_t lock_w(mutex_);
        const auto tid = std::this_thread::get_id();
        assert(storage_.count(tid) == 0);
        auto it = storage_.emplace(tid, std::forward<U>(value));
        assert(it.second);
        return it.first->second;
    }

    bool is_set() {
        utils::lock_read_t lock_r(mutex_);
        return storage_.find(std::this_thread::get_id()) != storage_.end();
    }

    T &get() {
        utils::lock_read_t lock_r(mutex_);
        return storage_.at(std::this_thread::get_id());
    }

    template <typename U>
    T &get(U &&def_value) {
        {
            utils::lock_read_t lock_r(mutex_);
            auto it = storage_.find(std::this_thread::get_id());
            if (it != storage_.end()) return it->second;
        }
        return set(std::forward<U>(def_value));
    }

private:
    std::unordered_map<std::thread::id, T> storage_;
    utils::rw_mutex_t mutex_;
};

} // namespace utils
} // namespace impl
} // namespace dnnl

#endif
