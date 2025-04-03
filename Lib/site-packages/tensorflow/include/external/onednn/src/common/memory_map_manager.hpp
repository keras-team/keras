/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef MEMORY_MAP_MANAGER_HPP
#define MEMORY_MAP_MANAGER_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include <functional>
#include <mutex>
#include <unordered_map>

namespace dnnl {
namespace impl {

// Service class to support RAII semantics with parameterized "finalization".
template <typename key_type, typename tag_type = void>
struct memory_map_manager_t : public c_compatible {
    using unmap_func_type = std::function<status_t(stream_t *, void *)>;

    memory_map_manager_t() = default;
    ~memory_map_manager_t() { assert(entries_.empty()); }

    static memory_map_manager_t &instance() {
        static memory_map_manager_t _instancwe;
        return _instancwe;
    }

    status_t map(const memory_storage_t *mem_storage, stream_t *stream,
            void *mapped_ptr, const unmap_func_type &unmap_func) {
        std::lock_guard<std::mutex> guard(mutex_);

        assert(entries_.count(mem_storage) == 0);
        entries_[mem_storage] = entry_t(stream, mapped_ptr, unmap_func);
        return status::success;
    }

    status_t unmap(const memory_storage_t *mem_storage, stream_t *stream,
            void *mapped_ptr) {
        std::lock_guard<std::mutex> guard(mutex_);

        auto it = entries_.find(mem_storage);
        if (it == entries_.end()) return status::runtime_error;
        CHECK(it->second.unmap(stream, mapped_ptr));
        entries_.erase(it);

        return status::success;
    }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(memory_map_manager_t);

    struct entry_t {
        entry_t() = default;
        entry_t(stream_t *stream, void *mapped_ptr,
                const unmap_func_type &unmap_func)
            : stream(stream), mapped_ptr(mapped_ptr), unmap_func(unmap_func) {}

        status_t unmap(stream_t *unmap_stream, void *unmap_mapped_ptr) {
            if (unmap_mapped_ptr != mapped_ptr) return status::runtime_error;
            return unmap_func(unmap_stream, unmap_mapped_ptr);
        }

        stream_t *stream = nullptr;
        void *mapped_ptr = nullptr;
        unmap_func_type unmap_func;
    };
    std::unordered_map<const memory_storage_t *, entry_t> entries_;
    std::mutex mutex_;
};

} // namespace impl
} // namespace dnnl

#endif
