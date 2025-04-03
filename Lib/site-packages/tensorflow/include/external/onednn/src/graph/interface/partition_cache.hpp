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

#ifndef GRAPH_INTERFACE_PARTITION_CACHE_HPP
#define GRAPH_INTERFACE_PARTITION_CACHE_HPP

#include <future>
#include <memory>
#include <thread>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.h"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/partition_hashing.hpp"

#include "common/cache_utils.hpp"
#include "common/rw_mutex.hpp"

namespace dnnl {
namespace impl {
namespace graph {

struct compiled_partition_cache_t {
    struct cache_value_t {
        bool is_empty() const { return value == nullptr; }
        compiled_partition_t &get_value() const { return *value; }
        std::shared_ptr<compiled_partition_t> value;
        status_t status;
    };
    using key_t = partition_hashing::key_t;
    using result_t = cache_value_t;
    using value_t = std::shared_future<cache_value_t>;
    using create_func_t = result_t (&)(void *);
    using create_func_ptr_t = result_t (*)(void *);

    compiled_partition_cache_t(int capacity) : cache_(capacity) {}

    ~compiled_partition_cache_t() = default;

    status_t set_capacity(int capacity) {
        return cache_.set_capacity(capacity);
    }
    int get_capacity() const { return cache_.get_capacity(); }
    int get_size() const { return cache_.get_size(); }

    result_t get_or_create(
            const key_t &key, create_func_t create, void *create_context) {
        return cache_.get_or_create(key, create, create_context);
    }

    const partition_t *get_partition(const key_t &key);

private:
    // No need to set key_merge here since update_entry function is not need in
    // partition cache
    utils::lru_cache_t<key_t, compiled_partition_t, result_t,
            /* key_merge */ nullptr>
            cache_;
};

compiled_partition_cache_t &compiled_partition_cache();

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
