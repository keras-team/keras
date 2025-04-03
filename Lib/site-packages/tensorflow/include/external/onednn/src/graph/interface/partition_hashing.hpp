/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef GRAPH_INTERFACE_PARTITION_HASHING_HPP
#define GRAPH_INTERFACE_PARTITION_HASHING_HPP

#include <algorithm>
#include <memory>
#include <thread>
#include <typeindex>
#include <vector>
#include <type_traits>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.h"

#include "common/engine_id.hpp"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/logical_tensor.hpp"
#include "graph/interface/op.hpp"

#include "graph/utils/id.hpp"
#include "graph/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace partition_hashing {

namespace {
inline std::vector<op_t *> get_raw_ptrs(
        const std::vector<std::shared_ptr<op_t>> &ops) {
    std::vector<op_t *> ret(ops.size(), nullptr);
    std::transform(ops.begin(), ops.end(), ret.begin(),
            [](const std::shared_ptr<op_t> &op_ptr) { return op_ptr.get(); });
    return ret;
}
} // namespace

struct key_t {
    key_t(const impl::engine_t *engine,
            const std::vector<std::shared_ptr<op_t>> &ops,
            const std::vector<const logical_tensor_t *> &ins,
            const std::vector<const logical_tensor_t *> &outs);
    key_t(const partition_t *partition, const impl::engine_t *engine,
            const std::vector<const logical_tensor_t *> &ins,
            const std::vector<const logical_tensor_t *> &outs);

    bool operator==(const key_t &other) const;
    const std::thread::id &thread_id() const { return thread_id_; }
    bool has_runtime_dependencies() const {
        return !(engine_id_.kind() == engine_kind::cpu
                && impl::is_native_runtime(engine_id_.runtime_kind()));
    }

    mutable std::vector<op_t *> ops_;
    mutable std::vector<logical_tensor_t> ins_;
    mutable std::vector<logical_tensor_t> outs_;
    int nthread_;
    impl::engine_id_t engine_id_;

private:
    // Thread ID is not used as part of the key, it's only used to get
    // information about what thread inserted the key and the corresponding
    // primitive to handle some multithread scenarios.
    std::thread::id thread_id_;
};

template <typename T>
size_t get_array_hash(size_t seed, const T *v, size_t size) {
    for (size_t i = 0; i < size; i++) {
        seed = hash_combine(seed, v[i]);
    }
    return seed;
}

template <typename Array>
size_t get_unordered_array_hash(size_t seed, const Array &array) {
    for (auto &&e : array) {
        seed = hash_combine(seed, std::hash<typename Array::value_type> {}(e));
    }
    return seed;
}

template <>
inline size_t get_array_hash<logical_tensor_t>(
        size_t seed, const logical_tensor_t *v, size_t size) {
    for (size_t i = 0; i < size; i++) {
        seed = hash_combine(seed, logical_tensor_wrapper_t(v[i]).hash());
    }
    return seed;
}

size_t get_op_hash(const op_t &op);

inline size_t get_array_hash(size_t seed, std::vector<op_t *> &ops) {
    for (const auto *op : ops)
        seed = hash_combine(seed, get_op_hash(*op));
    return seed;
}

template <>
inline size_t get_array_hash<float>(size_t seed, const float *v, size_t size) {
    for (size_t i = 0; i < size; i++) {
        seed = hash_combine(seed, float2int(v[i]));
    }
    return seed;
}

template <>
inline size_t get_unordered_array_hash<std::unordered_set<logical_tensor_t>>(
        size_t seed, const std::unordered_set<logical_tensor_t> &array) {
    for (auto &&e : array) {
        seed = hash_combine(seed, logical_tensor_wrapper_t(e).hash());
    }
    return seed;
}

} // namespace partition_hashing
} // namespace graph
} // namespace impl
} // namespace dnnl

// inject a specialization of std::hash for key_t in std namespace
namespace std {
template <>
struct hash<dnnl::impl::graph::partition_hashing::key_t> {
    using argument_type = dnnl::impl::graph::partition_hashing::key_t;
    using result_type = std::size_t;
    result_type operator()(const argument_type &key) const {
        using namespace dnnl::impl::graph;
        using namespace dnnl::impl::graph::partition_hashing;

        size_t seed = 0;
        // Compute hash for nthread_, engine_kind_
        seed = dnnl::impl::hash_combine(seed, key.nthread_);
        seed = dnnl::impl::hash_combine(seed, key.engine_id_.hash());

        // Combine hash for op_kinds & attributes with the computed hash
        seed = get_array_hash(seed, key.ops_);

        // Combine hash for input and output ports with the computed hash
        seed = get_array_hash(seed, key.ins_.data(), key.ins_.size());
        seed = get_array_hash(seed, key.outs_.data(), key.outs_.size());

        return seed;
    }
};

} // namespace std

#endif
