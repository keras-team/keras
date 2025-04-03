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

#ifndef COMMON_PRIMITIVE_HASHING_HPP
#define COMMON_PRIMITIVE_HASHING_HPP

#include <thread>
#include <typeindex>
#include <type_traits>

#include "c_types_map.hpp"
#include "engine_id.hpp"
#include "oneapi/dnnl/dnnl.h"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {

struct primitive_desc_t;
namespace primitive_hashing {

struct key_t {
    key_t(const engine_t *engine, const op_desc_t *op_desc,
            const primitive_attr_t *attr, int pd_iterator_offset,
            const std::vector<memory_desc_t> &hint_mds, int skip_idx);

    key_t(const primitive_desc_t *pd, const engine_t *engine);

    bool operator==(const key_t &other) const;
    const std::thread::id &thread_id() const { return thread_id_; }
    bool has_runtime_dependencies() const {
        return !(engine_id_.kind() == engine_kind::cpu
                && is_native_runtime(engine_id_.runtime_kind()));
    }

    primitive_kind_t primitive_kind_;
    // Make these data fields mutable to be able to update them without removing
    // and adding a key (extract is available in C++17 only).
    mutable const op_desc_t *op_desc_;
    mutable const primitive_attr_t *attr_;
    int pd_iterator_offset_;
    int impl_nthr_;
    int skip_idx_;
    std::vector<memory_desc_t> hint_mds_;
    engine_id_t engine_id_;

private:
    template <typename desc_t>
    static const desc_t &cast_to_desc(const void *p) {
        return *(reinterpret_cast<const desc_t *>(p));
    }

    static primitive_kind_t get_pkind(primitive_kind_t pkind);

    // Thread ID is not used as part of the key, it's only used to get
    // information about what thread inserted the key and the corresponding
    // primitive to handle some multithreaded scenarios.
    std::thread::id thread_id_;
};

size_t get_md_hash(const memory_desc_t &md);
size_t get_attr_hash(const primitive_attr_t &attr);
size_t get_desc_hash(const concat_desc_t &desc);
size_t get_desc_hash(const batch_normalization_desc_t &desc);
size_t get_desc_hash(const binary_desc_t &desc);
size_t get_desc_hash(const convolution_desc_t &desc);
size_t get_desc_hash(const eltwise_desc_t &desc);
size_t get_desc_hash(const gemm_desc_t &desc);
size_t get_desc_hash(const group_normalization_desc_t &desc);
size_t get_desc_hash(const inner_product_desc_t &desc);
size_t get_desc_hash(const layer_normalization_desc_t &desc);
size_t get_desc_hash(const lrn_desc_t &desc);
size_t get_desc_hash(const matmul_desc_t &desc);
size_t get_desc_hash(const pooling_desc_t &desc);
size_t get_desc_hash(const prelu_desc_t &desc);
size_t get_desc_hash(const reduction_desc_t &desc);
size_t get_desc_hash(const reorder_desc_t &desc);
size_t get_desc_hash(const resampling_desc_t &desc);
size_t get_desc_hash(const rnn_desc_t &desc);
size_t get_desc_hash(const shuffle_desc_t &desc);
size_t get_desc_hash(const softmax_desc_t &desc);
size_t get_desc_hash(const sum_desc_t &desc);
size_t get_desc_hash(const zero_pad_desc_t &desc);

template <typename T>
size_t get_array_hash(size_t seed, const T *v, int size) {
    for (int i = 0; i < size; i++) {
        seed = hash_combine(seed, v[i]);
    }
    return seed;
}

template <>
inline size_t get_array_hash<memory_desc_t>(
        size_t seed, const memory_desc_t *v, int size) {
    for (int i = 0; i < size; i++) {
        seed = hash_combine(seed, get_md_hash(v[i]));
    }
    return seed;
}

inline size_t get_array_hash(
        size_t seed, const std::vector<const memory_desc_t *> &mds) {
    for (const auto *md : mds)
        seed = hash_combine(seed, get_md_hash(*md));
    return seed;
}

template <>
inline size_t get_array_hash<data_type_t>(
        size_t seed, const data_type_t *v, int size) {
    for (int i = 0; i < size; i++) {
        seed = hash_combine(seed, static_cast<size_t>(v[i]));
    }
    return seed;
}

} // namespace primitive_hashing
} // namespace impl
} // namespace dnnl

// inject a specialization of std::hash for key_t in std namespace
namespace std {
template <>
struct hash<dnnl::impl::primitive_hashing::key_t> {
    using argument_type = dnnl::impl::primitive_hashing::key_t;
    using result_type = std::size_t;
    result_type operator()(const argument_type &key) const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        // Compute hash for primitive_kind_, attr_, impl_id_ and impl_nthr_
        seed = hash_combine(seed,
                hash_combine(0, static_cast<size_t>(key.primitive_kind_)));
        seed = hash_combine(seed, get_attr_hash(*key.attr_));
        seed = hash_combine(seed, hash_combine(0, key.pd_iterator_offset_));
        seed = hash_combine(seed, hash_combine(0, key.impl_nthr_));
        seed = hash_combine(seed, hash_combine(0, key.skip_idx_));

        seed = hash_combine(seed, key.engine_id_.hash());
        // Combine hash for op_desc with the computed hash
#define CASE(pkind) \
    case primitive_kind::pkind: \
        seed = hash_combine( \
                seed, get_desc_hash(*(pkind##_desc_t *)key.op_desc_)); \
        break;

        // clang-format off
        switch ((int)key.primitive_kind_) {
            CASE(batch_normalization)
            CASE(binary)
            CASE(concat)
            CASE(convolution)
            CASE(deconvolution)
            CASE(eltwise)
            CASE(gemm)
            CASE(group_normalization)
            CASE(inner_product)
            CASE(layer_normalization)
            CASE(lrn)
            CASE(matmul)
            CASE(pooling)
            CASE(prelu)
            CASE(reduction)
            CASE(reorder)
            CASE(resampling)
            CASE(rnn)
            CASE(shuffle)
            CASE(softmax)
            CASE(sum)
            CASE(zero_pad)
            default: assert(!"unknown primitive_kind");
        }
            // clang-format on
#undef CASE
        seed = get_array_hash(
                seed, key.hint_mds_.data(), (int)key.hint_mds_.size());

        return seed;
    }
};

} // namespace std

#endif
