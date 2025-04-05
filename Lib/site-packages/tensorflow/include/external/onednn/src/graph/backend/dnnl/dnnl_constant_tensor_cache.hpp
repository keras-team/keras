/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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
#ifndef GRAPH_BACKEND_DNNL_DNNL_CONSTANT_TENSOR_CACHE_HPP
#define GRAPH_BACKEND_DNNL_DNNL_CONSTANT_TENSOR_CACHE_HPP

#include "graph/interface/constant_tensor_cache.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/dnnl_backend.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct dnnl_constant_buffer_t : public graph::constant_buffer_t {
    dnnl_constant_buffer_t(
            size_t size, dnnl::engine &engine, graph::allocator_t *alc)
        : graph::constant_buffer_t(
                size, engine.get(), alc, malloc_func, free_func) {}

    static void *malloc_func(
            size_t size, impl::engine_t *eng, graph::allocator_t *alc) {
        dnnl::engine engine;
        engine.reset(eng, true); // not own
        return dnnl_allocator_t::malloc(
                size, engine, alc, allocator_t::mem_type_t::persistent);
    }

    static void free_func(
            void *data, impl::engine_t *eng, graph::allocator_t *alc) {
        dnnl::engine engine;
        engine.reset(eng, true); // not own
        if (eng->kind() == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            dnnl_allocator_t::free(data, engine, alc, ::sycl::event());
#else
            dnnl_allocator_t::free(data, engine, alc);
#endif
        } else if (eng->kind() == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            dnnl_allocator_t::free(data, engine, alc, ::sycl::event());
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
            dnnl_allocator_t::free(data, engine, alc, cl_event());
#endif
        }
    }
};

inline graph::constant_tensor_cache_t::value_t dnnl_constant_cache_get_or_add(
        const dnnl::engine &eng, graph::constant_tensor_cache_t::key_t key,
        size_t size, const graph::constant_tensor_cache_t::value_t &value) {
    auto cache = graph::get_constant_tensor_cache(
            eng.get()->kind(), eng.get()->index());
    assertm(cache,
            "no available constant cache for specified engine kind and index");
    return cache->get_or_add(
            dnnl_backend::get_singleton().get_id(), key, size, value);
}

inline void dnnl_constant_cache_remove_if_exist(
        const dnnl::engine &eng, graph::constant_tensor_cache_t::key_t key) {
    auto cache = graph::get_constant_tensor_cache(
            eng.get()->kind(), eng.get()->index());
    assertm(cache,
            "no available constant cache for specified engine kind and index");
    cache->remove_if_exist(dnnl_backend::get_singleton().get_id(), key);
}

inline bool is_constant_cache_enabled(const dnnl::engine &eng) {
    auto cache = graph::get_constant_tensor_cache(
            eng.get()->kind(), eng.get()->index());
    return cache && cache->get_capacity() != 0;
}

inline void dnnl_constant_cache_retain(const dnnl::engine &eng) {
    auto cache = graph::get_constant_tensor_cache(
            eng.get()->kind(), eng.get()->index());
    cache->retain();
}

inline void dnnl_constant_cache_release(const dnnl::engine &eng) {
    auto cache = graph::get_constant_tensor_cache(
            eng.get()->kind(), eng.get()->index());
    cache->release();
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
