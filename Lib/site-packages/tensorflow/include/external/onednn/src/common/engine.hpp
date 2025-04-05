/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
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

#ifndef COMMON_ENGINE_HPP
#define COMMON_ENGINE_HPP

#include "oneapi/dnnl/dnnl.h"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#endif

#include "c_types_map.hpp"
#include "engine_id.hpp"
#include "memory.hpp"
#include "memory_storage.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#ifdef ONEDNN_BUILD_GRAPH
#include "graph/interface/allocator.hpp"
#endif

#define VERROR_ENGINE(cond, stat, msg, ...) \
    do { \
        if (!(cond)) { \
            VERROR(common, runtime, msg, ##__VA_ARGS__); \
            return stat; \
        } \
    } while (0)

/** \brief An abstraction of an execution unit with shared resources
 *
 * Responsibilities:
 *   - Provide engine specific memory allocation
 *   - Provide engine specific primitive_desc_t creators
 */
struct dnnl_engine : public dnnl::impl::c_compatible {
    dnnl_engine(dnnl::impl::engine_kind_t kind,
            dnnl::impl::runtime_kind_t runtime_kind, size_t index)
        : kind_(kind)
        , runtime_kind_(runtime_kind)
        , index_(index)
        , counter_(1) {}

    /** get kind of the current engine */
    dnnl::impl::engine_kind_t kind() const { return kind_; }

    /** get the runtime kind of the current engine */
    dnnl::impl::runtime_kind_t runtime_kind() const { return runtime_kind_; }

    /** get index of the current engine */
    size_t index() const { return index_; }

    virtual dnnl::impl::device_id_t device_id() const = 0;

    virtual dnnl::impl::engine_id_t engine_id() const = 0;

    /** create memory storage */
    virtual dnnl::impl::status_t create_memory_storage(
            dnnl::impl::memory_storage_t **storage, unsigned flags, size_t size,
            void *handle)
            = 0;
    dnnl::impl::status_t create_memory_storage(
            dnnl::impl::memory_storage_t **storage, size_t size) {
        return create_memory_storage(
                storage, dnnl::impl::memory_flags_t::alloc, size, nullptr);
    }

    /** create stream */
    virtual dnnl::impl::status_t create_stream(
            dnnl::impl::stream_t **stream, unsigned flags)
            = 0;

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    virtual dnnl::impl::status_t create_stream(dnnl::impl::stream_t **stream,
            dnnl::threadpool_interop::threadpool_iface *threadpool) {
        return dnnl::impl::status::invalid_arguments;
    }
#endif

    virtual dnnl::impl::status_t get_service_stream(
            dnnl::impl::stream_t *&stream) {
        stream = nullptr;
        return dnnl::impl::status::success;
    }

    /* implementation section */

    /** return the list of reorder implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const dnnl::impl::impl_list_item_t *get_reorder_implementation_list(
            const dnnl::impl::memory_desc_t *src_md,
            const dnnl::impl::memory_desc_t *dst_md) const = 0;

    /** return the list of concat implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const dnnl::impl::impl_list_item_t *
    get_concat_implementation_list() const = 0;

    /** return the list of sum implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const dnnl::impl::impl_list_item_t *
    get_sum_implementation_list() const = 0;

    /** return the list of implementations for a given descriptor.
     * engine guarantees to return a NULL-terminated list */

    virtual const dnnl::impl::impl_list_item_t *get_implementation_list(
            const dnnl::impl::op_desc_t *desc) const = 0;

    virtual dnnl::impl::status_t serialize_device(
            dnnl::impl::serialization_stream_t &sstream) const {
        assert(!"unexpected");
        return dnnl::impl::status::runtime_error;
    }

    virtual bool mayiuse_f16_accumulator_with_f16() const { return false; }

#ifdef ONEDNN_BUILD_GRAPH
    /** only used in graph implementation **/
    void *get_allocator() const { return (void *)(&allocator_); };
    void set_allocator(dnnl::impl::graph::allocator_t *alloc) {
        allocator_ = *alloc;
    }
#endif

    void retain() { counter_++; }

    void release() {
        if (--counter_ == 0) { delete this; }
    }

protected:
    dnnl::impl::engine_kind_t kind_;
    dnnl::impl::runtime_kind_t runtime_kind_;
    size_t index_;

#ifdef ONEDNN_BUILD_GRAPH
    /** only used in graph implementation **/
    dnnl::impl::graph::allocator_t allocator_;
#endif

    virtual ~dnnl_engine() = default;

private:
    std::atomic<int> counter_;
};

namespace dnnl {
namespace impl {

inline runtime_kind_t get_default_runtime(engine_kind_t kind) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (kind == engine_kind::gpu) return runtime_kind::ocl;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (kind == engine_kind::gpu) return runtime_kind::sycl;
#endif
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SEQ
    return runtime_kind::seq;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    return runtime_kind::omp;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_TBB
    return runtime_kind::tbb;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    return runtime_kind::threadpool;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    return runtime_kind::sycl;
#else
    return runtime_kind::none;
#endif
}

inline runtime_kind_t get_cpu_native_runtime() {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
    return runtime_kind::seq;
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    return runtime_kind::omp;
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
    return runtime_kind::tbb;
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    return runtime_kind::threadpool;
#else
    return runtime_kind::none;
#endif
}

struct engine_factory_t : public c_compatible {
    virtual size_t count() const = 0;
    virtual status_t engine_create(engine_t **engine, size_t index) const = 0;
    virtual ~engine_factory_t() = default;
};

struct engine_deleter_t {
    void operator()(engine_t *e) const { e->release(); }
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
