/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GRAPH_INTERFACE_ALLOCATOR_HPP
#define GRAPH_INTERFACE_ALLOCATOR_HPP

#include <atomic>
#include <cstdlib>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.h"

#include "common/rw_mutex.hpp"

#include "graph/interface/c_types_map.hpp"

#include "graph/utils/alloc.hpp"
#include "graph/utils/id.hpp"
#include "graph/utils/utils.hpp"
#include "graph/utils/verbose.hpp"

#ifdef DNNL_WITH_SYCL
#include "graph/utils/sycl_check.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "graph/utils/ocl_check.hpp"
#include "oneapi/dnnl/dnnl_graph_ocl.h"
#endif

struct dnnl_graph_allocator final : public dnnl::impl::graph::utils::id_t {
public:
    dnnl_graph_allocator() = default;

    dnnl_graph_allocator(dnnl_graph_host_allocate_f host_malloc,
            dnnl_graph_host_deallocate_f host_free)
        : host_malloc_(host_malloc), host_free_(host_free) {}

#ifdef DNNL_WITH_SYCL
    dnnl_graph_allocator(dnnl_graph_sycl_allocate_f sycl_malloc,
            dnnl_graph_sycl_deallocate_f sycl_free)
        : sycl_malloc_(sycl_malloc), sycl_free_(sycl_free) {}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    dnnl_graph_allocator(dnnl_graph_ocl_allocate_f ocl_malloc,
            dnnl_graph_ocl_deallocate_f ocl_free)
        : ocl_malloc_(ocl_malloc), ocl_free_(ocl_free) {}
#endif

    dnnl_graph_allocator(const dnnl_graph_allocator &alloc) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        ocl_malloc_ = alloc.ocl_malloc_;
        ocl_free_ = alloc.ocl_free_;
#endif

#ifdef DNNL_WITH_SYCL
        sycl_malloc_ = alloc.sycl_malloc_;
        sycl_free_ = alloc.sycl_free_;
#endif

        host_malloc_ = alloc.host_malloc_;
        host_free_ = alloc.host_free_;
    }

    ~dnnl_graph_allocator() = default;

    dnnl_graph_allocator &operator=(const dnnl_graph_allocator &alloc) {
        // check self-assignment
        if (this == &alloc) return *this;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        ocl_malloc_ = alloc.ocl_malloc_;
        ocl_free_ = alloc.ocl_free_;
#endif

#ifdef DNNL_WITH_SYCL
        sycl_malloc_ = alloc.sycl_malloc_;
        sycl_free_ = alloc.sycl_free_;
#endif
        host_malloc_ = alloc.host_malloc_;
        host_free_ = alloc.host_free_;
        return *this;
    }

    enum class mem_type_t {
        persistent = 0,
        output = 1,
        temp = 2,
    };

    /// Allocator attributes
    struct mem_attr_t {
        mem_type_t type_;
        size_t alignment_;

        /// Default constructor for an uninitialized attribute
        mem_attr_t() {
            type_ = mem_type_t::persistent;
            alignment_ = 0;
        }

        mem_attr_t(mem_type_t type, size_t alignment) {
            type_ = type;
            alignment_ = alignment;
        }

        /// Copy constructor
        mem_attr_t(const mem_attr_t &other) = default;

        /// Assign operator
        mem_attr_t &operator=(const mem_attr_t &other) = default;
    };

    struct mem_info_t {
        mem_info_t(size_t size, mem_type_t type) : size_(size), type_(type) {}
        size_t size_;
        mem_type_t type_;
    };

    struct monitor_t {
    private:
        size_t persist_mem_ = 0;

        std::unordered_map<const void *, mem_info_t> persist_mem_infos_;

        std::unordered_map<std::thread::id, size_t> temp_mem_;
        std::unordered_map<std::thread::id, size_t> peak_temp_mem_;
        std::unordered_map<std::thread::id,
                std::unordered_map<const void *, mem_info_t>>
                temp_mem_infos_;

        // Since the memory operation will be performed from multiple threads,
        // so we use the rw lock to guarantee the thread safety of the global
        // persistent memory monitoring.
        dnnl::impl::utils::rw_mutex_t rw_mutex_;

    public:
        void record_allocate(const void *buf, size_t size, mem_type_t type);

        void record_deallocate(const void *buf);

        void reset_peak_temp_memory();

        size_t get_peak_temp_memory();

        size_t get_total_persist_memory();

        void lock_write();
        void unlock_write();
    };

    void *allocate(size_t size, mem_attr_t attr = {}) const {
#ifndef NDEBUG
        monitor_.lock_write();
        void *buffer = host_malloc_(size, attr.alignment_);
        monitor_.record_allocate(buffer, size, attr.type_);
        monitor_.unlock_write();
#else
        void *buffer = host_malloc_(size, attr.alignment_);
#endif
        return buffer;
    }

#ifdef DNNL_WITH_SYCL
    void *allocate(size_t size, const ::sycl::device &dev,
            const ::sycl::context &ctx, mem_attr_t attr = {}) const {
#ifndef NDEBUG
        monitor_.lock_write();
        void *buffer = sycl_malloc_(size, attr.alignment_,
                static_cast<const void *>(&dev),
                static_cast<const void *>(&ctx));
        monitor_.record_allocate(buffer, size, attr.type_);
        monitor_.unlock_write();
#else
        void *buffer = sycl_malloc_(size, attr.alignment_,
                static_cast<const void *>(&dev),
                static_cast<const void *>(&ctx));
#endif
        return buffer;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void *allocate(size_t size, const cl_device_id dev, const cl_context ctx,
            mem_attr_t attr = {}) const {
#ifndef NDEBUG
        monitor_.lock_write();
        void *buffer = ocl_malloc_(size, attr.alignment_, dev, ctx);
        monitor_.record_allocate(buffer, size, attr.type_);
        monitor_.unlock_write();
#else
        void *buffer = ocl_malloc_(size, attr.alignment_, dev, ctx);
#endif
        return buffer;
    }
#endif

    template <typename T>
    T *allocate(size_t nelem, mem_attr_t attr = {}) {
        const size_t size = nelem * sizeof(T);
        void *buffer = allocate(size, attr);
        return reinterpret_cast<T *>(buffer);
    }

#ifdef DNNL_WITH_SYCL
    template <typename T>
    T *allocate(size_t nelem, const ::sycl::device &dev,
            const ::sycl::context &ctx, mem_attr_t attr = {}) {
        const size_t size = nelem * sizeof(T);
        void *buffer = allocate(size, dev, ctx, attr);
        return reinterpret_cast<T *>(buffer);
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    template <typename T>
    T *allocate(size_t nelem, const cl_device_id dev, const cl_context ctx,
            mem_attr_t attr = {}) {
        const size_t size = nelem * sizeof(T);
        void *buffer = allocate(size, dev, ctx, attr);
        return reinterpret_cast<T *>(buffer);
    }
#endif

    void deallocate(void *buffer) const {
        if (buffer) {
#ifndef NDEBUG
            monitor_.lock_write();
            monitor_.record_deallocate(buffer);
            host_free_(buffer);
            monitor_.unlock_write();
#else
            host_free_(buffer);
#endif
        }
    }

#ifdef DNNL_WITH_SYCL
    void deallocate(void *buffer, const ::sycl::device &dev,
            const ::sycl::context &ctx, ::sycl::event deps) const {
        if (buffer) {
#ifndef NDEBUG
            monitor_.lock_write();
            monitor_.record_deallocate(buffer);
            sycl_free_(buffer, static_cast<const void *>(&dev),
                    static_cast<const void *>(&ctx),
                    static_cast<void *>(&deps));
            monitor_.unlock_write();
#else
            sycl_free_(buffer, static_cast<const void *>(&dev),
                    static_cast<const void *>(&ctx),
                    static_cast<void *>(&deps));
#endif
        }
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void deallocate(void *buffer, const cl_device_id dev, const cl_context ctx,
            cl_event deps) const {
        if (buffer) {
#ifndef NDEBUG
            monitor_.lock_write();
            monitor_.record_deallocate(buffer);
            ocl_free_(buffer, dev, ctx, deps);
            monitor_.unlock_write();
#else
            ocl_free_(buffer, dev, ctx, deps);
#endif
            buffer = nullptr;
        }
    }
#endif

    monitor_t &get_monitor() { return monitor_; }

private:
    dnnl_graph_host_allocate_f host_malloc_ {
            dnnl::impl::graph::utils::cpu_allocator_t::malloc};
    dnnl_graph_host_deallocate_f host_free_ {
            dnnl::impl::graph::utils::cpu_allocator_t::free};

#ifdef DNNL_WITH_SYCL
    dnnl_graph_sycl_allocate_f sycl_malloc_ {
            dnnl::impl::graph::utils::sycl_allocator_t::malloc};
    dnnl_graph_sycl_deallocate_f sycl_free_ {
            dnnl::impl::graph::utils::sycl_allocator_t::free};
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    // By default, use the malloc and free functions provided by the library.
    dnnl_graph_ocl_allocate_f ocl_malloc_ {
            dnnl::impl::graph::utils::ocl_allocator_t::malloc};
    dnnl_graph_ocl_deallocate_f ocl_free_ {
            dnnl::impl::graph::utils::ocl_allocator_t::free};
#endif

    mutable monitor_t monitor_;
};

#endif
