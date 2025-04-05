/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#ifndef COMMON_MEMORY_HPP
#define COMMON_MEMORY_HPP

#include <assert.h>
#include <memory>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "memory_storage.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

struct exec_ctx_t;

enum memory_flags_t {
    alloc = 0x1,
    use_runtime_ptr = 0x2,
    prefer_device_usm = 0x4
};
} // namespace impl
} // namespace dnnl

struct dnnl_memory : public dnnl::impl::c_compatible {
    /** XXX: Parameter flags must contain either alloc or use_runtime_ptr from
     * memory_flags_t. */
    dnnl_memory(dnnl::impl::engine_t *engine,
            const dnnl::impl::memory_desc_t *md,
            const std::vector<unsigned> &flags,
            const std::vector<void *> &handles);
    dnnl_memory(dnnl::impl::engine_t *engine,
            const dnnl::impl::memory_desc_t *md, unsigned flags, void *handle)
        : dnnl_memory(engine, md, std::vector<unsigned> {flags},
                std::vector<void *> {handle}) {}
    dnnl_memory(dnnl::impl::engine_t *engine,
            const dnnl::impl::memory_desc_t *md,
            std::unique_ptr<dnnl::impl::memory_storage_t> &&memory_storage);
    virtual ~dnnl_memory() = default;

    /** returns memory's engine */
    dnnl::impl::engine_t *engine() const { return engine_; }
    /** returns memory's description */
    const dnnl::impl::memory_desc_t *md() const { return &md_; }
    /** returns the underlying memory storage */
    dnnl::impl::memory_storage_t *memory_storage(int index = 0) const {
        if (index >= (int)memory_storages_.size()) return nullptr;
        return memory_storages_[index].get();
    }

    /** returns the underlying memory storage */
    dnnl::impl::memory_storage_t *memory_storage_clean(
            const dnnl::impl::exec_ctx_t &ctx,
            dnnl::impl::status_t &status) const {
        status = zero_pad(ctx);
        return memory_storage(0);
    }

    /** returns data handle */
    dnnl::impl::status_t get_data_handle(void **handle, int index = 0) const {
        return memory_storage(index)->get_data_handle(handle);
    }

    /** sets data handle */
    dnnl::impl::status_t set_data_handle(void *handle, int index = 0) const;

    /** zeros padding */
    dnnl::impl::status_t zero_pad(const dnnl::impl::exec_ctx_t &ctx) const;

    dnnl::impl::status_t reset_memory_storage(
            std::unique_ptr<dnnl::impl::memory_storage_t> &&memory_storage);

    size_t get_num_handles() const { return memory_storages_.size(); }

protected:
    dnnl::impl::engine_t *engine_;
    const dnnl::impl::memory_desc_t md_;

private:
    dnnl_memory() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(dnnl_memory);

    // Number of storages is larger than 1 only for sparse memory.
    std::vector<std::unique_ptr<dnnl::impl::memory_storage_t>> memory_storages_;
};

#endif
