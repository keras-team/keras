/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_IFACE_HPP
#define COMMON_PRIMITIVE_IFACE_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "cache_blob.hpp"
#include "primitive_exec_types.hpp"
#include "resource.hpp"
#include "scratchpad.hpp"

namespace dnnl {
namespace impl {
status_t primitive_execute(
        const primitive_iface_t *primitive_iface, exec_ctx_t &ctx);
}
} // namespace dnnl

// dnnl_primitive is a user facing entity that has an alias primitive_iface_t
// for internal use.
// The primitive_iface_t is responsible for holding:
// 1. impl::primitive_t - a primitive implementation that can be
// stored in the primitive cache. Other data members are NOT stored in
// the cache
// 2. scratchpad_t - a memory for scratchpad
// 3. primitive_desc_iface_t - an alias for dnnl_primitive_desc and is
// a user facing primitive descriptor (the one a user should create prior
// creating a primitive)
// 4. resource_mapper_t - a resource mapper that provides a mapping between
// impl::primitive_t and its resource
//
// Note: primitive_desc_iface_t and impl::primitive_t share the same
// impl::primitive_desc_t
struct dnnl_primitive : public dnnl::impl::c_compatible {
    dnnl_primitive(const std::shared_ptr<dnnl::impl::primitive_t> &primitive,
            dnnl::impl::engine_t *engine);

    // This is a ctor for reorder
    dnnl_primitive(const std::shared_ptr<dnnl::impl::primitive_t> &primitive,
            dnnl::impl::engine_t *engine, dnnl::impl::engine_t *src_engine,
            dnnl::impl::engine_t *dst_engine);

    dnnl::impl::status_t init();
    dnnl::impl::engine_t *engine() const;
    const primitive_desc_iface_t *pd() const;
    dnnl::impl::status_t get_cache_blob_size(size_t *size) const;
    dnnl::impl::status_t get_cache_blob(
            dnnl::impl::cache_blob_t cache_blob) const;
    dnnl::impl::status_t execute(dnnl::impl::exec_ctx_t &ctx) const;

    void retain() { counter_++; }

    void release() {
        if (--counter_ == 0) { delete this; }
    }

protected:
    ~dnnl_primitive();

private:
    std::atomic<int> counter_;
    std::shared_ptr<dnnl::impl::primitive_t> primitive_;
    std::unique_ptr<dnnl::impl::scratchpad_t> scratchpad_;
    std::unique_ptr<primitive_desc_iface_t> pd_;
    dnnl::impl::resource_mapper_t resource_mapper_;

    dnnl_primitive() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(dnnl_primitive);
};

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
