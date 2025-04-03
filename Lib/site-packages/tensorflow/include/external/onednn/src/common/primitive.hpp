/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_HPP
#define COMMON_PRIMITIVE_HPP

#include <assert.h>
#include <atomic>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "cache_blob.hpp"
#include "memory_storage.hpp"
#include "memory_tracking.hpp"
#include "primitive_desc.hpp"
#include "primitive_exec_types.hpp"
#include "rw_mutex.hpp"
#include "scratchpad.hpp"

#include <future>
#include <type_traits>

namespace dnnl {
namespace impl {

struct resource_mapper_t;
// Primitive implementation
struct primitive_t : public c_compatible {
    using primitive_list_t = std::vector<const primitive_t *>;

    primitive_t(const primitive_desc_t *pd) : pd_(pd->clone()) {}
    virtual ~primitive_t() = default;

    virtual status_t init(engine_t *engine) { return status::success; }

    status_t init(engine_t *engine, bool use_global_scratchpad,
            const cache_blob_t &cache_blob) {
        cache_blob_ = cache_blob;
        CHECK(init(engine));
        use_global_scratchpad_ = use_global_scratchpad;
        // The `cache_blob_` is no longer needed after primitive creation.
        cache_blob_ = cache_blob_t();
        return status::success;
    }

    const std::shared_ptr<primitive_desc_t> &pd() const { return pd_; }
    primitive_kind_t kind() const { return pd_->kind(); }
    virtual status_t execute(const exec_ctx_t &ctx) const = 0;

    virtual status_t get_cache_blob(
            engine_t *engine, cache_blob_t &cache_blob) const {
        assert(!"unexpected");
        return status::runtime_error;
    }

    virtual status_t get_cache_blob_size(engine_t *engine, size_t *size) const {
        assert(!"unexpected");
        return status::runtime_error;
    }

    virtual status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const {
        return status::success;
    }

    bool use_global_scratchpad() const { return use_global_scratchpad_; }
    cache_blob_t cache_blob() const { return cache_blob_; }

protected:
    template <typename impl_type, typename pd_t>
    static status_t create_primitive_common(
            std::pair<std::shared_ptr<primitive_t>, bool> &primitive,
            const pd_t *pd, engine_t *engine, bool use_global_scratchpad,
            const cache_blob_t &cache_blob) {

        auto global_primitive_cache = primitive_cache();
        primitive_hashing::key_t key(pd, engine);

        struct create_context_t {
            engine_t *engine;
            const pd_t *pd;
            const cache_blob_t &cache_blob;
            bool use_global_scratchpad;
            bool is_create_called;
        };
        create_context_t context {
                engine, pd, cache_blob, use_global_scratchpad, false};

        primitive_cache_iface_t::create_func_ptr_t create = [](void *context) {
            auto &c = *static_cast<create_context_t *>(context);
            std::shared_ptr<primitive_t> p = std::make_shared<impl_type>(c.pd);
            status_t status
                    = p->init(c.engine, c.use_global_scratchpad, c.cache_blob);
            c.is_create_called = true;
            return primitive_cache_iface_t::result_t {std::move(p), status};
        };
        auto result
                = global_primitive_cache.get_or_create(key, *create, &context);
        primitive = {std::move(result.value), !context.is_create_called};
        return result.status;
    }

    std::shared_ptr<primitive_desc_t> pd_;
    bool use_global_scratchpad_ = false;
    cache_blob_t cache_blob_;

private:
    primitive_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(primitive_t);
};

// This is a helper class which is used for forwarding a scratchpad
// from master primitive to the nested ones.
struct nested_scratchpad_t {
    nested_scratchpad_t(const exec_ctx_t &master_ctx, int key,
            const std::shared_ptr<primitive_t> &nested_p);
    const memory_tracking::grantor_t *grantor() const { return grantor_.get(); }

    ~nested_scratchpad_t();

    DNNL_DISALLOW_COPY_AND_ASSIGN(nested_scratchpad_t);

private:
    std::unique_ptr<memory_storage_t> scratchpad_mem_storage_;
    std::unique_ptr<memory_tracking::grantor_t> grantor_;
};

} // namespace impl
} // namespace dnnl

#define ARG_TYPE(t) \
    typename std::remove_cv<typename std::remove_pointer<t>::type>::type

// Returns destination memory which has been zero pad initialized. This macro
// may result in a failure returned via the `status` input since zero pad
// may fail.
#define CTX_OUT_CLEAN_MEM(type, arg, status) \
    static_cast<ARG_TYPE(type) *>(ctx.host_ptr(arg, true, &status))

// Returns destination memory which may not have been zero pad initialized.
#define CTX_OUT_MEM_COMMON(type, arg, index) \
    static_cast<ARG_TYPE(type) *>(ctx.host_ptr(arg, false, nullptr, index))
#define CTX_OUT_MEm(type, arg) CTX_OUT_MEM_COMMON(type, arg, 0)
#define CTX_OUT_MEm0(type, arg) CTX_OUT_MEM_COMMON(type, arg, 0)
#define CTX_OUT_MEm1(type, arg) CTX_OUT_MEM_COMMON(type, arg, 1)
#define CTX_OUT_MEm2(type, arg) CTX_OUT_MEM_COMMON(type, arg, 2)

#define CTX_IN_MEM_COMMON(type, arg, index) \
    static_cast<const ARG_TYPE(type) *>( \
            ctx.host_ptr(arg, false, nullptr, index))
#define CTX_IN_MEm(type, arg) CTX_IN_MEM_COMMON(type, arg, 0)
#define CTX_IN_MEm0(type, arg) CTX_IN_MEM_COMMON(type, arg, 0)
#define CTX_IN_MEm1(type, arg) CTX_IN_MEM_COMMON(type, arg, 1)
#define CTX_IN_MEm2(type, arg) CTX_IN_MEM_COMMON(type, arg, 2)

// __VA_ARGS__here is an index of the buffer. It is empty unless the memory
// argument is sparse.
#define CTX_IN_MEM(type, arg, ...) CTX_IN_MEm##__VA_ARGS__(type, arg)
#define CTX_OUT_MEM(type, arg, ...) CTX_OUT_MEm##__VA_ARGS__(type, arg)

#endif
