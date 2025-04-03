/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef COMMON_MEMORY_STORAGE_HPP
#define COMMON_MEMORY_STORAGE_HPP

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/utils.hpp"

#include <assert.h>

namespace dnnl {
namespace impl {

// Memory storage is an abstraction providing interfaces to:
// - set/get the underlying data handle (in form of void pointer)
// - map/unmap the data to the host
//
// Memory storage is engine-specific and has different implementations for
// different engines.
struct memory_storage_t : public c_compatible {
    memory_storage_t(engine_t *engine, const memory_storage_t *parent_storage);
    memory_storage_t(engine_t *engine) : memory_storage_t(engine, this) {}

    virtual ~memory_storage_t();

    status_t init(unsigned flags, size_t size, void *handle);

    engine_t *engine() const { return engine_; }

    void *data_handle() const {
        void *handle;
        status_t status = get_data_handle(&handle);
        assert(status == status::success);
        MAYBE_UNUSED(status);
        return handle;
    }

    virtual status_t get_data_handle(void **handle) const = 0;
    virtual status_t set_data_handle(void *handle) = 0;

    size_t offset() const { return offset_; }
    void set_offset(size_t offset) { offset_ = offset; }

    virtual size_t base_offset() const { return 0; }

    virtual status_t map_data(
            void **mapped_ptr, stream_t *stream, size_t size) const = 0;

    virtual status_t unmap_data(void *mapped_ptr, stream_t *stream) const = 0;

    virtual bool is_host_accessible() const { return false; }

    /** returns slice of memory storage
     *
     * @note: sub-storage lifetime shall not exceed one of the base memory storage
     * @note: (offset + size) shall not be greater than base memory storage size */
    virtual std::unique_ptr<memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const = 0;

    /** returns shallow copy */
    virtual std::unique_ptr<memory_storage_t> clone() const = 0;

    /** returns true if the pointer associated with the storage is NULL */
    bool is_null() const {
        void *ptr;
        status_t status = get_data_handle(&ptr);
        assert(status == status::success);
        MAYBE_UNUSED(status);
        return !ptr;
    }

    operator bool() const { return !is_null(); }

    static memory_storage_t &empty_storage();

protected:
    virtual status_t init_allocate(size_t size) = 0;

    const memory_storage_t *parent_storage() const { return parent_storage_; }

private:
    engine_t *engine_;
    size_t offset_ = 0;

    const memory_storage_t *parent_storage_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(memory_storage_t);
};

struct empty_memory_storage_t : public memory_storage_t {
    empty_memory_storage_t() : memory_storage_t(nullptr) {}

    status_t get_data_handle(void **handle) const override {
        *handle = nullptr;
        return status::success;
    }

    status_t set_data_handle(void *handle) override {
        assert(!"not expected");
        return status::runtime_error;
    }

    status_t map_data(
            void **mapped_ptr, stream_t *stream, size_t size) const override {
        UNUSED(mapped_ptr);
        UNUSED(stream);
        UNUSED(size);
        return status::success;
    }

    status_t unmap_data(void *mapped_ptr, stream_t *stream) const override {
        UNUSED(mapped_ptr);
        UNUSED(stream);
        return status::success;
    }

    std::unique_ptr<memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const override {
        assert(!"not expected");
        return nullptr;
    }

    std::unique_ptr<memory_storage_t> clone() const override {
        assert(!"not expected");
        return nullptr;
    }

protected:
    status_t init_allocate(size_t) override { return status::success; }
};

inline memory_storage_t &memory_storage_t::empty_storage() {
    static empty_memory_storage_t instance;
    return instance;
}

} // namespace impl
} // namespace dnnl

#endif
