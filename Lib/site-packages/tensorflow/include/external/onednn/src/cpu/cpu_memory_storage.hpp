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

#ifndef CPU_CPU_MEMORY_STORAGE_HPP
#define CPU_CPU_MEMORY_STORAGE_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_storage.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

class cpu_memory_storage_t : public memory_storage_t {
public:
    cpu_memory_storage_t(engine_t *engine)
        : memory_storage_t(engine), data_(nullptr, release) {}

    status_t get_data_handle(void **handle) const override {
        *handle = data_.get();
        return status::success;
    }

    status_t set_data_handle(void *handle) override {
        data_ = decltype(data_)(handle, release);
        return status::success;
    }

    status_t map_data(
            void **mapped_ptr, stream_t *stream, size_t size) const override {
        UNUSED(size);
        // This function is called for non-SYCL CPU engines only, where the
        // runtime_kind is constant for a specific build, and engine_kind is
        // only cpu. However, at the same time, the stream engine and memory
        // object engine may have different memory locations. Therefore, at
        // most, we need to ensure that the indexes of these engines are
        // identical.
        if (stream != nullptr && stream->engine()->index() != engine()->index())
            return status::invalid_arguments;
        return get_data_handle(mapped_ptr);
    }

    status_t unmap_data(void *mapped_ptr, stream_t *stream) const override {
        UNUSED(mapped_ptr);
        if (stream != nullptr && stream->engine()->index() != engine()->index())
            return status::invalid_arguments;
        return status::success;
    }

    bool is_host_accessible() const override { return true; }

    std::unique_ptr<memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const override {
        void *sub_ptr = reinterpret_cast<uint8_t *>(data_.get()) + offset;
        auto sub_storage = new cpu_memory_storage_t(this->engine());
        sub_storage->init(memory_flags_t::use_runtime_ptr, size, sub_ptr);
        return std::unique_ptr<memory_storage_t>(sub_storage);
    }

    std::unique_ptr<memory_storage_t> clone() const override {
        auto storage = new cpu_memory_storage_t(engine());
        if (storage)
            storage->init(memory_flags_t::use_runtime_ptr, 0, data_.get());
        return std::unique_ptr<memory_storage_t>(storage);
    }

protected:
    status_t init_allocate(size_t size) override {
        void *ptr = malloc(size, platform::get_cache_line_size());
        if (!ptr) return status::out_of_memory;
        data_ = decltype(data_)(ptr, destroy);
        return status::success;
    }

private:
    std::unique_ptr<void, void (*)(void *)> data_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(cpu_memory_storage_t);

    static void release(void *ptr) {}
    static void destroy(void *ptr) { free(ptr); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
