/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef CPU_CPU_STREAM_HPP
#define CPU_CPU_STREAM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#endif

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/stream.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct cpu_stream_t : public stream_t {
    cpu_stream_t(engine_t *engine, unsigned flags) : stream_t(engine, flags) {}
    virtual ~cpu_stream_t() = default;

    dnnl::impl::status_t wait() override {
        // CPU execution is synchronous so return immediately
        return dnnl::impl::status::success;
    }

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    cpu_stream_t(engine_t *engine,
            dnnl::threadpool_interop::threadpool_iface *threadpool)
        : stream_t(engine, threadpool) {}

    void before_exec_hook() override {
        dnnl::threadpool_interop::threadpool_iface *tp;
        auto rc = this->get_threadpool(&tp);
        if (rc == status::success) threadpool_utils::activate_threadpool(tp);
    }

    void after_exec_hook() override {
        threadpool_utils::deactivate_threadpool();
    }
#endif
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
