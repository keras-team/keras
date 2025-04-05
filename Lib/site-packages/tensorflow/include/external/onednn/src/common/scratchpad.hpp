/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#ifndef COMMON_SCRATCHPAD_HPP
#define COMMON_SCRATCHPAD_HPP

#include "c_types_map.hpp"
#include "memory_storage.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

struct scratchpad_t {
    virtual ~scratchpad_t() {}
    virtual const memory_storage_t *get_memory_storage() const = 0;
    virtual size_t size() const = 0;
};

scratchpad_t *create_scratchpad(
        engine_t *engine, size_t size, bool use_global_scratchpad);

} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
