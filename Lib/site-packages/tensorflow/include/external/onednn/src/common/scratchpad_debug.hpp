/*******************************************************************************
 * Copyright 2020 Intel Corporation
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

#ifndef COMMON_SCRATCHPAD_DEBUG_HPP
#define COMMON_SCRATCHPAD_DEBUG_HPP

#include <stdint.h>

#include "memory_debug.hpp"
#include "memory_tracking.hpp"

namespace dnnl {
namespace impl {
namespace scratchpad_debug {
// Static inline for optimization purposes when memory_debug is disabled
static inline bool is_protect_scratchpad() {
    return memory_debug::is_mem_debug();
}
void protect_scratchpad_buffer(void *scratchpad_ptr, engine_kind_t engine_kind,
        const memory_tracking::registry_t &registry);
void unprotect_scratchpad_buffer(const void *scratchpad_ptr,
        engine_kind_t engine_kind, const memory_tracking::registry_t &registry);
void protect_scratchpad_buffer(const memory_storage_t *storage,
        const memory_tracking::registry_t &registry);
void unprotect_scratchpad_buffer(const memory_storage_t *storage,
        const memory_tracking::registry_t &registry);

} // namespace scratchpad_debug
} // namespace impl
} // namespace dnnl
#endif
