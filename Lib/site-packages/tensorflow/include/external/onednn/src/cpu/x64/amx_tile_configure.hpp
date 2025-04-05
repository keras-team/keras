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

#ifndef CPU_X64_AMX_TILE_CONFIGURE_HPP
#define CPU_X64_AMX_TILE_CONFIGURE_HPP

#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static constexpr size_t AMX_PALETTE_SIZE = 64;
status_t DNNL_API amx_tile_configure(const char palette[AMX_PALETTE_SIZE]);
status_t DNNL_API amx_tile_lazy_configure(const char palette[AMX_PALETTE_SIZE]);
status_t DNNL_API amx_tile_release();

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
