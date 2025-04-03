/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_X64_RNN_BRGEMM_CELL_COMMON_UTILS_HPP
#define CPU_X64_RNN_BRGEMM_CELL_COMMON_UTILS_HPP

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct amx_tile_configuration_loader_t {
    /*
     * Tile configurations are prepared in init phase. In execute we must load
     * proper configuration for given situation. Tile configure is an expensive
     * performance operation. We should avoid multiple reconfigurations as well
     * as loading same configuration if it is already loaded.
     */
    void operator()(const char *requested_cfg_addr);
    ~amx_tile_configuration_loader_t();

private:
    const char *current_cfg_addr = nullptr;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
