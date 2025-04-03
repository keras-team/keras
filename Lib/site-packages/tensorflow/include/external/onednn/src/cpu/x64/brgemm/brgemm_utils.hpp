/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef CPU_X64_BRGEMM_BRGEMM_UTILS_HPP
#define CPU_X64_BRGEMM_BRGEMM_UTILS_HPP

#include "cpu/x64/brgemm/brgemm.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace brgemm_utils {

bool can_dispatch_uker(const brgemm_desc_t *brg);

void maybe_try_bf32(brgemm_desc_t *brg);

status_t brgemm_blocking(brgemm_desc_t *brg);

status_t brdgmm_blocking(brgemm_desc_t *brg);

/* The purpose of this function is to enable initialization of brgemm values
 * and then call additional functions like blocking heuristics without
 * having to depend on BRGeMM's API. An additional feature is that this
 * function can be modified depending on needs without requiring changes
 * at the API level. */
void init_brgemm_conf(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, brgemm_layout_t layout, float alpha, float beta,
        dim_t LDA, dim_t LDB, dim_t LDC, dim_t M, dim_t N, dim_t K,
        const brgemm_strides_t *strides = nullptr, bool is_bf32 = false);

/* The purpose of this function is to enable initialization of brgemm values
 * and then call additional functions like blocking heuristics without
 * having to depend on BRDGeMM's API. An additional feature is that this
 * function can be modified depending on needs without requiring changes
 * at the API level. */
void init_brdgmm_conf(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, brgemm_layout_t layout, float alpha, float beta,
        dim_t LDA, dim_t LDC, dim_t M, dim_t N,
        const brgemm_strides_t *strides = nullptr);

} // namespace brgemm_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_BRGEMM_BRGEMM_UTILS_HPP

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
