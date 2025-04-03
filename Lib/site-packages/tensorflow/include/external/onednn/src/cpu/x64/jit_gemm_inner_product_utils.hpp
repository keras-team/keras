/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_GEMM_INNER_PRODUCT_UTILS_HPP
#define CPU_X64_JIT_GEMM_INNER_PRODUCT_UTILS_HPP

#include "cpu/gemm_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace inner_product_utils {

cpu::inner_product_utils::pp_kernel_t *jit_pp_kernel_create(size_t OC,
        size_t MB, dim_t dst_mb_stride, const primitive_attr_t *attr,
        data_type_t bias_dt, data_type_t acc_dt, const memory_desc_t *dst_md,
        bool skip_sum);

inline cpu_isa_t get_max_jit_pp_kernel_supported_isa() {
#define CASE(isa) \
    do { \
        if (mayiuse(isa)) return isa; \
    } while (false)
    CASE(avx512_core_bf16);
    CASE(avx512_core);
    CASE(avx2);
#undef CASE
    return sse41;
}

} // namespace inner_product_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
