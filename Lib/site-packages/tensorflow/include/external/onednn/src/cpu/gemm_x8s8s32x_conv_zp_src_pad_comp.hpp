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

#ifndef CPU_GEMM_X8S8S32X_CONV_ZP_SRC_PAD_COMP_HPP_
#define CPU_GEMM_X8S8S32X_CONV_ZP_SRC_PAD_COMP_HPP_

#include "common/c_types_map.hpp"
#include "cpu/gemm_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

void compute_zp_src_comp_pad(const conv_gemm_conf_t &jcp,
        int32_t *const zp_src_pad_buf, const int32_t *const zp_src,
        const int8_t *weights, const memory_desc_wrapper &weights_md,
        const bool with_groups);

void apply_zp_src_comp_pad(const conv_gemm_conf_t &jcp, const dim_t g,
        const dim_t d_offset, const dim_t h_offset, const dim_t w_offset,
        const dim_t h_size, const dim_t w_size,
        int32_t *__restrict gemm_conv_result,
        const int32_t *__restrict zp_src_pad_buf);

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif