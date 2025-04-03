/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef CPU_X64_JIT_GEMM_X8S8S32X_CONVOLUTION_UTILS_HPP
#define CPU_X64_JIT_GEMM_X8S8S32X_CONVOLUTION_UTILS_HPP

#include "cpu/gemm_x8s8s32x_convolution_utils.hpp"
#include "cpu/x64/injectors/injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace gemm_x8s8s32x_convolution_utils {

cpu::gemm_x8s8s32x_convolution_utils::pp_ker_t *jit_pp_ker_create(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp);
bool mayiuse_jit_pp_kernel(data_type_t dst_dt) noexcept;
bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_wrapper *dst_d);

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
