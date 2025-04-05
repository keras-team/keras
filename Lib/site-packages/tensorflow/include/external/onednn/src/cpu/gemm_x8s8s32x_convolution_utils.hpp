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

#ifndef CPU_GEMM_X8S8S32X_CONVOLUTION_UTILS_HPP
#define CPU_GEMM_X8S8S32X_CONVOLUTION_UTILS_HPP

#include "cpu/gemm_convolution_utils.hpp"
#if DNNL_X64
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace gemm_x8s8s32x_convolution_utils {

struct pp_ker_t {
    static pp_ker_t *create(
            const convolution_pd_t *pd, const conv_gemm_conf_t &jcp);
    virtual ~pp_ker_t() = default;

    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void operator()(void *dst, const acc_data_t *acc, const char *bias,
            const float *scales, float dst_scale, float sum_scale,
            float signed_scale, int g, size_t start, size_t end,
            const zero_point_call_params_t &zp,
            const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
            const exec_ctx_t &ctx, const memory_desc_t &dst_md,
            const single_gemm_conv_chunk_desc_t &chunk_desc) const = 0;

    virtual status_t create_kernel() { return status::success; }

protected:
    pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp);

    const conv_gemm_conf_t &jcp_;
};

bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_wrapper *dst_d);
bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_t *dst_d);
bool mayiuse_jit_pp_kernel(data_type_t dst_dt) noexcept;

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
