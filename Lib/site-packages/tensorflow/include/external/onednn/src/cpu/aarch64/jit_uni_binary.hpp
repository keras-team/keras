/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
* Copyright 2022-2023 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_BINARY_HPP
#define CPU_AARCH64_JIT_UNI_BINARY_HPP

#include "common/primitive.hpp"

#include "cpu/cpu_eltwise_pd.hpp"

#include "cpu/aarch64/jit_uni_binary_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct binary_kernel_t;

using op_t = binary_op_t;
using bcast_t = binary_bcast_t;

struct jit_uni_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_binary_t);

        status_t init(engine_t *engine);

        jit_binary_conf_t get_conf() const { return conf_; };

    private:
        op_t get_op_type(const memory_desc_wrapper &src0_d);
        bool is_only_dim0_bcasted(const dims_t &bcast_dims, const int ndims);
        bcast_t get_bcast_type(
                const memory_desc_wrapper &src1_d, const dims_t &bcast_dims);

        // alg_preserves_zero returns true if operation preserves zero in case
        // of both inputs contain zero.
        bool alg_preserves_zero() const;
        bool check_scales_mask() const;
        bool is_bcast_pattern(const dims_t &bcast_dims, const dim_t ndims,
                const dim_t N_bcast, const dim_t C_bcast,
                const dim_t W_bcast) const;
        bool is_bcast_pattern(const dims_t &bcast_dims, const dim_t N_bcast,
                const dim_t C_bcast) const;
        bool is_bcast_allowed(const int ndims) const;
        bool is_format_non_blocked(const memory_desc_wrapper &mdw) const;
        bool is_different_layouts_allowed(const memory_desc_wrapper &src0_d,
                const memory_desc_wrapper &src1_d) const;
        bool is_applicable();

        jit_binary_conf_t conf_;
    };

    jit_uni_binary_t(const pd_t *apd);
    ~jit_uni_binary_t() = default;

    status_t init(engine_t *engine) override;

    using data_t = int8_t;

    void execute_no_bcast_strategy(const data_t *src0, const data_t *src1,
            data_t *dst, const float *scale0, const float *scale1,
            const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
            const bcast_t bcast_type) const;
    void execute_bcast_per_batch_strategy(const data_t *src0,
            const data_t *src1, data_t *dst, const float *scale0,
            const float *scale1,
            const std::vector<const void *> &post_ops_binary_rhs_arg_vec) const;
    void execute_bcast_per_c_strategy(const data_t *src0, const data_t *src1,
            data_t *dst, const float *scale0, const float *scale1,
            const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
            const op_t op_type, const bcast_t bcast_type,
            const bool blocked_oc_tail) const;
    void execute_bcast_per_w_strategy(const data_t *src0, const data_t *src1,
            data_t *dst, const float *scale0, const float *scale1,
            const std::vector<const void *> &post_ops_binary_rhs_arg_vec,
            const op_t op_type, const bool blocked_oc_tail) const;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    static bool post_ops_ok(const primitive_attr_t *attr,
            const memory_desc_wrapper &src0_d, const memory_desc_wrapper &dst_d,
            const bool is_src1_different_layouts, const cpu_isa_t isa);

    std::unique_ptr<binary_kernel_t> kernel_;
    // used only in bcast_c_blocked strategy if tail exists
    std::unique_ptr<binary_kernel_t> kernel_tail_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
