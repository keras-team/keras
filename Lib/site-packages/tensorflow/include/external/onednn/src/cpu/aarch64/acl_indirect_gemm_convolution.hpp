/*******************************************************************************
* Copyright 2021-2024 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_INDIRECT_GEMM_CONVOLUTION_HPP
#define CPU_AARCH64_ACL_INDIRECT_GEMM_CONVOLUTION_HPP

#include "cpu/cpu_convolution_pd.hpp"

#include "cpu/aarch64/acl_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_indirect_gemm_resource_t : public resource_t {
    acl_indirect_gemm_resource_t()
        : acl_obj_(utils::make_unique<acl_obj_t<arm_compute::NEGEMMConv2d>>()) {
    }

    status_t configure(const acl_conv_conf_t &acp) {
        if (!acl_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_obj_->src_tensor.allocator()->init(acp.src_tensor_info);
        acl_obj_->wei_tensor.allocator()->init(acp.wei_tensor_info);
        acl_obj_->dst_tensor.allocator()->init(acp.dst_tensor_info);
        acl_obj_->bia_tensor.allocator()->init(acp.bia_tensor_info);

        // clang-format off
        acl_obj_->conv.configure(
            &acl_obj_->src_tensor,
            &acl_obj_->wei_tensor,
            acp.with_bias ? &acl_obj_->bia_tensor : nullptr,
            &acl_obj_->dst_tensor,
            arm_compute::Conv2dInfo(acp.padstride_info,
                                    acp.dilation_info,
                                    acp.act_info,
                                    acp.fast_math,
                                    1,
                                    acp.weights_info));
        // clang-format on

        return status::success;
    }

    acl_obj_t<arm_compute::NEGEMMConv2d> &get_acl_obj() const {
        return *acl_obj_;
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_indirect_gemm_resource_t);

private:
    std::unique_ptr<acl_obj_t<arm_compute::NEGEMMConv2d>> acl_obj_;

}; // acl_indirect_gemm_resource_t

struct acl_indirect_gemm_convolution_fwd_t : public primitive_t {

    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), acp_() {}

        DECLARE_COMMON_PD_T("indirect_gemm:acl",
                acl_indirect_gemm_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            const bool is_fp16_ok = expect_data_types(f16, f16, f16, f16, undef)
                    && attr()->has_default_values(smask_t::post_ops, f16);
            const bool is_bf16_ok
                    = expect_data_types(bf16, bf16, bf16, bf16, undef)
                    && attr_.post_ops_.len() == 0;
            const bool is_fp32_ok = expect_data_types(f32, f32, f32, f32, undef)
                    && attr()->has_default_values(
                            smask_t::post_ops | smask_t::fpmath_mode, f32);
            bool ok = is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && utils::one_of(true, is_fp16_ok, is_bf16_ok, is_fp32_ok)
                    && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            CHECK(acl_convolution_utils::init_conf_indirect_gemm(acp_, src_md_,
                    weights_md_, dst_md_, bias_md_, *desc(), *attr()));

            CHECK(post_ops.init(
                    engine, attr_.post_ops_, dst_md_, acp_.act_info));
            acp_.use_dst_acc = post_ops.has_sum();

            return status::success;
        }

        acl_conv_conf_t acp_;

        acl_post_ops_t post_ops;
    };

    acl_indirect_gemm_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_indirect_gemm_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        CHECK(r->configure(pd()->acp_));
        mapper.add(this, std::move(r));

        CHECK(pd()->post_ops.create_resource(engine, mapper));

        return status::success;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    // To guard the const execute_forward(), the mutex must be 'mutable'
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_INDIRECT_GEMM_CONVOLUTION_HPP
