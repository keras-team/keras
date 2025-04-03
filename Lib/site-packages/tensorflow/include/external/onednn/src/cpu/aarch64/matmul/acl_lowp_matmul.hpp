/*******************************************************************************
* Copyright 2024 Arm Ltd. and affiliates
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

#ifndef ACL_LOWP_MATMUL_HPP
#define ACL_LOWP_MATMUL_HPP

#include "cpu/cpu_primitive.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/matmul/matmul_utils.hpp"

#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct acl_lowp_matmul_obj_t {
    arm_compute::NEGEMMLowpMatrixMultiplyCore gemm;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_lowp_matmul_conf_t {
    arm_compute::TensorInfo src_tensor_info;
    arm_compute::TensorInfo wei_tensor_info;
    bool with_bias;
    arm_compute::TensorInfo bia_tensor_info;
    arm_compute::TensorInfo dst_tensor_info;
};

status_t configure_gemm(
        acl_lowp_matmul_obj_t &acl_obj, const acl_lowp_matmul_conf_t &almc) {

    acl_obj.src_tensor.allocator()->init(almc.src_tensor_info);
    acl_obj.wei_tensor.allocator()->init(almc.wei_tensor_info);
    if (almc.with_bias) {
        acl_obj.bia_tensor.allocator()->init(almc.bia_tensor_info);
    }
    acl_obj.dst_tensor.allocator()->init(almc.dst_tensor_info);

    acl_obj.gemm.configure(&acl_obj.src_tensor, &acl_obj.wei_tensor,
            almc.with_bias ? &acl_obj.bia_tensor : nullptr,
            &acl_obj.dst_tensor);

    return status::success;
}

struct acl_lowp_matmul_resource_t : public resource_t {
    acl_lowp_matmul_resource_t()
        : acl_obj_(utils::make_unique<acl_lowp_matmul_obj_t>()) {}

    status_t configure(const acl_lowp_matmul_conf_t &almc) {

        if (!acl_obj_) return status::out_of_memory;

        acl_obj_->src_tensor.allocator()->init(almc.src_tensor_info);
        acl_obj_->wei_tensor.allocator()->init(almc.wei_tensor_info);
        if (almc.with_bias) {
            acl_obj_->bia_tensor.allocator()->init(almc.bia_tensor_info);
        }
        acl_obj_->dst_tensor.allocator()->init(almc.dst_tensor_info);

        acl_obj_->gemm.configure(&acl_obj_->src_tensor, &acl_obj_->wei_tensor,
                almc.with_bias ? &acl_obj_->bia_tensor : nullptr,
                &acl_obj_->dst_tensor);

        return status::success;
    }

    acl_lowp_matmul_obj_t &get_acl_obj() const { return *acl_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_lowp_matmul_resource_t);

private:
    std::unique_ptr<acl_lowp_matmul_obj_t> acl_obj_;
};

struct acl_lowp_matmul_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {

        pd_t(const matmul_desc_t *adesc, const primitive_attr_t *attr,
                const cpu_matmul_pd_t *hint_fwd_pd)
            : cpu_matmul_pd_t(adesc, attr, hint_fwd_pd), almc_() {}

        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T(
                "lowp_gemm:acl", acl_lowp_matmul_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {

            VDISPATCH_MATMUL(
                    set_default_formats(), "failed to set default formats");
            using smask_t = primitive_attr_t::skip_mask_t;
            VDISPATCH_MATMUL(attr()->has_default_values(smask_t::scales_runtime
                                     | smask_t::zero_points_runtime),
                    "only scale and zero point attrs supported");

            // Note that has_default_values checks the argument for default zero
            // points but skips the argument for scales. Hence they are the
            // opposite but mean similar things
            VDISPATCH_MATMUL(attr()->scales_.has_default_values(
                                     {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS}),
                    "only src and weights scales are supported");
            VDISPATCH_MATMUL(
                    attr()->zero_points_.has_default_values(DNNL_ARG_DST),
                    "only src and weights zero points are supported");

            VDISPATCH_MATMUL(attr()->scales_.get(DNNL_ARG_SRC).mask_ == 0
                            && attr()->zero_points_.get(DNNL_ARG_SRC) == 0
                            && attr()->scales_.get(DNNL_ARG_WEIGHTS).mask_ == 0
                            && attr()->zero_points_.get(DNNL_ARG_WEIGHTS) == 0,
                    "common scales and zero points only");

            VDISPATCH_MATMUL(!has_runtime_dims_or_strides(),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);

            const memory_desc_wrapper src_d(src_md_);
            const memory_desc_wrapper wei_d(weights_md_);
            const memory_desc_wrapper bia_d(bias_md_);
            const memory_desc_wrapper dst_d(dst_md_);

            using namespace data_type;
            VDISPATCH_MATMUL(src_d.data_type() == s8 && wei_d.data_type() == s8
                            && dst_d.data_type() == f32
                            && utils::one_of(bia_d.data_type(), f32, undef),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_MATMUL(src_d.matches_tag(format_tag::ab)
                            && wei_d.matches_tag(format_tag::ab)
                            && dst_d.matches_tag(format_tag::ab),
                    VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_MATMUL_SC(
                    memory_desc_init_by_tag(bias_md_, bias_md_.ndims,
                            bias_md_.dims, bias_md_.data_type, format_tag::ab),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);

            // We set the QuantizationInfo to be dynamic because it is re-set in run()
            almc_.src_tensor_info = arm_compute::TensorInfo(
                    arm_compute::TensorShape(K(), M()), 1,
                    arm_compute::DataType::QASYMM8_SIGNED,
                    arm_compute::QuantizationInfo(1.0, 0, true));
            almc_.src_tensor_info.set_are_values_constant(false);

            almc_.wei_tensor_info = arm_compute::TensorInfo(
                    arm_compute::TensorShape(N(), K()), 1,
                    arm_compute::DataType::QASYMM8_SIGNED,
                    arm_compute::QuantizationInfo(1.0, 0, true));
            almc_.wei_tensor_info.set_are_values_constant(false);

            almc_.bia_tensor_info = arm_compute::TensorInfo(
                    arm_compute::TensorShape(), 1, arm_compute::DataType::F32);
            almc_.with_bias = bia_d.format_kind() != format_kind::undef;
            if (almc_.with_bias) {
                // This is not currently guarded in ACL
                VDISPATCH_MATMUL(bia_d.ndims() == 2 && bia_d.dims()[0] == 1
                                && bia_d.dims()[1] == N(),
                        "Only 1xN bias is supported");
                almc_.bia_tensor_info.set_tensor_shape(arm_compute::TensorShape(
                        bia_d.dims()[1], bia_d.dims()[0]));
            }

            almc_.dst_tensor_info = arm_compute::TensorInfo(
                    arm_compute::TensorShape(N(), M()),
                    arm_compute::Format::F32);

            ACL_CHECK_VALID(arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(
                    &almc_.src_tensor_info, &almc_.wei_tensor_info,
                    almc_.with_bias ? &almc_.bia_tensor_info : nullptr,
                    &almc_.dst_tensor_info, arm_compute::GEMMInfo()));

            return status::success;
        }

        acl_lowp_matmul_conf_t almc_;
    };

    acl_lowp_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const {

        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_lowp_matmul_resource_t>();
        if (!r) return status::out_of_memory;

        CHECK(r->configure(pd()->almc_));

        mapper.add(this, std::move(r));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const {
        std::lock_guard<std::mutex> _lock {this->mtx};

        bool with_bias = pd()->almc_.with_bias;

        acl_lowp_matmul_obj_t &acl_obj
                = ctx.get_resource_mapper()
                          ->get<acl_lowp_matmul_resource_t>(this)
                          ->get_acl_obj();

        auto src = CTX_IN_MEM(const int8_t *, DNNL_ARG_SRC);
        auto wei = CTX_IN_MEM(const int8_t *, DNNL_ARG_WEIGHTS);
        auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

        acl_obj.src_tensor.allocator()->import_memory(
                const_cast<int8_t *>(src));
        acl_obj.wei_tensor.allocator()->import_memory(
                const_cast<int8_t *>(wei));
        if (with_bias) {
            auto bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
            acl_obj.bia_tensor.allocator()->import_memory(
                    const_cast<float *>(bias));
        }
        acl_obj.dst_tensor.allocator()->import_memory(dst);

        DEFINE_ARG_SCALES_BUFFER(src_scale, DNNL_ARG_SRC);
        DEFINE_ZERO_POINT_VALUE(src_zero_point, DNNL_ARG_SRC);
        DEFINE_ARG_SCALES_BUFFER(wei_scale, DNNL_ARG_WEIGHTS);
        DEFINE_ZERO_POINT_VALUE(wei_zero_point, DNNL_ARG_WEIGHTS);

        // Note that we set the offset to be -zero_point, this is a known
        // inconsistency with most other operators in the ACL API
        acl_obj.src_tensor.info()->set_quantization_info(
                arm_compute::QuantizationInfo(
                        *src_scale, -src_zero_point, true));

        acl_obj.wei_tensor.info()->set_quantization_info(
                arm_compute::QuantizationInfo(
                        *wei_scale, -wei_zero_point, true));

        acl_obj.gemm.run();

        // free() here tells ACL it can no longer use it, it does not deallocate
        acl_obj.src_tensor.allocator()->free();
        acl_obj.wei_tensor.allocator()->free();
        if (with_bias) { acl_obj.bia_tensor.allocator()->free(); }
        acl_obj.dst_tensor.allocator()->free();

        return status::success;
    };

private:
    mutable std::mutex mtx;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_LOWP_MATMUL_HPP