/*******************************************************************************
* Copyright 2022 Arm Ltd. and affiliates
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
#ifndef CPU_AARCH64_ACL_PRELU_HPP
#define CPU_AARCH64_ACL_PRELU_HPP

#include "cpu/aarch64/acl_utils.hpp"
#include "cpu/cpu_prelu_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_prelu_obj_t {
    arm_compute::NEPReluLayer prelu;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor weights_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_prelu_conf_t {
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo weights_info;
    arm_compute::TensorInfo dst_info;
};

struct acl_prelu_resource_t : public resource_t {
    acl_prelu_resource_t() : acl_obj_(utils::make_unique<acl_prelu_obj_t>()) {}

    status_t configure(const acl_prelu_conf_t &app) {
        if (!acl_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_obj_->src_tensor.allocator()->init(app.src_info);
        acl_obj_->weights_tensor.allocator()->init(app.weights_info);
        acl_obj_->dst_tensor.allocator()->init(app.dst_info);

        // clang-format off
        acl_obj_->prelu.configure(
            &acl_obj_->src_tensor,
            &acl_obj_->weights_tensor,
            &acl_obj_->dst_tensor);
        // clang-format on

        return status::success;
    }

    acl_prelu_obj_t &get_acl_obj() const { return *acl_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_prelu_resource_t);

private:
    std::unique_ptr<acl_prelu_obj_t> acl_obj_;
}; // acl_prelu_resource_t

struct acl_prelu_fwd_t : public primitive_t {
    struct pd_t : public cpu_prelu_fwd_pd_t {
        using cpu_prelu_fwd_pd_t::cpu_prelu_fwd_pd_t;

        DECLARE_COMMON_PD_T("acl", acl_prelu_fwd_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace acl_utils;

            // Forward only
            if (!prelu_pd_t::is_fwd()) return status::unimplemented;

            // Only support f32 and f16 for now
            data_type_t ddt = dst_md(0)->data_type;
            if (!utils::one_of(ddt, data_type::f32, data_type::f16))
                return status::unimplemented;

            if (!set_default_formats()) return status::unimplemented;

            if (!attr()->has_default_values()) return status::unimplemented;

            // ACL pointwise arithmetic operators assume that the innermost
            // dimensions are dense for src, weights and dst. Reordering the
            // logical dimensions by stride does this (if reordered_dims >= 1 )
            // and also makes memory accesses contiguous in ACL (without any
            // data reordering).
            memory_desc_t src_d_permed, weights_d_permed, dst_d_permed;
            int reordered_dims = reorder_dimensions_by_stride(
                    {&src_d_permed, &weights_d_permed, &dst_d_permed},
                    {src_md(0), weights_md(0), dst_md(0)});
            if (reordered_dims < 1) return status::unimplemented;

            // Create ACL tensor infos with permuted descs
            CHECK(tensor_info(app_.src_info, src_d_permed));
            CHECK(tensor_info(app_.weights_info, weights_d_permed));
            CHECK(tensor_info(app_.dst_info, dst_d_permed));

            // This forces ACL not to parallelise with small workloads, this is
            // a temporary fix and should be removed in future versions (TODO)
            memory_desc_wrapper dst_d(dst_md(0));
            if (dst_d.nelems() < 40000) {
                size_t acl_y_axis_i = 1;
                CHECK(insert_singleton_dimension(app_.src_info, acl_y_axis_i));
                CHECK(insert_singleton_dimension(
                        app_.weights_info, acl_y_axis_i));
                CHECK(insert_singleton_dimension(app_.dst_info, acl_y_axis_i));
            }

            ACL_CHECK_VALID(arm_compute::NEPReluLayer::validate(
                    &app_.src_info, &app_.weights_info, &app_.dst_info));

            return status::success;
        }

        acl_prelu_conf_t app_;
    }; // pd_t

    acl_prelu_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_prelu_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        CHECK(r->configure(pd()->app_));

        mapper.add(this, std::move(r));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    // To guard the const execute_forward, the mutex must be 'mutable'
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // acl_prelu_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_PRELU_HPP
