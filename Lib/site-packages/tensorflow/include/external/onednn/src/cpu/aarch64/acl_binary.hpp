/*******************************************************************************
* Copyright 2022 Arm Ltd. and affiliates
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed tos in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_AARCH64_ACL_BINARY_HPP
#define CPU_AARCH64_ACL_BINARY_HPP

#include "cpu/cpu_binary_pd.hpp"

#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_binary_obj_t {
    std::unique_ptr<arm_compute::IFunction> binary_op;
    arm_compute::Tensor src0_tensor;
    arm_compute::Tensor src1_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_binary_conf_t {
    arm_compute::TensorInfo src0_info;
    arm_compute::TensorInfo src1_info;
    arm_compute::TensorInfo dst_info;
    alg_kind_t alg;
};

struct acl_binary_resource_t : public resource_t {
    acl_binary_resource_t()
        : acl_obj_(utils::make_unique<acl_binary_obj_t>()) {}

    status_t configure(const acl_binary_conf_t &asp) {
        if (!acl_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_obj_->src0_tensor.allocator()->init(asp.src0_info);
        acl_obj_->src1_tensor.allocator()->init(asp.src1_info);
        acl_obj_->dst_tensor.allocator()->init(asp.dst_info);

        switch (asp.alg) {
            case alg_kind::binary_add: {
                auto add_op
                        = std::make_unique<arm_compute::NEArithmeticAddition>();
                add_op->configure(&acl_obj_->src0_tensor,
                        &acl_obj_->src1_tensor, &acl_obj_->dst_tensor,
                        arm_compute::ConvertPolicy::SATURATE);
                acl_obj_->binary_op = std::move(add_op);
                break;
            }
            case alg_kind::binary_sub: {
                auto sub_op = std::make_unique<
                        arm_compute::NEArithmeticSubtraction>();
                sub_op->configure(&acl_obj_->src0_tensor,
                        &acl_obj_->src1_tensor, &acl_obj_->dst_tensor,
                        arm_compute::ConvertPolicy::SATURATE);
                acl_obj_->binary_op = std::move(sub_op);
                break;
            }
            case alg_kind::binary_div: {
                auto div_op = std::make_unique<
                        arm_compute::NEElementwiseDivision>();
                div_op->configure(&acl_obj_->src0_tensor,
                        &acl_obj_->src1_tensor, &acl_obj_->dst_tensor);
                acl_obj_->binary_op = std::move(div_op);
                break;
            }
            case alg_kind::binary_mul: {
                auto mul_op = std::make_unique<
                        arm_compute::NEPixelWiseMultiplication>();
                mul_op->configure(&acl_obj_->src0_tensor,
                        &acl_obj_->src1_tensor, &acl_obj_->dst_tensor, 1.0f,
                        arm_compute::ConvertPolicy::SATURATE,
                        arm_compute::RoundingPolicy::TO_ZERO);
                acl_obj_->binary_op = std::move(mul_op);
                break;
            }
            case alg_kind::binary_min: {
                auto min_op = std::make_unique<arm_compute::NEElementwiseMin>();
                min_op->configure(&acl_obj_->src0_tensor,
                        &acl_obj_->src1_tensor, &acl_obj_->dst_tensor);
                acl_obj_->binary_op = std::move(min_op);
                break;
            }
            case alg_kind::binary_max: {
                auto max_op = std::make_unique<arm_compute::NEElementwiseMax>();
                max_op->configure(&acl_obj_->src0_tensor,
                        &acl_obj_->src1_tensor, &acl_obj_->dst_tensor);
                acl_obj_->binary_op = std::move(max_op);
                break;
            }
            default: return status::runtime_error;
        }

        return status::success;
    }

    acl_binary_obj_t &get_acl_obj() const { return *acl_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_binary_resource_t);

private:
    std::unique_ptr<acl_binary_obj_t> acl_obj_;
}; // acl_binary_resource_t

struct acl_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("acl", acl_binary_t);

        status_t init(engine_t *engine) {

            using namespace acl_utils;

            // Only support f16/f32/s32 for now
            data_type_t ddt = dst_md(0)->data_type;
            if (!utils::one_of(
                        ddt, data_type::f16, data_type::f32, data_type::s32))
                return status::unimplemented;

            // Only support src and dst all matching for now
            if (ddt != src_md(0)->data_type || src_md(1)->data_type != ddt)
                return status::unimplemented;

            // Sets the memory format of dst from any to src_md(0) blocking desc
            CHECK(set_default_params());

            if (!attr()->has_default_values()) return status::unimplemented;

            asp_.alg = desc()->alg_kind;

            // All the algorithms we support
            if (!utils::one_of(asp_.alg, alg_kind::binary_add,
                        alg_kind::binary_sub, alg_kind::binary_mul,
                        alg_kind::binary_div, alg_kind::binary_max,
                        alg_kind::binary_min))
                return status::unimplemented;

            // s32 div in ACL does not round as oneDNN expects
            if (ddt == data_type::s32 && asp_.alg == alg_kind::binary_div)
                return status::unimplemented;

            // ACL pointwise arithmetic operators assume that the innermost
            // dimensions are dense for src0, src1 and dst. Reordering the
            // logical dimensions by stride does this (if reordered_dims >= 1 )
            // and also makes memory accesses contiguous in ACL (without any
            // data reordering).
            memory_desc_t src_d0_permed, src_d1_permed, dst_d_permed;
            int reordered_dims = reorder_dimensions_by_stride(
                    {&src_d0_permed, &src_d1_permed, &dst_d_permed},
                    {src_md(0), src_md(1), dst_md()});
            if (reordered_dims < 1) return status::unimplemented;

            // Create ACL tensor infos with permuted descs
            CHECK(tensor_info(asp_.src0_info, src_d0_permed));
            CHECK(tensor_info(asp_.src1_info, src_d1_permed));
            CHECK(tensor_info(asp_.dst_info, dst_d_permed));

            // In this case ACL tries to treat src0 and src1 as a 1D array, but
            // fails because the strides aren't equal. TODO: remove when fixed
            // in ACL
            if (asp_.alg == alg_kind::binary_add
                    && asp_.src0_info.tensor_shape()
                            == asp_.src1_info.tensor_shape()
                    && asp_.src0_info.strides_in_bytes()
                            != asp_.src1_info.strides_in_bytes()) {
                return status::unimplemented;
            }

            // This forces ACL not to parallelise with small workloads, this is
            // a temporary fix and should be removed in future versions (TODO)
            memory_desc_wrapper dst_d(dst_md());
            if (dst_d.nelems() < 40000) {
                size_t acl_y_axis_i = 1;
                CHECK(insert_singleton_dimension(asp_.src0_info, acl_y_axis_i));
                CHECK(insert_singleton_dimension(asp_.src1_info, acl_y_axis_i));
                CHECK(insert_singleton_dimension(asp_.dst_info, acl_y_axis_i));
            }

            // Call operator specific validate function to check support
            ACL_CHECK_VALID(validate(asp_));

            return status::success;
        }

        acl_binary_conf_t asp_;

    private:
        arm_compute::Status validate(const acl_binary_conf_t &asp) {
            switch (asp.alg) {
                case alg_kind::binary_add:
                    return arm_compute::NEArithmeticAddition::validate(
                            &asp.src0_info, &asp.src1_info, &asp.dst_info,
                            arm_compute::ConvertPolicy::SATURATE);
                case alg_kind::binary_sub:
                    return arm_compute::NEArithmeticSubtraction::validate(
                            &asp.src0_info, &asp.src1_info, &asp.dst_info,
                            arm_compute::ConvertPolicy::SATURATE);
                case alg_kind::binary_div:
                    return arm_compute::NEElementwiseDivision::validate(
                            &asp.src0_info, &asp.src1_info, &asp.dst_info);
                case alg_kind::binary_mul:
                    return arm_compute::NEPixelWiseMultiplication::validate(
                            &asp.src0_info, &asp.src1_info, &asp.dst_info, 1.0f,
                            arm_compute::ConvertPolicy::SATURATE,
                            arm_compute::RoundingPolicy::TO_ZERO);
                case alg_kind::binary_min:
                    return arm_compute::NEElementwiseMin::validate(
                            &asp.src0_info, &asp.src1_info, &asp.dst_info);
                case alg_kind::binary_max:
                    return arm_compute::NEElementwiseMax::validate(
                            &asp.src0_info, &asp.src1_info, &asp.dst_info);
                default:
                    return arm_compute::Status(
                            arm_compute::ErrorCode::RUNTIME_ERROR,
                            "unsupported alg_kind");
            }
        }

        friend struct acl_post_ops_t;
    }; // pd_t

    acl_binary_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_binary_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        auto st = r->configure(pd()->asp_);
        if (st == status::success) { mapper.add(this, std::move(r)); }

        return st;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    // To guard the const execute_forward, the mutex must be 'mutable'
    mutable std::mutex mtx;

    status_t execute_forward(const exec_ctx_t &ctx) const;
    // Execute forward with arbitrary src0, src1 and dst, used by acl_post_ops_t
    status_t execute_forward(const exec_ctx_t &ctx, const void *src0,
            const void *src1, void *dst) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    friend struct acl_post_ops_t;

}; // acl_binary_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
