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

#ifndef CPU_AARCH64_ACL_SOFTMAX_HPP
#define CPU_AARCH64_ACL_SOFTMAX_HPP

#include "cpu/cpu_softmax_pd.hpp"

#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_softmax_obj_t {
    std::unique_ptr<arm_compute::IFunction> softmax;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_softmax_conf_t {
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo dst_info;
    float beta;
    int32_t axis;
    bool is_logsoftmax;
};

struct acl_softmax_resource_t : public resource_t {
    acl_softmax_resource_t()
        : acl_obj_(utils::make_unique<acl_softmax_obj_t>()) {}

    status_t configure(const acl_softmax_conf_t &asp) {
        if (!acl_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_obj_->src_tensor.allocator()->init(asp.src_info);
        acl_obj_->dst_tensor.allocator()->init(asp.dst_info);

        if (asp.is_logsoftmax) {
            auto logsoftmax
                    = std::make_unique<arm_compute::NELogSoftmaxLayer>();
            // clang-format off
            logsoftmax->configure(
                &acl_obj_->src_tensor,
                &acl_obj_->dst_tensor,
                asp.beta,
                asp.axis);
            // clang-format on
            acl_obj_->softmax = std::move(logsoftmax);
        } else {
            auto softmax = std::make_unique<arm_compute::NESoftmaxLayer>();
            // clang-format off
            softmax->configure(
                &acl_obj_->src_tensor,
                &acl_obj_->dst_tensor,
                asp.beta,
                asp.axis);
            // clang-format on
            acl_obj_->softmax = std::move(softmax);
        }

        return status::success;
    }

    acl_softmax_obj_t &get_acl_obj() const { return *acl_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_softmax_resource_t);

private:
    std::unique_ptr<acl_softmax_obj_t> acl_obj_;
}; // acl_softmax_resource_t

struct acl_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("acl", acl_softmax_fwd_t);

        status_t init(engine_t *engine) {

            bool ok = is_fwd()
                    && set_default_formats() == status::success
                    // ACL only supports matching src/dst (this must come after
                    // set_default_formats() to handle format_kind::any)
                    && *src_md() == *dst_md()
                    && utils::one_of(
                            src_md()->data_type, data_type::f32, data_type::f16)
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            // Get memory desc to find sizes and dims
            const memory_desc_wrapper src_d(src_md());
            const data_type_t data_type = src_d.data_type();

            // ACL only supports plain tensors, can be permuted but not blocked
            if (!src_d.is_plain()) return status::unimplemented;

            // Guards against a 0-sized dimension
            if (src_d.has_zero_dim()) return status::unimplemented;

            // No scaling
            asp_.beta = 1;

            asp_.is_logsoftmax = is_logsoftmax();

            // The strides give us the in memory inner size
            dim_t inner_size_ = src_d.blocking_desc().strides[axis()];

            dim_t axis_size_ = axis_size();

            // The outer size is any left-over dimensions not inner or on the axis
            dim_t outer_size_ = src_d.nelems() / (inner_size_ * axis_size_);

            // In this context, NHWC tells ACL that the logical and physical
            // dimensions are the same
            arm_compute::DataLayout acl_layout = arm_compute::DataLayout::NHWC;

            const arm_compute::DataType acl_data_t
                    = acl_utils::get_acl_data_t(data_type);

            const int threads = dnnl_get_max_threads();
            if (inner_size_ == 1) {
                // A rough empirical heuristic created by fitting a polynomial
                // of the tensor sizes and thread count to the run time of the
                // ref and ACL softmax. This variable is greater than zero when
                // ref is faster, and less than zero when ACL is faster. We can
                // interpret the constant term as the constant overhead
                // associated with calling the external library and the negative
                // coefficient on total_size as ACL being faster at processing
                // each element
                double acl_ref_performance_diff = 1 + 0.005 * outer_size_
                        - 0.0027 * axis_size_
                                * std::ceil(double(outer_size_) / threads);
                if (threads > 1 || outer_size_ > 1) {
                    // Using threads within ACL adds another constant overhead
                    acl_ref_performance_diff += 17;
                }
                if (acl_ref_performance_diff > 0) return status::unimplemented;

                // If the inner size is 1, we can get rid of the dimension.
                // This stops ACL doing a unnecessary permute
                arm_compute::TensorShape acl_tensor_shape
                        = arm_compute::TensorShape(axis_size_, outer_size_);
                asp_.axis = 0;

                asp_.src_info = arm_compute::TensorInfo(
                        acl_tensor_shape, 1, acl_data_t, acl_layout);
                asp_.dst_info = arm_compute::TensorInfo(
                        acl_tensor_shape, 1, acl_data_t, acl_layout);
            } else {
                // A rough empirical heuristic, see comment above
                // The only difference here is that ACL does a reorder, and so
                // is considerably better
                double acl_ref_performance_diff = 1 + 0.005 * outer_size_
                        - 0.01 * inner_size_ * axis_size_
                                * std::ceil(double(outer_size_) / threads);
                if (threads > 1 || outer_size_ > 1) {
                    // Using threads within ACL adds another constant overhead
                    acl_ref_performance_diff += 17;
                }

                if (acl_ref_performance_diff > 0) return status::unimplemented;

                // Irrespective of the input dimensions, we construct a tensor
                // with dimensions such that softmax can be applied over the
                // middle axis (1), with the correct stride and vector length.
                arm_compute::TensorShape acl_tensor_shape
                        = arm_compute::TensorShape(
                                inner_size_, axis_size_, outer_size_);
                asp_.axis = 1;

                asp_.src_info = arm_compute::TensorInfo(
                        acl_tensor_shape, 1, acl_data_t, acl_layout);
                asp_.dst_info = arm_compute::TensorInfo(
                        acl_tensor_shape, 1, acl_data_t, acl_layout);
            }

            // Validate manually to check for return status
            if (asp_.is_logsoftmax) {
                ACL_CHECK_VALID(arm_compute::NELogSoftmaxLayer::validate(
                        &asp_.src_info, &asp_.dst_info, asp_.beta, asp_.axis));
            } else {
                ACL_CHECK_VALID(arm_compute::NESoftmaxLayer::validate(
                        &asp_.src_info, &asp_.dst_info, asp_.beta, asp_.axis));
            }

            return status::success;
        }

        acl_softmax_conf_t asp_;
    }; // pd_t

    acl_softmax_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_softmax_resource_t>();
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
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // acl_softmax_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
