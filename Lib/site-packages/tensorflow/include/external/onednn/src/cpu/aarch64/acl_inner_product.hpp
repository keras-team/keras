/*******************************************************************************
* Copyright 2021-2023 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_INNER_PRODUCT_HPP
#define CPU_AARCH64_ACL_INNER_PRODUCT_HPP

#include "cpu/aarch64/acl_utils.hpp"
#include "cpu/cpu_inner_product_pd.hpp"

#include "cpu/aarch64/acl_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_ip_obj_t {
    arm_compute::NEFullyConnectedLayer fc;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_ip_conf_t {
    bool with_bias;
    // If this is true, the result of the inner product goes into a temporarily
    // allocated ACL tensor to be accumulated into the oneDNN dst during postops
    bool use_dst_acc;
    arm_compute::TensorInfo src_tensor_info;
    arm_compute::TensorInfo wei_tensor_info;
    arm_compute::TensorInfo bia_tensor_info;
    arm_compute::TensorInfo dst_tensor_info;
    arm_compute::FullyConnectedLayerInfo fc_info;
    // Additional information about the weights not included in wei_tensor_info
    arm_compute::WeightsInfo weights_info;
};
struct acl_ip_resource_t : public resource_t {
    acl_ip_resource_t() : acl_ip_obj_(utils::make_unique<acl_ip_obj_t>()) {}

    status_t configure(const acl_ip_conf_t &aip) {
        if (!acl_ip_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_ip_obj_->src_tensor.allocator()->init(aip.src_tensor_info);
        acl_ip_obj_->wei_tensor.allocator()->init(aip.wei_tensor_info);
        acl_ip_obj_->dst_tensor.allocator()->init(aip.dst_tensor_info);
        acl_ip_obj_->bia_tensor.allocator()->init(aip.bia_tensor_info);

        // clang-format off
        acl_ip_obj_->fc.configure(
            &acl_ip_obj_->src_tensor,
            &acl_ip_obj_->wei_tensor,
            aip.with_bias ? &acl_ip_obj_->bia_tensor : nullptr,
            &acl_ip_obj_->dst_tensor,
            aip.fc_info,
            aip.weights_info);
        // clang-format on

        return status::success;
    }

    acl_ip_obj_t &get_acl_obj() const { return *acl_ip_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_ip_resource_t);

private:
    std::unique_ptr<acl_ip_obj_t> acl_ip_obj_;
}; // acl_ip_resource_t

struct acl_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_inner_product_fwd_pd_t(adesc, attr, hint_fwd_pd), aip() {}

        DECLARE_COMMON_PD_T("acl", acl_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            const format_kind_t weights_format_kind_received
                    = weights_md_.format_kind;
            const bool is_fp16_ok = expect_data_types(f16, f16, f16, f16, undef)
                    && attr()->has_default_values(smask_t::post_ops, f16);
            const bool is_fp32_ok = expect_data_types(f32, f32, f32, f32, undef)
                    && attr()->has_default_values(
                            smask_t::post_ops | smask_t::fpmath_mode, f32);
            const bool is_fp32_bf16_ok
                    = expect_data_types(f32, bf16, f32, f32, undef)
                    && attr()->has_default_values(
                            smask_t::post_ops | smask_t::fpmath_mode, f32);
            const bool is_weights_md_format_ok
                    = utils::one_of(weights_format_kind_received,
                            format_kind::any, format_kind::blocked);
            const bool ok = is_fwd() && !has_zero_dim_memory()
                    && utils::one_of(
                            true, is_fp16_ok, is_fp32_ok, is_fp32_bf16_ok)
                    && is_weights_md_format_ok
                    && set_default_params(true) == status::success;

            if (!ok) return status::unimplemented;

            CHECK(init_conf_ip(engine, weights_format_kind_received));

            return status::success;
        }

        acl_ip_conf_t aip;

        acl_post_ops_t post_ops;

        status_t init_conf_ip(
                engine_t *engine, format_kind_t weights_format_kind_received) {

            ACL_CHECK_SUPPORT(src_md()->ndims != weights_md()->ndims,
                    "source and weights dimensions must match");

            const int ndims = src_md()->ndims;

            const bool is_2d = (ndims == 2);
            const bool is_4d = (ndims == 4);

            ACL_CHECK_SUPPORT(
                    !(is_2d || is_4d), "ACL supports only 2d or 4d cases");

            using namespace format_tag;
            auto src_tag
                    = memory_desc_matches_one_of_tag(src_md_, nhwc, nchw, nc);
            auto dst_tag = memory_desc_matches_one_of_tag(dst_md_, nc);

            ACL_CHECK_SUPPORT(
                    utils::one_of(format_tag::undef, src_tag, dst_tag),
                    "unsupported memory layout");

            ACL_CHECK_SUPPORT(is_2d && src_tag != dst_tag,
                    "for src and dst layouts must match");

            const dim_t ic_total = IC_total();
            const dim_t n = MB();
            const dim_t oc = OC();

            aip.src_tensor_info = arm_compute::TensorInfo(
                    arm_compute::TensorShape(ic_total, n), 1,
                    acl_utils::get_acl_data_t(src_md()->data_type));

            // ACL requires the weights to be in 2D flattened shape
            aip.wei_tensor_info = arm_compute::TensorInfo(
                    arm_compute::TensorShape(oc, ic_total), 1,
                    acl_utils::get_acl_data_t(weights_md(0)->data_type));

            auto acl_dst_data_t
                    = acl_utils::get_acl_data_t(dst_md()->data_type);
            aip.dst_tensor_info = arm_compute::TensorInfo(
                    arm_compute::TensorShape(oc, n), 1, acl_dst_data_t);

            aip.with_bias = desc()->bias_desc.format_kind != format_kind::undef;
            auto acl_bia_data_t = aip.with_bias
                    ? acl_utils::get_acl_data_t(weights_md(1)->data_type)
                    : acl_dst_data_t;
            aip.bia_tensor_info = arm_compute::TensorInfo(aip.with_bias
                            ? arm_compute::TensorShape(oc)
                            : arm_compute::TensorShape(),
                    1, acl_bia_data_t);

            aip.fc_info.transpose_weights = false;

            aip.fc_info.enable_fast_math = utils::one_of(
                    attr()->fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any);

            CHECK(post_ops.init(engine, attr_.post_ops_, dst_md_,
                    aip.fc_info.activation_info));
            aip.use_dst_acc = post_ops.has_sum();

            // WeightFormat::ANY tells ACL we can handle any format
            aip.weights_info = arm_compute::WeightsInfo(false, 1, 1, ic_total,
                    false, arm_compute::WeightFormat::ANY);

            // Get the format that the ACL kernel will expect the weights to be
            // in (if a kernel exists) Note that these are referred to as fixed
            // format kernels, because they require one specific weights format
            arm_compute::WeightFormat expected_weight_format;
            ACL_CHECK_VALID(arm_compute::NEFullyConnectedLayer::has_opt_impl(
                    expected_weight_format, &aip.src_tensor_info,
                    &aip.wei_tensor_info,
                    aip.with_bias ? &aip.bia_tensor_info : nullptr,
                    &aip.dst_tensor_info, aip.fc_info, aip.weights_info));

            // Set weights info to the one returned by has_opt_impl
            aip.weights_info.set_weight_format(expected_weight_format);

            // has_opt_impl may return a non fast math kernel, even if requested
            aip.fc_info.enable_fast_math
                    = arm_compute::is_fixed_format_fast_math(
                            expected_weight_format);

            // Inner product is the same as the matmul n x (chw) * (ihw) x o
            // (note that the src c and weights i both correspond to the input
            // channel). ACL FullyConnectedLayer assumes the chw dimensions of
            // src and ihw dimensions of weights are collapsed, so we need to
            // make sure that they have the same layout. Given that weights are
            // more often fixed, (so reorders can be hoisted) it makes sense to
            // reorder the weights to fit the src.

            // For 4D tensors we need to:
            // - reorder the ihw of the weights to match the src chw
            // - collapse ihw
            // - pad the collapsed ihw
            // But there is not yet a way to express this collapse+pad as a
            // reorder. So we try to reorder the weights to match the src,
            // implicitly collapse ihw in our definition of the weights
            // TensorInfo and hope that the inner_dim has zero padding
            // (weights_md_.dims[inner_dim] % block_by == 0). If it does, we
            // fall back to a kernel without blocking (currently this is
            // equivalent to non-fastmath).

            // 2D just works because we just pad the only dimension.

            // o_dim is always the first logical dimension (oihw, ohwi, oi)
            dim_t o_dim = 0;
            dim_t inner_dim;
            // Rest of logical dimensions in order of innermost to outermost
            std::vector<dim_t> remaining_dims = {};

            if (src_tag == nchw) {
                inner_dim = 3; // w
                remaining_dims = {2, 1}; // h, i
            } else if (src_tag == nhwc) {
                inner_dim = 1; // i
                remaining_dims = {3, 2}; // w, h
            } else { // Only remaining case is 2D (nc)
                inner_dim = 1; // i
                remaining_dims = {}; // No other dimensions for 2D
            }

            // Fallback
            int block_by = arm_compute::block_by(expected_weight_format);
            if (is_4d && weights_md_.dims[inner_dim] % block_by != 0
                    && aip.fc_info.enable_fast_math) {
                aip.fc_info.enable_fast_math = false;
                aip.weights_info.set_weight_format(
                        arm_compute::WeightFormat::ANY);
                ACL_CHECK_VALID(
                        arm_compute::NEFullyConnectedLayer::has_opt_impl(
                                expected_weight_format, &aip.src_tensor_info,
                                &aip.wei_tensor_info,
                                aip.with_bias ? &aip.bia_tensor_info : nullptr,
                                &aip.dst_tensor_info, aip.fc_info,
                                aip.weights_info));
                aip.weights_info.set_weight_format(expected_weight_format);
                block_by = arm_compute::block_by(expected_weight_format);
                if (weights_md_.dims[inner_dim] % block_by != 0)
                    return status::unimplemented;
            }

            const memory_desc_t weights_md_received = weights_md_;
            acl_utils::reorder_to_weight_format(aip.wei_tensor_info,
                    weights_md_, expected_weight_format, inner_dim, o_dim,
                    remaining_dims, {});

            ACL_CHECK_SUPPORT(
                    (weights_format_kind_received == format_kind::blocked)
                            && !(dnnl_memory_desc_equal(
                                    &weights_md_received, &weights_md_)),
                    "specific blocked format not supported by ACL, use "
                    "format_kind_t::any to find a supported blocked format for "
                    "your platform");

            // clang-format off

            // Validate fully connected layer manually to check for return status
            ACL_CHECK_VALID(arm_compute::NEFullyConnectedLayer::validate(
                &aip.src_tensor_info,
                &aip.wei_tensor_info,
                aip.with_bias ? &aip.bia_tensor_info : nullptr,
                &aip.dst_tensor_info,
                aip.fc_info,
                aip.weights_info));
            // clang-format on

            return status::success;
        }
    }; // pd_t

    acl_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_ip_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        CHECK(r->configure(pd()->aip));
        mapper.add(this, std::move(r));

        CHECK(pd()->post_ops.create_resource(engine, mapper));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    //To guard the const execute_forward, the mutex must be 'mutable'
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // acl_inner_product_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_INNER_PRODUCT_HPP
