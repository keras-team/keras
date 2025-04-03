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

#ifndef CPU_AARCH64_ACL_BATCH_NORMALIZATION_HPP
#define CPU_AARCH64_ACL_BATCH_NORMALIZATION_HPP

#include "cpu/cpu_batch_normalization_pd.hpp"

#include "cpu/aarch64/acl_post_ops.hpp"
#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_batch_normalization_obj_t {
    arm_compute::NEBatchNormalizationLayer bnorm;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;
    arm_compute::Tensor mean_tensor;
    arm_compute::Tensor var_tensor;
    arm_compute::Tensor beta_tensor; // shift
    arm_compute::Tensor gamma_tensor; // scale
};

struct acl_batch_normalization_conf_t {
    // src and dst
    arm_compute::TensorInfo data_info;
    // TensorInfo for statistics (mean and/or variance), shift and scale
    arm_compute::TensorInfo stats_info;
    arm_compute::ActivationLayerInfo act_info;
};

struct acl_batch_normalization_resource_t : public resource_t {
    acl_batch_normalization_resource_t()
        : acl_obj_(utils::make_unique<acl_batch_normalization_obj_t>()) {}

    status_t configure(const acl_batch_normalization_conf_t &abp,
            const batch_normalization_pd_t *pd) {
        if (!acl_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_obj_->src_tensor.allocator()->init(abp.data_info);
        acl_obj_->dst_tensor.allocator()->init(abp.data_info);

        acl_obj_->mean_tensor.allocator()->init(abp.stats_info);
        acl_obj_->var_tensor.allocator()->init(abp.stats_info);
        if (pd->use_shift())
            acl_obj_->beta_tensor.allocator()->init(abp.stats_info);
        if (pd->use_scale())
            acl_obj_->gamma_tensor.allocator()->init(abp.stats_info);

        // clang-format off
        acl_obj_->bnorm.configure(
            &acl_obj_->src_tensor,
            &acl_obj_->dst_tensor,
            &acl_obj_->mean_tensor,
            &acl_obj_->var_tensor,
            pd->use_shift() ? &acl_obj_->beta_tensor : nullptr,
            pd->use_scale() ? &acl_obj_->gamma_tensor : nullptr,
            pd->desc()->batch_norm_epsilon,
            abp.act_info);
        // clang-format on

        return status::success;
    }

    acl_batch_normalization_obj_t &get_acl_obj() const { return *acl_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_batch_normalization_resource_t);

private:
    std::unique_ptr<acl_batch_normalization_obj_t> acl_obj_;
}; // acl_batch_normalization_resource_t

struct acl_batch_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_batch_normalization_fwd_pd_t {
        using cpu_batch_normalization_fwd_pd_t::
                cpu_batch_normalization_fwd_pd_t;

        pd_t(const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_batch_normalization_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , abp() {}

        DECLARE_COMMON_PD_T("acl", acl_batch_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;

            // Get memory desc to find sizes and dims
            const memory_desc_wrapper src_d(src_md_);
            const memory_desc_wrapper stat_d(stat_md_);

            using smask_t = primitive_attr_t::skip_mask_t;
            ACL_CHECK_SUPPORT(!attr()->has_default_values(smask_t::post_ops),
                    "attr must have default values");

            ACL_CHECK_SUPPORT(!set_default_formats_common(),
                    "Failed to set default formats");
            ACL_CHECK_SUPPORT(src_d != memory_desc_wrapper(dst_md()),
                    "Source and destination must have the same layout");

            // Stats must already have been calculated
            ACL_CHECK_SUPPORT(!use_global_stats(),
                    "stats must already have been computed (use global stats)");

            ACL_CHECK_SUPPORT(!is_fwd(), "must be forward mode");

            ACL_CHECK_SUPPORT(!utils::one_of(src_d.data_type(), data_type::f32,
                                      data_type::f16),
                    "data type must be f32/f16");

            // Dimensions can be permuted but no blocking or padding
            ACL_CHECK_SUPPORT((!src_d.is_plain() || !stat_d.is_plain()),
                    "tensors must be plain");
            ACL_CHECK_SUPPORT((!src_d.is_dense() || !stat_d.is_dense()),
                    "tensors must be dense");

            ACL_CHECK_SUPPORT(src_d.is_zero() || src_d.has_zero_dim(),
                    "zero sized src tensor");
            ACL_CHECK_SUPPORT(stat_d.is_zero() || stat_d.has_zero_dim(),
                    "zero sized stats tensor");

            // mean, var, scale and shift all have a single dimension with size
            // equal to the number of channels
            auto acl_stats_dt = acl_utils::get_acl_data_t(stat_d.data_type());
            abp.stats_info
                    = arm_compute::TensorInfo(arm_compute::TensorShape(C()), 1,
                            acl_stats_dt, arm_compute::DataLayout::NHWC);

            // We can simplify into two cases: channels are dense or not dense
            auto channel_stride = src_d.blocking_desc().strides[1];
            if (channel_stride == 1) {
                // Channels are dense => NHWC
                // Collapse all dims except channels into W
                auto elems = src_d.nelems();
                dim_t w = elems / C();

                // For small problems, it is better to disable threads with ACL
                bool use_acl_threads = elems > 20000;

                // The number of threads each candidate implementation
                // (acl,nspc) is able to use for these problem parameters
                int max_threads = dnnl_get_max_threads();
                int acl_threads
                        = use_acl_threads ? std::min((int)w, max_threads) : 1;
                int nspc_threads = std::min((int)MB(), max_threads);

                // A simple empirical heuristic which is negative when ACL is
                // faster, positive when slower. The first term represents a
                // constant overhead when calling ACL as an external library,
                // the second term represents the cost per tensor elem for the
                // two candidate implementations (scaling for the number of
                // threads used)
                double acl_ref_time_diff = 33
                        + elems * (0.11 / acl_threads - 0.48 / nspc_threads);

                // Threads in ACL incur an additional cost
                if (use_acl_threads) {
                    acl_ref_time_diff += 1500.0 + 130.0 * (acl_threads - 1);
                }

                // If ACL is slower, don't use it, fall back to nspc_bnorm
                if (acl_ref_time_diff > 0) return status::unimplemented;

                // We disable threading in ACL by moving the width into height
                auto data_shape = use_acl_threads
                        ? arm_compute::TensorShape(C(), w, 1, 1)
                        : arm_compute::TensorShape(C(), 1, w, 1);
                auto acl_data_dt = acl_utils::get_acl_data_t(src_d.data_type());
                abp.data_info = arm_compute::TensorInfo(data_shape, 1,
                        acl_data_dt, arm_compute::DataLayout::NHWC);
            } else {
                // Implemented in ACL but not yet optimal
                return status::unimplemented;
            }

            ACL_CHECK_SUPPORT(fuse_norm_relu()
                            && desc()->prop_kind == prop_kind::forward_training,
                    "forward training with fused ReLU is not supported");

            // oneDNN only supports eltwise post ops with bnorm
            for (int i = 0; i < attr()->post_ops_.len(); ++i)
                if (!attr()->post_ops_.entry_[i].is_eltwise())
                    return status::unimplemented;

            if (fuse_norm_relu()) {
                abp.act_info = arm_compute::ActivationLayerInfo(arm_compute::
                                ActivationLayerInfo::ActivationFunction::RELU);
                // ACL BNorm supports fusing ReLU. If this validate fails, it
                // will be for a different, unrecoverable reason
                CHECK(validate(abp.act_info));
                // init any additional post ops
                CHECK(post_ops.init(engine, attr_.post_ops_, src_md_));
            } else {
                // init post ops, removing first eltwise for fusion
                arm_compute::ActivationLayerInfo act_info;
                CHECK(post_ops.init(
                        engine, attr_.post_ops_, src_md_, act_info));
                // ACL BNorm doesn't support all the same eltwise ops as the
                // standalone ACL operator, so fall back to unfused eltwise if
                // validate fails
                if (validate(act_info) == status::success) {
                    // validate succeeded => we can fuse the eltwise
                    abp.act_info = act_info;
                } else {
                    // validate unfused eltwise + remaining post ops
                    CHECK(validate());
                    CHECK(post_ops.init(engine, attr_.post_ops_, src_md_));
                }
            }

            return status::success;
        }

        status_t validate() { return validate(abp.act_info); }

        status_t validate(arm_compute::ActivationLayerInfo &act_info) {
            ACL_CHECK_VALID(arm_compute::NEBatchNormalizationLayer::validate(
                    &abp.data_info, &abp.data_info, &abp.stats_info,
                    &abp.stats_info, use_shift() ? &abp.stats_info : nullptr,
                    use_scale() ? &abp.stats_info : nullptr,
                    desc()->batch_norm_epsilon, act_info));
            return status::success;
        }

        acl_batch_normalization_conf_t abp;

        acl_post_ops_t post_ops;
    }; // pd_t

    acl_batch_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_batch_normalization_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        CHECK(r->configure(pd()->abp, pd()));
        mapper.add(this, std::move(r));

        CHECK(pd()->post_ops.create_resource(engine, mapper));

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
}; // acl_batch_normalization_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
