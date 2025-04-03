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

#ifndef ACL_MATMUL_HPP
#define ACL_MATMUL_HPP

#include "common/utils.hpp"
#include "cpu/aarch64/acl_post_ops.hpp"
#include "cpu/aarch64/matmul/acl_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct acl_resource_t : public resource_t {
    acl_resource_t() : acl_obj_(utils::make_unique<acl_matmul_obj_t>()) {}

    status_t configure(const acl_matmul_conf_t &amp,
            const dnnl::impl::format_kind_t weights_format_kind_) {
        if (!acl_obj_) return status::out_of_memory;
        acl_obj_->src_tensor.allocator()->init(amp.src_tensor_info);
        acl_obj_->wei_tensor.allocator()->init(amp.wei_tensor_info);
        acl_obj_->dst_tensor.allocator()->init(amp.dst_tensor_info);

        // Configure transpose kernel for src, wei, or dst
        if (amp.is_transA && !amp.do_transC) {
            acl_obj_->src_acc_tensor.allocator()->init(amp.src_acc_info);
            acl_obj_->transA.configure(
                    &acl_obj_->src_acc_tensor, &acl_obj_->src_tensor);
        }

        if (amp.is_transB && !amp.do_transC) {
            acl_obj_->wei_acc_tensor.allocator()->init(amp.wei_acc_info);
            acl_obj_->transB.configure(
                    &acl_obj_->wei_acc_tensor, &acl_obj_->wei_tensor);
        }

        if (amp.do_transC) {
            acl_obj_->dst_acc_tensor.allocator()->init(amp.dst_acc_info);
            acl_obj_->transC.configure(
                    &acl_obj_->dst_acc_tensor, &acl_obj_->dst_tensor);
        }

        // Configure GEMM
        if (amp.do_transC) {
            acl_obj_->gemm.configure(&acl_obj_->wei_tensor,
                    &acl_obj_->src_tensor, nullptr, &acl_obj_->dst_acc_tensor,
                    1.0f, 0.0f, amp.gemm_info);
        } else {
            acl_obj_->gemm.configure(&acl_obj_->src_tensor,
                    &acl_obj_->wei_tensor, nullptr, &acl_obj_->dst_tensor, 1.0f,
                    0.0f, amp.gemm_info);
        }

        return status::success;
    }
    acl_matmul_obj_t &get_acl_obj() const { return *acl_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_resource_t);

private:
    std::unique_ptr<acl_matmul_obj_t> acl_obj_;
};

struct acl_matmul_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {

        pd_t(const matmul_desc_t *adesc, const primitive_attr_t *attr,
                const cpu_matmul_pd_t *hint_fwd_pd)
            : cpu_matmul_pd_t(adesc, attr, hint_fwd_pd)
            , amp_()
            , acl_post_ops() {}

        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("gemm:acl", acl_matmul_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using smask_t = primitive_attr_t::skip_mask_t;
            const bool is_fp32_ok
                    = utils::everyone_is(data_type::f32, src_md()->data_type,
                              weights_md()->data_type, dst_md()->data_type,
                              desc()->accum_data_type)
                    && platform::has_data_type_support(data_type::f32);
            const bool is_fp16_ok
                    = utils::everyone_is(data_type::f16, src_md()->data_type,
                              weights_md()->data_type, dst_md()->data_type)
                    && platform::has_data_type_support(data_type::f16);
            const bool is_bf16_ok
                    = utils::everyone_is(data_type::bf16, src_md()->data_type,
                              weights_md()->data_type, dst_md()->data_type)
                    && platform::has_data_type_support(data_type::bf16);

            // we need to save this state as it can change inside set_default_formats()
            weights_format_kind_ = weights_md_.format_kind;

            VDISPATCH_MATMUL(
                    is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(
                    utils::one_of(true, is_fp32_ok, is_fp16_ok, is_bf16_ok),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(
                    attr()->has_default_values(smask_t::oscale
                            | smask_t::post_ops | smask_t::fpmath_mode),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(attr_oscale_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_MATMUL(!has_runtime_dims_or_strides(),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);

            if (weights_format_kind_ == format_kind::any) {
                CHECK(acl_matmul_utils::init_conf_matmul<true>(
                        amp_, src_md_, weights_md_, dst_md_, *desc(), *attr()));
            } else {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
                // to avoid seg. fault in case threadpool is enabled and its pointer is null
                if (threadpool_utils::get_active_threadpool() == nullptr)
                    return status::unimplemented;
#endif
                CHECK(acl_matmul_utils::init_conf_matmul<false>(
                        amp_, src_md_, weights_md_, dst_md_, *desc(), *attr()));
            }

            // We can only fuse sum if it is the first post op and we aren't
            // transposing dst after
            if (attr_.post_ops_.contain(primitive_kind::sum, 0)
                    && !amp_.do_transC) {
                // Check there isn't another sum after the first
                VDISPATCH_MATMUL(
                        attr_.post_ops_.find(primitive_kind::sum, 1, -1) < 0,
                        "cannot contain multiple sum post-ops");
                VDISPATCH_MATMUL(attr_.post_ops_.entry_[0].sum.scale == 1.0f,
                        "sum post op scale must be 1 (no scale)");
                VDISPATCH_MATMUL(attr_.post_ops_.entry_[0].sum.zero_point == 0,
                        "sum post op zero point must be 0 (no shift)");
                amp_.gemm_info.set_accumulate(true);
            }

            arm_compute::ActivationLayerInfo act_info;
            CHECK(acl_post_ops.init(engine, attr_.post_ops_, dst_md_, act_info,
                    amp_.gemm_info.accumulate() ? 1 : 0));
            amp_.gemm_info.set_activation_info(act_info);
            amp_.use_dst_acc_for_sum = acl_post_ops.has_sum();

            // Validate ACL GEMM
            if (amp_.do_transC) {
                ACL_CHECK_VALID(arm_compute::NEGEMM::validate(
                        &amp_.wei_tensor_info, &amp_.src_tensor_info, nullptr,
                        &amp_.dst_acc_info, 1.0f, 0.0f, amp_.gemm_info));
            } else {
                ACL_CHECK_VALID(arm_compute::NEGEMM::validate(
                        &amp_.src_tensor_info, &amp_.wei_tensor_info, nullptr,
                        &amp_.dst_tensor_info, 1.0f, 0.0f, amp_.gemm_info));
            }

            auto scratchpad = scratchpad_registry().registrar();
            CHECK(acl_matmul_utils::init_scratchpad(scratchpad, amp_, dst_md_));

            return status::success;
        }

        acl_matmul_conf_t amp_;
        acl_post_ops_t acl_post_ops;
        dnnl::impl::format_kind_t weights_format_kind_;

    protected:
        bool attr_oscale_ok() const {
            const auto &oscale = attr()->output_scales_;
            return oscale.mask_ == 0;
        }
    };

    acl_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;
        auto r = utils::make_unique<acl_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        CHECK(r->configure(pd()->amp_, pd()->weights_format_kind_));
        mapper.add(this, std::move(r));

        CHECK(pd()->acl_post_ops.create_resource(engine, mapper));

        return status::success;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->weights_format_kind_ == format_kind::any) {
            return execute_forward<true>(ctx);
        } else {
            return execute_forward<false>(ctx);
        }
    }

private:
    // To guard the const execute_forward(), the mutex must be 'mutable'
    mutable std::mutex mtx;
    template <bool IsFixedFormat>
    status_t execute_forward(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // acl_matmul_t

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
