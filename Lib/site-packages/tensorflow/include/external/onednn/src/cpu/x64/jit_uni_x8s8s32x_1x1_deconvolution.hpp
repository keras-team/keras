/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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
* *******************************************************************************/

#ifndef CPU_X64_JIT_UNI_X8S8S32X_1X1_DECONVOLUTION_HPP
#define CPU_X64_JIT_UNI_X8S8S32X_1X1_DECONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_deconvolution_pd.hpp"

#include "cpu/x64/jit_uni_1x1_conv_utils.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_1x1_convolution.hpp"
#include "cpu/zero_point_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct jit_uni_x8s8s32x_1x1_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(
                name_.c_str(), jit_uni_x8s8s32x_1x1_deconvolution_fwd_t);

        status_t init_convolution(engine_t *engine) {
            convolution_desc_t cd;

            auto dd = desc();
            CHECK(conv_desc_init(&cd, prop_kind::forward_training,
                    alg_kind::convolution_direct, &(dd->src_desc),
                    &(dd->weights_desc), &(dd->bias_desc), &(dd->dst_desc),
                    dd->strides, dd->dilates, dd->padding[0], dd->padding[1]));

            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;
            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;

            while (++it != it.end()) {
                conv_pd_ = *it;
                // XXX: find another way to create required implementation.
                if (dynamic_cast<conv_pd_t *>(conv_pd_.get()))
                    return set_default_params();
            }

            return status::unimplemented;
        };

        status_t init(engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;
            VDISPATCH_DECONVOLUTION(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_DECONVOLUTION(
                    desc()->alg_kind == alg_kind::deconvolution_direct,
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_DECONVOLUTION(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_DECONVOLUTION(utils::one_of(src_md(0)->data_type, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(
                    weights_md(0)->data_type == s8, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(
                    IMPLICATION(with_bias(),
                            utils::one_of(weights_md(1)->data_type, f32, s32,
                                    s8, u8)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(
                    utils::one_of(dst_md(0)->data_type, f32, s32, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(
                    desc()->accum_data_type == s32, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(
                    attr()->has_default_values(skip_mask_t::scales_runtime
                            | skip_mask_t::post_ops
                            | skip_mask_t::zero_points_runtime),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_DECONVOLUTION(
                    zero_points_valid(attr(), true /*per_oc_bcast_accepted*/),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            CHECK(init_convolution(engine));
            CHECK(attr_.set_default_formats(dst_md(0)));
            init_name();
            init_scratchpad();

            return status::success;
        }

    protected:
        status_t set_default_params() {
            auto conv_1x1_pd_ = static_cast<conv_pd_t *>(conv_pd_.get());
            src_md_ = *conv_1x1_pd_->src_md();
            dst_md_ = *conv_1x1_pd_->dst_md();
            weights_md_ = *conv_1x1_pd_->weights_md();
            if (with_bias()) bias_md_ = *conv_1x1_pd_->weights_md(1);
            return status::success;
        }

        using conv_pd_t =
                typename jit_uni_x8s8s32x_1x1_convolution_fwd_t<isa>::pd_t;
        friend jit_uni_x8s8s32x_1x1_deconvolution_fwd_t;

        std::shared_ptr<primitive_desc_t> conv_pd_;

    private:
        std::string name_ = JIT_IMPL_NAME_HELPER("jit_deconvolution:", isa, "");

        void init_name() {
            name_.append("+");
            name_.append(conv_pd_->name());
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    jit_uni_x8s8s32x_1x1_deconvolution_fwd_t(const pd_t *apd)
        : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        pd()->conv_pd_->create_primitive(conv_p_, engine);
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        nested_scratchpad_t ns(
                ctx, memory_tracking::names::key_nested, conv_p_);
        // XXX: create a new ctx for convolution?
        auto &tmp_ctx = const_cast<exec_ctx_t &>(ctx);
        tmp_ctx.set_scratchpad_grantor(ns.grantor());
        return conv_p_->execute(tmp_ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_p_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif /* CPU_X64_JIT_UNI_X8S8S32X_1X1_DECONVOLUTION_HPP */
