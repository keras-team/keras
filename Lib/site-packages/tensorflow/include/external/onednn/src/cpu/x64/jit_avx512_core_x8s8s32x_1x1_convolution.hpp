/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_X8S8S32X_1X1_CONVOLUTION_HPP
#define CPU_X64_JIT_AVX512_CORE_X8S8S32X_1X1_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/primitive_hashing.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/dw_convolution_utils.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/jit_avx512_core_x8s8s32x_1x1_conv_kernel.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_convolution.hpp"
#include "cpu/x64/jit_uni_1x1_conv_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using dw_conv_pd_type = cpu_convolution_fwd_pd_t;
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_()
            , jcp_dw_(nullptr) {}

        pd_t(const pd_t &other) : cpu_convolution_fwd_pd_t(other) {
            if (copy(other) != status::success) is_initialized_ = false;
        }

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_int8_1x1:",
                        ((jcp_.has_vnni) ? avx512_core_vnni : avx512_core), ""),
                jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);

            VDISPATCH_CONV(utils::one_of(src_md(0)->data_type, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(
                    weights_md(0)->data_type == s8, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(
                    IMPLICATION(with_bias(), weights_md(1)->data_type == f32),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_CONV(
                    utils::one_of(dst_md(0)->data_type, f32, s32, s8, u8, bf16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(
                    desc()->accum_data_type == s32, VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            VDISPATCH_CONV(
                    attr()->has_default_values(smask_t::scales_runtime
                                    | smask_t::zero_points_runtime
                                    | smask_t::post_ops | smask_t::sum_dt,
                            dst_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(attr()->scales_.has_default_values({DNNL_ARG_SRC,
                                   DNNL_ARG_WEIGHTS, DNNL_ARG_DST,
                                   DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS,
                                   DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_DST}),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_CONV(set_default_formats_common(
                                   dat_tag(), format_tag::any, dat_tag()),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_CONV(attr()->post_ops_.check_sum_consistency(
                                   dst_md(0)->data_type,
                                   /* is_int8 */ true),
                    VERBOSE_UNSUPPORTED_POSTOP);

            VDISPATCH_CONV(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_CONV(zero_points_ok(), VERBOSE_UNSUPPORTED_ZP_CFG);

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, dst_md(), weights_md());

            CHECK(jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_conf(jcp_,
                    *conv_d, src_d, weights_md_, dst_md_, bias_md_, *attr(),
                    dnnl_get_max_threads(), rtus_.reduce_src_));
            if (jcp_.with_dw_conv) CHECK(depthwise_po_init(engine));

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_scratchpad(
                    scratchpad, jcp_, *attr());

            rtus_prepare_space_info(this, scratchpad, jcp_.nthr);

            return status::success;
        }

        const memory_desc_t *dst_1x1_md(int index = 0) const {
            return cpu_convolution_fwd_pd_t::dst_md(index);
        }

        const memory_desc_t *dst_md(
                int index = 0, bool user_input = false) const override {
            return jcp_.with_dw_conv
                    ? dw_conv_pd_->dst_md(index, user_input)
                    : cpu_convolution_fwd_pd_t::dst_md(index, user_input);
        }

        const memory_desc_t *arg_md(
                int arg, bool user_input = false) const override {
            if (jcp_.with_dw_conv) {
                switch (arg) {
                    case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_SRC:
                        return cpu_convolution_fwd_pd_t::dst_md(0, user_input);
                    case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS:
                        return dw_conv_pd_->weights_md(0);
                    case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS:
                        return dw_conv_pd_->weights_md(1);
                    default: break;
                }
            }
            return convolution_fwd_pd_t::arg_md(arg, user_input);
        }

        arg_usage_t arg_usage(int arg) const override {
            if (arg == (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS))
                return arg_usage_t::input;

            if (arg == (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS)
                    && attr_post_op_dw_inputs() > 1)
                return arg_usage_t::input;

            if (arg == (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_ATTR_OUTPUT_SCALES)
                    && jcp_.with_dw_conv)
                return arg_usage_t::input;
            return convolution_fwd_pd_t::arg_usage(arg);
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;
        jit_conv_conf_t *jcp_dw_; // doesn't own a resource
        std::unique_ptr<cpu_convolution_fwd_pd_t> dw_conv_pd_;
        using dw_pd_t =
                typename jit_avx512_core_x8s8s32x_convolution_fwd_t::pd_t;

    protected:
        format_tag_t dat_tag() const {
            return utils::pick(src_md_.ndims - 3, format_tag::nwc,
                    format_tag::nhwc, format_tag::ndhwc);
        }

        bool zero_points_ok() const {
            // Only common zero points are supported -> mask should only be 0
            int mask_src = 0, mask_dst = 0;
            attr()->zero_points_.get(DNNL_ARG_SRC, &mask_src);
            attr()->zero_points_.get(DNNL_ARG_DST, &mask_dst);
            return attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)
                    && mask_src == 0 && mask_dst == 0;
        }

        status_t copy(const pd_t &other) {
            jcp_ = other.jcp_;
            rtus_ = other.rtus_;
            jcp_dw_ = nullptr;

            if (other.dw_conv_pd_) {
                dw_conv_pd_.reset(static_cast<cpu_convolution_fwd_pd_t *>(
                        other.dw_conv_pd_->clone()));
                if (!dw_conv_pd_) return status::out_of_memory;

                jcp_dw_ = &(static_cast<dw_pd_t *>(dw_conv_pd_.get())->jcp_);
            }
            return status::success;
        }

        status_t depthwise_po_init(engine_t *engine) {
            using namespace memory_tracking;
            auto &jcp_1x1 = jcp_;
            primitive_attr_t attr_1x1(*attr());
            if (!attr_1x1.is_initialized()) return status::out_of_memory;

            const auto &src_md = dst_md_;
            const memory_desc_wrapper src_d(src_md);
            const auto nthr = dnnl_get_max_threads();
            auto l2_cache = platform::get_per_core_cache_size(2) * nthr;

            // Note: A robust fusion implementation would be to check if both
            // 1x1 conv and dw conv that are considered here for fusion are
            // optimal independently. This would require creating a new
            // primitive_desc through primitive_iterator & check if they match.
            // Due to concern that these creations and/or checks could be heavy,
            // for 1x1: Check that no better ISA is available.
            // for dw: Always fuse with same ISA.
            // Caveat: May be a better dw conv exists.
            VDISPATCH_CONV_IC(!mayiuse(avx512_core_amx),
                    VERBOSE_1x1CONV_HEURISTIC_FAIL, "higher ISA is supported");

            VDISPATCH_CONV_IC(
                    attr_1x1.post_ops_.find(primitive_kind::sum) == -1,
                    VERBOSE_UNSUPPORTED_FEATURE, "unsupported sum post-op");

            // TODO: Below may be further tuned.
            VDISPATCH_CONV_IC(l2_cache * 2 < src_d.size(),
                    VERBOSE_1x1CONV_HEURISTIC_FAIL, "cache size check failed");

            // load_grp_count check can be redundant due to l2 check
            // above. Adding it explicitly as the current driver doesn't
            // work if this condition fails.
            VDISPATCH_CONV_IC(jcp_1x1.load_grp_count < 2,
                    VERBOSE_1x1CONV_HEURISTIC_FAIL, "load group count > 1");

            int dw_po_index
                    = attr_1x1.post_ops_.find(primitive_kind::convolution);

            convolution_desc_t cd_dw;
            primitive_attr_t attr_dw;
            CHECK(get_depthwise_conv_desc(
                    cd_dw, src_md, attr_1x1, attr_dw, dw_po_index));

            auto fusable_pd
                    = make_unique_pd<dw_pd_t>(&cd_dw, &attr_dw, nullptr);
            CHECK(fusable_pd->init(engine));
            jcp_dw_ = &(fusable_pd->jcp_);
            dw_conv_pd_ = std::move(fusable_pd);

            VDISPATCH_CONV_IC(
                    dnnl_memory_desc_equal(&src_md, dw_conv_pd_->src_md(0)),
                    VERBOSE_INCONSISTENT_MDS, "src_md", "dw_conv_pd_->src_md");
            VDISPATCH_CONV_IC(
                    jcp_1x1.oc_without_padding % jcp_1x1.oc_block == 0,
                    VERBOSE_1x1CONV_HEURISTIC_FAIL,
                    "output-channel is not an exact multiple of oc_block");
            VDISPATCH_CONV_IC(IMPLICATION(jcp_dw_->ow_block,
                                      jcp_dw_->ow_block == jcp_dw_->ow),
                    VERBOSE_1x1CONV_HEURISTIC_FAIL,
                    "ow_block does not equal output-width");

            assert(jcp_dw_);
            assert(dw_conv_pd_->dst_md(0)->format_kind != format_kind::any);
            assert(dw_conv_pd_->weights_md(0)->format_kind != format_kind::any);
            assert(IMPLICATION(
                    dw_conv_pd_->weights_md(1)->data_type != data_type::undef,
                    dw_conv_pd_->weights_md(1)->format_kind
                            != format_kind::any));

            jcp_dw_->is_fused_conv = true;
            // TODO: Support/experiment arbitary oc_work in dw conv.
            // Until then we keep ch_work perfectly divisible.
            while (jcp_1x1.nb_load % jcp_1x1.nb_load_blocking != 0)
                --jcp_1x1.nb_load_blocking;
            jcp_1x1.nb_load_blocking_max = jcp_1x1.nb_load_blocking;

            while (jcp_1x1.nb_load_blocking % jcp_dw_->nb_ch_blocking != 0)
                --jcp_dw_->nb_ch_blocking;

            jcp_dw_->dw_conv_buffer_oc
                    = jcp_1x1.nb_load_blocking * jcp_1x1.oc_block;
            jcp_1x1.bcast_loop_output_step = jcp_1x1.ur
                    * (jcp_1x1.nb_load_blocking * jcp_1x1.oc_block)
                    * jcp_1x1.typesize_out;

            registrar_t scratchpad(scratchpad_registry_);
            registrar_t dw_scratchpad(scratchpad, names::prefix_fusion);

            size_t dw_conv_buffer_size_ = (size_t)nthr * jcp_dw_->kh
                    * jcp_dw_->iw * jcp_dw_->dw_conv_buffer_oc;
            assert(dw_conv_buffer_size_);
            dw_scratchpad.book(memory_tracking::names::key_fusion_inout_buffer,
                    dw_conv_buffer_size_,
                    types::data_type_size(dw_conv_pd_->src_md()->data_type));

            dw_conv_kernel_t::init_scratchpad(
                    dw_scratchpad, *jcp_dw_, *(dw_conv_pd_->attr()));
            return status::success;
        }
    };
    template <cpu_isa_t isa, typename conv_t>
    friend status_t init_rtus_driver(conv_t *self);

    jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd) {}

    // Note: In case of fused depthwise convolution, the final output data type
    // after fusion may not be same as for dst.
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_avx512_core_x8s8s32x_1x1_conv_kernel(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_1x1_md(0))));
        CHECK(kernel_->create_kernel());

        if (pd()->jcp_.with_dw_conv) {
            CHECK(safe_ptr_assign(kernel_dw_,
                    new dw_conv_kernel_t(*(pd()->jcp_dw_),
                            *(pd()->dw_conv_pd_->attr()), *pd()->dst_md(0))));
            CHECK(kernel_dw_->create_kernel());
        }

        CHECK(init_rtus_driver<avx512_core>(this));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr, const char *src,
            const char *weights, const char *bias, const char *weights_dw,
            const char *bias_dw, char *dst, const float *oscales,
            const float *dst_scales, const float *dw_oscales,
            const float *dw_dst_scales, const int32_t *src_zero_point,
            const int32_t *dst_zero_point,
            const memory_tracking::grantor_t &scratchpad,
            const void *post_ops_binary_rhs_arg_vec,
            const void *post_ops_binary_rhs_arg_vec_dw) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_avx512_core_x8s8s32x_1x1_conv_kernel> kernel_;
    std::unique_ptr<rtus_driver_t<avx512_core>> rtus_driver_;
    using dw_conv_kernel_t = jit_avx512_core_x8s8s32x_fwd_kernel;
    std::unique_ptr<dw_conv_kernel_t> kernel_dw_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
