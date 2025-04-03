/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX2_1X1_CONVOLUTION_HPP
#define CPU_X64_JIT_AVX2_1X1_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/primitive_hashing.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/dw_convolution_utils.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/cpu_reducer.hpp"
#include "cpu/x64/jit_avx2_1x1_conv_kernel_f32.hpp"
#include "cpu/x64/jit_uni_1x1_conv_utils.hpp"
#include "cpu/x64/jit_uni_dw_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx2_1x1_convolution_fwd_t : public primitive_t {
    // TODO: (Roma) Code duplication duplication! Remove with templates
    //              (maybe...)!
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_()
            , jcp_dw_(nullptr) {}

        pd_t(const pd_t &other) : cpu_convolution_fwd_pd_t(other) {
            if (copy(other) != status::success) is_initialized_ = false;
        }

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_1x1:", jcp_.isa, ""),
                jit_avx2_1x1_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(expect_data_types(f32, f32, f32, f32, f32),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, f32),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, dst_md(), weights_md());

            CHECK(jit_avx2_1x1_conv_kernel_f32::init_conf(
                    jcp_, *conv_d, *src_d, *weights_md(), *dst_md(), *attr()));
            if (jcp_.with_dw_conv) CHECK(depthwise_po_init(engine));

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_1x1_conv_kernel_f32::init_scratchpad(scratchpad, jcp_);

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

            return convolution_fwd_pd_t::arg_usage(arg);
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;
        jit_conv_conf_t *jcp_dw_;
        std::unique_ptr<cpu_convolution_fwd_pd_t> dw_conv_pd_;

    protected:
        template <cpu_isa_t isa>
        using dw_pd_t = typename jit_uni_dw_convolution_fwd_t<isa,
                data_type::f32>::pd_t;

        bool set_default_formats() {
            using namespace format_tag;

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper dst_d(&dst_md_);

            const auto dat_tag_nxc = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            const auto dat_tag_nCx8c
                    = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            const auto curr_src_tag
                    = src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
            const auto curr_dst_tag
                    = dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
            const auto is_data_layout_nxc
                    = IMPLICATION(curr_src_tag != dat_tag_nxc,
                              src_d.format_kind() == format_kind::any)
                    && IMPLICATION(curr_dst_tag != dat_tag_nxc,
                            dst_d.format_kind() == format_kind::any)
                    && utils::one_of(dat_tag_nxc, curr_src_tag, curr_dst_tag);
            auto dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, gOIw8i8o, gOIhw8i8o, gOIdhw8i8o)
                    : utils::pick(ndims() - 3, OIw8i8o, OIhw8i8o, OIdhw8i8o);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        status_t copy(const pd_t &other) {
            jcp_ = other.jcp_;
            rtus_ = other.rtus_;
            jcp_dw_ = nullptr;
            if (other.dw_conv_pd_) {
                dw_conv_pd_.reset(static_cast<cpu_convolution_fwd_pd_t *>(
                        other.dw_conv_pd_->clone()));
                if (!dw_conv_pd_) return status::out_of_memory;
                if (jcp_.isa == avx2) {
                    jcp_dw_ = &(static_cast<dw_pd_t<avx2> *>(dw_conv_pd_.get())
                                        ->jcp_);
                } else { // sse41
                    jcp_dw_ = &(static_cast<dw_pd_t<sse41> *>(dw_conv_pd_.get())
                                        ->jcp_);
                }
            }

            return status::success;
        }

        status_t depthwise_po_init(engine_t *engine) {

            using namespace memory_tracking;
            auto &jcp_1x1 = jcp_;
            primitive_attr_t attr_1x1(*attr());
            if (!attr_1x1.is_initialized()) return status::out_of_memory;
            jit_conv_conf_t *jcp_dw = nullptr;

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
            VDISPATCH_CONV_IC(!mayiuse(avx512_core),
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

            if (jcp_1x1.isa == avx2) {
                auto fusable_pd = make_unique_pd<dw_pd_t<avx2>>(
                        &cd_dw, &attr_dw, nullptr);
                CHECK(fusable_pd->init(engine));
                jcp_dw = &(fusable_pd->jcp_);
                dw_conv_pd_ = std::move(fusable_pd);
            } else {
                // Special case for this primitive, as we dont have dw<avx>.
                // In this case fuse with sse41 depthwise conv
                // NOTE: Currently dw f32 kernel is similar for all ISA and can
                // be fused regardless of ISA if inter-connecting md_ matches.
                auto fusable_pd = make_unique_pd<dw_pd_t<sse41>>(
                        &cd_dw, &attr_dw, nullptr);
                CHECK(fusable_pd->init(engine));
                jcp_dw = &(fusable_pd->jcp_);
                dw_conv_pd_ = std::move(fusable_pd);
            }

            VDISPATCH_CONV_IC(
                    dnnl_memory_desc_equal(&src_md, dw_conv_pd_->src_md(0)),
                    VERBOSE_INCONSISTENT_MDS, "src_md", "dw_conv_pd_->src_md");
            VDISPATCH_CONV_IC(
                    jcp_1x1.oc_without_padding % jcp_1x1.oc_block == 0,
                    VERBOSE_1x1CONV_HEURISTIC_FAIL,
                    "output-channel is not an exact multiple of oc_block");
            VDISPATCH_CONV_IC(IMPLICATION(jcp_dw->ow_block,
                                      jcp_dw->ow_block == jcp_dw->ow),
                    VERBOSE_1x1CONV_HEURISTIC_FAIL,
                    "ow_block does not equal output-width");

            assert(dw_conv_pd_->dst_md(0)->format_kind != format_kind::any);
            assert(dw_conv_pd_->weights_md(0)->format_kind != format_kind::any);
            assert(IMPLICATION(
                    dw_conv_pd_->weights_md(1)->data_type != data_type::undef,
                    dw_conv_pd_->weights_md(1)->format_kind
                            != format_kind::any));

            jcp_dw->is_fused_conv = true;
            // TODO: Support/experiment arbitary oc_work in dw conv.
            // Until then we keep oc_work perfectly divisible.
            while (jcp_1x1.nb_load % jcp_1x1.nb_load_blocking != 0)
                --jcp_1x1.nb_load_blocking;
            jcp_1x1.nb_load_blocking_max = jcp_1x1.nb_load_blocking;

            while (jcp_1x1.nb_load_blocking % jcp_dw->nb_ch_blocking != 0)
                --jcp_dw->nb_ch_blocking;

            jcp_dw->dw_conv_buffer_oc
                    = jcp_1x1.nb_load_blocking * jcp_1x1.oc_block;

            const auto dat_tag_nxc = utils::pick(ndims() - 3, format_tag::nwc,
                    format_tag::nhwc, format_tag::ndhwc);
            const bool is_data_nxc = utils::everyone_is(
                    dat_tag_nxc, jcp_1x1.src_tag, jcp_1x1.dst_tag);
            if (!is_data_nxc)
                jcp_1x1.bcast_loop_output_step = jcp_1x1.ur * jcp_1x1.load_block
                        * jcp_1x1.typesize_out;

            registrar_t scratchpad(scratchpad_registry_);
            registrar_t dw_scratchpad(scratchpad, names::prefix_fusion);

            size_t dw_conv_buffer_size_ = (size_t)nthr * jcp_dw->kh * jcp_dw->iw
                    * jcp_dw->dw_conv_buffer_oc;
            assert(dw_conv_buffer_size_);
            dw_scratchpad.book(memory_tracking::names::key_fusion_inout_buffer,
                    dw_conv_buffer_size_,
                    types::data_type_size(dw_conv_pd_->src_md()->data_type));

            if (jcp_1x1.isa == avx2)
                dw_conv_kernel_t<avx2>::init_scratchpad(dw_scratchpad, *jcp_dw);
            else
                dw_conv_kernel_t<sse41>::init_scratchpad(
                        dw_scratchpad, *jcp_dw);

            return status::success;
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend status_t init_rtus_driver(conv_t *self);

    jit_avx2_1x1_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_avx2_1x1_conv_kernel_f32(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_1x1_md(0))));
        CHECK(kernel_->create_kernel());
        CHECK(init_rtus_driver<avx2>(this));
        if (pd()->jcp_.with_dw_conv) {
            auto &isa = pd()->jcp_.isa;

            if (isa == avx2) {
                CHECK(safe_ptr_assign(kernel_dw_avx2,
                        new dw_conv_kernel_t<avx2>(
                                *(pd()->jcp_dw_), *pd()->dst_md(0))));
                CHECK(kernel_dw_avx2->create_kernel());
            } else {
                CHECK(safe_ptr_assign(kernel_dw_sse41,
                        new dw_conv_kernel_t<sse41>(
                                *(pd()->jcp_dw_), *pd()->dst_md(0))));
                CHECK(kernel_dw_sse41->create_kernel());
            }
        }

        return status::success;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr, const data_t *src,
            const data_t *weights, const data_t *bias, const data_t *weights_dw,
            const data_t *bias_dw, data_t *dst,
            const memory_tracking::grantor_t &scratchpad,
            const void *post_ops_binary_rhs_arg_vec,
            const void *post_ops_binary_rhs_arg_vec_dw) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx2_1x1_conv_kernel_f32> kernel_;
    std::unique_ptr<rtus_driver_t<avx2>> rtus_driver_;

    template <cpu_isa_t isa>
    using dw_conv_kernel_t = jit_uni_dw_conv_fwd_kernel<isa, data_type::f32>;

    std::unique_ptr<dw_conv_kernel_t<avx2>> kernel_dw_avx2;
    std::unique_ptr<dw_conv_kernel_t<sse41>> kernel_dw_sse41;
};

struct jit_avx2_1x1_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_1x1:", avx2, ""),
                jit_avx2_1x1_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_data,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(
                    expect_data_types(f32, f32, data_type::undef, f32, f32),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *diff_src_d = diff_src_md();
            rtus_prepare(this, conv_d, diff_src_d, diff_dst_md(), weights_md());

            status_t status = jit_avx2_1x1_conv_kernel_f32::init_conf(jcp_,
                    *conv_d, *diff_src_d, *weights_md(), *diff_dst_md(),
                    *attr());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_1x1_conv_kernel_f32::init_scratchpad(scratchpad, jcp_);

            rtus_prepare_space_info(this, scratchpad, jcp_.nthr);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            const memory_desc_wrapper diff_src_d(&diff_src_md_);
            const memory_desc_wrapper diff_dst_d(&diff_dst_md_);

            const auto dat_tag_nxc = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            const auto dat_tag_nCx8c
                    = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            const auto curr_src_tag
                    = diff_src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
            const auto curr_dst_tag
                    = diff_dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
            const auto is_data_layout_nxc
                    = IMPLICATION(curr_src_tag != dat_tag_nxc,
                              diff_src_d.format_kind() == format_kind::any)
                    && IMPLICATION(curr_dst_tag != dat_tag_nxc,
                            diff_dst_d.format_kind() == format_kind::any)
                    && utils::one_of(dat_tag_nxc, curr_src_tag, curr_dst_tag);
            auto dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, gOIw8o8i, gOIhw8o8i, gOIdhw8o8i)
                    : utils::pick(ndims() - 3, OIw8o8i, OIhw8o8i, OIdhw8o8i);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend status_t init_rtus_driver(conv_t *self);

    jit_avx2_1x1_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_avx2_1x1_conv_kernel_f32(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));
        CHECK(kernel_->create_kernel());
        CHECK(init_rtus_driver<avx2>(this));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx2_1x1_conv_kernel_f32> kernel_;
    std::unique_ptr<rtus_driver_t<avx2>> rtus_driver_;
};

struct jit_avx2_1x1_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_1x1:", avx2, ""),
                jit_avx2_1x1_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_weights,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(expect_data_types(f32, f32, f32, f32, f32),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, diff_dst_md(), diff_weights_md());

            status_t status = jit_avx2_1x1_conv_kernel_f32::init_conf(jcp_,
                    *conv_d, *src_d, *diff_weights_md(), *diff_dst_md(),
                    *attr());
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_1x1_conv_kernel_f32::init_scratchpad(scratchpad, jcp_);

            rtus_prepare_space_info(this, scratchpad, jcp_.nthr);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);

            auto reducer_wei_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_wei);
            reducer_wei_conf_.init_scratchpad(reducer_wei_scratchpad);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        cpu_reducer_t<data_type::f32>::conf_t reducer_bia_conf_;
        cpu_reducer_2d_t<data_type::f32>::conf_t reducer_wei_conf_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper diff_dst_d(&diff_dst_md_);

            const auto dat_tag_nxc = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            const auto dat_tag_nCx8c
                    = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            const auto curr_src_tag
                    = src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
            const auto curr_dst_tag
                    = diff_dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
            const auto is_data_layout_nxc
                    = IMPLICATION(curr_src_tag != dat_tag_nxc,
                              src_d.format_kind() == format_kind::any)
                    && IMPLICATION(curr_dst_tag != dat_tag_nxc,
                            diff_dst_d.format_kind() == format_kind::any)
                    && utils::one_of(dat_tag_nxc, curr_src_tag, curr_dst_tag);

            auto dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, gOIw8i8o, gOIhw8i8o, gOIdhw8i8o)
                    : utils::pick(ndims() - 3, OIw8i8o, OIhw8i8o, OIdhw8i8o);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

    private:
        void init_balancers() {
            const int ic_block = jcp_.bcast_block;
            const int nb_ic = jcp_.nb_bcast;
            const int nb_ic_blocking = jcp_.nb_bcast_blocking;
            const int bcast_work = utils::div_up(nb_ic, nb_ic_blocking);

            const int oc_block = jcp_.load_block;
            const int nb_oc = jcp_.nb_load;
            const int nb_oc_blocking = jcp_.nb_load_blocking;
            const int load_work = utils::div_up(nb_oc, nb_oc_blocking);

            const int job_size
                    = nb_oc_blocking * nb_ic_blocking * ic_block * oc_block;
            const int njobs_x = bcast_work;
            const int njobs_y = jcp_.ngroups * load_work;

            const int max_threads = dnnl_get_max_threads();
            const size_t max_buffer_size = (size_t)max_threads * job_size * 8;

            if (with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(max_threads, oc_block,
                        jcp_.ngroups * nb_oc, jcp_.mb, max_buffer_size, true));
            }

            reducer_wei_conf_.init(
                    reduce_balancer_t(max_threads, job_size, njobs_y * njobs_x,
                            jcp_.mb * jcp_.nb_reduce, max_buffer_size, true),
                    job_size / nb_oc_blocking, nb_oc_blocking, ic_block,
                    nb_ic * ic_block * oc_block, nb_oc);
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend status_t init_rtus_driver(conv_t *self);

    jit_avx2_1x1_convolution_bwd_weights_t(const pd_t *apd)
        : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx2_1x1_conv_kernel_f32> kernel_;
    std::unique_ptr<cpu_reducer_2d_t<data_type::f32>> reducer_weights_;
    std::unique_ptr<cpu_reducer_t<data_type::f32>> reducer_bias_;
    std::unique_ptr<rtus_driver_t<avx2>> rtus_driver_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
