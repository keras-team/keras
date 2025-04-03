/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX2_CONVOLUTION_HPP
#define CPU_X64_JIT_AVX2_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/x64/cpu_reducer.hpp"

#include "cpu/x64/jit_avx2_conv_kernel_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx2_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", jcp_.isa, ""),
                jit_avx2_convolution_fwd_t);

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

            CHECK(jit_avx2_conv_fwd_kernel_f32::init_conf(
                    jcp_, *desc(), src_md(), weights_md(), dst_md(), *attr()));

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_conv_fwd_kernel_f32::init_scratchpad(scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper dst_d(&dst_md_);

            const auto dat_tag_nxc = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            const auto dat_tag_ncx = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            const auto dat_tag_nCx8c
                    = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            const auto curr_src_tag = src_d.matches_one_of_tag(
                    dat_tag_nxc, dat_tag_ncx, dat_tag_nCx8c);
            const auto curr_dst_tag = dst_d.matches_one_of_tag(
                    dat_tag_nxc, dat_tag_ncx, dat_tag_nCx8c);
            const auto is_data_layout_nxc
                    = IMPLICATION(curr_src_tag != dat_tag_nxc,
                              src_d.format_kind() == format_kind::any)
                    && IMPLICATION(curr_dst_tag != dat_tag_nxc,
                            dst_d.format_kind() == format_kind::any)
                    && utils::one_of(dat_tag_nxc, curr_src_tag, curr_dst_tag);

            const bool flat = IC() < 8;
            auto src_tag = is_data_layout_nxc ? dat_tag_nxc
                    : flat                    ? dat_tag_ncx
                                              : dat_tag_nCx8c;
            auto dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
            auto wei_tag = with_groups()
                    ? utils::pick(2 * ndims() - 6 + flat, gOIw8i8o, gOwi8o,
                            gOIhw8i8o, gOhwi8o, gOIdhw8i8o, gOdhwi8o)
                    : utils::pick(2 * ndims() - 6 + flat, OIw8i8o, Owi8o,
                            OIhw8i8o, Ohwi8o, OIdhw8i8o, Odhwi8o);

            return set_default_formats_common(src_tag, wei_tag, dst_tag);
        }
    };

    jit_avx2_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_avx2_conv_fwd_kernel_f32(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx2_conv_fwd_kernel_f32> kernel_;
};

struct jit_avx2_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", avx2, ""),
                jit_avx2_convolution_bwd_data_t);

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

            status_t status = jit_avx2_conv_bwd_data_kernel_f32::init_conf(jcp_,
                    *desc(), *diff_src_md(), *weights_md(), *diff_dst_md());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_conv_bwd_data_kernel_f32::init_scratchpad(
                    scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

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

    jit_avx2_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(
                kernel_, new jit_avx2_conv_bwd_data_kernel_f32(pd()->jcp_)));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx2_conv_bwd_data_kernel_f32> kernel_;
};

struct jit_avx2_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", avx2, ""),
                jit_avx2_convolution_bwd_weights_t);

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

            status_t status = jit_avx2_conv_bwd_weights_kernel_f32::init_conf(
                    jcp_, *desc(), *src_md(), *diff_weights_md(),
                    *diff_dst_md());
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_conv_bwd_weights_kernel_f32::init_scratchpad(
                    scratchpad, jcp_);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);

            auto reducer_wei_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_wei);
            reducer_wei_conf_.init_scratchpad(reducer_wei_scratchpad);

            return status::success;
        }

        jit_conv_conf_t jcp_;
        cpu_reducer_t<data_type::f32>::conf_t reducer_bia_conf_;
        cpu_reducer_t<data_type::f32>::conf_t reducer_wei_conf_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            const bool flat = IC() == 3;

            const memory_desc_wrapper src_d(&src_md_);
            const memory_desc_wrapper diff_dst_d(&diff_dst_md_);

            const auto dat_tag_nxc = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            const auto dat_tag_ncx = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            const auto dat_tag_nCx8c
                    = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            const auto curr_src_tag = src_d.matches_one_of_tag(
                    dat_tag_nxc, dat_tag_ncx, dat_tag_nCx8c);
            const auto curr_dst_tag = diff_dst_d.matches_one_of_tag(
                    dat_tag_nxc, dat_tag_ncx, dat_tag_nCx8c);
            const auto is_data_layout_nxc
                    = IMPLICATION(curr_src_tag != dat_tag_nxc,
                              src_d.format_kind() == format_kind::any)
                    && IMPLICATION(curr_dst_tag != dat_tag_nxc,
                            diff_dst_d.format_kind() == format_kind::any)
                    && utils::one_of(dat_tag_nxc, curr_src_tag, curr_dst_tag);

            auto src_tag = is_data_layout_nxc ? dat_tag_nxc
                    : flat                    ? dat_tag_ncx
                                              : dat_tag_nCx8c;
            auto dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
            auto wei_tag = with_groups()
                    ? utils::pick(2 * ndims() - 6 + flat, gOIw8i8o, gOwi8o,
                            gOIhw8i8o, gOhwi8o, gOIdhw8i8o, gOdhwi8o)
                    : utils::pick(2 * ndims() - 6 + flat, OIw8i8o, Owi8o,
                            OIhw8i8o, Ohwi8o, OIdhw8i8o, Odhwi8o);

            return set_default_formats_common(src_tag, wei_tag, dst_tag);
        }

    private:
        void init_balancers() {
            const int max_threads = dnnl_get_max_threads();
            const size_t max_buffer_size = 1 << 21; /* just a heuristic */

            if (with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(max_threads,
                        jcp_.oc_block, jcp_.ngroups * jcp_.nb_oc, jcp_.mb,
                        max_buffer_size, true));
            }

            reducer_wei_conf_.init(reduce_balancer_t(max_threads,
                    jcp_.kd * jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block,
                    jcp_.ngroups * jcp_.nb_ic * jcp_.nb_oc, jcp_.mb * jcp_.od,
                    max_buffer_size, true));
        }
    };

    jit_avx2_convolution_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(
                kernel_, new jit_avx2_conv_bwd_weights_kernel_f32(pd()->jcp_)));
        CHECK(safe_ptr_assign(reducer_bias_,
                new cpu_reducer_t<data_type::f32>(pd()->reducer_bia_conf_)));
        CHECK(safe_ptr_assign(reducer_weights_,
                new cpu_reducer_t<data_type::f32>(pd()->reducer_wei_conf_)));
        CHECK(kernel_->create_kernel());
        CHECK(reducer_weights_->create_kernel());
        CHECK(reducer_bias_->create_kernel());
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx2_conv_bwd_weights_kernel_f32> kernel_;
    std::unique_ptr<cpu_reducer_t<data_type::f32>> reducer_weights_,
            reducer_bias_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
