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

#ifndef CPU_X64_JIT_AVX512_COMMON_CONVOLUTION_HPP
#define CPU_X64_JIT_AVX512_COMMON_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_reducer.hpp"

#include "cpu/x64/jit_avx512_common_conv_kernel.hpp"
#include "cpu/x64/jit_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
        impl::data_type_t dst_type = src_type>
struct jit_avx512_common_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", avx512_core, ""),
                jit_avx512_common_convolution_fwd_t);

        status_t init(engine_t *engine) {
            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(expect_data_types(src_type, wei_type, dst_type,
                                   dst_type, data_type::undef),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, dst_type),
                    VERBOSE_UNSUPPORTED_ATTR);

            CHECK(jit_avx512_common_conv_fwd_kernel::init_conf(jcp_, *desc(),
                    src_md_, weights_md_, dst_md_, bias_md_, attr_,
                    dnnl_get_max_threads()));

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_conv_fwd_kernel::init_scratchpad(
                    scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;
    };

    jit_avx512_common_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_avx512_common_conv_fwd_kernel(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->ndims() == 3)
            execute_forward_1d(ctx);
        else if (pd()->ndims() == 4)
            execute_forward_2d(ctx);
        else if (pd()->ndims() == 5)
            execute_forward_3d(ctx);
        else
            assert(false);

        if (pd()->wants_zero_pad_dst()) ctx.zero_pad_output(DNNL_ARG_DST);
        return status::success;
    }

private:
    void prepare_padded_bias(const dst_data_t *&bias,
            const memory_tracking::grantor_t &scratchpad) const;
    void execute_forward_1d(const exec_ctx_t &ctx) const;
    void execute_forward_2d(const exec_ctx_t &ctx) const;
    void execute_forward_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx512_common_conv_fwd_kernel> kernel_;
};

template <impl::data_type_t diff_dst_type,
        impl::data_type_t wei_type = diff_dst_type,
        impl::data_type_t diff_src_type = diff_dst_type>
struct jit_avx512_common_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", avx512_core, ""),
                jit_avx512_common_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_data,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(
                    expect_data_types(diff_src_type, wei_type, data_type::undef,
                            diff_dst_type, data_type::undef),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            status_t status
                    = jit_avx512_common_conv_bwd_data_kernel_f32::init_conf(
                            jcp_, *desc(), diff_src_md_, weights_md_,
                            diff_dst_md_, dnnl_get_max_threads());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_conv_bwd_data_kernel_f32::init_scratchpad(
                    scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;
    };

    jit_avx512_common_convolution_bwd_data_t(const pd_t *apd)
        : primitive_t(apd) {}

    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_avx512_common_conv_bwd_data_kernel_f32(pd()->jcp_)));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->ndims() == 3)
            execute_backward_data_1d(ctx);
        else if (pd()->ndims() == 4)
            execute_backward_data_2d(ctx);
        else if (pd()->ndims() == 5)
            execute_backward_data_3d(ctx);
        else
            assert(false);
        return status::success;
    }

private:
    void execute_backward_data_1d(const exec_ctx_t &ctx) const;
    void execute_backward_data_2d(const exec_ctx_t &ctx) const;
    void execute_backward_data_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx512_common_conv_bwd_data_kernel_f32> kernel_;
};

template <impl::data_type_t src_type,
        impl::data_type_t diff_dst_type = src_type,
        impl::data_type_t diff_weights_type = src_type>
struct jit_avx512_common_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", avx512_core, ""),
                jit_avx512_common_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_weights,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(
                    expect_data_types(src_type, diff_weights_type,
                            diff_weights_type, diff_dst_type, data_type::undef),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            status_t status
                    = jit_avx512_common_conv_bwd_weights_kernel_f32::init_conf(
                            jcp_, *desc(), src_md_, diff_weights_md_,
                            diff_bias_md_, diff_dst_md_,
                            dnnl_get_max_threads());
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_conv_bwd_weights_kernel_f32::init_scratchpad(
                    scratchpad, jcp_);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);

            return status;
        }

        jit_conv_conf_t jcp_;
        typename cpu_reducer_t<diff_weights_type>::conf_t reducer_bia_conf_;

    private:
        void init_balancers() {
            const size_t max_buffer_size = jcp_.nthr * 3 * 5 * 5 * 16 * 16;
            if (with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(jcp_.nthr,
                        jcp_.oc_block, jcp_.ngroups * jcp_.nb_oc, jcp_.mb,
                        max_buffer_size, true));
            }
        }
    };

    jit_avx512_common_convolution_bwd_weights_t(const pd_t *apd)
        : primitive_t(apd) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<diff_weights_type>::type diff_weights_data_t;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    void prepare_scratchpad_data(const exec_ctx_t &ctx) const;
    struct thread_info_t;
    void compute_diff_weights_nxc(const thread_info_t *) const;
    void compute_diff_weights(const thread_info_t *) const;
    void compute_diff_weights_2d(const thread_info_t *) const;
    void compute_diff_weights_3d(const thread_info_t *) const;
    void reduce_diff_weights(const thread_info_t *) const;
    void reduce_diff_weights_3d(const thread_info_t *) const;
    void compute_diff_bias(const thread_info_t *) const;
    void reduce_diff_bias(const thread_info_t *) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    int nthr_, nthr_mb_, nthr_g_, nthr_oc_b_, nthr_ic_b_;

    std::unique_ptr<jit_avx512_common_conv_bwd_weights_kernel_f32> kernel_;
    std::unique_ptr<cpu_accumulator_1d_t<diff_weights_type>> acc_ker_;
    std::unique_ptr<cpu_reducer_t<diff_weights_type>> reducer_bias_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
