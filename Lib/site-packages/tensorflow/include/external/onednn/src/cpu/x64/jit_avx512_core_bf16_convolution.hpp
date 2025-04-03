/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_BF16_CONVOLUTION_HPP
#define CPU_X64_JIT_AVX512_CORE_BF16_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_reducer.hpp"

#include "cpu/x64/jit_avx512_core_bf16_conv_kernel.hpp"
#include "cpu/x64/jit_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_bf16_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_bf16:", jcp_.isa, ""),
                jit_avx512_core_bf16_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            // disabling verbose dispatch messages for unsupported isa for better readability
            if (!mayiuse(avx512_core)) return status::unimplemented;

            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(
                    (expect_data_types(bf16, bf16, data_type::undef, bf16,
                             data_type::undef)
                            || expect_data_types(bf16, bf16, data_type::undef,
                                    f32, data_type::undef)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(attr()->has_default_values(
                                   primitive_attr_t::skip_mask_t::post_ops,
                                   dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_CONV(
                    IMPLICATION(with_bias(),
                            utils::one_of(weights_md(1)->data_type, f32, bf16)),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);

            CHECK(jit_avx512_core_bf16_fwd_kernel::init_conf(jcp_, *desc(),
                    src_md_, weights_md_, dst_md_, bias_md_, attr_,
                    dnnl_get_max_threads()));

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_bf16_fwd_kernel::init_scratchpad(scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;
    };

    jit_avx512_core_bf16_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd) {}

    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_avx512_core_bf16_fwd_kernel(
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
            return status::unimplemented;

        if (pd()->wants_zero_pad_dst()) ctx.zero_pad_output(DNNL_ARG_DST);
        return status::success;
    }

private:
    void prepare_padded_bias(const char *&bias,
            const memory_tracking::grantor_t &scratchpad) const;
    void execute_forward_1d(const exec_ctx_t &ctx) const;
    void execute_forward_2d(const exec_ctx_t &ctx) const;
    void execute_forward_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx512_core_bf16_fwd_kernel> kernel_;
};

struct jit_avx512_core_bf16_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_bf16:", jcp_.isa, ""),
                jit_avx512_core_bf16_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            // disabling verbose dispatch messages for unsupported isa for better readability
            if (!mayiuse(avx512_core)) return status::unimplemented;

            VDISPATCH_CONV(is_bwd_d(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(
                    (expect_data_types(f32, bf16, data_type::undef, bf16,
                             data_type::undef)
                            || expect_data_types(bf16, bf16, data_type::undef,
                                    bf16, data_type::undef)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            status_t status = jit_avx512_core_bf16_bwd_data_kernel::init_conf(
                    jcp_, *desc(), diff_src_md_, weights_md_, diff_dst_md_,
                    dnnl_get_max_threads());
            return status;
        }

        jit_conv_conf_t jcp_;
    };

    jit_avx512_core_bf16_convolution_bwd_data_t(const pd_t *apd)
        : primitive_t(apd) {}

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(
                kernel_, new jit_avx512_core_bf16_bwd_data_kernel(pd()->jcp_)));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->ndims() < 5)
            execute_backward_data(ctx);
        else if (pd()->ndims() == 5)
            execute_backward_data_3d(ctx);
        else
            assert(!"invalid dimension");

        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    void execute_backward_data_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_avx512_core_bf16_bwd_data_kernel> kernel_;
};

struct jit_avx512_core_bf16_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_bf16:", jcp_.isa, ""),
                jit_avx512_core_bf16_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            // disabling verbose dispatch messages for unsupported isa for better readability
            if (!mayiuse(avx512_core)) return status::unimplemented;

            VDISPATCH_CONV(is_bwd_w(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(
                    (expect_data_types(bf16, bf16, data_type::undef, bf16,
                             data_type::undef)
                            || expect_data_types(bf16, f32, data_type::undef,
                                    bf16, data_type::undef)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    IMPLICATION(with_bias(),
                            utils::one_of(diff_bias_md_.data_type, f32, bf16)),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            status_t status = jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
                    init_conf(jcp_, *desc(), src_md_, diff_weights_md_,
                            diff_bias_md_, diff_dst_md_,
                            dnnl_get_max_threads());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_scratchpad(
                    scratchpad, jcp_);

            return status;
        }

        jit_conv_conf_t jcp_;
    };

    jit_avx512_core_bf16_convolution_bwd_weights_t(const pd_t *apd)
        : primitive_t(apd) {}

    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    void prepare_scratchpad_data(const exec_ctx_t &ctx) const;
    struct thread_info_t;
    void compute_diff_weights_2d(const thread_info_t *) const;
    void compute_diff_weights_3d(const thread_info_t *) const;
    void compute_diff_weights(const thread_info_t *) const;
    void reduce_and_convert_diff_weights_and_bias(const thread_info_t *) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    size_t tr_src_buf_number(const thread_info_t *ti, int g, int ic) const;
    size_t tr_diff_dst_buf_number(const thread_info_t *ti, int g, int oc) const;
    void trans_src(
            src_data_t *tr_src1, const src_data_t *src1, int my_work) const;
    void trans_dst(diff_dst_data_t *tr_diff_dst1,
            const diff_dst_data_t *diff_dst1, int my_work) const;
    void trans_src_nxc(src_data_t *tr_src, const src_data_t *src_base,
            int spatial_start, dim_t spatial_start_offset, int icb_start,
            dim_t chb_stride, int my_work) const;
    void trans_dst_nxc(diff_dst_data_t *tr_diff_dst,
            const diff_dst_data_t *diff_dst_base, int spatial_start,
            dim_t spatial_start_offset, int ocb_start, dim_t chb_stride,
            int my_work) const;

    int nthr_ = 0, nthr_mb_ = 0, nthr_g_ = 0, nthr_oc_b_ = 0, nthr_ic_b_ = 0;

    std::unique_ptr<jit_avx512_core_bf16_conv_bwd_weights_kernel_f32> kernel_;

    std::unique_ptr<cpu_accumulator_1d_t<data_type::f32>> acc_ker_;

    std::unique_ptr<jit_trans_src_t> trans_kernel_;
    std::unique_ptr<jit_trans_dst_t> trans_dst_kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
