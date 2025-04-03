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

#ifndef CPU_GEMM_CONVOLUTION_HPP
#define CPU_GEMM_CONVOLUTION_HPP

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm_convolution_utils.hpp"
#include "cpu/primitive_attr_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct gemm_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(
                GEMM_IMPL_STR, gemm_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;

            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(expect_data_types(f32, f32, f32, f32, f32),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, f32),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);

            auto scratchpad = scratchpad_registry().registrar();

            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, attr_,
                    dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;

    protected:
        bool post_ops_ok() const {
            auto const &po = attr()->post_ops_;
            auto is_sum_ok = [&](int idx) {
                return IMPLICATION(po.entry_[idx].kind == primitive_kind::sum,
                        idx == 0 && po.entry_[idx].is_sum());
            };
            auto is_binary
                    = [&](int idx) { return po.entry_[idx].is_binary(); };
            auto is_prelu = [&](int idx) { return po.entry_[idx].is_prelu(); };
            auto is_binary_or_prelu_supported = [&](int idx) {
                bool ok = dnnl::impl::get_rhs_arg_broadcasting_strategy(
                                  binary_injector_utils::get_src1_desc(
                                          po.entry_[idx], dst_md_),
                                  dst_md_,
                                  {broadcasting_strategy_t::scalar,
                                          broadcasting_strategy_t::per_oc})
                        != broadcasting_strategy_t::unsupported;
                return ok;
            };

            if (!ref_post_ops_t::primitive_kind_ok(attr()->post_ops_))
                return false;

            for (int idx = 0; idx < po.len(); idx++) {
                bool ok = is_sum_ok(idx)
                        && IMPLICATION(is_binary(idx) || is_prelu(idx),
                                is_binary_or_prelu_supported(idx));
                if (!ok) return false;
            }

            return true;
        }
    };

    gemm_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd), post_ops_(nullptr) {}

    status_t init(engine_t *engine) override {
        const data_t one = 1.0, zero = 0.0;
        const auto &jcp = pd()->jcp_;
        beta_ = jcp.with_sum ? one : zero;

        if (jcp.with_eltwise || jcp.with_binary) {
            CHECK(safe_ptr_assign(post_ops_, new ref_post_ops_t(jcp.post_ops)));
            CHECK(post_ops_->init(pd()->dst_md()));
        }
        return status::success;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        bool is_nspc = pd()->jcp_.is_nspc;
        return is_nspc ? execute_forward_nspc(ctx) : execute_forward_ncsp(ctx);
    }

private:
    status_t execute_forward_ncsp(const exec_ctx_t &ctx) const;
    status_t execute_forward_nspc(const exec_ctx_t &ctx) const;
    status_t execute_forward_thr_nspc(const exec_ctx_t &ctx, const int ithr,
            const int nthr, const data_t *src_base, const data_t *wei_base,
            const data_t *bia_base, data_t *dst_base,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    data_t beta_;

    std::unique_ptr<ref_post_ops_t> post_ops_;
};

struct gemm_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_data_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace dnnl::impl::data_type;

            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_data,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(expect_data_types(f32, f32, undef, f32, f32),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            auto scratchpad = scratchpad_registry().registrar();

            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), diff_src_md_, weights_md_, diff_dst_md_, bias_md_,
                    attr_, dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;
    };

    gemm_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        bool is_nspc = pd()->jcp_.is_nspc;
        return is_nspc ? execute_backward_data_nspc(ctx)
                       : execute_backward_data_ncsp(ctx);
    }

private:
    status_t execute_backward_data_nspc(const exec_ctx_t &ctx) const;
    status_t execute_backward_data_ncsp(const exec_ctx_t &ctx) const;
    status_t execute_backward_data_thr_nspc(const int ithr, const int nthr,
            const data_t *diff_dst_base, const data_t *wei_base,
            const data_t *bia_base, data_t *diff_src_base,
            const memory_tracking::grantor_t &scratchpad) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct gemm_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_weights_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace dnnl::impl::data_type;

            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_weights,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(expect_data_types(f32, f32, f32, f32, f32),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md_, diff_weights_md_, diff_dst_md_,
                    diff_bias_md_, attr_, dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;
    };

    gemm_convolution_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        const bool is_nspc = pd()->jcp_.is_nspc;
        return is_nspc ? execute_backward_weights_nspc(ctx)
                       : execute_backward_weights_ncsp(ctx);
    }

private:
    status_t execute_backward_weights_ncsp(const exec_ctx_t &ctx) const;
    status_t execute_backward_weights_nspc(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
