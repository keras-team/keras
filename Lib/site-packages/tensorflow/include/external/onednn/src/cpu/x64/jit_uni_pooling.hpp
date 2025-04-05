/*******************************************************************************
* Copyright 2017-2024 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_POOLING_HPP
#define CPU_X64_JIT_UNI_POOLING_HPP

#include <assert.h>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/x64/jit_uni_pool_kernel.hpp"
#include "cpu/x64/jit_uni_reorder.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace jit_uni_pooling_utils {
struct trans_wrapper_t;
struct trans_context_t;
} // namespace jit_uni_pooling_utils

template <cpu_isa_t isa, impl::data_type_t d_type>
struct jit_uni_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", jpp_.isa, ""),
                jit_uni_pooling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;

            VDISPATCH_POOLING(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_POOLING(everyone_is(d_type, src_md()->data_type,
                                      dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(
                    attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, d_type),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
                    "does not support dilations");
            VDISPATCH_POOLING(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);

            const bool is_training
                    = desc_.prop_kind == prop_kind::forward_training;
            if (desc()->alg_kind == alg_kind::pooling_max && is_training)
                init_default_ws();

            auto scratchpad = scratchpad_registry().registrar();

            CHECK(jit_uni_pool_kernel<isa>::init_conf(
                    jpp_, scratchpad, attr_, this));

            return status::success;
        }

        jit_pool_conf_t jpp_;
    };

    explicit jit_uni_pooling_fwd_t(const pd_t *apd);
    jit_uni_pooling_fwd_t(jit_uni_pooling_fwd_t &&) = default;
    jit_uni_pooling_fwd_t &operator=(jit_uni_pooling_fwd_t &&) = default;
    ~jit_uni_pooling_fwd_t();

    using data_t = typename prec_traits<d_type>::type;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
        auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
        auto ws = CTX_OUT_MEM(char *, DNNL_ARG_WORKSPACE);

        if (pd()->ndims() == 5)
            execute_forward_3d(src, dst, ws, ctx);
        else
            execute_forward(src, dst, ws, ctx);

        return status::success;
    }

private:
    void execute_forward(const data_t *src, data_t *dst, char *indices,
            const exec_ctx_t &ctx) const;
    void execute_forward_3d(const data_t *src, data_t *dst, char *indices,
            const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t init_ncsp_trans_ctx();

    std::unique_ptr<jit_uni_pool_kernel<isa>> kernel_;
    std::unique_ptr<jit_uni_pooling_utils::trans_context_t> trans_ctx_;
    static constexpr data_type_t wsp_dt_ = data_type::f32;
};

template <cpu_isa_t isa, impl::data_type_t d_type>
struct jit_uni_pooling_bwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_bwd_pd_t {
        using cpu_pooling_bwd_pd_t::cpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", jpp_.isa, ""),
                jit_uni_pooling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;

            VDISPATCH_POOLING(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_POOLING(everyone_is(d_type, diff_src_md()->data_type,
                                      diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
                    "does not support dilations");

            if (desc()->alg_kind == alg_kind::pooling_max) {
                const auto ws_dt = hint_fwd_pd_->workspace_md()->data_type;
                init_default_ws(ws_dt);
                VDISPATCH_POOLING(
                        compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);
            }

            auto scratchpad = scratchpad_registry().registrar();

            CHECK(jit_uni_pool_kernel<isa>::init_conf(
                    jpp_, scratchpad, attr_, this));

            return status::success;
        }

        jit_pool_conf_t jpp_;
    };

    explicit jit_uni_pooling_bwd_t(const pd_t *apd);
    jit_uni_pooling_bwd_t(jit_uni_pooling_bwd_t &&) = default;
    jit_uni_pooling_bwd_t &operator=(jit_uni_pooling_bwd_t &&) = default;
    ~jit_uni_pooling_bwd_t();

    using data_t = typename prec_traits<d_type>::type;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
        auto ws = CTX_IN_MEM(const char *, DNNL_ARG_WORKSPACE);
        auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

        if (pd()->ndims() == 5)
            execute_backward_3d(diff_dst, ws, diff_src, ctx);
        else
            execute_backward(diff_dst, ws, diff_src, ctx);

        return status::success;
    }

private:
    void execute_backward(const data_t *diff_dst, const char *indices,
            data_t *diff_src, const exec_ctx_t &ctx) const;
    void execute_backward_3d(const data_t *diff_dst, const char *indices,
            data_t *diff_src, const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t init_ncsp_trans_ctx();

    std::unique_ptr<jit_uni_pool_kernel<isa>> kernel_;
    std::unique_ptr<jit_uni_pooling_utils::trans_context_t> trans_ctx_;
    static constexpr data_type_t wsp_dt_ = data_type::f32;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
