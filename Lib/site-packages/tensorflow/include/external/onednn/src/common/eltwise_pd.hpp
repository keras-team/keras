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

#ifndef COMMON_ELTWISE_PD_HPP
#define COMMON_ELTWISE_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"

#define VDISPATCH_ELTWISE(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, eltwise, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_ELTWISE_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, eltwise, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct eltwise_fwd_pd_t;

status_t eltwise_desc_init(eltwise_desc_t *eltwise_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, float alpha, float beta);

struct eltwise_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::eltwise;

    const eltwise_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::alg_kind:
                *(alg_kind_t *)result = desc()->alg_kind;
                break;
            case query::alpha_f32: *(float *)result = desc()->alpha; break;
            case query::beta_f32: *(float *)result = desc()->beta; break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common eltwise aux functions */

    dim_t MB() const { return data_md()->dims[0]; }
    dim_t C() const { return ndims() >= 2 ? data_md()->dims[1] : 1; }
    dim_t D() const { return ndims() >= 5 ? data_md()->dims[ndims() - 3] : 1; }
    dim_t H() const { return ndims() >= 4 ? data_md()->dims[ndims() - 2] : 1; }
    dim_t W() const { return ndims() >= 3 ? data_md()->dims[ndims() - 1] : 1; }

    int ndims() const { return data_md()->ndims; }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(data_md()).has_zero_dim();
    }

    bool use_dst() const {
        using namespace alg_kind;
        return !is_fwd()
                && utils::one_of(desc_.alg_kind, eltwise_relu_use_dst_for_bwd,
                        eltwise_tanh_use_dst_for_bwd,
                        eltwise_elu_use_dst_for_bwd,
                        eltwise_sqrt_use_dst_for_bwd,
                        eltwise_logistic_use_dst_for_bwd,
                        eltwise_exp_use_dst_for_bwd,
                        eltwise_clip_v2_use_dst_for_bwd);
    }

protected:
    eltwise_desc_t desc_;
    const eltwise_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t src_md_;
    memory_desc_t dst_md_;

    eltwise_pd_t(const eltwise_desc_t *adesc, const primitive_attr_t *attr,
            const eltwise_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , dst_md_(desc_.dst_desc) {}

private:
    const memory_desc_t *data_md(int index = 0) const {
        return use_dst() ? dst_md(index) : src_md(index);
    }
};

struct eltwise_fwd_pd_t : public eltwise_pd_t {
    typedef eltwise_fwd_pd_t base_class;
    typedef eltwise_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return eltwise_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 1 + n_binary_po_inputs(); }
    int n_outputs() const override { return 1; }

    static bool eltwise_preserves_zero(
            alg_kind_t alg, float alpha, float beta) {
        using namespace alg_kind;
        using namespace utils;
        return one_of(alg, eltwise_relu, eltwise_tanh, eltwise_elu,
                       eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_swish,
                       eltwise_gelu_tanh, eltwise_gelu_erf, eltwise_round,
                       eltwise_hardswish)
                || one_of(alg, eltwise_relu_use_dst_for_bwd,
                        eltwise_tanh_use_dst_for_bwd,
                        eltwise_elu_use_dst_for_bwd,
                        eltwise_sqrt_use_dst_for_bwd)
                || (one_of(alg, eltwise_clip, eltwise_clip_v2) && alpha <= 0
                        && beta >= 0)
                || (alg == eltwise_linear && beta == 0)
                || (alg == eltwise_pow && beta > 0);
    }

    static bool eltwise_preserves_zero(
            const post_ops_t::entry_t::eltwise_t &eltwise) {
        return eltwise_preserves_zero(eltwise.alg, eltwise.alpha, eltwise.beta);
    }

    bool is_zero_preserved() const {
        return eltwise_preserves_zero(desc_.alg_kind, desc_.alpha, desc_.beta);
    }

protected:
    eltwise_fwd_pd_t(const eltwise_desc_t *adesc, const primitive_attr_t *attr,
            const eltwise_fwd_pd_t *hint_fwd_pd)
        : eltwise_pd_t(adesc, attr, hint_fwd_pd) {}

    bool set_default_formats_common() {
        return IMPLICATION(dst_md_.format_kind == format_kind::any,
                memory_desc_init_by_md_and_dt(
                        dst_md_, src_md_, dst_md_.data_type)
                        == status::success);
    }
};

struct eltwise_bwd_pd_t : public eltwise_pd_t {
    typedef eltwise_bwd_pd_t base_class;
    typedef eltwise_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (use_dst() ? arg == DNNL_ARG_DST : arg == DNNL_ARG_SRC)
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DIFF_DST) return arg_usage_t::input;
        if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0, user_input);
            default: return eltwise_pd_t::arg_md(arg);
        }
    }

    // To avoid additional logic in implementations
    const memory_desc_t *data_md(int index = 0) const {
        return use_dst() ? dst_md(index) : src_md(index);
    }
    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0 && !use_dst())
            return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0 && use_dst())
            return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_dst_desc : &diff_dst_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *diff_src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_src_desc : &diff_src_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 2; }
    int n_outputs() const override { return 1; }

    static bool eltwise_preserves_zero(
            alg_kind_t alg, float alpha, float beta) {
        // Unlike forward counterpart, bwd works on two tensors (with same formats)
        // and if alg moves zero to non-zero, it's fine, because diff_dst will
        // still have zeros in padding and multiplication of zero and non-zero
        // gives desired result. However, it doesn't work in case of special fp
        // values which are NaN or infinity which give NaN when multiplying on
        // zero, so excluding all those algs from here.
        using namespace alg_kind;
        using namespace utils;
        return one_of(alg, eltwise_abs, eltwise_clip, eltwise_clip_v2,
                       eltwise_elu, eltwise_exp, eltwise_gelu_erf,
                       eltwise_gelu_tanh, eltwise_hardsigmoid, eltwise_linear,
                       eltwise_logistic, eltwise_mish, eltwise_relu,
                       eltwise_soft_relu, eltwise_square, eltwise_swish,
                       eltwise_tanh)
                || one_of(alg, eltwise_elu_use_dst_for_bwd,
                        eltwise_exp_use_dst_for_bwd,
                        eltwise_logistic_use_dst_for_bwd,
                        eltwise_relu_use_dst_for_bwd,
                        eltwise_tanh_use_dst_for_bwd,
                        eltwise_clip_v2_use_dst_for_bwd)
                || (alg == eltwise_pow && beta >= 1);
    }

    bool is_zero_preserved() const {
        return eltwise_preserves_zero(desc_.alg_kind, desc_.alpha, desc_.beta);
    }

protected:
    memory_desc_t diff_src_md_;
    memory_desc_t diff_dst_md_;

    eltwise_bwd_pd_t(const eltwise_desc_t *adesc, const primitive_attr_t *attr,
            const eltwise_fwd_pd_t *hint_fwd_pd)
        : eltwise_pd_t(adesc, attr, hint_fwd_pd)
        , diff_src_md_(desc_.diff_src_desc)
        , diff_dst_md_(desc_.diff_dst_desc) {}

    bool set_default_formats_common() {
        return IMPLICATION(diff_dst_md_.format_kind == format_kind::any,
                       memory_desc_init_by_md_and_dt(
                               diff_dst_md_, *data_md(), diff_dst_md_.data_type)
                               == status::success)
                && IMPLICATION(diff_src_md_.format_kind == format_kind::any,
                        memory_desc_init_by_md_and_dt(diff_src_md_, *data_md(),
                                diff_src_md_.data_type)
                                == status::success);
    }
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
