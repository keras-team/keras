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

#ifndef COMMON_DECONVOLUTION_PD_HPP
#define COMMON_DECONVOLUTION_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "convolution_pd.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VDISPATCH_DECONVOLUTION(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, deconvolution, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_DECONVOLUTION_IC(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, deconvolution, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)

#define VDISPATCH_DECONVOLUTION_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, deconvolution, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct deconvolution_fwd_pd_t;

struct deconvolution_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::deconvolution;

    const deconvolution_desc_t *desc() const { return &desc_; }
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
            case query::strides:
                *(const dims_t **)result = &desc()->strides;
                break;
            case query::dilations:
                *(const dims_t **)result = &desc()->dilates;
                break;
            case query::padding_l:
                *(const dims_t **)result = &desc()->padding[0];
                break;
            case query::padding_r:
                *(const dims_t **)result = &desc()->padding[1];
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common deconv aux functions (note that conv_desc_t == deconv_desc_t) */

    dim_t MB() const { return invariant_src_md()->dims[0]; }

    dim_t IC() const { return invariant_src_md()->dims[1]; }
    dim_t OC() const { return invariant_dst_md()->dims[1]; }
    dim_t G() const { return with_groups() ? invariant_wei_md()->dims[0] : 1; }

    dim_t ID() const {
        return ndims() >= 5 ? invariant_src_md()->dims[ndims() - 3] : 1;
    }
    dim_t IH() const {
        return ndims() >= 4 ? invariant_src_md()->dims[ndims() - 2] : 1;
    }
    dim_t IW() const { return invariant_src_md()->dims[ndims() - 1]; }

    dim_t OD() const {
        return ndims() >= 5 ? invariant_dst_md()->dims[ndims() - 3] : 1;
    }
    dim_t OH() const {
        return ndims() >= 4 ? invariant_dst_md()->dims[ndims() - 2] : 1;
    }
    dim_t OW() const { return invariant_dst_md()->dims[ndims() - 1]; }

    dim_t KD() const {
        const int w_ndims = ndims() + with_groups();
        return ndims() >= 5 ? invariant_wei_md()->dims[w_ndims - 3] : 1;
    }
    dim_t KH() const {
        const int w_ndims = ndims() + with_groups();
        return ndims() >= 4 ? invariant_wei_md()->dims[w_ndims - 2] : 1;
    }
    dim_t KW() const {
        const int w_ndims = ndims() + with_groups();
        return invariant_wei_md()->dims[w_ndims - 1];
    }

    dim_t KSD() const { return ndims() >= 5 ? desc_.strides[ndims() - 5] : 1; }
    dim_t KSH() const { return ndims() >= 4 ? desc_.strides[ndims() - 4] : 1; }
    dim_t KSW() const { return desc_.strides[ndims() - 3]; }

    dim_t KDD() const { return ndims() >= 5 ? desc_.dilates[ndims() - 5] : 0; }
    dim_t KDH() const { return ndims() >= 4 ? desc_.dilates[ndims() - 4] : 1; }
    dim_t KDW() const { return desc_.dilates[ndims() - 3]; }

    dim_t padFront() const {
        return ndims() >= 5 ? desc_.padding[0][ndims() - 5] : 0;
    }
    dim_t padBack() const {
        return ndims() >= 5 ? desc_.padding[1][ndims() - 5] : 0;
    }
    dim_t padT() const {
        return ndims() >= 4 ? desc_.padding[0][ndims() - 4] : 0;
    }
    dim_t padB() const {
        return ndims() >= 4 ? desc_.padding[1][ndims() - 4] : 0;
    }
    dim_t padL() const { return desc_.padding[0][ndims() - 3]; }
    dim_t padR() const { return desc_.padding[1][ndims() - 3]; }

    bool with_bias() const {
        auto *bia_d = desc()->prop_kind == prop_kind::backward_weights
                ? &desc()->diff_bias_desc
                : &desc()->bias_desc;
        return !memory_desc_wrapper(bia_d).is_zero();
    }

    bool with_groups() const {
        return invariant_wei_md()->ndims == ndims() + 1;
    }

    int ndims() const { return invariant_src_md()->ndims; }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    bool has_zero_dim_memory() const {
        const auto s_d = memory_desc_wrapper(*invariant_src_md());
        const auto d_d = memory_desc_wrapper(*invariant_dst_md());
        return s_d.has_zero_dim() || d_d.has_zero_dim();
    }

protected:
    deconvolution_desc_t desc_;
    const deconvolution_fwd_pd_t *hint_fwd_pd_;

    deconvolution_pd_t(const deconvolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const deconvolution_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}

    bool attr_scales_ok(const std::vector<int> &supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) const {
        bool ok = attr()->scales_.has_default_values(supported_args);
        for (int arg : supported_args) {
            const auto &mask = attr()->scales_.get(arg).mask_;
            if (arg == DNNL_ARG_WEIGHTS)
                ok = ok && (mask == 0 || mask == (with_groups() ? 3 : 1));
            else
                ok = ok && (mask == 0);
        }
        return ok;
    }

    /// This function body follows
    /// convolution_pd_t::expect_data_types(...)
    bool expect_data_types(data_type_t src_dt, data_type_t wei_dt,
            data_type_t bia_dt, data_type_t dst_dt) const {
        bool ok = true
                && (src_dt == data_type::undef
                        || invariant_src_md()->data_type == src_dt)
                && (wei_dt == data_type::undef
                        || invariant_wei_md()->data_type == wei_dt)
                && (dst_dt == data_type::undef
                        || invariant_dst_md()->data_type == dst_dt);
        if (with_bias() && bia_dt != data_type::undef)
            ok = ok && invariant_bia_md()->data_type == bia_dt;
        return ok;
    }
};

struct deconvolution_fwd_pd_t : public deconvolution_pd_t {
    typedef deconvolution_fwd_pd_t base_class;
    typedef deconvolution_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_WEIGHTS))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_BIAS && with_bias()) return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_WEIGHTS: return weights_md(0);
            case DNNL_ARG_BIAS: return weights_md(1);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return deconvolution_pd_t::arg_md(arg);
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
    const memory_desc_t *weights_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->weights_desc : &weights_md_;
        if (index == 1) return user_input ? &desc()->bias_desc : &bias_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override {
        return 2 + with_bias() + n_prelu_po_inputs() + n_binary_po_inputs();
    }
    int n_outputs() const override { return 1; }

protected:
    memory_desc_t src_md_;
    memory_desc_t weights_md_;
    memory_desc_t bias_md_;
    memory_desc_t dst_md_;

    deconvolution_fwd_pd_t(const deconvolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const deconvolution_fwd_pd_t *hint_fwd_pd)
        : deconvolution_pd_t(adesc, attr, hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , weights_md_(desc_.weights_desc)
        , bias_md_(desc_.bias_desc)
        , dst_md_(desc_.dst_desc) {}
};

struct deconvolution_bwd_data_pd_t : public deconvolution_pd_t {
    typedef deconvolution_bwd_data_pd_t base_class;
    typedef deconvolution_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_WEIGHTS, DNNL_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_WEIGHTS: return weights_md(0);
            case DNNL_ARG_BIAS: return weights_md(1);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0, user_input);
            default: return deconvolution_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *diff_src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_src_desc : &diff_src_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_dst_desc : &diff_dst_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *weights_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->weights_desc : &weights_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 2; }
    int n_outputs() const override { return 1; }

protected:
    memory_desc_t diff_src_md_;
    memory_desc_t weights_md_;
    memory_desc_t diff_dst_md_;

    deconvolution_bwd_data_pd_t(const deconvolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const deconvolution_fwd_pd_t *hint_fwd_pd)
        : deconvolution_pd_t(adesc, attr, hint_fwd_pd)
        , diff_src_md_(desc_.diff_src_desc)
        , weights_md_(desc_.weights_desc)
        , diff_dst_md_(desc_.diff_dst_desc) {}
};

struct deconvolution_bwd_weights_pd_t : public deconvolution_pd_t {
    typedef deconvolution_bwd_weights_pd_t base_class;
    typedef deconvolution_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DIFF_WEIGHTS) return arg_usage_t::output;

        if (arg == DNNL_ARG_DIFF_BIAS && with_bias())
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DIFF_WEIGHTS: return diff_weights_md(0);
            case DNNL_ARG_DIFF_BIAS: return diff_weights_md(1);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0, user_input);
            default: return deconvolution_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_dst_desc : &diff_dst_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *diff_weights_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_weights_desc : &diff_weights_md_;
        if (index == 1)
            return user_input ? &desc()->diff_bias_desc : &diff_bias_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 2; }
    int n_outputs() const override { return 1 + with_bias(); }

protected:
    memory_desc_t src_md_;
    memory_desc_t diff_weights_md_;
    memory_desc_t diff_bias_md_;
    memory_desc_t diff_dst_md_;

    deconvolution_bwd_weights_pd_t(const deconvolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const deconvolution_fwd_pd_t *hint_fwd_pd)
        : deconvolution_pd_t(adesc, attr, hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , diff_weights_md_(desc_.diff_weights_desc)
        , diff_bias_md_(desc_.diff_bias_desc)
        , diff_dst_md_(desc_.diff_dst_desc) {}
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
