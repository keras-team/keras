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

#ifndef COMMON_INNER_PRODUCT_PD_HPP
#define COMMON_INNER_PRODUCT_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VDISPATCH_INNER_PRODUCT(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, inner_product, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_INNER_PRODUCT_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, inner_product, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

status_t ip_desc_init(inner_product_desc_t *ip_desc, prop_kind_t prop_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc);

struct inner_product_fwd_pd_t;

struct inner_product_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::inner_product;

    inner_product_pd_t(const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}

    const inner_product_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common inner_product aux functions */

    dim_t MB() const { return invariant_src_md()->dims[0]; }
    dim_t IC() const { return invariant_src_md()->dims[1]; }
    dim_t OC() const { return invariant_dst_md()->dims[1]; }

    dim_t ID() const {
        return ndims() >= 5 ? invariant_src_md()->dims[ndims() - 3] : 1;
    }
    dim_t IH() const {
        return ndims() >= 4 ? invariant_src_md()->dims[ndims() - 2] : 1;
    }
    dim_t IW() const {
        return ndims() >= 3 ? invariant_src_md()->dims[ndims() - 1] : 1;
    }

    dim_t OD() const {
        return ndims() >= 5 ? invariant_dst_md()->dims[ndims() - 3] : 1;
    }
    dim_t OH() const {
        return ndims() >= 4 ? invariant_dst_md()->dims[ndims() - 2] : 1;
    }
    dim_t OW() const {
        return ndims() >= 3 ? invariant_dst_md()->dims[ndims() - 1] : 1;
    }

    dim_t KD() const {
        return ndims() >= 5 ? invariant_wei_md()->dims[ndims() - 3] : 1;
    }
    dim_t KH() const {
        return ndims() >= 4 ? invariant_wei_md()->dims[ndims() - 2] : 1;
    }
    dim_t KW() const {
        return ndims() >= 3 ? invariant_wei_md()->dims[ndims() - 1] : 1;
    }

    dim_t IC_total() const {
        return utils::array_product(&invariant_src_md()->dims[1], ndims() - 1);
    }

    dim_t IC_total_padded() const {
        auto src_d = desc()->prop_kind == prop_kind::backward_data
                ? memory_desc_wrapper(diff_src_md())
                : memory_desc_wrapper(src_md());
        assert(src_d.is_blocking_desc());
        if (!src_d.is_blocking_desc()) return -1;
        return utils::array_product(src_d.padded_dims() + 1, ndims() - 1);
    }

    int ndims() const { return invariant_src_md()->ndims; }

    bool with_bias() const {
        auto *bia_d = desc()->prop_kind == prop_kind::backward_weights
                ? &desc()->diff_bias_desc
                : &desc()->bias_desc;
        return !memory_desc_wrapper(bia_d).is_zero();
    }

    bool has_zero_dim_memory() const {
        const auto s_d = memory_desc_wrapper(*invariant_src_md());
        const auto d_d = memory_desc_wrapper(*invariant_dst_md());
        return s_d.has_zero_dim() || d_d.has_zero_dim();
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

protected:
    inner_product_desc_t desc_;
    const inner_product_fwd_pd_t *hint_fwd_pd_;

    bool set_default_formats_common_template(memory_desc_t &src_md,
            format_tag_t src_tag, memory_desc_t &wei_md, format_tag_t wei_tag,
            memory_desc_t &dst_md, format_tag_t dst_tag,
            memory_desc_t &bia_md) {
        using namespace format_tag;

#define IS_OK(f) \
    do { \
        if ((f) != status::success) return false; \
    } while (0)
        if (src_md.format_kind == format_kind::any
                && !utils::one_of(src_tag, any, undef))
            IS_OK(memory_desc_init_by_tag(src_md, src_tag));
        if (dst_md.format_kind == format_kind::any
                && !utils::one_of(dst_tag, any, undef))
            IS_OK(memory_desc_init_by_tag(dst_md, dst_tag));
        if (wei_md.format_kind == format_kind::any
                && !utils::one_of(wei_tag, any, undef))
            IS_OK(memory_desc_init_by_tag(wei_md, wei_tag));
        if (with_bias() && bia_md.format_kind == format_kind::any)
            IS_OK(memory_desc_init_by_tag(bia_md, x));
#undef IS_OK

        return true;
    }

    bool expect_data_types(data_type_t src_dt, data_type_t wei_dt,
            data_type_t bia_dt, data_type_t dst_dt, data_type_t acc_dt) const {
        bool ok = true
                && (src_dt == data_type::undef
                        || invariant_src_md()->data_type == src_dt)
                && (wei_dt == data_type::undef
                        || invariant_wei_md()->data_type == wei_dt)
                && (dst_dt == data_type::undef
                        || invariant_dst_md()->data_type == dst_dt)
                && (acc_dt == data_type::undef
                        || desc_.accum_data_type == acc_dt);
        if (with_bias() && bia_dt != data_type::undef)
            ok = ok && invariant_bia_md()->data_type == bia_dt;
        return ok;
    }

    bool attr_scales_ok(const std::vector<int> &supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) const {
        bool ok = attr()->scales_.has_default_values(supported_args);
        for (auto arg : supported_args) {
            int mask = attr()->scales_.get(arg).mask_;
            if (arg == DNNL_ARG_WEIGHTS)
                ok = ok && (mask == 0 || mask == (1 << 0));
            else
                ok = ok && (mask == 0);
        }
        return ok;
    }
};

struct inner_product_fwd_pd_t : public inner_product_pd_t {
    typedef inner_product_fwd_pd_t base_class;
    typedef inner_product_fwd_pd_t hint_class;

    inner_product_fwd_pd_t(const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_pd_t(adesc, attr, hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , weights_md_(desc_.weights_desc)
        , bias_md_(desc_.bias_desc)
        , dst_md_(desc_.dst_desc) {}

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
            default: return inner_product_pd_t::arg_md(arg);
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
        return 2 + with_bias() + n_binary_po_inputs() + n_prelu_po_inputs();
    }
    int n_outputs() const override { return 1; }

protected:
    memory_desc_t src_md_;
    memory_desc_t weights_md_;
    memory_desc_t bias_md_;
    memory_desc_t dst_md_;

    bool set_default_formats_common(
            format_tag_t src_tag, format_tag_t wei_tag, format_tag_t dst_tag) {
        return set_default_formats_common_template(src_md_, src_tag,
                weights_md_, wei_tag, dst_md_, dst_tag, bias_md_);
    }
};

struct inner_product_bwd_data_pd_t : public inner_product_pd_t {
    typedef inner_product_bwd_data_pd_t base_class;
    typedef inner_product_fwd_pd_t hint_class;

    inner_product_bwd_data_pd_t(const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_pd_t(adesc, attr, hint_fwd_pd)
        , diff_src_md_(desc_.diff_src_desc)
        , weights_md_(desc_.weights_desc)
        , diff_dst_md_(desc_.diff_dst_desc) {}

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
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0, user_input);
            default: return inner_product_pd_t::arg_md(arg);
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

    bool set_default_formats_common(format_tag_t diff_src_tag,
            format_tag_t wei_tag, format_tag_t diff_dst_tag) {
        memory_desc_t dummy_md;
        return set_default_formats_common_template(diff_src_md_, diff_src_tag,
                weights_md_, wei_tag, diff_dst_md_, diff_dst_tag, dummy_md);
    }
};

struct inner_product_bwd_weights_pd_t : public inner_product_pd_t {
    typedef inner_product_bwd_weights_pd_t base_class;
    typedef inner_product_fwd_pd_t hint_class;

    inner_product_bwd_weights_pd_t(const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_pd_t(adesc, attr, hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , diff_weights_md_(desc_.diff_weights_desc)
        , diff_bias_md_(desc_.diff_bias_desc)
        , diff_dst_md_(desc_.diff_dst_desc) {}

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
            default: return inner_product_pd_t::arg_md(arg);
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

    bool set_default_formats_common(format_tag_t src_tag,
            format_tag_t diff_wei_tag, format_tag_t diff_dst_tag) {
        return set_default_formats_common_template(src_md_, src_tag,
                diff_weights_md_, diff_wei_tag, diff_dst_md_, diff_dst_tag,
                diff_bias_md_);
    }
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
